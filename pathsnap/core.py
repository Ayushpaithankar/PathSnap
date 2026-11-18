import heapq
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

EARTH_M = 6371008.8

def metres(a, b):
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlat, dlon = lat2-lat1, lon2-lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*EARTH_M*math.asin(min(1, math.sqrt(h)))

PARTITIONS = {'train', 'student_test', 'evaluator_test'}
TRIP_EXTENSIONS = ('.track', '.nodes', '.arcs')


def valid_trip_id(trip_id):
    return len(trip_id) == 8 and trip_id.isascii() and trip_id.isdigit()


def trip_stem(partition, trip_id):
    if partition not in PARTITIONS or not valid_trip_id(trip_id):
        raise ValueError('Invalid partition or trip ID')
    return f'{partition}/{trip_id}/{trip_id}'


def list_trip_ids(root, partition):
    if partition not in PARTITIONS:
        raise ValueError('Invalid partition')
    root = Path(root)
    if root.is_file():
        with ZipFile(root) as archive:
            files = set(archive.namelist())
        prefix = f'Student folder/{partition}/'
        ids = {name[len(prefix):].split('/')[0] for name in files if name.startswith(prefix)}
        return sorted(t for t in ids if valid_trip_id(t) and
                      all(f'{prefix}{t}/{t}{ext}' in files for ext in TRIP_EXTENSIONS))
    folder = root / partition
    return sorted(p.name for p in folder.glob('*/') if valid_trip_id(p.name) and
                  all((p / (p.name + ext)).is_file() for ext in TRIP_EXTENSIONS))


def trip_files(root, stem, extensions):
    root = Path(root)
    if root.is_file():
        try:
            with ZipFile(root) as archive:
                return [archive.read('Student folder/' + stem + ext).decode('ascii') for ext in extensions]
        except KeyError as exc:
            raise FileNotFoundError(stem) from exc
    return [(root / (stem + ext)).read_text(encoding='ascii') for ext in extensions]


def load_reference(root, partition, trip_id):
    try:
        raw, = trip_files(root, trip_stem(partition, trip_id), ('.route',))
    except FileNotFoundError:
        return None
    return [int(token) for token in raw.split()]


def load_trip(root, partition, trip_id):
    track_text, node_text, arc_text = trip_files(root, trip_stem(partition, trip_id), TRIP_EXTENSIONS)
    def rows(content):
        return [line.split() for line in content.splitlines() if line.strip()]
    nodes = [tuple(map(float,row)) for row in rows(node_text)]
    arcs = [tuple(map(int,row)) for row in rows(arc_text)]
    track = [dict(index=i,lon=float(r[0]),lat=float(r[1]),timestamp=float(r[2])) for i,r in enumerate(rows(track_text))]
    if not (track and nodes and arcs):
        raise ValueError('Trip requires nonempty track, nodes and arcs')
    if any(len(r)!=2 for r in arcs) or any(not 0<=u<len(nodes) or not 0<=v<len(nodes) for u,v in arcs):
        raise ValueError('Invalid arc endpoint')
    return track,nodes,arcs

class RoadGraph:
    def __init__(self,nodes,arcs):
        self.nodes,self.arcs=nodes,arcs
        self.lengths=[metres(nodes[u],nodes[v]) for u,v in arcs]
        self.out=defaultdict(list)
        self.pairs=defaultdict(list)
        self._path_cache={}
        for i,(u,v) in enumerate(arcs):
            self.out[u].append(i)
            self.pairs[(u,v)].append(i)

    def path(self,start,end,limit_m=10000):
        if start==end:return (0.,[])
        key=(start,end)
        if key in self._path_cache:
            distance, links = self._path_cache[key]
            return (distance, links) if distance <= limit_m else (float('inf'), [])
        queue=[(0.,start)]
        dist={start:0.}; previous={}
        while queue:
            d,u=heapq.heappop(queue)
            if d!=dist[u]:continue
            if u==end:
                path=[]
                while u!=start:
                    p,arc=previous[u];path.append(arc);u=p
                result=(d,list(reversed(path)))
                if len(self._path_cache)<100000:self._path_cache[key]=result
                return result
            if d>limit_m:break
            for arc in self.out[u]:
                v=self.arcs[arc][1];nd=d+self.lengths[arc]
                if nd<dist.get(v,float('inf')) and nd<=limit_m:
                    dist[v]=nd;previous[v]=(u,arc);heapq.heappush(queue,(nd,v))
        return float('inf'),[]

    def transition(self,a,b,limit_m):
        """Directed travel from the end of arc a to the start of arc b."""
        d,connector=self.path(self.arcs[a][1],self.arcs[b][0],limit_m)
        return d,connector

    def diagnostics(self):
        pairs=self.pairs
        return dict(nodes=len(self.nodes),arcs=len(self.arcs),repeated_directed_pairs=sum(len(v)-1 for v in pairs.values()),opposite_direction_pairs=sum(1 for u,v in pairs if u<v and (v,u) in pairs))

def validate_candidates(track,arcs,rows):
    grouped=defaultdict(list);seen=set()
    for r in rows:
        i=r['observation_index'];a=r['arc_id'];p=float(r['probability'])
        if not isinstance(i,int) or not 0<=i<len(track) or not isinstance(a,int) or not 0<=a<len(arcs) or (i,a) in seen:
            raise ValueError('Invalid or duplicate candidate index/arc ID')
        if r['trip_id'] not in (None,'') and not isinstance(r['trip_id'],str):raise ValueError('Invalid trip ID')
        if not math.isfinite(p) or not 0<=p<=1 or not math.isclose(float(r['timestamp']),track[i]['timestamp'],abs_tol=1e-6):
            raise ValueError('Invalid candidate score or timestamp')
        for key in ('snapped_lon','snapped_lat','distance_m'):
            if not math.isfinite(float(r[key])):raise ValueError('Nonfinite '+key)
        if float(r['distance_m'])<0:raise ValueError('Negative distance')
        seen.add((i,a));grouped[i].append(r)
    return grouped

def arc_fraction(graph,row):
    """Approximate snapped position as a fraction along its directed arc."""
    u,v=graph.arcs[row['arc_id']]
    lon1,lat1=graph.nodes[u];lon2,lat2=graph.nodes[v]
    scale=math.cos(math.radians((lat1+lat2)/2))
    dx=(lon2-lon1)*scale;dy=lat2-lat1
    denom=dx*dx+dy*dy
    if denom<1e-20:return 0.
    return max(0.,min(1.,(((float(row['snapped_lon'])-lon1)*scale)*dx+(float(row['snapped_lat'])-lat1)*dy)/denom))

def decode(track, graph, grouped, max_gap_s=30., max_speed_mps=45., beta_m=100.,
           backward_jitter_m=8., emission_weight=1., reversal_penalty=0.,
           stationary_speed=None, bridge_gaps=False):
    """Viterbi over candidate arcs; empty sets and large gaps split runs."""
    predictions=[None]*len(track); connectors=[[] for _ in track];warnings=[]
    previous=None; scores={}; back={}; run=[];previous_fractions={}
    def finish():
        if not run:return
        end=max(scores,key=scores.get)
        for i in reversed(run):
            predictions[i]=end
            prev,link=back[(i,end)];connectors[i]=link
            end=prev
            if end is None:break
    for i,obs in enumerate(track):
        candidates=grouped.get(i,[])
        gap=obs['timestamp']-track[previous]['timestamp'] if previous is not None else 0
        if not candidates or (previous is not None and (gap<=0 or (gap>max_gap_s and not bridge_gaps))):
            finish();scores={};run=[];previous=None;previous_fractions={}
            back.clear()
            if not candidates:
                warnings.append(f'observation {i}: no candidates')
                continue
            warnings.append(f'observation {i}: time gap {gap:g}s; new segment')
        total=sum(max(float(r['probability']),1e-12) for r in candidates)
        emissions={r['arc_id']:emission_weight * math.log(max(float(r['probability']),1e-12)/total)
                   for r in candidates}
        fractions={r['arc_id']:arc_fraction(graph,r) for r in candidates}
        stationary = False
        if previous is not None:
            straight=metres((track[previous]['lon'],track[previous]['lat']),(obs['lon'],obs['lat']))
            long_gap = gap > max_gap_s
            cap = max_speed_mps * gap + (0. if long_gap else 100.)
            stationary = stationary_speed is not None and not long_gap and straight <= stationary_speed * gap
            shared = scores.keys() & fractions.keys() if stationary else set()
        updated={}; local_back={}
        for r in candidates:
            a=r['arc_id']; best=(-float('inf'),None,[])
            if previous is None:best=(emissions[a],None,[])
            else:
                for old,score in scores.items():
                    # Hold a stationary state if its previous arc is still a candidate.
                    if old in shared and old != a:
                        continue
                    old_frac=previous_fractions[old];new_frac=fractions[a]
                    progress=(new_frac-old_frac)*graph.lengths[a]
                    if old==a and (stationary or progress>=-backward_jitter_m):
                        travel=0. if stationary else max(0.,progress)
                        link=[]
                    else:
                        residual=(1-old_frac)*graph.lengths[old]
                        prefix=new_frac*graph.lengths[a]
                        distance,link=graph.transition(old,a,max(0.,cap-residual-prefix))
                        if not math.isfinite(distance):continue
                        travel=residual+distance+prefix
                    if travel>cap:continue
                    reversals = 0
                    if reversal_penalty and (old != a or link):
                        pair = [old, *link, a]
                        reversals = sum(graph.arcs[u] == graph.arcs[v][::-1]
                                        for u, v in zip(pair, pair[1:]))
                    likelihood_scale = beta_m * (math.sqrt(gap/max_gap_s) if long_gap else 1.)
                    emission = 0. if stationary and old == a else emissions[a]
                    value=score+emission-abs(travel-straight)/likelihood_scale-reversal_penalty*reversals
                    if value>best[0]:best=(value,old,link)
            if math.isfinite(best[0]):updated[a]=best[0];local_back[(i,a)]=(best[1],best[2])
        if not updated:
            finish();scores={a:emissions[a] for a in emissions};run=[i]
            back.clear()
            for a in scores:back[(i,a)]=(None,[])
            warnings.append(f'observation {i}: disconnected candidates; new segment')
        else:
            scores=updated;back.update(local_back);run.append(i)
            if bridge_gaps and previous is not None and gap > max_gap_s:
                warnings.append(f'observation {i}: time gap {gap:g}s bridged via directed route')
        previous=i
        previous_fractions=fractions
    finish()
    return predictions,connectors,warnings

def baseline(track,grouped):
    return [max(grouped[i], key=lambda r: float(r['probability']))['arc_id']
            if grouped.get(i) else None for i in range(len(track))]

def continuity(ids,graph):
    good=total=0
    for a,b in zip(ids,ids[1:]):
        if a is None or b is None:continue
        total+=1;good+=graph.arcs[a][1]==graph.arcs[b][0] or a==b
    return dict(connected=good,comparable=total,rate=good/total if total else None)

def lines(ids,connectors,graph):
    features=[]
    for i,a in enumerate(ids):
        if a is None:continue
        for arc, is_connector in [(arc, True) for arc in connectors[i]] + [(a, False)]:
            u,v=graph.arcs[arc]
            features.append(dict(type='Feature',properties=dict(observation_index=i,arc_id=arc,connector=is_connector),geometry=dict(type='LineString',coordinates=[graph.nodes[u],graph.nodes[v]])))
    return dict(type='FeatureCollection',features=features)
