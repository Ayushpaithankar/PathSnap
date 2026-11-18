"""Decode Member 1's real trip 00000052 handoff tables."""
import csv
import json
import argparse
import os
from pathlib import Path
from pathsnap.core import RoadGraph,baseline,continuity,decode,load_trip,validate_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path(os.getenv('PATHSNAP_DATA_ROOT', 'Student folder')))
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args()
    track,nodes,arcs=load_trip(args.data_root,'student_test','00000052')
    with (args.results/'example_observations.csv').open() as f: obs=list(csv.DictReader(f))
    assert len(track)==len(obs)
    for i,o in enumerate(obs):
        assert o['trip_id']=='00000052' and int(o['observation_index'])==i
        assert abs(float(o['timestamp'])-track[i]['timestamp'])<1e-6
    with (args.results/'example_candidates.csv').open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        r['observation_index']=int(r['observation_index']);r['arc_id']=int(r['arc_id'])
    graph=RoadGraph(nodes,arcs)
    grouped=validate_candidates(track,arcs,rows)
    raw=baseline(track,grouped)
    chosen,connecting,warnings=decode(track,graph,grouped)
    result={'trip_id':'00000052','observations':len(track),'candidate_rows':len(rows),
            'baseline_continuity':continuity(raw,graph),'decoded_continuity':continuity(chosen,graph),
            'unmatched_observations':[i for i,a in enumerate(chosen) if a is None],
            'route_breaks':warnings,
            'predictions':[{'observation_index':i,'timestamp':track[i]['timestamp'],'baseline_arc_id':raw[i],
                            'decoded_arc_id':chosen[i],'connecting_arc_ids':connecting[i]} for i in range(len(track))]}
    path=args.results/'checkpoint_00000052.json'
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))
    print('first_10_predictions:',json.dumps(result['predictions'][:10]))
    print('saved:',path)

if __name__=='__main__':main()
