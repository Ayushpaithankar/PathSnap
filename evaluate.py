"""Run genuine, labeled development diagnostics after integrating the teammate's scorer."""
import argparse
import csv
import importlib
import os
from pathlib import Path
from pathsnap.core import RoadGraph,baseline,continuity,decode,load_trip,validate_candidates

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--partition',choices=['train','student_test'],default='student_test')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    spec=os.environ['PATHSNAP_SCORER']
    module,name=spec.split(':',1);score=getattr(importlib.import_module(module),name)
    root=Path(os.getenv('PATHSNAP_DATA_ROOT','/Users/ayushpaithankar/Downloads/Student folder'))
    result=[]
    for directory in sorted((root/a.partition).iterdir()):
        if not directory.is_dir():continue
        trip_id=directory.name
        track,nodes,arcs=load_trip(root,a.partition,trip_id)
        rows=score(trip_id,a.partition,track,nodes,arcs)
        if any(r['trip_id']!=trip_id for r in rows):raise ValueError('Trip mismatch')
        group=validate_candidates(track,arcs,rows);graph=RoadGraph(nodes,arcs)
        simple=baseline(track,group);decoded,links,warnings=decode(track,graph,group)
        reference=[int(x) for x in (directory/(trip_id+'.route')).read_text().split()]
        ref=set(reference)
        record={'trip_id':trip_id,'observations':len(track),'reference_arcs':len(ref),'missing_candidates':sum(i not in group for i in range(len(track))),'segments_or_gaps':len(warnings)}
        for label,ids in [('baseline',simple),('decoded',decoded)]:
            c=continuity(ids,graph)
            record[label+'_direct_continuity']=c['rate']
            record[label+'_reference_arc_recall']=len(ref.intersection(x for x in ids if x is not None))/len(ref) if ref else ''
        result.append(record)
    with open(a.output,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=result[0].keys());writer.writeheader();writer.writerows(result)
    print('Wrote',len(result),'real trip diagnostics to',a.output)
    print('These measures are not observation-level accuracy or F1.')

if __name__=='__main__':main()
