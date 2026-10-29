#!/usr/bin/env python3
"""Safe external 4-bone weight-draft lineage propagation. Never overwrite.
Example:
 python skin_lineage_cli.py --source-skin input.npz --lineage r001.glb.lineage.npz \
   --geometry hm08_body/primitive_0 --output r001.weights.npz
"""
import argparse,json
from pathlib import Path
import numpy as np
from lineage_io import read_round_lineage,propagate_skin_weights

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--source-skin',required=True)
    p.add_argument('--lineage',required=True)
    p.add_argument('--geometry',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    dst=Path(a.output)
    if dst.exists():raise FileExistsError(dst)
    with np.load(a.source_skin,allow_pickle=False) as d:ids=d['indices'];weights=d['weights']
    lineage=read_round_lineage(a.lineage)
    if a.geometry not in lineage:raise ValueError('계보에 해당 geometry 이름이 없습니다')
    ids_out,w_out=propagate_skin_weights(ids,weights,lineage[a.geometry])
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('xb') as f:np.savez_compressed(f,indices=ids_out,weights=w_out)
    print(json.dumps({'output':str(dst),'before':len(ids),'after':len(ids_out),'max_influences':4}))
if __name__=='__main__':main()
