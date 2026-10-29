"""Explicit, reversible-on-paper coordinate-frame adapter for a static 3D part.

We NEVER guess a character's facing direction from geometry. Front information
comes from user metadata. Transform into ViewSlice canonical: +Y up, +Z front.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from scene_core import load_static_scene,export_static_scene

CANONICAL={'up':'+Y','front':'+Z','character_left':'+X','units':'m'}

def align_scene(scene, source_front='-Z', scale=1.0, offset=(0,0,0)):
    """Rotate around bbox center, then uniformly scale and translate.
    No re-mesh or UV deletion. Scene graph receives a root transform per node.
    """
    if source_front not in ('+Z','-Z','+X','-X'):
        raise ValueError('source_front는 +Z, -Z, +X, -X 중 하나입니다')
    if not 0<scale<100:raise ValueError('scale must be > 0 and <100')
    center=np.asarray(scene.bounds).mean(axis=0)
    degrees={'+Z':0,'-Z':180,'+X':-90,'-X':90}[source_front]
    rot=trimesh.transformations.rotation_matrix(np.deg2rad(degrees),[0,1,0],point=center)
    transform=trimesh.transformations.translation_matrix(np.asarray(offset,dtype=float)) @ (
        trimesh.transformations.translation_matrix(center) @
        trimesh.transformations.scale_matrix(scale) @
        trimesh.transformations.translation_matrix(-center) @rot)
    result=scene.copy()
    result.apply_transform(transform)
    return result,{'source_front':source_front,'target_frame':CANONICAL,'rotation_y_deg':degrees,
                   'uniform_scale':scale,'offset_m':[float(n) for n in offset],
                   'note':'정적 메시 뷰 변환만 수행. 연결/리깅/정확한 목 정렬은 수행하지 않음'}

def main():
    p=argparse.ArgumentParser(description='Static part frame normalization; original is never overwritten')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source-front',choices=['+Z','-Z','+X','-X'],required=True)
    p.add_argument('--scale',type=float,default=1.0)
    p.add_argument('--offset',nargs=3,type=float,default=[0,0,0],metavar=('X','Y','Z'))
    a=p.parse_args()
    src=Path(a.input).resolve();dst=Path(a.output).resolve()
    if src==dst or dst.exists():p.error('원본이나 기존 파일 덮어쓰기 금지')
    sc=load_static_scene(src)
    out,meta=align_scene(sc,a.source_front,a.scale,a.offset)
    export_static_scene(out,dst)
    dst.with_name(dst.name+'.frame.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(dst),'frame':meta},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
