#!/usr/bin/env python3
"""Stable CLI for independent character layers, file transfer and garment re-fit."""
import argparse
import json
import sys
from pathlib import Path
from layers import (create_project,add_part,load_project,transfer_part,set_visibility,
                    set_body_mask,export_parts_preview,inventory_scene,extract_scene_object,prepare_part_skin,inventory_components,extract_component)


def main(argv=None):
    p=argparse.ArgumentParser(description='ViewSlice v0.6 Layer & Wardrobe CLI')
    s=p.add_subparsers(dest='cmd',required=True)
    mk=s.add_parser('new');mk.add_argument('--project',required=True);mk.add_argument('--body',required=True);mk.add_argument('--name',default='Character');mk.add_argument('--profile',default='humanoid');mk.add_argument('--axis',choices=['y','z'],default='y')
    ap=s.add_parser('add');ap.add_argument('--project',required=True);ap.add_argument('--mesh',required=True);ap.add_argument('--slot',choices=['hair','clothing','accessory','extra'],required=True);ap.add_argument('--subtype',default='other');ap.add_argument('--name',default='');ap.add_argument('--coverage',default='')
    ls=s.add_parser('list');ls.add_argument('--project',required=True)
    tx=s.add_parser('wear');tx.add_argument('--donor',required=True);tx.add_argument('--receiver',required=True);tx.add_argument('--part',required=True);tx.add_argument('--strength',type=float,default=1.0);tx.add_argument('--clearance',type=float,default=0.006);tx.add_argument('--no-collision',action='store_true')
    vis=s.add_parser('visible');vis.add_argument('--project',required=True);vis.add_argument('--part',required=True);vis.add_argument('--show',choices=['on','off'],required=True)
    mk2=s.add_parser('mask');mk2.add_argument('--project',required=True);mk2.add_argument('--part',required=True);mk2.add_argument('--regions',required=True)
    prev=s.add_parser('preview');prev.add_argument('--project',required=True);prev.add_argument('--out',required=True);prev.add_argument('--apply-masks',action='store_true')
    skin=s.add_parser('skin-draft');skin.add_argument('--project',required=True);skin.add_argument('--part',required=True);skin.add_argument('--rig-plan',default='')
    inv=s.add_parser('inspect-scene');inv.add_argument('--input',required=True)
    ex=s.add_parser('extract');ex.add_argument('--input',required=True);ex.add_argument('--index',type=int,required=True);ex.add_argument('--out',required=True)
    comp=s.add_parser('inspect-components');comp.add_argument('--input',required=True);comp.add_argument('--min-faces',type=int,default=1)
    ce=s.add_parser('extract-component');ce.add_argument('--input',required=True);ce.add_argument('--index',type=int,required=True);ce.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.cmd=='new': result=create_project(a.project,a.body,a.name,a.profile,a.axis)
    elif a.cmd=='add':result=add_part(a.project,a.mesh,a.slot,a.subtype,a.name or None,[x.strip() for x in a.coverage.split(',') if x.strip()])
    elif a.cmd=='list':result=load_project(a.project)
    elif a.cmd=='wear':
        part,stats=transfer_part(a.donor,a.part,a.receiver,strength=a.strength,clearance=a.clearance,collision=not a.no_collision)
        result={'new_part':part,'fit_stats':stats}
    elif a.cmd=='visible':set_visibility(a.project,a.part,a.show=='on');result={'changed':True}
    elif a.cmd=='mask':set_body_mask(a.project,a.part,[x.strip() for x in a.regions.split(',') if x.strip()]);result={'saved':True}
    elif a.cmd=='preview':result={'preview':str(export_parts_preview(a.project,a.out,a.apply_masks))}
    elif a.cmd=='skin-draft':result=prepare_part_skin(a.project,a.part,a.rig_plan or None)
    elif a.cmd=='inspect-scene':result=inventory_scene(a.input)
    elif a.cmd=='extract':result={'file':str(extract_scene_object(a.input,a.index,a.out))}
    elif a.cmd=='inspect-components':result=inventory_components(a.input,a.min_faces)
    elif a.cmd=='extract-component':result={'file':str(extract_component(a.input,a.index,a.out))}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc: print('ERROR: '+str(exc),file=sys.stderr);sys.exit(1)
