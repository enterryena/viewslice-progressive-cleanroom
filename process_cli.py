#!/usr/bin/env python3
"""Independent Windows CLI; no Modly update required."""
from __future__ import annotations
import argparse
import json
import shutil
from format_io import load_geometry, export_geometry, ALL_INPUTS, ALL_OUTPUTS, RIG_OUT
import sys
from pathlib import Path
from mesh_core import read_mesh,process_mesh
from benchmark import run_benchmark,per_iteration_budget,profile_path
from rig_process import write_plan,apply_rig_blender
from rigging import PROFILES


def _guard_project_skin(path):
    """Prevent stale skin drafts from reaching PMX/GLB export entry points.

    Only applicable when the source is a managed part in a layered project.
    Independent unbound static meshes do not have a sidecar to validate.
    """
    from layers import load_project,validate_part_skin
    import warnings
    path=Path(path).resolve()
    root=path.parent.parent if path.parent.name=='assets' else None
    if root is None or not (root/'character.layers.json').is_file():return None
    project=load_project(root)
    for item in [project['body']]+project['parts']:
        if (root/item['file']).resolve()!=path or not item.get('skin_draft'):
            continue
        result=validate_part_skin(root,item['id'])
        if result.get('action')=='block' or result.get('reason') in ('no_skin_draft','legacy_draft_missing_vertex_identity'):
            raise ValueError('PMX/리깅 출력 중단: 정점 가중치의 구조적 대응이 유효하지 않습니다: '+str(result.get('reason')))
        if result.get('action')=='warn':
            warnings.warn('정점 위치 변경으로 스키닝 검증 권장: '+item['id'],RuntimeWarning)
        return result
    return None


def main(argv=None):
    a=argparse.ArgumentParser(description='ViewSlice v0.7: scene-preserving incremental editing & PMX draft export')
    a.add_argument('--input',required=True)
    a.add_argument('--output',required=True)
    a.add_argument('--mode',choices=['analyze','progressive','fit','fit_subdivide','rig_plan','rig_apply','mmd_export'],default='progressive')
    a.add_argument('--steps',type=int,choices=[1,2,3],default=1)
    a.add_argument('--reference-dir',default='');a.add_argument('--strength',type=float,default=.20)
    a.add_argument('--detail-strength',type=float,default=.12)
    a.add_argument('--target-part',default='',help='Scene geometry name to modify; required for multi-part height maps')
    a.add_argument('--vertex-mask',default='',help='Boolean per-vertex .npy mask; requires --target-part')
    a.add_argument('--height-map',default='',help='Explicit grayscale height map; requires UV (not inferred from color image)')
    a.add_argument('--height-amplitude',type=float,default=0.0,help='Full peak-to-peak displacement in model units')
    a.add_argument('--height-ratio',type=float,default=0.0,help='Full peak-to-peak displacement as a fraction of model height')
    a.add_argument('--region',choices=['all','head','torso','lower'],default='all')
    a.add_argument('--up-axis',choices=['y','z'],default='y')
    a.add_argument('--effort',type=float,default=1.,help='0.2 to 2.0; fraction of cached benchmark budget')
    a.add_argument('--rig-profile',choices=PROFILES,default='humanoid');a.add_argument('--rig-segments',type=int,default=7)
    a.add_argument('--rig-appendages',type=int,default=6);a.add_argument('--blender',default='')
    a.add_argument('--rig-plan',default='',help='Provide edited rig JSON to rig_apply (optional)')
    a.add_argument('--mmd-scale',type=float,default=12.5,help='PMX-only geometry scale, default 12.5')
    a.add_argument('--mmd-name',default='',help='MMD model display name')
    a.add_argument('--mmd-no-texture',action='store_true',help='Skip texture sidecar')
    args=a.parse_args(argv)
    src=Path(args.input).resolve();dst=Path(args.output).resolve()
    if src==dst or dst.exists():a.error('원본 덮어쓰기 및 기존 출력 파일 덮어쓰기 금지')
    if src.suffix.lower() not in ALL_INPUTS:a.error('지원하지 않는 입력 확장자: '+src.suffix)
    if dst.suffix.lower() not in ALL_OUTPUTS:a.error('지원하지 않는 출력 확장자: '+dst.suffix)
    if args.mode=='rig_apply' and dst.suffix.lower() not in RIG_OUT:a.error('리깅 출력은 GLB / FBX / BLEND / PMX 중 하나로 선택해 주세요')
    if args.mode=='mmd_export' and dst.suffix.lower()!='.pmx':a.error('MMD 출력은 PMX 형식을 선택해 주세요')
    # Analyze and copy identical GLB bytes without decoding: preserve skins,
    # animations, morphs, extensions and unknown authored glTF data exactly.
    if args.mode=='analyze' and src.suffix.lower()==dst.suffix.lower()=='.glb':
        from scene_core import _gltf_json
        data=_gltf_json(src)
        if data is None:raise ValueError('GLB JSON을 읽을 수 없습니다')
        dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as inp, dst.open('xb') as out:
            shutil.copyfileobj(inp,out)
        stat={'mode':'analyze','lossless_binary_copy':True,'mesh_only_export':False,
              'glb_meshes':len(data.get('meshes',[])), 'glb_skins':len(data.get('skins',[])),
              'glb_animations':len(data.get('animations',[])),'benchmark_executed_this_run':False}
        print(json.dumps({'saved_to':str(dst),'stats':stat},ensure_ascii=False,indent=2))
        return 0
    if args.mode in ('rig_plan','rig_apply','mmd_export'):
        _guard_project_skin(src)
        mesh=load_geometry(src,args.blender)
        planpath=Path(args.rig_plan).resolve() if args.rig_plan else dst.with_suffix('.rig.json')
        if not args.rig_plan:
            plan=write_plan(mesh,planpath,args.rig_profile,args.up_axis,args.rig_segments,args.rig_appendages)
        else:
            plan=json.loads(planpath.read_text(encoding='utf-8'))
        if args.mode=='mmd_export' or (args.mode=='rig_apply' and dst.suffix.lower()=='.pmx'):
            from pmx_export import write_pmx
            stat=write_pmx(mesh,dst,plan,up_axis=args.up_axis,
                           name=args.mmd_name or dst.stem,texture=not args.mmd_no_texture,scale=args.mmd_scale)
            stat['rig_template']=plan['profile']
            stat['benchmark_executed_this_run']=False
        elif args.mode=='rig_plan':
            dst.parent.mkdir(parents=True,exist_ok=True)
            export_geometry(mesh,dst,args.blender)
            stat={'rig_applied':False,'rig_template':plan['profile'],'plan':str(planpath),'bones':len(plan['bones'])}
        else:
            # Blender bridge handles non-native source formats via a geometry-only GLB intermediate.
            if src.suffix.lower() not in {'.glb','.gltf','.obj','.ply','.stl','.fbx','.blend','.usd','.usda','.usdc','.abc'}:
                raise ValueError('리깅 입력은 GLB/GLTF/OBJ/PLY/STL/FBX/BLEND/USD/Alembic입니다. 기타 형식은 먼저 GLB로 변환해 주세요')
            blend_sidecar=str(dst.with_suffix('.blend')) if dst.suffix.lower()!='.blend' else ''
            stat=apply_rig_blender(src,planpath,dst,args.blender,blend_sidecar)
            stat['bones']=len(plan['bones'])
    else:
        _guard_project_skin(src)
        # Protect material/part hierarchy when editing scenes. Non-GLB static mesh
        # output remains supported for single-mesh sources only.
        from scene_core import load_static_scene,process_static_scene,export_static_scene
        from format_io import DIRECT_IN
        scene=None
        if args.target_part and src.suffix.lower() not in DIRECT_IN:
            raise ValueError('파츠 선택은 GLB/glTF 등의 직접 지원 장면에서만 사용할 수 있습니다')
        vertex_mask=None
        if args.vertex_mask:
            import numpy as np
            vertex_mask=np.load(args.vertex_mask,allow_pickle=False)
            if vertex_mask.dtype!=np.bool_ or vertex_mask.ndim!=1:
                raise ValueError('마스크는 정점별 bool 1차원 numpy 배열이어야 합니다')
        if vertex_mask is not None and not args.target_part:
            raise ValueError('영역 마스크에는 --target-part 이름이 필요합니다')
        scene=None
        if src.suffix.lower()=='.gltf' and args.mode!='analyze':
            raise ValueError('glTF 외부/분리 버퍼의 원본 속성 보존 편집은 아직 미검증입니다. 원본 보호를 위해 중단합니다')
        # G1/G6: every GLB->GLB progressive operation uses the raw accessor
        # editor, including a *single* primitive with UV but no material.
        # Other mesh-changing GLB operations are fail-closed until their raw
        # attribute-preserving equivalent exists (analyze is byte-copy above).
        if src.suffix.lower()=='.glb' and args.mode!='analyze':
            if dst.suffix.lower()!='.glb':
                raise ValueError('GLB 원본 속성 보존을 확인할 수 없는 다른 형식으로는 편집할 수 없습니다')
            if args.mode!='progressive' or args.reference_dir or args.height_map:
                raise ValueError('GLB 무손실 편집은 현재 progressive 모드만 지원합니다. 속성 손실 방지로 중단합니다')
            from gltf_raw_editor import read_glb,refine_raw_glb,validate_raw_edit
            j,_=read_glb(src)
            validate_raw_edit(j)
            face_count=sum((j['accessors'][p['indices']]['count']//3 if 'indices' in p
                            else j['accessors'][p['attributes']['POSITION']]['count']//3)
                           for m in j.get('meshes',[]) for p in m['primitives'])
            profile,bench_ran=run_benchmark()
            budget=per_iteration_budget(face_count,profile,args.effort)
            lineage={}
            stats=refine_raw_glb(src,dst,target_mesh=args.target_part or None,
                                 steps=args.steps,budget=budget,vertex_mask=vertex_mask,
                                 lineage_sink=lineage)
            from lineage_io import save_round_lineage
            stats['lineage_file']=str(save_round_lineage(dst,lineage))
            stats['benchmark_executed_this_run']=bench_ran
            stats['estimated_add_faces_per_iteration']=budget
            operation=dst.with_name(dst.name+'.operation.json')
            operation.write_text(json.dumps({'input':str(src),'output':str(dst),'stats':stats},ensure_ascii=False,indent=2),encoding='utf8')
            print(json.dumps({'saved_to':str(dst),'stats':stats},ensure_ascii=False,indent=2))
            return 0
        if src.suffix.lower() in DIRECT_IN:
            scene=load_static_scene(src)
        if (args.target_part or args.vertex_mask) and dst.suffix.lower()!='.glb':
            raise ValueError('파츠·영역 선택 편집은 계층 보존 GLB 출력에서만 지원합니다')
        if scene is not None and len(scene.geometry)>1 and dst.suffix.lower()!='.glb':
            raise ValueError('다중 파츠·재질을 보존하려면 출력은 GLB를 선택해야 합니다.')
        # Blender/converter-only inputs must be loaded before the budget calculation.
        # Previously an FBX source passed face_count=1, silently reducing the budget.
        converted_mesh=None
        if scene is None:
            converted_mesh=load_geometry(src,args.blender)
        count=(sum(len(m.faces) for m in scene.geometry.values()) if scene is not None
               else len(converted_mesh.faces))
        if args.mode in ('progressive','fit_subdivide'):
            profile,bench_ran=run_benchmark()
            budget=per_iteration_budget(count,profile,args.effort)
        else:bench_ran=False;budget=None
        lineage={}
        if scene is not None and dst.suffix.lower()=='.glb':
            result,stat=process_static_scene(scene,args.mode,args.steps,args.reference_dir,
                args.strength,args.up_axis,budget,args.region,args.detail_strength,
                args.height_map,args.height_amplitude,target_part=args.target_part,
                vertex_mask=vertex_mask,lineage_sink=lineage,height_ratio=args.height_ratio)
            export_static_scene(result,dst)
        else:
            mesh=converted_mesh if converted_mesh is not None else load_geometry(src,args.blender)
            steps_lineage=[]
            result,stat=process_mesh(mesh,args.mode,args.steps,args.reference_dir,args.strength,args.up_axis,
                                     budget,args.region,args.detail_strength,
                                     height_map=args.height_map,height_amplitude=args.height_amplitude,
                                     lineage_sink=steps_lineage,height_ratio=args.height_ratio)
            lineage={'main':steps_lineage}
            dst.parent.mkdir(parents=True,exist_ok=True)
            export_geometry(result,dst,args.blender)
        from lineage_io import save_round_lineage
        linfile=save_round_lineage(dst,lineage) if any(lineage.values()) else None
        stat['lineage_file']=str(linfile) if linfile else None
        stat.update({'input_format':src.suffix.lower(),'output_format':dst.suffix.lower(),'mesh_only_export':True,
                     'warning':'이 도구의 점진적 메시 편집은 리깅/애니메이션을 보존하지 않습니다; 리깅은 편집 후 다시 적용하세요.',
                     'cached_benchmark_file':str(profile_path()),'benchmark_executed_this_run':bench_ran,
                     'estimated_add_faces_per_iteration':budget})
    if args.mode not in ('rig_plan','rig_apply','mmd_export'):
        # A report beside each immutable output makes future round-history repair possible.
        report=dst.with_name(dst.name+'.operation.json')
        report.write_text(json.dumps({'input':str(src),'output':str(dst),'stats':stat},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'saved_to':str(dst),'stats':stat},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print('ERROR:',e,file=sys.stderr);sys.exit(1)
