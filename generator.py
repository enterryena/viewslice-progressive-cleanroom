#!/usr/bin/env python3
"""Local Modly process-node adapter, with cached benchmarking and optional Blender rigging.
Does not overwrite any incoming mesh. Uses JSON-lines process protocol.
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
try:sys.stdout.reconfigure(encoding='utf-8',errors='replace')
except (AttributeError,OSError):pass

def send(message):print(json.dumps(message,ensure_ascii=False),flush=True)
def progress(pct,label):send({'type':'progress','percent':int(pct),'label':str(label)})
def log(msg):send({'type':'log','message':str(msg)})

def main(payload):
    from mesh_core import read_mesh,process_mesh
    from benchmark import run_benchmark,per_iteration_budget,profile_path
    from rig_process import write_plan,apply_rig_blender
    inp=payload.get('input') or {}
    path=inp.get('filePath') if isinstance(inp,dict) else None
    if not path or not os.path.isfile(path):raise ValueError('입력 메시 파일이 없습니다')
    p=payload.get('params') or {}
    action=str(p.get('operation','progressive'))
    steps=max(1,min(3,int(p.get('steps',1))))
    axis=str(p.get('up_axis','y'))
    outdir=Path(payload.get('workspaceDir') or payload.get('tempDir') or tempfile.gettempdir())/'Workflows'
    outdir.mkdir(parents=True,exist_ok=True)
    suffix=f'{int(time.time()*1000)}-{os.getpid()}'
    dest=outdir/f'viewslice-{action}-{suffix}.glb'
    progress(4,'기존 3D 모델 분석')
    # The Modly process node must use the same lossless accessor path as the
    # Windows CLI. Never silently route a one-primitive GLB through trimesh.
    if Path(path).suffix.lower()=='.glb' and action in ('progressive','analyze','fit','fit_subdivide'):
        if action=='analyze':
            shutil.copy2(path,dest)
            progress(100,'완료')
            send({'type':'done','result':{'filePath':str(dest),'stats':{'operation':'analyze','lossless_binary_copy':True}}})
            return
        if action!='progressive' or p.get('reference_dir') or p.get('height_map'):
            raise ValueError('GLB 무손실 편집은 현재 progressive만 지원합니다. 원본 UV와 사용자 속성 손실 방지로 중단합니다')
        from gltf_raw_editor import read_glb,validate_raw_edit,refine_raw_glb
        from lineage_io import save_round_lineage
        j,_=read_glb(path)
        validate_raw_edit(j)
        count=sum((j['accessors'][q['indices']]['count']//3 if 'indices' in q
                   else j['accessors'][q['attributes']['POSITION']]['count']//3)
                  for m in j.get('meshes',[]) for q in m['primitives'])
        profile,executed=run_benchmark()
        budget=per_iteration_budget(count,profile,float(p.get('effort',1.0)))
        progress(35,'원본 glTF 속성·UV·재질을 유지하며 세분화')
        lineage={}
        stat=refine_raw_glb(path,dest,steps=steps,budget=budget,lineage_sink=lineage)
        stat['lineage_file']=str(save_round_lineage(dest,lineage))
        stat['benchmark_executed_this_run']=executed
        extra=str(p.get('extra_export','none')).lower().lstrip('.')
        if extra in {'obj','ply','stl','off','3mf'}:
            if len(j.get('meshes',[]))==1 and len(j['meshes'][0]['primitives'])==1:
                from format_io import export_geometry
                import trimesh
                extra_path=dest.with_suffix('.'+extra)
                mesh=trimesh.load(str(dest),force='mesh',process=False)
                stat['extra_export_files']=export_geometry(mesh,extra_path)
                log('추가 형식은 GLB의 UV/ID를 보존하지 않을 수 있습니다. 원본 GLB를 정본으로 사용하세요')
            else:
                stat['extra_export_files']=[]
                log('여러 메시/재질은 추가 형식에서 정보가 손실될 수 있어 생략합니다')
        progress(100,'완료')
        send({'type':'done','result':{'filePath':str(dest),'stats':stat}})
        return
    from scene_core import load_static_scene,process_static_scene,export_static_scene
    scene=load_static_scene(path)
    mesh=read_mesh(path) if action in ('rig_plan','rig_apply') else None
    log(f'원본 {sum(len(g.faces) for g in scene.geometry.values()):,}면 / {len(scene.geometry)}개 메시 파츠')
    if action in ('rig_plan','rig_apply'):
        profile=str(p.get('rig_profile','humanoid'))
        plan_path=outdir/f'viewslice-rig-plan-{suffix}.json'
        plan=write_plan(mesh,plan_path,profile,axis,int(p.get('rig_segments',7)),int(p.get('rig_appendages',6)))
        log(f'리깅 계획: {len(plan["bones"])}개 뼈, {profile}. 형태 자동 판별이 아닌 선택한 템플릿 기반입니다.')
        progress(35,'뼈대 생성 완료')
        if action=='rig_plan':
            if Path(path).suffix.lower()=='.glb':shutil.copy2(path,dest)
            else:mesh.export(str(dest))
        else:
            blend_path=outdir/f'viewslice-rig-{suffix}.blend'
            progress(50,'Blender 뼈대 및 정점 가중치 생성 (첫 사용 시 Blender 필요)')
            apply_rig_blender(path,plan_path,dest,str(p.get('blender_path','')),str(blend_path))
        progress(100,'완료')
        send({'type':'done','result':{'filePath':str(dest),'stats':{'operation':action,'profile':profile,'bones':len(plan['bones']),'rig_plan_file':str(plan_path),'rigging_applied':action=='rig_apply','blend_file':str(blend_path) if action=='rig_apply' else None}}})
        return
    if action not in ('analyze','progressive','fit','fit_subdivide'):
        raise ValueError('알 수 없는 operation: '+action)
    if action in ('progressive','fit_subdivide'):
        profile,executed=run_benchmark()
        log(('최초 벤치마크를 수행하여 저장했습니다: ' if executed else '저장된 벤치마크 재사용: ')+str(profile_path()))
        added=per_iteration_budget(sum(len(g.faces) for g in scene.geometry.values()),profile,float(p.get('effort',1.0)))
        log(f'회차당 새 폴리곤 목표 약 {added:,}개; 총 50만 개 제한 없음')
    else:added=None
    progress(25,'기존 메시를 유지한 형상 수정/국소 세분화')
    mode='progressive' if action=='progressive' else action
    result,stat=process_static_scene(scene,mode,steps,str(p.get('reference_dir','')),float(p.get('fit_strength',0.2)),axis,
                             budget=added,region=str(p.get('region','all')),
                             detail_strength=float(p.get('detail_strength',0.12)))
    progress(80,'원본과 다른 결과 파일 저장')
    if action=='analyze' and Path(path).suffix.lower()=='.glb':shutil.copy2(path,dest)
    else:export_static_scene(result,dest)
    if not dest.exists():raise RuntimeError('출력 실패')
    extra=str(p.get('extra_export','none')).lower().lstrip('.')
    if action!='analyze' and extra in {'obj','ply','stl','off','3mf'}:
        from format_io import export_geometry
        extra_path=dest.with_suffix('.'+extra)
        if len(result.geometry)>1:
            log('다중 파츠 장면은 추가 출력에서 병합될 수 있어 extra_export를 생략했습니다. GLB에 파츠를 보존했습니다.')
            files=[]
        else:
            files=export_geometry(next(iter(result.geometry.values())),extra_path)
        stat['extra_export_files']=files
        log('추가 형식 내보내기 완료: '+str(extra_path))
    progress(100,'완료')
    send({'type':'done','result':{'filePath':str(dest),'stats':stat}})

if __name__=='__main__':
    try:main(json.loads(sys.stdin.readline()))
    except Exception as exc:
        send({'type':'error','message':f'{type(exc).__name__}: {exc}'})
        sys.exit(1)
