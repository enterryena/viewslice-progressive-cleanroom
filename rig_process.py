"""One-place rig plan and Blender bridge. Blender is optional for planning; required for skinning."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from rigging import rig_plan


def locate_blender(user_path=''):
    candidates=[user_path,shutil.which('blender')]
    if sys.platform=='win32':
        import glob
        root=os.environ.get('PROGRAMFILES','C:\\Program Files')
        candidates+=sorted(glob.glob(os.path.join(root,'Blender Foundation','Blender *','blender.exe')),reverse=True)
    for candidate in candidates:
        if candidate and Path(candidate).is_file():return str(Path(candidate).resolve())
    return None

def write_plan(mesh,path,profile='humanoid',up_axis='y',segments=7,appendages=6):
    plan=rig_plan(mesh.vertices,profile,up_axis,segments,appendages)
    path=Path(path)
    if path.exists():raise FileExistsError('리깅 계획 파일 덮어쓰기 금지: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    return plan

def apply_rig_blender(src,plan_path,dest,blender='',blend_path=''):
    executable=locate_blender(blender)
    if not executable:raise FileNotFoundError('Blender가 없습니다. https://www.blender.org/download/ 에서 설치하거나 --blender 경로를 지정하세요.')
    dest=Path(dest);src=Path(src)
    if dest.resolve()==src.resolve() or dest.exists():raise FileExistsError('원본 파일 보호: 출력 경로는 새 파일이어야 합니다')
    command=[executable,'--background','--factory-startup','--python',str(Path(__file__).with_name('rig_blender.py')),'--',
             '--input',str(src.resolve()),'--plan',str(Path(plan_path).resolve()),'--output',str(dest.resolve())]
    if blend_path:command+=['--blend',str(Path(blend_path).resolve())]
    res=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800)
    if res.returncode!=0 or not dest.is_file():raise RuntimeError('Blender 리깅 실패\n'+res.stdout[-3500:])
    return {'rigged_glb':str(dest),'blend':str(blend_path) if blend_path else None,'blender_output_tail':res.stdout[-700:]}
