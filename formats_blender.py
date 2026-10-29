"""Optional Blender background import/export of STATIC scene geometry.
Intentional limitation: this conversion does not preserve imported skeletons/animation.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

def main():
    import bpy
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    a=ap.parse_args(argv)
    src=Path(a.input).resolve();dst=Path(a.output).resolve()
    if src==dst or dst.exists():raise RuntimeError('결과 파일은 원본과 달라야 하고 기존 파일 덮어쓰기는 불가능합니다')
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    ext=src.suffix.lower()
    if ext=='.blend':
        bpy.ops.wm.open_mainfile(filepath=str(src))
    elif ext in ('.glb','.gltf'):
        bpy.ops.import_scene.gltf(filepath=str(src))
    elif ext=='.fbx':
        bpy.ops.import_scene.fbx(filepath=str(src))
    elif ext in ('.usd','.usda','.usdc'):
        bpy.ops.wm.usd_import(filepath=str(src))
    elif ext=='.abc':
        bpy.ops.wm.alembic_import(filepath=str(src))
    elif ext=='.dae' and hasattr(bpy.ops.wm,'collada_import'):
        bpy.ops.wm.collada_import(filepath=str(src))
    else:raise RuntimeError('Blender 브리지 입력 형식 미지원: '+ext)
    count=sum(o.type=='MESH' for o in bpy.context.scene.objects)
    if count<1:raise RuntimeError('메시 객체가 없습니다')
    # Export only evaluated geometry: do not advertise preservation of armatures here.
    dst.parent.mkdir(parents=True,exist_ok=True)
    fmt=dst.suffix.lower()
    if fmt=='.glb':
        bpy.ops.export_scene.gltf(filepath=str(dst),export_format='GLB',export_animations=False,export_skins=False)
    elif fmt=='.fbx':
        bpy.ops.export_scene.fbx(filepath=str(dst),use_selection=False,bake_anim=False)
    elif fmt=='.blend':
        bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    elif fmt in ('.usd','.usda','.usdc'):
        bpy.ops.wm.usd_export(filepath=str(dst),export_animation=False)
    elif fmt=='.abc':
        bpy.ops.wm.alembic_export(filepath=str(dst),start=1,end=1)
    else:raise RuntimeError('Blender 브리지 출력 형식 미지원: '+fmt)
    print('VIEWSLICE_CONVERTED='+str(dst),flush=True)

if __name__=='__main__':main()
