"""Headless Blender application of a previously saved rig_plan.json.
Usage: blender -b --python rig_blender.py -- --input mesh.glb --plan plan.json --output result.glb --blend result.blend
Blender runtime required; not tested under Blender in authoring environment.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np


def main():
    import bpy
    from rigging import skinning_weights
    a=argparse.ArgumentParser()
    a.add_argument('--input',required=True);a.add_argument('--plan',required=True);a.add_argument('--output',required=True)
    a.add_argument('--blend',default='');opts=a.parse_args(sys.argv[sys.argv.index('--')+1:])
    src=Path(opts.input).resolve();dest=Path(opts.output).resolve();plan=json.loads(Path(opts.plan).read_text(encoding='utf-8'))
    if src==dest or dest.exists():raise ValueError('원본과 출력 파일 분리 및 기존 파일 덮어쓰기 금지')
    if dest.suffix.lower() not in {'.glb','.fbx','.blend'}:raise ValueError('리깅 출력 형식은 GLB, FBX, BLEND입니다')
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    ext=src.suffix.lower()
    if ext in ('.glb','.gltf'):bpy.ops.import_scene.gltf(filepath=str(src))
    elif ext=='.obj':bpy.ops.wm.obj_import(filepath=str(src))
    elif ext=='.stl':bpy.ops.wm.stl_import(filepath=str(src))
    elif ext=='.ply':bpy.ops.wm.ply_import(filepath=str(src))
    elif ext=='.fbx':bpy.ops.import_scene.fbx(filepath=str(src))
    elif ext=='.blend':bpy.ops.wm.open_mainfile(filepath=str(src))
    elif ext in ('.usd','.usda','.usdc'):bpy.ops.wm.usd_import(filepath=str(src))
    elif ext=='.abc':bpy.ops.wm.alembic_import(filepath=str(src))
    else:raise ValueError('지원하지 않는 파일: '+ext)
    meshes=[obj for obj in bpy.context.scene.objects if obj.type=='MESH']
    if not meshes:raise ValueError('가져온 메시가 없습니다')
    # glTF and Blender use different up axes: glTF Y up -> Blender Z up.
    if ext in ('.glb','.gltf') and plan.get('up_axis')=='y':
        for bone in plan['bones']:
            bone['head']=[bone['head'][0],-bone['head'][2],bone['head'][1]]
            bone['tail']=[bone['tail'][0],-bone['tail'][2],bone['tail'][1]]
    armdata=bpy.data.armatures.new('ViewSliceArmatureData')
    arm=bpy.data.objects.new('ViewSliceAutoRig',armdata)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active=arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb={}
    for b in plan['bones']:
        bone=armdata.edit_bones.new(b['name']);bone.head=b['head'];bone.tail=b['tail'];bone.use_deform=True
        eb[b['name']]=bone
    for b in plan['bones']:
        if b.get('parent'):eb[b['name']].parent=eb[b['parent']]
    bpy.ops.object.mode_set(mode='OBJECT')
    names=[b['name'] for b in plan['bones']]
    for obj in meshes:
        groups=[obj.vertex_groups.get(n) or obj.vertex_groups.new(name=n) for n in names]
        old_world=obj.matrix_world.copy()
        obj.parent=arm
        obj.matrix_world=old_world
        mod=obj.modifiers.new('ViewSlice Skinning','ARMATURE');mod.object=arm
        # Sparse grouped assignment avoids millions of per-vertex bpy API calls.
        # Quantization slightly perturbs weights, but 20 levels keep runtime manageable.
        count=len(obj.data.vertices)
        coords=np.empty(count*3,dtype=np.float32)
        obj.data.vertices.foreach_get('co',coords)
        coords=coords.reshape(-1,3)
        matrix=np.asarray(obj.matrix_world,dtype=float)
        for start in range(0,count,4000):
            xyz=coords[start:start+4000].astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
            ids,ws=skinning_weights(xyz,plan['bones'],max_influences=2)
            bins=np.rint(ws*20).astype(np.int32)
            grouped={}
            for row in range(len(xyz)):
                for j in range(ids.shape[1]):
                    b=int(bins[row,j]);group=int(ids[row,j]);
                    if b:grouped.setdefault((group,b),[]).append(start+row)
            for (group,b),indices in grouped.items():groups[group].add(indices,b/20.,'REPLACE')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if opts.blend:
        outblend=Path(opts.blend).resolve()
        if outblend.exists():raise FileExistsError(str(outblend))
        bpy.ops.wm.save_as_mainfile(filepath=str(outblend))
    if dest.suffix.lower()=='.glb':
        bpy.ops.export_scene.gltf(filepath=str(dest),export_format='GLB',export_skins=True,export_animations=False)
    elif dest.suffix.lower()=='.fbx':
        bpy.ops.export_scene.fbx(filepath=str(dest),use_selection=False,add_leaf_bones=False,bake_anim=False)
    elif dest.suffix.lower()=='.blend':
        bpy.ops.wm.save_as_mainfile(filepath=str(dest))
    print('VIEWSLICE_RIG_SAVED='+str(dest),flush=True)

if __name__=='__main__':main()
