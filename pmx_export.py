"""ViewSlice native PMX 2.0 export. Clean-room binary serializer from the public PMX 2.0 format.

No MMD Tools/Blender dependency. Creates geometry, UV (if present), one material,
initial bones with BDEF2 binding, and display frames. NOT a motion-ready IK/physics rig.
Never modifies input files. Does not create facial morphs or physics.
"""
from __future__ import annotations
import json
import os
import struct
import tempfile
from pathlib import Path
import numpy as np

JP_HUMANOID={
  'hips':'センター','spine':'下半身','chest':'上半身','neck':'首','head':'頭',
  'arm_L_01':'左腕','arm_L_02':'左ひじ','arm_R_01':'右腕','arm_R_02':'右ひじ',
  'leg_L_01':'左足','leg_L_02':'左ひざ','leg_R_01':'右足','leg_R_02':'右ひざ',
}


def _txt(s):
    data=str(s).encode('utf-16-le')
    return struct.pack('<i',len(data))+data


def _vec3(point,up_axis):
    x,y,z=(float(i) for i in point)
    # PMX/MMD coordinates are Y-up and left-handed; reverse the depth axis.
    # For Z-up inputs rotate up to Y first, then reverse depth.
    return (x,y,-z) if up_axis=='y' else (x,z,y)


def _image_from_mesh(mesh):
    try:
        vis=mesh.visual
        if getattr(vis,'kind',None)!='texture':return None
        mat=getattr(vis,'material',None)
        im=getattr(mat,'baseColorTexture',None)
        if im is None:im=getattr(mat,'image',None)
        return im if hasattr(im,'save') else None
    except Exception:return None


def _material_color(mesh):
    try:
        mat=getattr(mesh.visual,'material',None)
        rgba=getattr(mat,'baseColorFactor',None)
        if rgba is None:rgba=getattr(mat,'diffuse',None)
        if rgba is not None:
            vals=np.asarray(rgba,dtype=float).ravel()[:4]
            if vals.size>=3:
                if float(vals.max())>1.001:vals=vals/255.0
                return tuple(float(min(max(x,0),1)) for x in (list(vals[:3])+[vals[3] if vals.size>=4 else 1.0]))
    except Exception:pass
    try:
        c=getattr(mesh.visual,'vertex_colors',None)
        if c is not None and len(c):
            rgba=np.asarray(c).reshape(-1,4).mean(axis=0)/255.
            return tuple(float(x) for x in rgba)
    except Exception:pass
    return (.78,.80,.83,1.)


def write_pmx(mesh,dest,rig_plan=None,up_axis='y',name=None,source_format='',texture=True,scale=12.5):
    """Write a PMX 2.0 file. In native mode exports one material and at most one albedo image.

    If rig_plan is None the root bone alone receives all vertices. Plans yield an
    initial template-based rig, not a complete MMD-standard IK rig.
    """
    dst=Path(dest).resolve(); dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.suffix.lower()!='.pmx':raise ValueError('PMX 출력 경로는 .pmx여야 합니다')
    if dst.exists():raise FileExistsError(f'기존 PMX 덮어쓰기 금지: {dst}')
    if up_axis not in ('y','z'):raise ValueError('up_axis는 y 또는 z이어야 합니다')
    scale=float(scale)
    if not np.isfinite(scale) or scale<=0 or scale>100000:raise ValueError('유효하지 않은 MMD 크기 배율')
    verts=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.faces,dtype=np.int64)
    if verts.ndim!=2 or verts.shape[1]!=3 or not len(verts):raise ValueError('정점이 없는 메시입니다')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces):raise ValueError('삼각형 면이 없습니다')
    if not np.isfinite(verts).all():raise ValueError('정점에 NaN/Inf가 있습니다')
    if faces.min()<0 or faces.max()>=len(verts):raise ValueError('잘못된 정점 참조가 있습니다')
    n=len(verts);m=len(faces)
    if n>2_147_483_647 or m>715_827_882:raise ValueError('PMX 32비트 인덱스 범위 초과')
    from rigging import skinning_weights
    orig_bones=rig_plan['bones'] if rig_plan else []
    if len(orig_bones)>3000:raise ValueError('뼈대가 비정상적으로 많습니다')
    bones=[{'name':'全ての親','english':'Root','parent':None,'head':[0,0,0],'tail':[0,0.05,0]}]
    for b in orig_bones:
        bones.append({'name':JP_HUMANOID.get(b['name'],b['name']) if rig_plan.get('profile')=='humanoid' else b['name'],
                      'english':b['name'],'parent':b.get('parent'),'head':_vec3(b['head'],up_axis),
                      'tail':_vec3(b['tail'],up_axis)})
    if orig_bones:
        lookup={b['name']:i+1 for i,b in enumerate(orig_bones)}
        for b in bones[1:]:b['parent_index']=lookup.get(b['parent'],0)
    else:bones[0]['parent_index']=-1
    for b in bones:
        if 'parent_index' not in b:b['parent_index']=-1
        b['head']=[float(v)*scale for v in b['head']]
        b['tail']=[float(v)*scale for v in b['tail']]
    uv=np.zeros((n,2),dtype=np.float32)
    vis=getattr(mesh,'visual',None)
    candidate=getattr(vis,'uv',None)
    if candidate is not None and len(candidate)==n:
        uv[:]=np.asarray(candidate,dtype=np.float32)[:,:2];uv[:,1]=1.0-uv[:,1]
    image=_image_from_mesh(mesh) if texture else None
    sidecar=dst.with_name(dst.stem+'_albedo.png') if image is not None else None
    if sidecar is not None and sidecar.exists():raise FileExistsError(f'텍스처 파일 덮어쓰기 금지: {sidecar}')
    normals=np.asarray(mesh.vertex_normals,dtype=np.float32)
    if normals.shape!=(n,3):raise ValueError('정점 법선 개수 불일치')
    mat_color=_material_color(mesh)
    label=name or dst.stem
    tmp_path=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=dst.parent,prefix=dst.stem+'.',suffix='.tmp',delete=False) as f:
            tmp_path=Path(f.name)
            def put(fmt,*args):f.write(struct.pack(fmt,*args))
            # PMX 2.0 header: UTF-16LE; no extra UV; 32-bit vertex indices;
            # 32-bit signed indices for texture, material, bone, morph, rigidbody.
            f.write(b'PMX ');put('<f',2.0);put('<B',8)
            f.write(bytes([0,0,4,4,4,4,4,4]))
            f.write(_txt(label));f.write(_txt(label))
            f.write(_txt('ViewSlice CleanRoom 초안 — 뼈/가중치 자동 배치. 모프·물리·IK 미설정.'))
            f.write(_txt('ViewSlice CleanRoom draft. Basic template skinning; no morphs/physics/IK.'))
            # vertices: position, normal, uv, BDEF2, edge scale
            put('<i',n)
            chunk=2048
            for off in range(0,n,chunk):
                vv=verts[off:off+chunk];vn=normals[off:off+chunk];uvv=uv[off:off+chunk]
                if orig_bones:
                    ids,weights=skinning_weights(np.asarray(vv,dtype=np.float64),orig_bones,2)
                for j in range(len(vv)):
                    put('<3f',*(c*scale for c in _vec3(vv[j],up_axis)));put('<3f',*_vec3(vn[j],up_axis));put('<2f',*uvv[j])
                    if orig_bones:
                        put('<B',1);put('<ii',int(ids[j,0])+1,int(ids[j,-1])+1)
                        put('<f',float(weights[j,0]))
                    else:
                        put('<B',0);put('<i',0)
                    put('<f',1.0)
            # 3 indices for each triangle. Mirror depth changes chirality => reverse winding.
            put('<i',m*3)
            flipped=np.ascontiguousarray(faces[:,[0,2,1]],dtype='<i4')
            f.write(flipped.tobytes())
            put('<i',1 if sidecar is not None else 0)
            if sidecar is not None:f.write(_txt(sidecar.name))
            # One material using per-material triangle count; metadata is preserved only
            # to extent supported by base material colour and optional albedo image.
            put('<i',1);f.write(_txt('材質'));f.write(_txt('Material'))
            put('<4f',*mat_color);put('<3f',.18,.18,.18);put('<f',15.)
            put('<3f',.48,.48,.48);put('<B',0x1f)
            put('<4f',0.,0.,0.,1.);put('<f',1.)
            put('<i',0 if sidecar is not None else -1);put('<i',-1)
            put('<B',0);put('<B',1);put('<B',0)
            f.write(_txt('ViewSlice draft material'));put('<i',m*3)
            # bones; no IK/additional transforms, offsets for tail to prevent invalid references
            put('<i',len(bones))
            for i,b in enumerate(bones):
                f.write(_txt(b['name']));f.write(_txt(b['english']))
                put('<3f',*b['head']);put('<i',b['parent_index']);put('<i',0)
                put('<H',0x001e if i else 0x001e) # rotating/moving + visible/operable
                tip=np.asarray(b['tail'],dtype=float)-np.asarray(b['head'],dtype=float)
                if np.linalg.norm(tip)<1e-5:tip=np.array([0.,.05,0.])
                put('<3f',*tip)
            # zero morphs, display frames (Root, Expressions, and Other)
            put('<i',0)
            put('<i',3)
            f.write(_txt('Root'));f.write(_txt('Root'));put('<B',1);put('<i',1);put('<B',0);put('<i',0)
            f.write(_txt('表情'));f.write(_txt('Expressions'));put('<B',1);put('<i',0)
            f.write(_txt('操作'));f.write(_txt('Bones'));put('<B',0);put('<i',max(len(bones)-1,0))
            for bone_index in range(1,len(bones)):put('<B',0);put('<i',bone_index)
            put('<i',0) # rigid body count
            put('<i',0) # joint count
            f.flush();os.fsync(f.fileno())
        if sidecar is not None:
            image.convert('RGBA').save(sidecar,format='PNG')
        tmp_path.replace(dst)
    except Exception:
        if tmp_path is not None:tmp_path.unlink(missing_ok=True)
        raise
    return {'format':'PMX 2.0','vertices':n,'faces':m,'bones':len(bones),
            'rigged':bool(orig_bones),'mmd_scale':scale,'material_count':1,'albedo_texture':sidecar.name if sidecar else None,
            'morphs':0,'ik_chains':0,'physics_bodies':0,
            'limitations':['initial template-based rig only','no IK','no facial shape-key morphs','no MMD physics','single simplified material']}
