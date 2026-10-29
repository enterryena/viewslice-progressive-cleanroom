"""Independent clean-room geometry operations: midpoint subdivision and silhouette fitting.

No vendor algorithms or code are included. Input topology is maintained for all
existing vertices; subdivision adds edge midpoints; fitting moves existing vertices.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np



def mesh_stats(mesh):
    v=np.asarray(mesh.vertices); f=np.asarray(mesh.faces)
    extent=np.ptp(v,axis=0) if len(v) else np.zeros(3)
    return {'vertices':int(len(v)), 'faces':int(len(f)), 'extents':[round(float(x),6) for x in extent],
            'watertight':bool(mesh.is_watertight) if len(f)<200_000 else None}


def read_mesh(path):
    import trimesh
    src=Path(path)
    if src.suffix.lower() not in {'.glb','.gltf','.obj','.ply','.stl','.off','.3mf','.dae'}:raise ValueError('지원되지 않는 메시 형식')
    loaded=trimesh.load(str(src),force='scene',process=False)
    if isinstance(loaded,trimesh.Scene):
        # Bake scene transform to world coordinates; never touch original.
        meshes=[]
        for node in loaded.graph.nodes_geometry:
            mat,geom_name=loaded.graph[node]
            geo=loaded.geometry[geom_name].copy()
            geo.apply_transform(mat)
            meshes.append(geo)
        if not meshes:raise ValueError('메시 형상이 없는 파일')
        mesh=trimesh.util.concatenate(meshes)
    else:mesh=loaded.copy()
    if not len(mesh.vertices) or not len(mesh.faces):raise ValueError('비어 있는 메시')
    return mesh


def midpoint_once(vertices, faces):
    """True topological refinement: old vertices unchanged, new vertices at unique edge midpoints."""
    v=np.asarray(vertices,dtype=np.float64); f=np.asarray(faces,dtype=np.int64)
    edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]),axis=0)
    sorted_edges=np.sort(edges,axis=1)
    unique, inverse=np.unique(sorted_edges,axis=0,return_inverse=True)
    mid=(v[unique[:,0]]+v[unique[:,1]])*.5
    e01=inverse[:len(f)]+len(v); e12=inverse[len(f):2*len(f)]+len(v); e20=inverse[2*len(f):]+len(v)
    a,b,c=f[:,0],f[:,1],f[:,2]
    new_faces=np.stack((np.stack((a,e01,e20),axis=1),np.stack((b,e12,e01),axis=1),
                        np.stack((c,e20,e12),axis=1),np.stack((e01,e12,e20),axis=1)),axis=1).reshape(-1,3)
    return np.concatenate((v,mid),axis=0),new_faces


def silhouette_bounds(image_path,bins=128):
    """Visual hull *outline* only; does not synthesize occluded anatomy.

    Returns a full-height sampled horizontal interval in *units of body height*.
    """
    from PIL import Image
    im=Image.open(image_path).convert('RGBA')
    im.thumbnail((1024,1024))
    p=np.asarray(im,dtype=np.uint8)
    visible=(p[:,:,3]>90)&(np.min(p[:,:,:3],axis=2)<238)
    # Transparent alpha is authoritative. RGB threshold is safety for white opaque canvas.
    if np.count_nonzero(visible)<25:raise ValueError(f'윤곽을 찾지 못했습니다: {image_path}')
    ys,xs=np.where(visible)
    left=int(xs.min());right=int(xs.max());top=int(ys.min());bottom=int(ys.max());height=max(bottom-top+1,1)
    center=(left+right)*0.5
    bounds=np.full((bins,2),np.nan,dtype=np.float64)
    for i in range(bins):
        yy=top+(i+.5)*(height/bins)
        y0=max(0,int(yy-height/bins));y1=min(len(visible),int(yy+height/bins)+1)
        cols=np.where(np.any(visible[y0:y1],axis=0))[0]
        if len(cols):bounds[i]=((cols.min()-center)/height,(cols.max()-center)/height)
    valid=np.where(np.isfinite(bounds[:,0]))[0]
    if len(valid)<3:raise ValueError('윤곽 데이터가 너무 적습니다')
    for dim in range(2):bounds[:,dim]=np.interp(np.arange(bins),valid,bounds[valid,dim])
    return bounds


def find_references(folder):
    root=Path(folder).expanduser()
    if not root.is_dir():raise ValueError(f'참고 이미지 폴더가 없습니다: {root}')
    by_role={}
    rolefile=root/'viewslice_roles.json'
    if rolefile.is_file():
        metadata=json.loads(rolefile.read_text(encoding='utf-8'))
        for row in metadata.get('views',[]):
            role=str(row.get('role','')).lower()
            f=root/Path(row.get('image_file','')).name
            if f.is_file() and role in {'front','back','right','left'}:by_role[role]=f
    for f in sorted(root.glob('*.png')):
        for role in ('front','back','right','left'):
            if role not in by_role and role in f.stem.lower():by_role[role]=f
    if 'front' not in by_role or not ('left' in by_role or 'right' in by_role):
        raise ValueError('참고 폴더에 front와 left/right 이미지를 지정해 주세요')
    return by_role


def proportion_fit(vertices, folder, strength=0.35, up_axis='y', bins=128):
    """Conservative per-slice shape correction, not full re-generation.

    Uses existing 3D vertices and moves only x/z coordinates. May distort meshes
    with poorly matched view poses. Keep strength low and preserve original file.
    """
    if strength<=0:return np.asarray(vertices,dtype=np.float64).copy(),{'moved':0,'skipped_bins':0}
    ref=find_references(folder)
    bounds_front=silhouette_bounds(ref['front'],bins)
    side_role='right' if 'right' in ref else 'left'
    bounds_side=silhouette_bounds(ref[side_role],bins)
    # A left-profile image is horizontally mirrored relative to the right view.
    if side_role=='left': bounds_side=-bounds_side[:, ::-1]
    v=np.asarray(vertices,dtype=np.float64)
    out=v.copy()
    # axes: x horizontal, y up, z depth; optional Blender z-up.
    up_axis=str(up_axis).lower()
    up_idx=2 if up_axis=='z' else 1
    side_idx=1 if up_idx==2 else 2
    vheight=float(np.ptp(v[:,up_idx]))
    if vheight<1e-9:raise ValueError('모델 높이가 0입니다')
    # image y runs top->bottom, mesh y runs bottom->top
    bi=np.clip(((v[:,up_idx].max()-v[:,up_idx])/vheight*bins).astype(np.int32),0,bins-1)
    moved=0;skipped=0
    # row local min and max of current surface; reject degenerate depth bins instead
    # of exploding near-flat AI meshes in an unconstrained direction.
    for axis, target in ((0,bounds_front),(side_idx,bounds_side)):
        for b in range(bins):
            ix=np.flatnonzero(bi==b)
            if len(ix)<3:continue
            vals=v[ix,axis];old_lo=float(vals.min());old_hi=float(vals.max());old_span=old_hi-old_lo
            desired_lo=float(target[b,0]*vheight);desired_hi=float(target[b,1]*vheight)
            target_span=desired_hi-desired_lo
            if old_span<max(0.015*vheight,1e-7):skipped+=1;continue
            # preserve row centroid, only change ratio; clamp excessive deformation.
            scale=np.clip(target_span/old_span,0.65,1.45)
            old_center=(old_lo+old_hi)*.5
            # Image coordinates are bbox-relative. Anchor the reference to the
            # model's original axis center instead of teleporting it to 0.
            anchor=(float(v[:,axis].min())+float(v[:,axis].max()))*.5
            desired_center=anchor+(desired_lo+desired_hi)*.5
            delta=(vals-old_center)*(scale-1)+desired_center-old_center
            delta=np.clip(delta,-0.12*vheight,0.12*vheight)
            out[ix,axis]+=strength*delta
            moved+=int(np.count_nonzero(np.abs(delta)>1e-9))
    return out,{'moved':moved,'skipped_degenerate_slice_axes':skipped,'side_view':side_role,
                'note':'윤곽 비율만 보정; 코·손가락 등 숨겨진 세부 형상을 생성하지 않음'}


def process_mesh(mesh,mode='progressive',steps=1,reference_dir='',strength=0.35,up_axis='y',
                 budget=None,region='all',detail_strength=0.12,
                 height_map='',height_amplitude=0,vertex_mask=None,lineage_sink=None,height_ratio=0):
    """Cached benchmark-guided partial refinement. Never AI reset; source file untouched.
    Shape refinement is conservative silhouette fitting, NOT learned semantic detail.
    """
    import trimesh
    from adaptive import adaptive_refine,expand_seam_mask
    if mode not in {'analyze','progressive','subdivide','fit','fit_subdivide'}:raise ValueError('알 수 없는 모드')
    before=mesh_stats(mesh)
    if mode=='analyze':return mesh.copy(),{'mode':mode,'before':before,'after':before}
    v=np.asarray(mesh.vertices,dtype=np.float64).copy();f=np.asarray(mesh.faces,dtype=np.int64).copy()
    info={};levels=[]
    original_visual=getattr(mesh,'visual',None)
    visual_kind=getattr(original_visual,'kind',None)
    uv=None;vertex_colors=None;face_colors=None;face_materials=None
    if visual_kind=='texture' and getattr(original_visual,'uv',None) is not None:
        if len(original_visual.uv)==len(v):uv=np.array(original_visual.uv,copy=True)
        fm=getattr(original_visual,'face_materials',None)
        if fm is not None and len(fm)==len(f):face_materials=np.array(fm,copy=True)
    elif visual_kind=='vertex':
        vertex_colors=np.asarray(original_visual.vertex_colors).copy()
    elif visual_kind=='face':
        face_colors=np.asarray(original_visual.face_colors).copy()
    if mode in ('fit','fit_subdivide'):
        if not reference_dir:raise ValueError('비율 보정에는 reference_dir 폴더가 필요합니다')
        v,info=proportion_fit(v,reference_dir,strength=strength,up_axis=up_axis)
    from benchmark import per_iteration_budget
    if mode in ('progressive','subdivide','fit_subdivide'):
        if budget is not None and int(budget)<1:raise ValueError('회차당 새 폴리곤 수가 유효하지 않습니다')
        for i in range(int(steps)):
            increment=per_iteration_budget(len(f)) if budget is None else int(budget)
            if vertex_mask is not None:vertex_mask=expand_seam_mask(v,vertex_mask,f)
            v2,f2,step,sources,edges=adaptive_refine(v,f,max_added_faces=increment,region=region,
                                                      up_axis=up_axis,vertex_mask=vertex_mask)
            if lineage_sink is not None:
                lineage_sink.append({'round':i+1,'old_vertices':len(v),'new_vertices':len(v2),
                                     'roots':edges[:,0].copy(),'tips':edges[:,1].copy()})
            if vertex_mask is not None:
                vertex_mask=np.concatenate((np.asarray(vertex_mask,dtype=bool),
                    np.asarray(vertex_mask,dtype=bool)[edges[:,0]] &
                    np.asarray(vertex_mask,dtype=bool)[edges[:,1]]))
            if uv is not None:
                uv=np.concatenate((uv,(uv[edges[:,0]]+uv[edges[:,1]])*.5),axis=0)
            if vertex_colors is not None:
                vc=(vertex_colors[edges[:,0]].astype(float)+vertex_colors[edges[:,1]].astype(float))*.5
                vertex_colors=np.concatenate((vertex_colors,np.rint(vc).astype(np.uint8)),axis=0)
            if face_colors is not None:face_colors=face_colors[sources]
            if face_materials is not None:face_materials=face_materials[sources]
            v,f=v2,f2;levels.append(step)
            if reference_dir and mode in ('progressive','fit_subdivide') and detail_strength>0:
                v,fit=proportion_fit(v,reference_dir,strength=min(detail_strength,0.35),up_axis=up_axis)
                levels[-1]['silhouette_detail']=fit
    result=trimesh.Trimesh(vertices=v,faces=f,process=False)
    visual_message='없음/확인 필요'
    if uv is not None:
        result.visual=trimesh.visual.TextureVisuals(uv=uv,material=original_visual.material,face_materials=face_materials)
        visual_message='UV/재질 구조 전파 시도. 복잡한 GLB 다중 소재의 외관 일치 여부는 별도 검증 필요'
    elif vertex_colors is not None:
        result.visual=trimesh.visual.ColorVisuals(mesh=result,vertex_colors=vertex_colors)
        visual_message='정점 색상 에지 보간'
    elif face_colors is not None:
        result.visual=trimesh.visual.ColorVisuals(mesh=result,face_colors=face_colors)
        visual_message='면 색상 복제'
    # glTF custom attribute: stable vertex IDs, including seam duplicates.
    # glTF custom vertex attributes use FLOAT, with exactly representable
    # integer values <=2**24. A new vertex has _SOURCE_VERTEX = -1.
    source_vertex=getattr(mesh,'vertex_attributes',{}).get('_SOURCE_VERTEX')
    if source_vertex is not None:
        source_vertex=np.asarray(source_vertex,dtype=np.float64).reshape(-1)
        source_vertex=np.where(source_vertex==np.iinfo(np.uint32).max,-1,source_vertex)
        if len(source_vertex)!=len(mesh.vertices):raise ValueError('_SOURCE_VERTEX 길이 불일치')
        if np.any(~np.isfinite(source_vertex)) or np.any(source_vertex!=np.rint(source_vertex)) or np.any(source_vertex>16777216):
            raise ValueError('원본 정점 ID 정밀도 손실 위험')
        extra_count=len(v)-len(source_vertex)
        if extra_count>0:source_vertex=np.concatenate((source_vertex,np.full(extra_count,-1,dtype=np.float64)))
        result.vertex_attributes['_SOURCE_VERTEX']=source_vertex.astype(np.float32)
    inherited=getattr(mesh,'vertex_attributes',{}).get('_VERTEX_ID')
    if inherited is None:
        inherited=np.arange(len(mesh.vertices),dtype=np.float64)
    else:
        inherited=np.asarray(inherited,dtype=np.float64).reshape(-1)
        if len(inherited)!=len(mesh.vertices) or len(np.unique(inherited))!=len(inherited):
            raise ValueError('원본 정점 ID가 누락·중복돼 편집을 중단합니다')
    if np.any(~np.isfinite(inherited)) or np.any(inherited<0) or np.any(inherited!=np.rint(inherited)):
        raise ValueError('_VERTEX_ID의 양의 정수 규약 위반')
    extra=len(v)-len(inherited)
    if extra>0:
        start=int(inherited.max())+1 if len(inherited) else 0
        new_ids=np.concatenate((inherited,np.arange(start,start+extra,dtype=np.float64)))
    else:new_ids=inherited
    if len(new_ids) and np.max(new_ids)>16777216:
        raise ValueError('glTF FLOAT 정점 ID 정밀도 한도 초과; ID 사이드카가 필요합니다')
    result.vertex_attributes['_VERTEX_ID']=new_ids.astype(np.float32)
    # On >32-bit-ID scenes, preserve ancestry separately in an NPZ sidecar.
    geometry_detail=None
    if height_map:
        from detail_layer import apply_height_displacement
        if height_ratio and height_amplitude:
            raise ValueError('높이맵 절대 진폭과 키 대비 비율을 동시에 사용할 수 없습니다')
        amp=float(height_amplitude)
        if height_ratio:
            if not 0<float(height_ratio)<.5:
                raise ValueError('높이 대비 진폭 비율은 0과 0.5 사이여야 합니다')
            up_idx=2 if up_axis=='z' else 1
            amp=float(height_ratio)*float(np.ptp(result.vertices[:,up_idx]))
        result,geometry_detail=apply_height_displacement(result,height_map,amp)
    # Density growth alone must never be misreported as new shape detail.
    changed=bool(geometry_detail) or bool(info.get('moved',0)) or any(
        lev.get('silhouette_detail',{}).get('moved',0)>0 for lev in levels)
    return result,{'mode':mode,'before':before,'after':mesh_stats(result),'fitting':info,
                   'levels':levels,'steps':int(steps),'visual_handling':visual_message,
                   'detail_changed':changed,'height_detail':geometry_detail,
                   'detail_note':('실제 정점 변위가 수행됨' if changed else
                                  '폴리곤 밀도만 증가했으며 새로운 형상 디테일은 생성되지 않았음')}
