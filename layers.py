"""ViewSlice v0.6 - independently authored, non-destructive layered assets and mesh refitting.

Limits: geometric envelope fitting, not learned cloth design/simulation. No copy of vendor code.
Coordinates: all assets must be in a common neutral-pose/world frame; Y-up default.
"""
from __future__ import annotations
import copy
import json
import hashlib
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from mesh_core import read_mesh
from format_io import load_geometry

SCHEMA = 'viewslice.layered-character.v1'
SLOTS = ('body', 'hair', 'clothing', 'accessory', 'extra')
SUBTYPES = ('top', 'bottom', 'dress', 'outerwear', 'shoes', 'gloves', 'hat', 'hair_front',
            'hair_back', 'hair_side', 'tail', 'wings', 'tentacles', 'other')


def _safe_name(text):
    clean = re.sub(r'[^\w-]+', '_', str(text), flags=re.UNICODE).strip('_')
    return (clean[:64] or 'item')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _write_json_exclusive(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def _save_glb(mesh, dest):
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f'이미 파일이 존재합니다: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dest)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise IOError('GLB 파일 출력 실패')
    return dest


def _store_asset_source(path,dest):
    """Keep whole GLB byte-for-byte, including material extensions and instances.

    This is an import/asset-storage path, NOT a promise that preview/retarget
    can edit an opaque extension without loss.
    """
    source=Path(path)
    if source.suffix.lower()=='.glb':
        from gltf_raw_editor import read_glb,validate_raw_edit
        meta,_=read_glb(source)
        validate_raw_edit(meta)
        dest=Path(dest)
        if dest.exists():raise FileExistsError(dest)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as inp,dest.open('xb') as out:
            shutil.copyfileobj(inp,out)
        return dest
    return _save_glb(_load_part_source(source),dest)


def _load_part_source(path):
    """Retain all static GLB instances/materials when initially storing a part."""
    from scene_core import load_static_scene
    path=Path(path)
    return load_static_scene(path) if path.suffix.lower() in ('.glb','.gltf') else load_geometry(path)


def _read_single_part(path,operation):
    """Legacy enveloping cannot safely work on a multi-node / multi-mesh part."""
    from scene_core import load_static_scene
    scene=load_static_scene(path)
    if len(scene.geometry)!=1 or len(scene.graph.nodes_geometry)!=1:
        raise ValueError(operation+'에는 단일 형상만 지원됩니다. 다중 파츠 병합을 방지하기 위해 중단합니다.')
    return read_mesh(path)


def load_project(folder):
    root = Path(folder).resolve()
    data = _read_json(root / 'character.layers.json')
    if data.get('schema') != SCHEMA:
        raise ValueError('지원되지 않는 레이어 프로젝트 버전')
    return data


def asset_path(folder, entry):
    root = Path(folder).resolve()
    rel = Path(entry['file'])
    dest = (root / rel).resolve()
    if not dest.is_relative_to(root) or not dest.is_file():
        raise ValueError('누락되거나 프로젝트 밖을 가리키는 파츠 파일: ' + str(rel))
    return dest


def create_project(folder, body, display_name='Character', profile='humanoid', up_axis='y'):
    if up_axis not in ('y','z'):
        raise ValueError('세로축은 y 또는 z입니다')
    root = Path(folder).resolve()
    if (root/'character.layers.json').exists():
        raise FileExistsError('기존 프로젝트를 덮어쓰지 않습니다')
    root.mkdir(parents=True, exist_ok=True)
    _store_asset_source(body, root/'assets'/'body.glb')
    data = {
        'schema': SCHEMA, 'name': display_name, 'profile': profile,
        'up_axis': up_axis, 'created_utc': _now(), 'edited_utc': _now(),
        'body': {'id':'body', 'slot':'body', 'name':'기본 신체', 'file':'assets/body.glb',
                 'source_file':str(Path(body).resolve()), 'visible':True,
                 'frame':{'up':'+'+up_axis.upper(),'front':'+Z' if up_axis=='y' else '+Y','units':'m'}},
        'parts':[], 'coverage_masks':[], 'anchors':{}, 'fit_history':[],
        'notes':'정면 기준 좌표계와 중립 포즈를 일치시켜야 자동 재적합이 유효합니다.'
    }
    _write_json_exclusive(root/'character.layers.json', data)
    return data


def add_part(folder, model, slot, subtype='other', name=None, coverage=None, compatible=None):
    if slot not in SLOTS or slot=='body':
        raise ValueError('헤어/의상/액세서리/추가 파츠 중 선택하세요')
    data = load_project(folder)
    root = Path(folder).resolve()
    key = _safe_name(name or Path(model).stem)
    ident = uuid4().hex[:10]
    rel = f'assets/{slot}_{key}_{ident}.glb'
    _store_asset_source(model, root/rel)
    record = {'id':ident, 'slot':slot, 'subtype':subtype, 'name':name or Path(model).stem,
              'file':rel, 'source_file':str(Path(model).resolve()),
              'visible':True, 'coverage':list(coverage or []),
              'compatible_profiles':list(compatible or [data['profile']]),
              'rig_bind':'unbound', 'fit':None,
              'frame':copy.deepcopy(data['body'].get('frame',{'up':'+Y','front':'+Z','units':'m'}))}
    data['parts'].append(record)
    data['edited_utc']=_now()
    _write_json_atomic(root/'character.layers.json',data)
    return record


def set_visibility(folder, part_id, visible):
    data = load_project(folder)
    matches = [p for p in [data['body']]+data['parts'] if p['id']==part_id]
    if len(matches)!=1:
        raise KeyError(part_id)
    matches[0]['visible'] = bool(visible)
    _write_json_atomic(Path(folder)/'character.layers.json', data)


def set_body_mask(folder, part_id, hidden_regions):
    data = load_project(folder)
    if not any(p['id']==part_id for p in data['parts']):
        raise KeyError(part_id)
    data['coverage_masks'] = [r for r in data['coverage_masks'] if r['part_id']!=part_id]
    data['coverage_masks'].append({'part_id':part_id,'regions':list(hidden_regions),
                                   'non_destructive':True})
    _write_json_atomic(Path(folder)/'character.layers.json',data)


def _body_slices(vertices, up_axis='y', steps=80):
    v = np.asarray(vertices, dtype=np.float64)
    ax = 1 if up_axis=='y' else 2
    depth = 2 if ax==1 else 1
    hmin, hmax = float(v[:,ax].min()), float(v[:,ax].max())
    if hmax-hmin < 1e-7: raise ValueError('몸체 높이가 너무 얇습니다')
    t = np.clip((v[:,ax]-hmin)/(hmax-hmin), 0,1)
    mid = np.array([np.median(v[:,0]),np.median(v[:,depth])])
    out = np.zeros((steps,4), dtype=np.float64)
    for i in range(steps):
        selected = np.abs(t-(i+0.5)/steps) <= 1.5/steps
        # Fallback to nearest N at sparse heights, no invented anatomy
        if selected.sum() < 5:
            order = np.argpartition(np.abs(t-(i+0.5)/steps), min(len(v)-1, 64))[:min(len(v),65)]
            sub = v[order]
        else:
            sub = v[selected]
        out[i,0]=.5*(np.quantile(sub[:,0],0.02)+np.quantile(sub[:,0],0.98))
        out[i,1]=max(.5*(np.quantile(sub[:,0],0.98)-np.quantile(sub[:,0],0.02)),1e-5)
        out[i,2]=.5*(np.quantile(sub[:,depth],0.02)+np.quantile(sub[:,depth],0.98))
        out[i,3]=max(.5*(np.quantile(sub[:,depth],0.98)-np.quantile(sub[:,depth],0.02)),1e-5)
    # Sampled contours can change abruptly around extremities: local smoothing
    # (center 0, radius 1, center 2, radius 3)
    padded = np.pad(out, ((2,2),(0,0)),mode='edge')
    for j in range(4):
        out[:,j] = np.convolve(padded[:,j], [1,2,3,2,1],mode='valid')/9
    return (hmin,hmax,ax,depth,out)


def _interp_slices(info, tn):
    samples=info[-1]
    q=np.clip(np.asarray(tn,dtype=np.float64),0,1)*(len(samples)-1)
    i=np.floor(q).astype(int)
    j=np.clip(i+1,0,len(samples)-1)
    u=(q-i)[:,None]
    return samples[i]*(1-u)+samples[j]*u


def fit_mesh(garment, source_body, target_body, *, up_axis='y', strength=1.0,
             clearance=0.006, collision=True, max_scale=2.2):
    """Reposition garment vertices using source vs target body height-sliced envelopes.

    Body coordinates and garment must share a neutral-pose frame. Non-destructive:
    returns new mesh; source unchanged. Works for arbitrary outer silhouettes, but
    does not infer seams, cuffs, cutouts, physical draping, cloth thickness or topology.
    """
    import trimesh
    if not 0<=strength<=1:raise ValueError('strength는 0~1 범위')
    if clearance<0:raise ValueError('clearance는 음수가 될 수 없음')
    src=_body_slices(source_body.vertices,up_axis)
    dst=_body_slices(target_body.vertices,up_axis)
    old=np.asarray(garment.vertices,dtype=np.float64)
    new=old.copy()
    if not len(old):raise ValueError('파츠 정점이 비어 있습니다')
    ymin,ymax,ax,depth,_=src
    t=(old[:,ax]-ymin)/(ymax-ymin)
    src_s=_interp_slices(src,t)
    dst_s=_interp_slices(dst,t)
    src_hw=np.maximum(src_s[:,1],1e-4*(ymax-ymin))
    src_hd=np.maximum(src_s[:,3],1e-4*(ymax-ymin))
    ratio_x=np.clip(dst_s[:,1]/src_hw,1/max_scale,max_scale)
    ratio_d=np.clip(dst_s[:,3]/src_hd,1/max_scale,max_scale)
    # Height changes + local volume changes, while preserving original vertex topology.
    new[:,ax]=dst[0]+(old[:,ax]-src[0])*(dst[1]-dst[0])/(src[1]-src[0])
    new[:,0]=dst_s[:,0]+(old[:,0]-src_s[:,0])*ratio_x
    new[:,depth]=dst_s[:,2]+(old[:,depth]-src_s[:,2])*ratio_d
    new=old*(1-strength)+new*strength
    displaced=0
    absolute_clearance=clearance*(dst[1]-dst[0])
    if collision and clearance>0 and strength>0:
        # Conservative 2D slice envelope push-out; this is not accurate mesh/triangle collision.
        target_now=_interp_slices(dst,(new[:,ax]-dst[0])/(dst[1]-dst[0]))
        dx=new[:,0]-target_now[:,0]; dz=new[:,depth]-target_now[:,2]
        radii_x=np.maximum(target_now[:,1]+absolute_clearance,1e-6)
        radii_z=np.maximum(target_now[:,3]+absolute_clearance,1e-6)
        radial=np.sqrt((dx/radii_x)**2+(dz/radii_z)**2)
        inside=(radial<1) & (t>=0) & (t<=1)
        if np.any(inside):
            # Prevent arbitrarily moving parts near the central axis when pose mismatch occurs.
            valid=inside&(radial>0.05)
            move_factor=np.minimum((1-np.maximum(radial[valid],0.05))/np.maximum(radial[valid],0.05),.35)
            new[valid,0] += dx[valid]*move_factor
            new[valid,depth] += dz[valid]*move_factor
            displaced=int(np.count_nonzero(valid))
    result=garment.copy()
    result.vertices=new
    return result, {'vertices':int(len(old)),'faces':int(len(garment.faces)),
                    'mean_displacement':float(np.linalg.norm(new-old,axis=1).mean()),
                    'max_displacement':float(np.linalg.norm(new-old,axis=1).max()),
                    'approx_collision_pushed_vertices':displaced,
                    'method':'height-sliced envelope deformation + optional approximate push-out',
                    'clearance_fraction_of_body_height':float(clearance),
                    'new_geometry_generated':False,'topology_preserved':True,
                    'limitations':'참고 두 캐릭터 중립 포즈/방향 일치 필수. 옷솔기·재봉·주름·실제 충돌·물리 시뮬레이션 미구현.'}


def export_parts_preview(folder, out_file, apply_masks=False):
    import trimesh
    root=Path(folder).resolve()
    data=load_project(root)
    scene=trimesh.Scene()
    for entry in [data['body']]+data['parts']:
        if entry.get('visible',True):
            if entry.get('skin_draft'):
                validation=validate_part_skin(root,entry['id'])
                if validation.get('action')=='block':
                    raise ValueError('스키닝 정점 구조가 달라져 미리보기를 중단합니다: '+entry['id'])
                if validation.get('action')=='warn':
                    import warnings
                    warnings.warn('스키닝 초안 메시 위치가 변경됐습니다. 가중치 확인 권장: '+entry['id'],RuntimeWarning)
            from scene_core import load_static_scene
            sub=load_static_scene(asset_path(root,entry))
            if apply_masks and entry['id']=='body':
                regions=[]
                visible_ids={p['id'] for p in data['parts'] if p.get('visible',True)}
                for record in data['coverage_masks']:
                    if record['part_id'] in visible_ids:regions.extend(record['regions'])
                if regions:
                    if len(sub.geometry)!=1 or len(sub.graph.nodes_geometry)!=1:
                        raise ValueError('다중 인스턴스 신체의 마스크는 보존 검증 전까지 적용 불가')
                    mesh=_read_single_part(asset_path(root,entry),'신체 마스크')
                    mesh,_=body_region_hide_preview(mesh,regions,data['profile'],data['up_axis'])
                    sub=trimesh.Scene()
                    sub.add_geometry(mesh,geom_name='masked_body',node_name='masked_body')
            prefix=entry['id']+'__'
            # Store each unique geometry once; node graph references remain shared.
            for geom_name,geometry in sub.geometry.items():
                scene.geometry[prefix+geom_name]=geometry.copy()
            for node in sub.graph.nodes_geometry:
                transform,geom_name=sub.graph[node]
                scene.graph.update(frame_from=scene.graph.base_frame,frame_to=prefix+node,
                                   matrix=transform.copy(),geometry=prefix+geom_name)
    outfile=Path(out_file)
    if outfile.exists():raise FileExistsError(outfile)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    # Preview, not guaranteed bone/morph-preserving; separate independent assets stay intact.
    scene.export(str(outfile),file_type='glb')
    return outfile


def transfer_part(donor_folder, part_id, receiver_folder, *, strength=1.0, clearance=0.006,
                  collision=True, copy_coverage=True):
    donor_root=Path(donor_folder).resolve()
    target_root=Path(receiver_folder).resolve()
    if donor_root==target_root:raise ValueError('동일 프로젝트 내 복제는 add_part 또는 별도 복제 기능을 사용하세요')
    donor=load_project(donor_root)
    receiver=load_project(target_root)
    if donor['up_axis']!=receiver['up_axis']:
        raise ValueError('두 프로젝트의 세로축을 일치시켜 주세요')
    matches=[p for p in donor['parts'] if p['id']==part_id]
    if len(matches)!=1:raise KeyError('해당 파츠 ID를 찾지 못했습니다: '+part_id)
    entry=matches[0]
    src_body=_read_single_part(asset_path(donor_root,donor['body']),'의상 전이')
    dst_body=_read_single_part(asset_path(target_root,receiver['body']),'의상 전이')
    original=_read_single_part(asset_path(donor_root,entry),'의상 전이')
    fitted,stats=fit_mesh(original,src_body,dst_body,up_axis=donor['up_axis'],
                          strength=strength,clearance=clearance,collision=collision)
    ident=uuid4().hex[:10]
    rel=f'assets/{entry["slot"]}_{_safe_name(entry["name"])}_fit_{ident}.glb'
    _save_glb(fitted,target_root/rel)
    new=copy.deepcopy(entry)
    new.update({'id':ident,'file':rel,'source_project':str(donor_root),
                'source_part_id':part_id,'rig_bind':'requires_transfer',
                'fit':{'timestamp':_now(),'strength':strength,'clearance':clearance,
                       'collision':collision, 'stats':stats}})
    receiver['parts'].append(new)
    if copy_coverage and entry.get('coverage'):
        receiver['coverage_masks'].append({'part_id':ident,'regions':entry['coverage'],'non_destructive':True})
    receiver['fit_history'].append({'part_id':ident,'donor':str(donor_root),'original_part_id':part_id,
                                    'stats':stats})
    receiver['edited_utc']=_now()
    _write_json_atomic(target_root/'character.layers.json',receiver)
    return new,stats


def transfer_skin_weights(part_vertices, target_body_vertices, body_bone_indices, body_bone_weights):
    """Nearest-vertex weight transfer; returns arrays and does not claim rig binding.
    Existing target-body weight arrays must be provided by caller; no fabricated bones.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as ex:
        raise RuntimeError('가중치 전이에만 scipy가 추가로 필요합니다') from ex
    src=np.asarray(target_body_vertices)
    idx=np.asarray(body_bone_indices)
    weight=np.asarray(body_bone_weights,dtype=float)
    if len(src)!=len(idx) or idx.shape!=weight.shape:
        raise ValueError('대상 바디 정점과 가중치 데이터가 일치하지 않습니다')
    _,nearest=cKDTree(src).query(np.asarray(part_vertices),k=1)
    out_idx=idx[nearest].copy()
    out_weight=weight[nearest].copy()
    sums=out_weight.sum(axis=1,keepdims=True)
    out_weight/=np.maximum(sums,1e-12)
    return out_idx,out_weight


def inventory_scene(source):
    """Inspect object names for manual semantic tagging; never auto-label arbitrary geometry."""
    import trimesh
    loaded=trimesh.load(str(source),force='scene',process=False)
    rows=[]
    for i,node in enumerate(sorted(loaded.graph.nodes_geometry)):
        mat,geom_name=loaded.graph[node]
        mesh=loaded.geometry[geom_name]
        rows.append({'index':i,'node':node,'geometry':geom_name,
                     'vertices':int(len(mesh.vertices)),'faces':int(len(mesh.faces))})
    return rows


def extract_scene_object(source, index, dest):
    import trimesh
    loaded=trimesh.load(str(source),force='scene',process=False)
    nodes=sorted(loaded.graph.nodes_geometry)
    if not 0<=index<len(nodes):raise IndexError('선택한 메시 번호를 찾지 못했습니다')
    mat,name=loaded.graph[nodes[index]]
    mesh=loaded.geometry[name].copy()
    mesh.apply_transform(mat)
    return _save_glb(mesh,dest)


def _vertex_identity(vertices,faces=None):
    v=np.asarray(vertices,dtype='<f8',order='C')
    info={'count':int(len(v)),
          'position_hash':'sha256:'+hashlib.sha256(v.tobytes()).hexdigest()}
    if faces is not None:
        f=np.asarray(faces,dtype='<i8',order='C')
        info['topology_hash']='sha256:'+hashlib.sha256(f.tobytes()).hexdigest()
        info['face_count']=int(len(f))
    return info


def validate_part_skin(folder,part_id):
    """Check for silently stale vertex-ordered weight drafts after refinement."""
    root=Path(folder).resolve()
    data=load_project(root)
    item=next((p for p in data['parts'] if p['id']==part_id),None)
    if item is None: raise KeyError(part_id)
    sidecar=root/'skins'/f'{part_id}.skin.json'
    if not sidecar.exists():return {'valid':False,'reason':'no_skin_draft'}
    metadata=_read_json(sidecar)
    expected=metadata.get('vertex_identity')
    if not expected:return {'valid':False,'reason':'legacy_draft_missing_vertex_identity'}
    mesh=read_mesh(asset_path(root,item))
    current=_vertex_identity(mesh.vertices,mesh.faces)
    count_diff=current['count']!=expected['count']
    topo_diff=('topology_hash' in expected and
               current['topology_hash']!=expected['topology_hash'])
    if count_diff or topo_diff:
        return {'valid':False,'action':'block','reason':'vertex_topology_changed_rebind_required',
                'expected':expected,'current':current}
    if current['position_hash']!=expected['position_hash']:
        return {'valid':False,'action':'warn','reason':'vertex_identity_changed_retransfer_required',
                'expected':expected,'current':current}
    return {'valid':True,'action':'continue','reason':'ok','expected':expected,'current':current}


def prepare_part_skin(folder, part_id, rig_plan_file=None, chunk_size=32768):
    """Create a portable rig-weight *draft* for one part; no Blender/PMX binding claimed.

    If an existing target-rig JSON is supplied, use its bones. Otherwise generate
    a profile template from the target body. Results are intentionally separate
    from geometry so future refinements do not silently erase stored authoring data.
    """
    from rigging import rig_plan, skinning_weights
    root=Path(folder).resolve()
    data=load_project(root)
    entry=next((p for p in data['parts'] if p['id']==part_id),None)
    if entry is None:raise KeyError(part_id)
    if rig_plan_file:
        plan=_read_json(rig_plan_file)
        if 'bones' not in plan or not plan['bones']:
            raise ValueError('뼈대 JSON에 bones 데이터가 없습니다')
    else:
        body=_read_single_part(asset_path(root,data['body']),'스키닝 초안')
        plan=rig_plan(body.vertices,profile=data['profile'],up_axis=data['up_axis'])
    mesh=_read_single_part(asset_path(root,entry),'스키닝 초안')
    v=np.asarray(mesh.vertices)
    bone_ids=[];weights=[]
    for start in range(0,len(v),max(1,int(chunk_size))):
        ids,w=skinning_weights(v[start:start+chunk_size],plan['bones'],max_influences=4)
        bone_ids.append(ids.astype(np.int32,copy=False))
        weights.append(w.astype(np.float32,copy=False))
    if not bone_ids:raise ValueError('빈 파츠 메시')
    dest=root/'skins'/f'{part_id}.npz'
    sidecar=root/'skins'/f'{part_id}.skin.json'
    if dest.exists() or sidecar.exists():
        raise FileExistsError('이미 스키닝 초안이 존재합니다. 덮어쓰지 않습니다')
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,indices=np.concatenate(bone_ids),weights=np.concatenate(weights))
    meta={
        'schema':'viewslice.skin-weight-draft.v1',
        'part_id':part_id, 'rig_profile':plan.get('profile',data['profile']),
        'bone_names':[b['name'] for b in plan['bones']],
        'bone_definitions':plan['bones'],
        'indices_file':f'skins/{part_id}.npz',
        'vertex_identity':_vertex_identity(v,mesh.faces),
        'source':'template_nearest_bone' if rig_plan_file is None else 'provided_rig_plan_nearest_bone',
        'status':'draft_only','notes':'블렌더 Armature에 적용된 상태가 아닙니다. 품질을 위해 weight paint 보정이 필요할 수 있습니다.'
    }
    _write_json_exclusive(sidecar,meta)
    entry['rig_bind']='weight_draft_only'
    entry['skin_draft']=f'skins/{part_id}.skin.json'
    data['edited_utc']=_now()
    _write_json_atomic(root/'character.layers.json',data)
    return {'skin':str(sidecar),'weights':str(dest),'bones':len(plan['bones']),'vertices':len(v)}


def body_region_hide_preview(mesh, coverage, profile='humanoid', up_axis='y'):
    """Non-destructive *approximate* region-based hiding for combined preview only.

    Uses simple position rules, not true cloth-surface coverage. Source mesh untouched.
    """
    if not coverage:return mesh.copy(),0
    import trimesh
    vertices=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.faces)
    ax=1 if up_axis=='y' else 2
    ymin,ymax=vertices[:,ax].min(),vertices[:,ax].max()
    y=(vertices[faces].mean(axis=1)[:,ax]-ymin)/max(ymax-ymin,1e-9)
    center=np.median(vertices[:,0])
    dx=np.abs(vertices[faces].mean(axis=1)[:,0]-center)
    lateral_scale=max(float(np.ptp(vertices[:,0])),1e-6)
    hidden=np.zeros(len(faces),dtype=bool)
    regions=set(coverage)
    if profile=='humanoid':
        if 'head' in regions:hidden|=(y>=.83)
        if 'neck' in regions:hidden|=((y>=.77)&(y<.87))
        if 'torso' in regions:hidden|=((y>=.45)&(y<.80)&(dx<.34*lateral_scale))
        if 'arms' in regions:hidden|=((y>=.45)&(y<.82)&(dx>=.34*lateral_scale))
        if 'legs' in regions or 'pants' in regions:hidden|=((y>=.12)&(y<.45))
        if 'feet' in regions:hidden|=(y<.12)
    else:
        if 'upper' in regions:hidden|=(y>=.67)
        if 'middle' in regions:hidden|=((y>=.33)&(y<.67))
        if 'lower' in regions:hidden|=(y<.33)
    if not np.any(hidden):return mesh.copy(),0
    # Keep vertex indices stable; preview only. Standalone body GLB never changed.
    subset=trimesh.Trimesh(vertices=vertices.copy(),faces=faces[~hidden].copy(),process=False)
    return subset,int(np.count_nonzero(hidden))


def inventory_components(source, min_faces=1):
    """List geometric connected components, without guessing their semantic role.

    Large fused or welded objects cannot be semantically separated by this method.
    Costs grow with face adjacency, so run on demand only (never in normal imports).
    """
    mesh=load_geometry(source)
    parts=sorted(mesh.split(only_watertight=False),key=lambda x:len(x.faces),reverse=True)
    return [{'index':i,'vertices':int(len(p.vertices)),'faces':int(len(p.faces)),
             'extents':[float(x) for x in p.extents]} for i,p in enumerate(parts) if len(p.faces)>=min_faces]


def extract_component(source,index,dest):
    mesh=load_geometry(source)
    parts=sorted(mesh.split(only_watertight=False),key=lambda x:len(x.faces),reverse=True)
    if not 0<=index<len(parts):raise IndexError('선택한 연결 구성 요소가 없습니다')
    return _save_glb(parts[index],dest)
