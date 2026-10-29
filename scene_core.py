"""Non-destructive scene-aware editing for unrigged GLB/glTF.

Preserve per-geometry identity, node transforms and materials. This is NOT a
rigged-scene round trip: glTF skins, animations and morph targets are rejected
before import so they are never silently discarded.
"""
from __future__ import annotations
import json
from pathlib import Path
import struct
import trimesh


def reject_unsupported_attributes(path):
    """Fail closed on glTF data that trimesh cannot promise to round-trip."""
    data=_gltf_json(path)
    if not data:return
    causes=[]
    if data.get('extensionsRequired') or data.get('extensionsUsed'):
        causes.append('glTF extensions')
    if data.get('extras') or data.get('extensions'):
        causes.append('root metadata')
    for item in data.get('nodes',[])+data.get('meshes',[]):
        extras=item.get('extras') or {}
        known_trimesh_extras={'shape','radius','units','from_gltf_primitive','extents','height','sections','processed','subdivisions','origin','transform','name','node'}
        if item.get('extensions') or any(key not in known_trimesh_extras for key in extras):
            causes.append('node/mesh metadata')
    for mesh in data.get('meshes',[]):
        if len({prim.get('material') for prim in mesh.get('primitives',[])})>1:
            causes.append('multiple primitive materials (unverified preservation)')
        for prim in mesh.get('primitives',[]):
            if prim.get('extras') or prim.get('extensions'):
                causes.append('primitive metadata')
            attrs=prim.get('attributes',{})
            if any(k not in {'POSITION','NORMAL','TANGENT','TEXCOORD_0','COLOR_0','_VERTEX_ID','_SOURCE_VERTEX'} for k in attrs):
                causes.append('extra vertex attributes')
    if causes:
        raise ValueError('보존 여부가 검증되지 않은 glTF 속성을 감지해 편집을 중단했습니다: '+', '.join(sorted(set(causes))))


def _gltf_json(path):
    path=Path(path)
    if path.suffix.lower()=='.gltf':
        return json.loads(path.read_text(encoding='utf-8'))
    if path.suffix.lower()!='.glb':return None
    with path.open('rb') as f:
        hdr=f.read(12)
        if len(hdr)!=12:return None
        magic,ver,total=struct.unpack('<4sII',hdr)
        if magic!=b'glTF' or ver!=2:raise ValueError('GLB 2.0 헤더가 아닙니다')
        chunk=f.read(8)
        if len(chunk)!=8:raise ValueError('GLB JSON 청크 없음')
        size,typ=struct.unpack('<I4s',chunk)
        if typ!=b'JSON' or size>32*1024*1024:raise ValueError('GLB JSON 청크가 유효하지 않습니다')
        return json.loads(f.read(size).decode('utf-8').rstrip(' \x00'))


def reject_rigged(path):
    data=_gltf_json(path)
    if not data:return
    present=[k for k in ('skins','animations') if data.get(k)]
    if any(prim.get('targets') for mesh in data.get('meshes',[]) for prim in mesh.get('primitives',[])):
        present.append('morph targets')
    if present:
        raise ValueError('리깅/모션/모프 포함 glTF는 현재 정적 메시 편집에서 보존할 수 없습니다: '
                         +', '.join(present)+'. 원본 보존을 위해 중단합니다. 별도 바인딩 해제 복사본을 사용하세요.')


def load_static_scene(path):
    p=Path(path)
    if p.suffix.lower() in ('.glb','.gltf'):
        reject_rigged(p)
        reject_unsupported_attributes(p)
    scene=trimesh.load(str(p),force='scene',process=False)
    if not scene.geometry:raise ValueError('형상이 없는 장면')
    for key,geo in scene.geometry.items():
        if not isinstance(geo,trimesh.Trimesh):raise ValueError(f'메시가 아닌 형상: {key}')
    return scene


def process_static_scene(scene, mode='progressive', steps=1, reference_dir='',
                         strength=.2, up_axis='y', budget=None,region='all',detail_strength=.12,
                         height_map='',height_amplitude=0,target_part='',vertex_mask=None,lineage_sink=None,height_ratio=0):
    """One geometry name can be instanced by multiple scene nodes; process it ONCE.
    Referential silhouette fitting on multi-instance scenes is unsafe and rejected.
    """
    from mesh_core import process_mesh
    if reference_dir and len(scene.geometry)>1:
        raise ValueError('다중 파츠 장면에서 전역 실루엣 보정을 실행하면 옷·헤어 형태가 뒤섞일 수 있습니다. 파츠를 따로 선택하세요.')
    if reference_dir:
        import numpy as np
        for node in scene.graph.nodes_geometry:
            transform,_=scene.graph[node]
            if not np.allclose(transform,np.eye(4)):
                raise ValueError('월드 변환된 장면은 전역 실루엣 보정 불가. 먼저 편집용 복사본에서 변환을 적용하세요.')
    if target_part and target_part not in scene.geometry:
        raise ValueError('요청된 파츠 이름이 장면에 없습니다: '+str(target_part))
    if height_map and not target_part and len(scene.geometry)>1:
        raise ValueError('다중 파츠 높이맵은 반드시 --target-part를 지정해야 합니다')
    if vertex_mask is not None and not target_part:
        raise ValueError('정점 영역 마스크를 지정할 때는 대상 파츠가 필요합니다')
    result=scene.copy()
    reports={}
    # Budget is a per-run goal, not a per-part multiplier.
    counts={name:max(1,len(mesh.faces)) for name,mesh in scene.geometry.items()
            if not target_part or name==target_part}
    weight_total=sum(counts.values())
    for name,geo in scene.geometry.items():
        if target_part and name!=target_part:
            reports[name]={'mode':'untouched','before_faces':len(geo.faces),
                           'after_faces':len(geo.faces),'detail_changed':False,'levels':[]}
            continue
        local_budget=None if budget is None else max(1,round(budget*counts[name]/weight_total))
        steps_lineage=[] if lineage_sink is not None else None
        edited,stats=process_mesh(geo,mode,steps,reference_dir,strength,up_axis,
                                  local_budget,region,detail_strength,
                                  height_map=height_map,height_amplitude=height_amplitude,
                                  vertex_mask=vertex_mask,lineage_sink=steps_lineage,height_ratio=height_ratio)
        if lineage_sink is not None:lineage_sink[name]=steps_lineage
        result.geometry[name]=edited
        reports[name]=stats
    return result,{'scene_preserved':True,'geometry_count':len(result.geometry),
                   'node_count':len(result.graph.nodes_geometry), 'parts':reports,
                   'added_faces':sum(sum(x.get('added_faces',0) for x in r.get('levels',[])) for r in reports.values()),
                   'detail_changed':any(r.get('detail_changed',False) for r in reports.values()),
                   'warning':'정적 메시 장면에 한정. 기존 뼈/애니메이션/모프는 로드 이전에 검사하여 거부함'}


def export_static_scene(scene, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if output.suffix.lower()!='.glb':
        raise ValueError('레이어 보존 출력은 현재 .glb만 지원합니다. .gltf 등으로 바꿀 때는 별도 변환을 사용하세요.')
    output.parent.mkdir(parents=True,exist_ok=True)
    import os,tempfile
    fd,name=tempfile.mkstemp(prefix='.viewslice_',suffix='.glb',dir=output.parent)
    os.close(fd)
    try:
        scene.export(name,file_type='glb')
        os.replace(name,output)
    finally:
        if os.path.exists(name):os.unlink(name)
    return output
