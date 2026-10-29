"""Format bridge for ViewSlice. Explicitly separates mesh-only conversion from rigged scenes.
No destructive writes. Blender bridge is optional for proprietary/scene formats.
"""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path

DIRECT_IN={'.glb','.gltf','.obj','.ply','.stl','.off','.3mf','.dae'}
DIRECT_OUT={'.glb','.gltf','.obj','.ply','.stl','.off','.3mf','.dae','.pmx'}
BLENDER_IN={'.fbx','.blend','.usd','.usda','.usdc','.abc'}
BLENDER_OUT={'.fbx','.blend','.usd','.usda','.usdc','.abc'}
RIG_OUT={'.glb','.fbx','.blend','.pmx'}
ALL_INPUTS=sorted(DIRECT_IN|BLENDER_IN)
ALL_OUTPUTS=sorted(DIRECT_OUT|BLENDER_OUT)

NO_RIG={'.obj','.ply','.stl','.off','.3mf','.dae','.abc'}


def _blender_convert(src:Path, dest:Path, blender_exe=''):
    from rig_process import locate_blender
    executable=locate_blender(blender_exe)
    if not executable:
        raise RuntimeError(f'{src.suffix}→{dest.suffix} 변환은 Blender가 필요합니다. Blender를 설치하거나 --blender 경로를 지정해 주세요.')
    bridge=Path(__file__).with_name('formats_blender.py')
    cmd=[executable,'--background','--factory-startup','--python',str(bridge),'--','--input',str(src),'--output',str(dest)]
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=1800)
    if result.returncode or not dest.exists():
        raise RuntimeError('Blender 형식 변환 실패\n'+result.stdout[-2500:])


def load_geometry(path, blender_exe=''):
    """Return triangle mesh. WARNING: this is not an armature-preserving import."""
    from mesh_core import read_mesh
    p=Path(path).resolve()
    if p.suffix.lower() in DIRECT_IN:
        return read_mesh(p)
    if p.suffix.lower() in BLENDER_IN:
        with tempfile.TemporaryDirectory(prefix='viewslice-import-') as d:
            intermediate=Path(d)/'mesh.glb'
            _blender_convert(p,intermediate,blender_exe)
            return read_mesh(intermediate)
    raise ValueError(f'지원하지 않는 입력 형식: {p.suffix}. 지원: {", ".join(ALL_INPUTS)}')


def _safe_write(path, content):
    path=Path(path)
    if path.exists():raise FileExistsError(f'출력 파일이 이미 존재합니다: {path}')
    if isinstance(content,str):path.write_text(content,encoding='utf-8')
    else:path.write_bytes(content)


def export_geometry(mesh, dest, blender_exe=''):
    """Export static edited mesh. Only GLB/GLTF/FBX/BLEND can carry rigs, but this is not a rig exporter."""
    import trimesh
    dest=Path(dest).resolve()
    ext=dest.suffix.lower()
    if ext not in DIRECT_OUT|BLENDER_OUT:
        raise ValueError(f'지원하지 않는 출력 형식: {ext}. 지원: {", ".join(ALL_OUTPUTS)}')
    if dest.exists():raise FileExistsError(f'이미 존재하는 결과: {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if ext=='.pmx':
        from pmx_export import write_pmx
        return write_pmx(mesh,dest)
    if ext in BLENDER_OUT:
        with tempfile.TemporaryDirectory(prefix='viewslice-export-') as d:
            intermediate=Path(d)/'mesh.glb'
            mesh.export(str(intermediate))
            _blender_convert(intermediate,dest,blender_exe)
        return [str(dest)]
    if ext=='.gltf':
        assets=trimesh.Scene(mesh).export(file_type='gltf')
        main=[name for name in assets if name.endswith('.gltf')]
        if len(main)!=1:raise RuntimeError('glTF 주 파일 생성 실패')
        data=json.loads(assets[main[0]])
        # Avoid buffer collision when exporting multiple models into one folder.
        mapping={name: f'{dest.stem}_{name}' for name in assets if name!=main[0]}
        for entry in data.get('buffers',[]):
            uri=entry.get('uri')
            if uri in mapping:entry['uri']=mapping[uri]
        for entry in data.get('images',[]):
            uri=entry.get('uri')
            if uri in mapping:entry['uri']=mapping[uri]
        targets=[dest.parent/new for new in mapping.values()]+[dest]
        if any(x.exists() for x in targets):
            raise FileExistsError('glTF 보조 파일 이름이 기존 파일과 충돌합니다')
        for old,new in mapping.items():_safe_write(dest.parent/new,assets[old])
        _safe_write(dest,json.dumps(data,ensure_ascii=False,indent=2))
        return [str(x) for x in targets]
    if ext=='.obj':
        from trimesh.exchange.obj import export_obj
        txt, materials=export_obj(mesh,return_texture=True,mtl_name=dest.stem+'.mtl')
        targets=[dest]+[dest.parent/k for k in materials]
        if any(p.exists() for p in targets):raise FileExistsError('OBJ 보조 파일 이름이 기존 파일과 충돌합니다')
        _safe_write(dest,txt)
        for k,v in materials.items():
            if Path(k).name!=k:raise ValueError('OBJ 출력에 안전하지 않은 리소스 이름이 포함되어 있습니다')
            _safe_write(dest.parent/k,v)
        return [str(p) for p in targets]
    if ext=='.dae':
        try:
            data=mesh.export(file_type='dae')
        except ImportError as ex:
            raise RuntimeError('DAE 출력은 pycollada가 필요합니다: python -m pip install pycollada') from ex
    else:data=mesh.export(file_type=ext[1:])
    _safe_write(dest,data)
    return [str(dest)]
