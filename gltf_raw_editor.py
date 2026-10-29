"""Conservative glTF2 GLB raw-buffer editor for static multi-primitive meshes.

Unlike a trimesh round-trip, preserves existing glTF JSON, material extensions,
extras, images, nodes, instancing, and original binary buffers byte-for-byte;
appends newly generated mesh buffers/accessors. One glTF mesh (all primitives)
is one topological edit domain, never join different glTF meshes.

Limits: float POSITION, ordinary triangles; known non-deforming materials/extensions;
not a skin/morph/animation editor, and never modifies an existing source file.
"""
from __future__ import annotations
import copy
import json
import struct
import os
from pathlib import Path
import numpy as np
from adaptive import adaptive_refine, expand_seam_mask

CTYPE={5120:np.dtype('<i1'),5121:np.dtype('<u1'),5122:np.dtype('<i2'),5123:np.dtype('<u2'),5125:np.dtype('<u4'),5126:np.dtype('<f4')}
CTYPE_R={np.dtype('<f4'):5126,np.dtype('<u4'):5125}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}
SAFE_ATTR={'POSITION','NORMAL','TEXCOORD_0','TEXCOORD_1','COLOR_0','_VERTEX_ID','_SOURCE_VERTEX'}
SAFE_MATERIAL_EXT={'KHR_texture_transform','KHR_materials_unlit','KHR_materials_emissive_strength',
                   'KHR_materials_pbrSpecularGlossiness','KHR_materials_clearcoat','KHR_materials_transmission',
                   'KHR_materials_ior','KHR_materials_sheen','KHR_materials_specular','KHR_materials_volume',
                   'KHR_materials_iridescence','KHR_materials_anisotropy','KHR_materials_dispersion',
                   'KHR_texture_basisu'}


def read_glb(path):
    buf=Path(path).read_bytes()
    if len(buf)<20 or buf[:4]!=b'glTF' or struct.unpack_from('<I',buf,4)[0]!=2:
        raise ValueError('GLB 2.0 헤더가 유효하지 않습니다')
    if struct.unpack_from('<I',buf,8)[0]!=len(buf):raise ValueError('GLB length mismatch')
    off=12;chunks=[]
    while off<len(buf):
        if off+8>len(buf):raise ValueError('GLB chunk truncated')
        n,kind=struct.unpack_from('<I4s',buf,off);off+=8
        if off+n>len(buf):raise ValueError('GLB chunk overflow')
        chunks.append((kind,buf[off:off+n]));off+=n
    if not chunks or chunks[0][0]!=b'JSON':raise ValueError('GLB JSON 청크가 없습니다')
    data=json.loads(chunks[0][1].decode('utf8').rstrip(' \x00'))
    bins=[c for k,c in chunks if k==b'BIN\x00']
    if len(bins)!=1 or len(data.get('buffers',[]))!=1:
        raise ValueError('다중/외부 버퍼 GLB는 현재 무손실 편집을 지원하지 않습니다')
    if data['buffers'][0].get('uri'):raise ValueError('외부 buffer URI는 편집을 지원하지 않습니다')
    return data,bins[0]


def write_glb(path,data,buffer):
    dest=Path(path)
    if dest.exists():raise FileExistsError(str(dest))
    payload=copy.deepcopy(data)
    while len(buffer)%4:buffer+=b'\x00'
    payload['buffers'][0]['byteLength']=len(buffer)
    j=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode('utf8')
    j+=b' '*((-len(j))%4)
    body=struct.pack('<I4s',len(j),b'JSON')+j+struct.pack('<I4s',len(buffer),b'BIN\x00')+buffer
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(b'glTF'+struct.pack('<II',2,12+len(body))+body)


def validate_raw_edit(data):
    if data.get('skins') or data.get('animations'):
        raise ValueError('스킨/애니메이션이 있는 GLB 편집은 중단합니다')
    used=set(data.get('extensionsUsed',[]))|set(data.get('extensionsRequired',[]))
    if not used.issubset(SAFE_MATERIAL_EXT):
        raise ValueError('검증되지 않은 glTF 확장: '+', '.join(sorted(used-SAFE_MATERIAL_EXT)))
    for m in data.get('meshes',[]):
        for p in m.get('primitives',[]):
            if p.get('targets'):raise ValueError('모프 타깃 있는 GLB 편집 불가')
            if p.get('mode',4)!=4:raise ValueError('삼각형이 아닌 primitive 편집 불가')
            if set(p.get('attributes',{}))-SAFE_ATTR:
                raise ValueError('미지원 정점 속성: '+str(set(p['attributes'])-SAFE_ATTR))
            if 'POSITION' not in p.get('attributes',{}):raise ValueError('POSITION 없음')
            if p.get('extensions'):raise ValueError('primitive geometry extension 미지원')
    for v in data.get('accessors',[]):
        if v.get('sparse'):raise ValueError('sparse accessor 미지원')


def get_accessor(data,buffer,i):
    a=data['accessors'][int(i)]
    dtype=CTYPE[a['componentType']]
    typ=a['type'];ncomp=NC[typ]
    if 'bufferView' not in a:raise ValueError('bufferView 없는 accessor 미지원')
    view=data['bufferViews'][a['bufferView']]
    if view.get('buffer',0)!=0:raise ValueError('두 번째 buffer를 지원하지 않습니다')
    stride=view.get('byteStride',ncomp*dtype.itemsize)
    if stride<ncomp*dtype.itemsize:raise ValueError('잘못된 accessor stride')
    start=view.get('byteOffset',0)+a.get('byteOffset',0)
    count=int(a['count'])
    if count and start+(count-1)*stride+ncomp*dtype.itemsize>len(buffer):
        raise ValueError('accessor가 buffer를 초과합니다')
    # np.ndarray noncontiguous stride and copy allow both interleaved and packed accessors.
    shape=(count,) if ncomp==1 else (count,ncomp)
    strides=(stride,) if ncomp==1 else (stride,dtype.itemsize)
    out=np.ndarray(shape=shape,dtype=dtype,buffer=buffer,offset=start,strides=strides).copy()
    if a.get('normalized'):
        if dtype.kind=='u':out=out.astype(np.float32)/np.iinfo(dtype).max
        elif dtype.kind=='i':out=np.maximum(-1.,out.astype(np.float32)/np.iinfo(dtype).max)
    return out


def put_accessor(data,binary,array,semantic=None):
    a=np.ascontiguousarray(array)
    # glTF 2.0 UNSIGNED_INT (5125) is for indices, not vertex attributes.
    # IEEE754 FLOAT stores integers exactly through 2**24. -1 is the source-ID
    # sentinel for newly generated vertices; unsigned 0xffffffff is not encodable.
    if semantic in ('_VERTEX_ID', '_SOURCE_VERTEX'):
        if a.dtype.kind not in 'iuf':
            raise ValueError('vertex identity attribute must be numeric')
        if semantic=='_SOURCE_VERTEX':
            # NumPy 2.x can keep uint32 here and wrap -1 to UINT32_MAX.
            # Cast before replacing the legacy sentinel to preserve its meaning.
            a=a.astype(np.float64)
            a=np.where(a==np.iinfo(np.uint32).max,-1.0,a)
        if np.any(~np.isfinite(a)) or np.any(a!=np.rint(a)) or np.any(a>16777216) or np.any(a < (-1 if semantic=='_SOURCE_VERTEX' else 0)):
            raise ValueError('float32 vertex-ID range exceeded; use immutable index sidecar')
        a=a.astype('<f4')

    if a.dtype==np.float64:a=a.astype('<f4')
    if a.dtype not in CTYPE_R:
        if a.dtype.kind in 'iu':a=a.astype('<u4')
        else:a=a.astype('<f4')
    ncomp=1 if a.ndim==1 else a.shape[1]
    inv={v:k for k,v in NC.items()}
    if ncomp not in inv:raise ValueError('attribute component count unsupported')
    ofs=(len(binary)+3)//4*4
    binary+=b'\x00'*(ofs-len(binary))
    raw=a.tobytes(order='C')
    binary+=raw
    view={'buffer':0,'byteOffset':ofs,'byteLength':len(raw)}
    data.setdefault('bufferViews',[]).append(view)
    acc={'bufferView':len(data['bufferViews'])-1,'componentType':CTYPE_R[a.dtype],
         'count':len(a),'type':inv[ncomp]}
    if semantic=='POSITION' and len(a):
        acc['min']=a.min(axis=0).astype(float).tolist();acc['max']=a.max(axis=0).astype(float).tolist()
    data.setdefault('accessors',[]).append(acc)
    return len(data['accessors'])-1,binary


def refine_raw_glb(source,dest,*,target_mesh=None,steps=1,budget=1000,vertex_mask=None,
                   lineage_sink=None):
    """Static geometry with arbitrary material primitives; save only via write_glb.

    File-level glTF mesh identity and primitive material boundaries are retained.
    Vertex_ID is local to each glTF mesh; ancestry references local primitive vertices.
    """
    data,buffer=read_glb(source);validate_raw_edit(data)
    new_data=copy.deepcopy(data);new_buffer=buffer
    reports={}
    meshes=data.get('meshes',[])
    names=[m.get('name',f'mesh_{i}') for i,m in enumerate(meshes)]
    if target_mesh and target_mesh not in names:raise ValueError('지정한 glTF mesh가 없습니다: '+target_mesh)
    for mi,mesh in enumerate(meshes):
        if target_mesh and names[mi]!=target_mesh:continue
        prims=mesh['primitives']
        source_attrs=[];positions=[];owners=[];owner_local=[];faces=[];parent_prim=[]
        keys=set()
        for pi,prim in enumerate(prims):
            attrs={name:get_accessor(data,buffer,idx) for name,idx in prim['attributes'].items()}
            n=len(attrs['POSITION']);keys.update(attrs.keys())
            if attrs['POSITION'].shape!=(n,3):raise ValueError('POSITION 형식이 잘못되었습니다')
            if 'indices' in prim:
                faces_local=get_accessor(data,buffer,prim['indices']).reshape(-1)
            else:faces_local=np.arange(n,dtype=np.uint32)
            if len(faces_local)%3 or (len(faces_local) and np.max(faces_local)>=n):
                raise ValueError('잘못된 primitive indices')
            source_attrs.append(attrs)
            faces.append(faces_local.reshape(-1,3).astype(np.int64)+sum(len(x) for x in positions))
            parent_prim.extend([pi]*(len(faces_local)//3))
            positions.append(np.asarray(attrs['POSITION'],dtype=np.float64))
            owners.extend([pi]*n);owner_local.extend(range(n))
        v=np.concatenate(positions,axis=0);f=np.concatenate(faces,axis=0)
        if not len(f):raise ValueError('비어 있는 glTF mesh')
        owners=np.asarray(owners,dtype=np.int64);local=np.asarray(owner_local,dtype=np.int64)
        parent_prim=np.asarray(parent_prim,dtype=np.int64)
        if vertex_mask is not None:
            mask=np.asarray(vertex_mask,dtype=bool)
            if len(mask)!=len(v):raise ValueError('glTF mesh 전체 정점 수와 영역 마스크 길이가 다릅니다')
        else:mask=None
        allattr={}
        # Preserve existing primitive-specific attributes and append interpolated edge attributes.
        for key in keys:
            example=next(a[key] for a in source_attrs if key in a)
            a=[]
            for pi,attrs in enumerate(source_attrs):
                if key in attrs:a.append(np.asarray(attrs[key]))
                else:a.append(np.zeros((len(positions[pi]),)+example.shape[1:],dtype=example.dtype))
            allattr[key]=np.concatenate(a,axis=0)
        if '_VERTEX_ID' not in allattr:
            allattr['_VERTEX_ID']=np.arange(len(v),dtype=np.float32)
        # Migration from earlier UINT32 custom attributes. Attribute FLOAT is
        # legal; a new source vertex is indicated by -1, not 0xffffffff.
        for identity in ('_VERTEX_ID','_SOURCE_VERTEX'):
            if identity in allattr:
                a=allattr[identity]
                if identity=='_SOURCE_VERTEX':
                    a=np.asarray(a,dtype=np.float64)
                    a=np.where(a==np.iinfo(np.uint32).max,-1.0,a)
                if np.any(a>16777216) or np.any(a!=np.rint(a)):
                    raise ValueError('GLB identity precision risk: ID over 2^24')
                allattr[identity]=a.astype('<f4')
        if len(np.unique(allattr['_VERTEX_ID']))!=len(v):
            raise ValueError('_VERTEX_ID는 동일 glTF mesh 내에서 고유해야 합니다')
        change=[];local_records={pi:[] for pi in range(len(prims))}
        for step in range(steps):
            before=len(v)
            if mask is not None:mask=expand_seam_mask(v,mask,f)
            v2,f2,stat,parents,edges=adaptive_refine(v,f,max_added_faces=budget,vertex_mask=mask)
            if not len(edges):raise ValueError('선택한 모서리가 없습니다')
            new_owners=owners[edges[:,0]]
            if np.any(new_owners!=owners[edges[:,1]]):
                raise ValueError('primitive를 가로지르는 공유 인덱스가 발견됐습니다')
            new_local=np.empty(len(edges),dtype=np.int64)
            for pi in range(len(prims)):
                is_part=np.flatnonzero(new_owners==pi)
                start=int((owners==pi).sum())
                new_local[is_part]=start+np.arange(len(is_part),dtype=np.int64)
                local_records[pi].append({'round':step+1,'old_vertices':start,
                    'new_vertices':start+len(is_part),'roots':local[edges[is_part,0]].copy(),
                    'tips':local[edges[is_part,1]].copy()})
            for key,arr in allattr.items():
                if key=='_VERTEX_ID':
                    first=int(np.max(arr))+1
                    added=np.arange(first,first+len(edges),dtype=np.float32)
                    if np.any(added>16777216):raise ValueError('FLOAT 정점 ID 정밀도 한도 초과')
                elif key=='_SOURCE_VERTEX':
                    added=np.full(len(edges),-1,dtype=np.float32)
                else:
                    added=(arr[edges[:,0]].astype(np.float64)+arr[edges[:,1]].astype(np.float64))*.5
                    if key=='NORMAL':
                        lengths=np.linalg.norm(added,axis=1,keepdims=True)
                        added=added/np.maximum(lengths,1e-12)
                    added=added.astype(arr.dtype)
                allattr[key]=np.concatenate([arr,added],axis=0)
            # Positions after splitting are computed geometrically, not from a stale attribute.
            allattr['POSITION']=v2.astype('<f4')
            if mask is not None:
                mask=np.concatenate([mask,mask[edges[:,0]]&mask[edges[:,1]]])
            parent_prim=parent_prim[parents]
            owners=np.concatenate((owners,new_owners))
            local=np.concatenate((local,new_local))
            v,f=v2,f2
            change.append(stat)
        # Writer: retain each primitive material and extras, replace only attribute and index accessors.
        for pi,prim in enumerate(prims):
            new_prim=new_data['meshes'][mi]['primitives'][pi]
            vertex_mask_owner=owners==pi
            glob=np.flatnonzero(vertex_mask_owner)
            remap=np.full(len(v),-1,dtype=np.int64)
            remap[glob]=local[glob]
            assert np.array_equal(np.sort(remap[glob]),np.arange(len(glob)))
            fpart=f[parent_prim==pi]
            indices=remap[fpart].astype('<u4').reshape(-1)
            idx,new_buffer=put_accessor(new_data,new_buffer,indices)
            new_prim['indices']=idx
            for key in sorted(set(source_attrs[pi])|{'_VERTEX_ID'}):
                if key=='POSITION':values=v[glob].astype('<f4')
                else:values=allattr[key][glob]
                ai,new_buffer=put_accessor(new_data,new_buffer,values,semantic=key)
                new_prim['attributes'][key]=ai
        report={'mesh':names[mi],'before_faces':int(len(np.concatenate(faces))),
            'after_faces':int(len(f)),'added_faces':int(len(f)-len(np.concatenate(faces))),
            'primitive_count':len(prims),'detail_changed':False,'levels':change}
        reports[names[mi]]=report
        if lineage_sink is not None:
            for pi,levels in local_records.items():lineage_sink[f'{names[mi]}/primitive_{pi}']=levels
    # Atomic, fail-closed roundtrip: never expose a corrupt destination.
    dest=Path(dest)
    if dest.exists():raise FileExistsError(str(dest))
    import tempfile
    with tempfile.NamedTemporaryFile(prefix='.viewslice_',suffix='.glb',dir=dest.parent,delete=False) as tf:
        temp=Path(tf.name)
    try:
        temp.unlink()
        write_glb(temp,new_data,new_buffer)
        reloaded,rbin=read_glb(temp)
        # Verify every input primitive's semantic set survived, plus all
        # original vertices' attribute values (including UV) by accessor.
        for mi,mesh in enumerate(meshes):
            for pi,prior in enumerate(mesh['primitives']):
                after=reloaded['meshes'][mi]['primitives'][pi]
                old_names=set(prior.get('attributes',{}))
                if not old_names.issubset(after.get('attributes',{})):
                    raise ValueError('GLB 원본 속성 손실: '+str(old_names-set(after['attributes'])))
                for key,idx in prior['attributes'].items():
                    before=get_accessor(data,buffer,idx)
                    now=get_accessor(reloaded,rbin,after['attributes'][key])
                    if key=='_SOURCE_VERTEX':
                        before=np.asarray(before,dtype=np.float64)
                        before=np.where(before==np.iinfo(np.uint32).max,-1.0,before)
                    if len(now)<len(before) or not np.array_equal(now[:len(before)],before):
                        raise ValueError('GLB 속성 내용 또는 정점 대응 손실: '+key)
        # Check protected JSON payloads, not only materials.
        for key in ('nodes','materials','textures','images','samplers','scenes','extensionsUsed',
                    'extensionsRequired','extensions','extras'):
            if data.get(key)!=reloaded.get(key):raise ValueError('GLB metadata loss: '+key)
        os.replace(temp,dest)
    finally:
        if temp.exists():temp.unlink()
    # Validate material reference order and node/instance counts after physical reload.
    reloaded,_=read_glb(dest)
    for k in ('nodes','materials','textures','images','samplers','scenes','extensionsUsed','extensionsRequired'):
        if data.get(k)!=reloaded.get(k):raise ValueError('GLB 왕복 중 보존해야 할 속성이 달라졌습니다: '+k)
    return {'raw_scene_preserved':True,'meshes':reports,'detail_changed':False,
            'added_faces':sum(x['added_faces'] for x in reports.values())}
