"""Watertight edge-consistent partial triangle subdivision, clean-room implementation.
Every selected edge is split on BOTH adjacent faces; untouched old vertices remain.
"""
from __future__ import annotations
import numpy as np


def _edge_codes(a,b):
    a=np.asarray(a,dtype=np.uint64);b=np.asarray(b,dtype=np.uint64)
    return (np.minimum(a,b)<<np.uint64(32))|np.maximum(a,b)


def _spatial_ids(vertices):
    """Scale-aware spatial identity; radius search avoids quantization grid splits."""
    from scipy.spatial import cKDTree
    v=np.asarray(vertices,dtype=np.float64)
    diag=float(np.linalg.norm(np.ptp(v,axis=0)))
    tol=max(diag*1.e-6,np.finfo(np.float64).eps*max(1.,np.max(np.abs(v))))
    tree=cKDTree(v)
    parent=np.arange(len(v))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in tree.query_pairs(tol,output_type='ndarray'):
        a=find(a);b=find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    labels=np.array([find(i) for i in range(len(v))],dtype=np.int64)
    return labels,tol


def _spatial_edge_codes(vertices,edges):
    labels,_=_spatial_ids(vertices)
    return _edge_codes(labels[edges[:,0]],labels[edges[:,1]])


def expand_seam_mask(vertices,mask,faces=None):
    """Promote both endpoint IDs of matched geometric EDGES, not lone points.

    Prevents asymmetric UV seam masks losing half of each geometric edge after
    a split. Never use this across separate glTF mesh identities or body layers.
    """
    m=np.asarray(mask,dtype=bool).copy()
    if faces is None:return m
    f=np.asarray(faces,dtype=np.int64)
    edges=np.unique(np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0)
    labels,_=_spatial_ids(vertices)
    geometric=_edge_codes(labels[edges[:,0]],labels[edges[:,1]])
    # A mask may be grown across only matched geometrical edges.
    while True:
        marked=m[edges[:,0]]&m[edges[:,1]]
        matched=np.isin(geometric,geometric[marked])
        involved=edges[matched].reshape(-1)
        new=m.copy();new[involved]=True
        if np.array_equal(new,m):return m
        m=new


def _seam_complete_edges(vertices,faces,selected_keys):
    """Expand an edge split across its geometrically identical seam counterparts."""
    if not len(selected_keys):return selected_keys
    v=np.asarray(vertices,dtype=np.float64)
    labels,_=_spatial_ids(v)
    roots=(selected_keys>>np.uint64(32)).astype(np.int64)
    tips=(selected_keys&np.uint64((1<<32)-1)).astype(np.int64)
    wanted=np.unique(_edge_codes(labels[roots],labels[tips]))
    all_edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0)
    spatial_edges=_edge_codes(labels[all_edges[:,0]],labels[all_edges[:,1]])
    pick=np.isin(spatial_edges,wanted)
    return np.unique(_edge_codes(all_edges[pick,0],all_edges[pick,1]))


def _mask_faces(faces,vertex_mask):
    if vertex_mask is None:return None
    mask=np.asarray(vertex_mask,dtype=bool)
    if mask.ndim!=1 or not np.any(mask):
        raise ValueError('영역 마스크가 비어 있거나 잘못되었습니다')
    if np.max(faces)>=len(mask):
        raise ValueError('영역 마스크가 원본 정점 수와 일치하지 않습니다')
    # A face belongs to a mask only when every corner is included.
    selected=np.all(mask[faces],axis=1)
    if not np.any(selected):raise ValueError('선택 영역 안에 완전한 삼각형이 없습니다')
    return selected


def adaptive_refine(vertices,faces,max_added_faces=10000,region='all',up_axis='y',vertex_mask=None):
    """Return (vertices, faces, info, source_face_index, new_vertex_sources).
    new_vertex_sources is an (E,2) list of source vertex indices for interpolating UV/color.
    No fixed total face-count ceiling; per-operation budget is supplied by benchmark.
    """
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces,dtype=np.int64)
    n=len(f);nv=len(v)
    if not n or not nv:raise ValueError('빈 메시')
    if nv>=2**32:raise ValueError('정점 인덱스가 32비트 범위를 초과했습니다')
    budget=max(1,int(max_added_faces))
    count=min(n,max(1,budget//4))
    tris=v[f]
    e01=np.sum((tris[:,0]-tris[:,1])**2,axis=1)
    e12=np.sum((tris[:,1]-tris[:,2])**2,axis=1)
    e20=np.sum((tris[:,2]-tris[:,0])**2,axis=1)
    longest=np.maximum(np.maximum(e01,e12),e20)
    area=np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)
    priority=area + longest*.04
    if region in ('head','torso','lower'):
        up=1 if up_axis=='y' else 2
        z=tris[:,:,up].mean(axis=1)
        span=max(np.ptp(v[:,up]),1e-9)
        normalized=(z-v[:,up].min())/span
        focus={'head':.84,'torso':.55,'lower':.22}[region]
        priority*=.15+3*np.exp(-((normalized-focus)/.20)**2)
    # Restrict priority selection to a user-defined vertex-ID region, never
    # silently reinterpret a head mask as a height interval on each clothing part.
    if vertex_mask is not None:vertex_mask=expand_seam_mask(v,vertex_mask,f)
    inside=_mask_faces(f,vertex_mask)
    if inside is not None:
        eligible=np.flatnonzero(inside)
        count=min(count,len(eligible))
        selected=eligible[np.argpartition(priority[eligible],-count)[-count:]] if count<len(eligible) else eligible
    else:
        selected=np.argpartition(priority, -count)[-count:] if count<n else np.arange(n)
    lengths=np.stack((e01[selected],e12[selected],e20[selected]),axis=1)
    if inside is not None:
        # Select only edges whose incident faces all belong to the requested mask.
        # Otherwise a conforming split would necessarily add outside-mask faces.
        all_codes=np.concatenate((_edge_codes(f[:,0],f[:,1]),
                                  _edge_codes(f[:,1],f[:,2]),
                                  _edge_codes(f[:,2],f[:,0])))
        _,inverse=np.unique(all_codes,return_inverse=True)
        n_edges=int(inverse.max())+1
        total=np.bincount(inverse,minlength=n_edges)
        qualifying=np.bincount(inverse,weights=np.tile(inside.astype(float),3),minlength=n_edges)
        good=np.flatnonzero(qualifying==total)
        unique_codes=np.unique(all_codes)
        good_keys=unique_codes[good]
        candidate=np.stack((_edge_codes(f[selected,0],f[selected,1]),
                            _edge_codes(f[selected,1],f[selected,2]),
                            _edge_codes(f[selected,2],f[selected,0])),axis=1)
        # Also verify geometrically identical UV/material seam copies; an index-
        # valid edge is not safe when its spatial twin borders an unmasked face.
        labels,_=_spatial_ids(v)
        all_edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]),axis=0)
        s_codes=_edge_codes(labels[all_edges[:,0]],labels[all_edges[:,1]])
        s_unique,s_inv=np.unique(s_codes,return_inverse=True)
        s_total=np.bincount(s_inv)
        s_masked=np.bincount(s_inv,weights=np.tile(inside.astype(float),3))
        safe_spatial=s_unique[s_total==s_masked]
        c0=(candidate>>np.uint64(32)).astype(np.int64)
        c1=(candidate & np.uint64((1<<32)-1)).astype(np.int64)
        candidate_spatial=_edge_codes(labels[c0],labels[c1])
        lengths=np.where(np.isin(candidate,good_keys)&np.isin(candidate_spatial,safe_spatial),lengths,-np.inf)
        permitted=np.any(np.isfinite(lengths),axis=1)
        selected=selected[permitted]
        lengths=lengths[permitted]
        if not len(selected):raise ValueError('영역 경계에서 안전하게 분할할 모서리가 없습니다')
    pref=np.argmax(lengths,axis=1)
    a=f[selected, pref];b=f[selected,(pref+1)%3]
    edges=_edge_codes(a,b)
    selected_keys=_seam_complete_edges(v,f,np.unique(edges))
    if inside is not None:
        # A seam split is permitted only when EVERY geometric copy can split
        # without touching a face outside the expanded mask. Otherwise drop the
        # entire seam-edge family rather than creating a hidden T-junction.
        all_e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]),axis=0)
        all_codes=_edge_codes(all_e[:,0],all_e[:,1])
        all_flags=np.tile(inside,3)
        unique_codes,inverse=np.unique(all_codes,return_inverse=True)
        all_n=np.bincount(inverse)
        inside_n=np.bincount(inverse,weights=all_flags.astype(np.int8))
        allowed=unique_codes[all_n==inside_n]
        labels,_=_spatial_ids(v)
        eroot=(selected_keys>>np.uint64(32)).astype(np.int64)
        etip=(selected_keys & np.uint64((1<<32)-1)).astype(np.int64)
        spatial=_edge_codes(labels[eroot],labels[etip])
        bad_spatial=np.unique(spatial[~np.isin(selected_keys,allowed)])
        selected_keys=selected_keys[~np.isin(spatial,bad_spatial)]
        if not len(selected_keys):raise ValueError('선택 영역 경계의 이음매가 외부 면에 걸쳐 있어 안전한 분할이 없습니다')
    roots=(selected_keys>>np.uint64(32)).astype(np.int64)
    tips=(selected_keys&np.uint64((1<<32)-1)).astype(np.int64)
    mids=(v[roots]+v[tips])*.5
    nkeys=len(selected_keys)
    # Search each original edge against selected edges, then conformingly triangulate.
    edge_markers=[]
    for ia,ib in ((0,1),(1,2),(2,0)):
        codes=_edge_codes(f[:,ia],f[:,ib])
        hit=np.searchsorted(selected_keys,codes)
        ok=(hit<nkeys)
        clipped=np.minimum(hit,nkeys-1)
        ok &= selected_keys[clipped]==codes
        edge_markers.append(np.where(ok,clipped+nv,-1).astype(np.int64))
    m0,m1,m2=edge_markers
    mask=(m0>=0).astype(np.uint8)+2*(m1>=0).astype(np.uint8)+4*(m2>=0).astype(np.uint8)
    faces_out=[];parents=[]
    for k in range(8):
        ix=np.flatnonzero(mask==k)
        if not len(ix):continue
        A=f[ix,0];B=f[ix,1];C=f[ix,2];AB=m0[ix];BC=m1[ix];CA=m2[ix]
        def emit(p,q,r):
            faces_out.append(np.column_stack((p,q,r)));parents.append(ix)
        if k==0:emit(A,B,C)
        elif k==1:emit(A,AB,C);emit(AB,B,C)
        elif k==2:emit(B,BC,A);emit(BC,C,A)
        elif k==4:emit(C,CA,B);emit(CA,A,B)
        elif k==3:emit(B,BC,AB);emit(A,AB,C);emit(AB,BC,C)
        elif k==5:emit(A,AB,CA);emit(AB,B,C);emit(CA,AB,C)
        elif k==6:emit(C,CA,BC);emit(CA,A,B);emit(BC,CA,B)
        else:emit(A,AB,CA);emit(AB,B,BC);emit(CA,BC,C);emit(AB,BC,CA)
    new_faces=np.concatenate(faces_out,axis=0)
    sources=np.concatenate(parents,axis=0)
    if inside is not None:
        counts=np.bincount(sources,minlength=n)-1
        # Counts refer to the original faces that generated each extra face.
        added_inside=int(counts[inside].sum())
        added_outside=int(counts[~inside].sum())
    else:
        added_inside=len(new_faces)-n
        added_outside=0
    new_vertices=np.concatenate((v,mids),axis=0)
    info={'old_faces':n,'new_faces':len(new_faces),'added_faces':len(new_faces)-n,
          'added_vertices':nkeys,'selected_initial_faces':int(len(selected)),
          'added_faces_in_mask':added_inside,'added_faces_outside_mask':added_outside,
          'region':region,'seam_synchronized':True,'note':'기존 정점 보존 + 인접 삼각형 무균열 정합 분할'}
    return new_vertices,new_faces,info,sources,np.column_stack((roots,tips))
