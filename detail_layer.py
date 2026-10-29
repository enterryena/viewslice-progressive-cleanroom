"""Explicit user-supplied height map => true geometric detail, not AI inference.
UV-bilinear grayscale sampling plus bounded normal displacement; never infer depth
from an ordinary albedo image. Intended for static unrigged UV meshes only.
"""
from __future__ import annotations
import numpy as np

def apply_height_displacement(mesh,map_path,amplitude):
    from PIL import Image
    import trimesh
    uv=getattr(mesh.visual,'uv',None)
    if uv is None or len(uv)!=len(mesh.vertices):
        raise ValueError('높이맵 디테일에는 정점별 UV가 필요합니다. UV 없이 진행하면 품질을 조작할 수 없습니다.')
    if amplitude<=0:raise ValueError('높이맵 변위 진폭은 양수여야 합니다 (m 단위)')
    uv=np.asarray(uv,dtype=np.float64)
    if not np.isfinite(uv).all():raise ValueError('UV 데이터에 유효하지 않은 값이 있습니다')
    image=np.asarray(Image.open(map_path).convert('L'),dtype=np.float64)/255.0
    h,w=image.shape
    # Repeat UVs outside 0..1. Image rows grow downward, UV v grows upward.
    u=np.mod(uv[:,0],1.0)*(w-1)
    v=(1.0-np.mod(uv[:,1],1.0))*(h-1)
    x0=np.floor(u).astype(np.int64);x1=np.minimum(x0+1,w-1)
    y0=np.floor(v).astype(np.int64);y1=np.minimum(y0+1,h-1)
    tx=u-x0;ty=v-y0
    values=((1-tx)*(1-ty)*image[y0,x0]+tx*(1-ty)*image[y0,x1]+
            (1-tx)*ty*image[y1,x0]+tx*ty*image[y1,x1])
    normals=np.asarray(mesh.vertex_normals,dtype=np.float64)
    if not np.isfinite(normals).all():raise ValueError('메시 법선이 유효하지 않습니다')
    mesh=mesh.copy()
    signed=(values-.5)*amplitude
    original=np.asarray(mesh.vertices,dtype=np.float64)
    displaced=original+normals*signed[:,None]
    # UV/seam duplicates at the same physical position must receive the SAME
    # resulting geometric point, even when their UVs or vertex normals differ.
    diagonal=float(np.linalg.norm(np.ptp(original,axis=0)))
    if diagonal>0:
        tol=diagonal*1.e-6
        xyz=np.rint((original-original.min(axis=0))/tol).astype(np.int64)
        _,inverse,counts=np.unique(xyz,axis=0,return_inverse=True,return_counts=True)
        repeats=counts[inverse]>1
        if np.any(repeats):
            for axis in range(3):
                sums=np.bincount(inverse,weights=displaced[:,axis],minlength=len(counts))
                displaced[repeats,axis]=sums[inverse[repeats]]/counts[inverse[repeats]]
    mesh.vertices=displaced
    return mesh,{'source':'explicit_height_map','height_map':str(map_path),'amplitude_m':float(amplitude),
                 'mean_absolute_displacement_m':float(np.mean(abs(signed))),
                 'displaced_vertices':int(np.count_nonzero(abs(signed)>1e-12)),
                 'seam_coincident_vertices_synchronized':bool(diagonal>0)}
