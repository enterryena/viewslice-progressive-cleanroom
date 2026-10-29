"""Explicit, immutable edge-midpoint ancestry sidecars for progressive mesh editing."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def save_round_lineage(output, lineage):
    path=Path(str(output)+'.lineage.npz')
    if path.exists():raise FileExistsError('계보 파일 덮어쓰기 금지: '+str(path))
    arrays={}
    # Store original node labels separately; non-ASCII names are safe.
    names=list(lineage)
    arrays['geometry_names']=np.array(names,dtype=str)
    for i,name in enumerate(names):
        for j,level in enumerate(lineage[name]):
            pref=f'geo{i}_step{j}'
            arrays[pref+'_roots']=np.asarray(level['roots'],dtype=np.int64)
            arrays[pref+'_tips']=np.asarray(level['tips'],dtype=np.int64)
            arrays[pref+'_old_count']=np.array(level['old_vertices'],dtype=np.int64)
            arrays[pref+'_new_count']=np.array(level['new_vertices'],dtype=np.int64)
    with path.open('xb') as f:
        np.savez_compressed(f,**arrays)
    return path


def read_round_lineage(path):
    """Recover geometry/primitive-local midpoint ancestry with strict count checks."""
    import re
    with np.load(path,allow_pickle=False) as z:
        names=[str(x) for x in z['geometry_names']]
        output={}
        for i,name in enumerate(names):
            prefix=f'geo{i}_step'
            steps=sorted(int(k[len(prefix):].split('_')[0]) for k in z.files
                         if k.startswith(prefix) and k.endswith('_roots'))
            if steps!=list(range(len(steps))):raise ValueError('계보 단계가 끊어졌습니다')
            output[name]=[]
            for j in steps:
                pref=f'geo{i}_step{j}'
                roots=z[pref+'_roots'].astype(np.int64)
                tips=z[pref+'_tips'].astype(np.int64)
                old=int(z[pref+'_old_count']);new=int(z[pref+'_new_count'])
                if roots.shape!=tips.shape or new-old!=len(roots) or roots.min(initial=0)<0 or tips.min(initial=0)<0 or max(roots.max(initial=0),tips.max(initial=0))>=old:
                    raise ValueError('계보 정점 인덱스에 오류가 있습니다')
                output[name].append({'round':j+1,'roots':roots,'tips':tips,
                                     'old_vertices':old,'new_vertices':new})
    return output


def propagate_skin_weights(indices,weights,steps,max_influences=4):
    """Interpolate each new edge-midpoint's weights, retaining strongest four.

    Preserves old vertex weights bit-for-bit. Does not invent missing lineage.
    """
    ids=np.asarray(indices,dtype=np.int32).copy()
    w=np.asarray(weights,dtype=np.float32).copy()
    if ids.shape!=w.shape or ids.ndim!=2:raise ValueError('스키닝 배열 형식이 다릅니다')
    if ids.shape[1]>max_influences:raise ValueError('초안의 본 영향 개수가 제한을 넘습니다')
    # Older 2-weight sketches can be read and expanded to 4 without changing values.
    if ids.shape[1]<max_influences:
        pad=max_influences-ids.shape[1]
        ids=np.pad(ids,((0,0),(0,pad)),constant_values=0)
        w=np.pad(w,((0,0),(0,pad)),constant_values=0)
    for step in steps:
        old=int(step['old_vertices']);new=int(step['new_vertices'])
        roots=np.asarray(step['roots'],dtype=np.int64)
        tips=np.asarray(step['tips'],dtype=np.int64)
        if len(ids)!=old or new!=old+len(roots):
            raise ValueError('스키닝과 계보 정점 수가 맞지 않아 중단합니다')
        ext_ids=np.zeros((len(roots),max_influences),dtype=np.int32)
        ext_w=np.zeros((len(roots),max_influences),dtype=np.float32)
        for i,(a,b) in enumerate(zip(roots,tips)):
            blend={}
            for src in (int(a),int(b)):
                for bone,value in zip(ids[src],w[src]):
                    if value>0:blend[int(bone)]=blend.get(int(bone),0.)+float(value)*.5
            best=sorted(blend.items(),key=lambda x:(-x[1],x[0]))[:max_influences]
            total=sum(value for _,value in best)
            if total<=0:raise ValueError('새 정점의 유효한 본 가중치가 없습니다')
            for j,(bone,value) in enumerate(best):
                ext_ids[i,j]=bone;ext_w[i,j]=value/total
        ids=np.vstack([ids,ext_ids]);w=np.vstack([w,ext_w])
    return ids,w
