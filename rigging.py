"""Species-agnostic procedural *initial* rig templates, authored independently.
Not a universal anatomical detector. Edit template/landmarks for unknown morphology.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

PROFILES=('humanoid','quadruped','bird','insect','fish','snake','plant','tentacle','jellyfish','generic')

def rig_plan(vertices,profile='humanoid',up_axis='y',segments=7,appendages=6,custom=None):
    if profile not in PROFILES:raise ValueError('지원하지 않는 리깅 형태: '+str(profile))
    if up_axis not in ('y','z'):raise ValueError('up_axis는 y 또는 z이어야 합니다')
    v=np.asarray(vertices,dtype=float)
    if len(v)==0:raise ValueError('메시 정점이 없습니다')
    mn=v.min(0);mx=v.max(0);span=np.maximum(mx-mn,1e-5);mid=(mn+mx)*.5
    h_axis=1 if up_axis=='y' else 2;depth_axis=2 if up_axis=='y' else 1
    H=span[h_axis];W=span[0];D=span[depth_axis]
    def pt(x=0,y=.5,z=0):
        p=mid.copy();p[0]+=x*W;p[h_axis]=mn[h_axis]+y*H;p[depth_axis]+=z*D
        return [round(float(t),6) for t in p]
    bones=[]
    def add(name,head,tail,parent=None,tag='deform'):
        if np.linalg.norm(np.asarray(head)-np.asarray(tail))<min(H,W,D)*0.003:
            tail=tail.copy();tail[h_axis]+=max(H*.025,1e-4)
        bones.append({'name':name,'parent':parent,'head':head,'tail':tail,'tag':tag})
        return name
    def chain(prefix,coords,parent=None):
        for j in range(len(coords)-1):
            name=f'{prefix}_{j+1:02d}'
            add(name,coords[j],coords[j+1],parent)
            parent=name
        return parent
    def vertical_spine(prefix,coords,parent=None):
        return chain(prefix,[pt(*p) for p in coords],parent)
    if profile=='humanoid':
        add('hips',pt(0,.43),pt(0,.55));add('spine',pt(0,.55),pt(0,.71),'hips');add('chest',pt(0,.71),pt(0,.81),'spine');
        add('neck',pt(0,.81),pt(0,.87),'chest');add('head',pt(0,.87),pt(0,.975),'neck')
        for side,s in [('L',-1),('R',1)]:
            chain(f'arm_{side}',[pt(s*.19,.78),pt(s*.31,.65),pt(s*.45,.54)],'chest')
            chain(f'leg_{side}',[pt(s*.12,.47),pt(s*.15,.28),pt(s*.15,.06)],'hips')
    elif profile in ('quadruped','insect'):
        add('pelvis',pt(-.26,.63),pt(-.10,.64));add('body',pt(-.10,.64),pt(.22,.66),'pelvis')
        add('neck',pt(.22,.66),pt(.35,.79),'body');add('head',pt(.35,.79),pt(.48,.72),'neck')
        chain('tail',[pt(-.27,.64),pt(-.40,.62),pt(-.50,.54)],'pelvis')
        positions=[-.29,.25] if profile=='quadruped' else [-.31,-.02,.29]
        for i,x in enumerate(positions):
            for side,s in [('L',-1),('R',1)]:
                chain(f'leg_{i+1}_{side}',[pt(x,.61,s*.23),pt(x+(.06 if x<0 else -.06),.32,s*.40),pt(x+.02,.035,s*.40)],'body')
    elif profile=='bird':
        add('hips',pt(0,.28),pt(0,.54));add('body',pt(0,.54),pt(0,.78),'hips');
        chain('neck',[pt(0,.78),pt(0,.87),pt(0,.92)],'body');add('head',pt(0,.92),pt(0,.995),'neck_02')
        for side,s in [('L',-1),('R',1)]:
            chain(f'wing_{side}',[pt(s*.15,.73),pt(s*.34,.69),pt(s*.5,.55)],'body')
            chain(f'leg_{side}',[pt(s*.10,.36),pt(s*.12,.18),pt(s*.15,.04)],'hips')
        chain('tail',[pt(0,.4),pt(0,.26,-.35)],'hips')
    elif profile=='fish':
        chain('spine',[pt(-.45,.50),pt(-.22,.50),pt(.06,.52),pt(.31,.54),pt(.48,.54)])
        for side,s in [('L',-1),('R',1)]:
            chain(f'fin_{side}',[pt(0,.52,0),pt(-.12,.33,s*.46)],'spine_02')
        chain('dorsal_fin',[pt(.02,.52),pt(-.09,.84)],'spine_02')
    elif profile=='snake':
        # Straight reference chain: coiled snakes must supply manually adjusted points.
        pts=[pt(-.48+i*.96/max(segments,2),.50) for i in range(max(segments,2)+1)]
        chain('body',pts)
    elif profile=='plant':
        chain('trunk',[pt(0,.02),pt(0,.30),pt(0,.56),pt(0,.85),pt(0,.98)])
        n=max(2,min(20,int(appendages)))
        for i in range(n):
            angle=i*2*math.pi/n
            t=.42+.34*(i%3)/2
            chain(f'branch_{i+1:02d}',[pt(0,t),pt(.26*math.cos(angle),min(.97,t+.18),.30*math.sin(angle)),
                                       pt(.47*math.cos(angle),min(1.,t+.27),.47*math.sin(angle))], 'trunk_02' if t<.72 else 'trunk_03')
    elif profile in ('tentacle','jellyfish'):
        add('core',pt(0,.52),pt(0,.69));add('crown',pt(0,.69),pt(0,.93),'core')
        n=max(2,min(24,int(appendages)))
        for i in range(n):
            a=i*2*math.pi/n;x=math.cos(a);z=math.sin(a)
            pts=[pt(.15*x,.60,.18*z),pt(.25*x,.44,.3*z),pt(.35*x,.23,.40*z),pt(.4*x,.025,.48*z)]
            chain(f'appendage_{i+1:02d}',pts,'core')
    else:
        chain('soft_body',[pt(0,.05),pt(0,.29),pt(0,.53),pt(0,.77),pt(0,.98)])
        n=max(2,min(16,int(appendages)))
        for i in range(n):
            a=2*math.pi*i/n;x=math.cos(a);z=math.sin(a)
            chain(f'limb_{i+1:02d}',[pt(0,.53),pt(.22*x,.55,.22*z),pt(.45*x,.53,.45*z)],'soft_body_02')
    if custom is not None:
        # Full user-authored list replaces template (for arbitrary anatomy)
        bones=custom.get('bones',bones)
    names={b['name'] for b in bones}
    if len(names)!=len(bones):raise ValueError('뼈 이름 중복')
    for bone in bones:
        if bone.get('parent') and bone['parent'] not in names:raise ValueError('없는 부모 뼈: '+bone['name'])
    return {'schema':'viewslice.rig_plan.v1','profile':profile,'up_axis':up_axis,
            'bounds':{'min':mn.tolist(),'max':mx.tolist()},'anatomy_detection':'template/assumed orientation, not automatic species recognition',
            'weights':'nearest-bone approximation; edit and weight paint for quality',
            'bones':bones}


def save_plan(mesh,output_path,profile='humanoid',up_axis='y',segments=7,appendages=6,custom_path=None):
    custom=json.loads(Path(custom_path).read_text(encoding='utf-8')) if custom_path else None
    plan=rig_plan(mesh.vertices,profile,up_axis,segments,appendages,custom)
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('리깅 계획 파일이 존재합니다. 덮어쓰기 방지: '+str(out))
    out.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    return plan


def skinning_weights(points,bones,max_influences=4):
    """Return (N,K) bone indices and normalized distance weights.
    Bone positions are spatial approximations. Chunked callers avoid excessive RAM.
    """
    p=np.asarray(points,dtype=float)
    if not len(bones):raise ValueError('bone 목록 비어있음')
    h=np.asarray([b['head'] for b in bones],dtype=float)
    t=np.asarray([b['tail'] for b in bones],dtype=float)
    vec=t-h;den=np.maximum(np.sum(vec**2,axis=1),1e-15)
    ds=[]
    for a,b,d in zip(h,t,den):
        s=np.clip(((p-a)@(b-a))/d,0,1)
        closest=a+s[:,None]*(b-a)
        ds.append(np.sum((p-closest)**2,axis=1))
    dists=np.array(ds).T
    k=min(max(1,int(max_influences)),len(bones))
    idx=np.argpartition(dists,k-1,axis=1)[:,:k]
    selected=np.take_along_axis(dists,idx,axis=1)
    weights=1/np.maximum(selected,1e-7)
    weights/=np.sum(weights,axis=1,keepdims=True)
    return idx,weights
