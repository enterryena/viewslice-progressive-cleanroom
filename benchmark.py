"""One-time, locally cached CPU/RAM calibration for progressive mesh budgets.
No telemetry. No recurring benchmarks. Profile can be refreshed explicitly.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile
import time

PROFILE_VERSION = 1

def profile_path():
    root = Path(os.environ.get('LOCALAPPDATA') or (Path.home() / '.config')) / 'ViewSliceCleanRoom'
    return root / 'hardware_profile.json'

def hardware_key():
    try:
        import psutil
        total = round(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        total = 0
    bits = [platform.system(), platform.machine(), platform.processor(), str(os.cpu_count()), str(total)]
    return hashlib.sha256('|'.join(bits).encode()).hexdigest()[:20]

def load_profile():
    path=profile_path()
    if not path.exists():return None
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema')==PROFILE_VERSION and data.get('hardware_key')==hardware_key():return data
    except (OSError,ValueError,TypeError):pass
    return None

def available_ram_mb():
    try:
        import psutil
        return int(psutil.virtual_memory().available/(1024**2))
    except ImportError:
        return 2048  # conservative fallback; not actual hardware detection

def run_benchmark(force=False):
    """Runs only if profile absent or explicit force. Small synthetic mesh; never on each edit."""
    old=load_profile()
    if old and not force:return old, False
    import numpy as np
    # 30k triangles triangle soup; benchmark the actual adaptive subdivision path.
    from adaptive import adaptive_refine
    nx,ny=155,105
    xx,yy=np.meshgrid(np.linspace(-1,1,nx),np.linspace(-1,1,ny),indexing='xy')
    v=np.column_stack((xx.ravel(), yy.ravel(), np.sin(xx.ravel()*3)*.03)).astype(np.float64)
    a=np.arange((ny-1)*(nx-1)).reshape(ny-1,nx-1)
    q=a//(nx-1)*nx+a%(nx-1)
    faces=np.stack([np.stack((q,q+1,q+nx),axis=-1), np.stack((q+1,q+nx+1,q+nx),axis=-1)],axis=1).reshape(-1,3).astype(np.int64)
    t0=time.perf_counter()
    result=adaptive_refine(v,faces,max_added_faces=4500,region='all',up_axis='y')
    elapsed=max(time.perf_counter()-t0,0.005)
    observed_faces=max(len(result[1])-len(faces),1)
    # Conservative extrapolation: model includes scanning entire mesh and edge stitching.
    scan_rate=int(len(faces)/elapsed)
    additional_rate=int(observed_faces/elapsed)
    ram=available_ram_mb()
    # Reserve >= 70% of currently available RAM and account for NumPy temp arrays.
    safe_memory_faces=max(2000,int(ram*.25*1024*1024/300))
    # 5 sec total nominal CPU pass including import/export; conservative target.
    recommended=max(800,min(int(additional_rate*2),int(safe_memory_faces*.20)))
    profile={'schema':PROFILE_VERSION,'hardware_key':hardware_key(), 'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
             'cpu':platform.processor() or platform.machine(), 'logical_cpus':os.cpu_count(),
             'available_ram_mb_at_calibration':ram,'sample_input_faces':int(len(faces)),
             'sample_added_faces':int(observed_faces),'sample_seconds':round(elapsed,4),
             'scan_faces_per_second_est':scan_rate,'added_faces_per_second_est':additional_rate,
             'recommended_added_faces_per_iteration':recommended,
             'notes':'Results cached. Recalibrate only via --refresh. Per-file memory preflight is not a benchmark.'}
    path=profile_path();path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='bench_',suffix='.json',dir=str(path.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(profile,f,indent=2,ensure_ascii=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return profile, True

def per_iteration_budget(face_count,profile=None,lightness=1.0):
    """Select incremental faces, not total polygon count. Dynamic memory preflight.
    No fixed face-count cutoff. A severe low-memory condition errors cleanly.
    """
    if profile is None:
        profile,_=run_benchmark()
    ram=available_ram_mb()
    # Operational check, not a benchmark. NumPy will allocate temporary full arrays.
    estimated_scan_mb = face_count * 210 / (1024**2) + 200
    if ram < estimated_scan_mb * 1.25:
        raise MemoryError(f'메시 처리를 위한 여유 RAM 부족: 예상 {estimated_scan_mb:.0f}MiB, 가용 {ram}MiB. 다른 프로그램 종료 후 다시 시도하세요.')
    base=int(profile['recommended_added_faces_per_iteration']*max(0.2,min(2.,float(lightness))))
    # Limit proportional growth in each iteration; never prohibit big source files.
    return int(max(300,min(base, max(500,face_count//5))))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='ViewSlice 1-time local benchmark with persistent JSON cache')
    p.add_argument('--refresh',action='store_true',help='Explicitly rerun benchmark and replace cache')
    p.add_argument('--show',action='store_true',help='Show existing profile only; never run')
    a=p.parse_args()
    if a.show:
        r=load_profile();print(json.dumps({'path':str(profile_path()),'profile':r},ensure_ascii=False,indent=2))
    else:
        r,ran=run_benchmark(force=a.refresh)
        print(json.dumps({'benchmark_executed':ran,'path':str(profile_path()),'profile':r},ensure_ascii=False,indent=2))
