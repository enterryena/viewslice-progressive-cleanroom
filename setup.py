"""Isolated installer for the Modly process extension. Does not touch existing models."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    if len(sys.argv)<2:raise ValueError('Modly setup JSON 인자가 필요합니다')
    args=json.loads(sys.argv[1])
    python_exe=args['python_exe']
    ext_dir=Path(args.get('ext_dir') or __file__).resolve()
    if ext_dir.is_file():ext_dir=ext_dir.parent
    venv=ext_dir/'venv'
    subprocess.check_call([python_exe,'-m','venv',str(venv)])
    py=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    subprocess.check_call([str(py),'-m','pip','install','numpy>=1.24,<3','trimesh>=4,<6','pillow>=10,<13','psutil>=5.9,<8','scipy>=1.11,<2'])
    subprocess.check_call([str(py),'-c','import numpy,trimesh,PIL,psutil,scipy;print("ViewSlice dependencies OK")'])
    print('ViewSlice Progressive CleanRoom installed',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:print('Setup error:',exc,flush=True);sys.exit(1)
