#!/usr/bin/env python3
"""Build the wheel and verify its CLI/import/report outside the source checkout."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='attention-pretrained-install-') as temporary:
        work=Path(temporary)
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(work),str(ROOT)],check=True)
        wheel=next(work.glob('*.whl'))
        package=work/'package'
        with zipfile.ZipFile(wheel) as archive:archive.extractall(package)
        results=work/'results';shutil.copytree(ROOT/'results/pretrained',results)
        script='''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from attention_lab import pretrained,pretrained_report
assert Path(pretrained.__file__).is_relative_to(Path(sys.argv[1]))
before=(Path(sys.argv[2])/'report.html').read_bytes()
pretrained_report.render(Path(sys.argv[2]),Path(sys.argv[3]))
assert (Path(sys.argv[2])/'report.html').read_bytes()==before
sys.argv=['attention-pretrained','--help']
pretrained.main()
'''
        subprocess.run([sys.executable,'-I','-c',script,str(package),str(results),str(ROOT/'experiments/pretrained')],cwd=work,check=True)
    print('Wheel imports, CLI and report rendering passed outside checkout.')

if __name__=='__main__':main()
