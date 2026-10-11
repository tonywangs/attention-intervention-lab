import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
import numpy as np
import pytest
from attention_lab.artifacts import (transaction, write_json, read_json, manifest,
                                    verify_manifest, Budget, MAX_FILE)
from attention_lab.model import Transformer, save_weights, load_weights


def test_transaction_collision_and_cleanup(tmp_path):
    out=tmp_path/'out'
    with transaction(out) as stage:
        write_json(stage/'ok.json',{'ok':True})
        write_json(stage/'manifest.json',manifest(stage))
    assert verify_manifest(out)
    with pytest.raises(FileExistsError):
        with transaction(out):pass
    assert read_json(out/'ok.json')=={'ok':True}
    for exception in (RuntimeError('fail'),KeyboardInterrupt()):
        failed=tmp_path/'failed'
        with pytest.raises(type(exception)):
            with transaction(failed) as stage:
                (stage/'partial').write_text('partial')
                raise exception
        assert not failed.exists() and not list(tmp_path.glob('.failed-*'))


def test_manifest_rejects_hash_extra_file_and_traversal(tmp_path):
    out=tmp_path/'data';out.mkdir()
    write_json(out/'one.json',{'a':1})
    write_json(out/'manifest.json',manifest(out))
    (out/'extra').write_text('unlisted')
    with pytest.raises(ValueError):verify_manifest(out)
    (out/'extra').unlink()
    (out/'one.json').write_text('{"a":2}\n')
    with pytest.raises(ValueError):verify_manifest(out)
    write_json(out/'manifest.json',{'../outside':{'bytes':0,'sha256':'x'}})
    with pytest.raises(ValueError):verify_manifest(out)
    write_json(out/'manifest.json',{'one.json':{'bytes':MAX_FILE+1,'sha256':'x'}})
    with pytest.raises(ValueError):verify_manifest(out)


def test_nested_manifest_is_included_and_verified(tmp_path):
    out=tmp_path/'outer';inner=out/'inner';inner.mkdir(parents=True)
    write_json(inner/'one.json',{'a':1})
    write_json(inner/'manifest.json',manifest(inner))
    write_json(out/'manifest.json',manifest(out))
    assert 'inner/manifest.json' in verify_manifest(out)


def test_resource_limits_and_malformed_json(tmp_path):
    file=tmp_path/'bad'
    for content in ('{"x":1,"x":2}','{"x":NaN}','[','['*10000):
        file.write_text(content)
        with pytest.raises(ValueError):read_json(file)
    file.write_bytes(b' '*1025)
    with pytest.raises(ValueError):read_json(file,1024)
    with pytest.raises(TimeoutError):Budget(1e-10).check()
    with pytest.raises(ValueError):Budget(float('nan'))
    with pytest.raises(ValueError):
        with transaction(tmp_path/'large') as stage:
            with (stage/'oversized').open('wb') as f:f.truncate(MAX_FILE+1)
    assert not (tmp_path/'large').exists()


def test_symlink_and_output_collision(tmp_path):
    target=tmp_path/'target';target.mkdir()
    (target/'data').write_text('original')
    link=tmp_path/'link';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(FileExistsError):
        with transaction(link):pass
    assert (target/'data').read_text()=='original'
    path=tmp_path/'linked';path.symlink_to(target/'data')
    with pytest.raises(ValueError):read_json(path)


def test_checkpoint_malformed_shape_dtype_and_nan(tmp_path):
    model=Transformer()
    state={k:v.numpy() for k,v in model.state_dict().items()}
    for replacement in (np.ones((100,),np.float32), np.ones(state['readout.bias'].shape,np.float64),
                        np.full(state['readout.bias'].shape,np.nan,dtype=np.float32)):
        damaged={**state,'readout.bias':replacement}
        np.savez(tmp_path/'bad.npz',**damaged)
        with pytest.raises(ValueError):load_weights(model,tmp_path/'bad.npz')
    np.savez(tmp_path/'bad.npz',malicious=np.array([{'x':'y'}],dtype=object))
    with pytest.raises(ValueError):load_weights(model,tmp_path/'bad.npz')


def test_checkpoint_declared_huge_shape_rejected_before_allocation(tmp_path,monkeypatch):
    model=Transformer();good=tmp_path/'good.npz';bad=tmp_path/'bad.npz'
    save_weights(model,good)
    with zipfile.ZipFile(good) as source, zipfile.ZipFile(bad,'w') as target:
        for item in source.infolist():
            if item.filename=='readout.bias.npy':
                stream=io.BytesIO()
                np.lib.format.write_array_header_1_0(stream,{'descr':'<f4','fortran_order':False,'shape':(10**12,)})
                target.writestr(item.filename,stream.getvalue())
            else:target.writestr(item.filename,source.read(item.filename))
    def forbidden(*a,**kw):raise AssertionError('np.load reached before header validation')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError,match='header'):load_weights(model,bad)


@pytest.mark.parametrize('signum',[signal.SIGINT,signal.SIGTERM])
def test_real_process_cancellation(tmp_path,signum):
    destination=tmp_path/'cancelled';ready=tmp_path/'ready'
    code='''
from pathlib import Path
import sys,time
from attention_lab.artifacts import transaction
with transaction(sys.argv[1]) as out:
    (out/'partial').write_text('partial')
    Path(sys.argv[2]).touch()
    time.sleep(30)
'''
    process=subprocess.Popen([sys.executable,'-c',code,str(destination),str(ready)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists()
        process.send_signal(signum)
        process.communicate(timeout=10)
        assert process.returncode!=0
        assert not destination.exists() and not list(tmp_path.glob('.cancelled-*'))
    finally:
        if process.poll() is None:process.kill();process.wait()
