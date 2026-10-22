"""Run an ORIGINAL training loop (setup_training + do_training) end to end on small data.

usage: python ref_tf_dotraining.py in.npz out.npz
in.npz : net ('lamp' -> reference/gmlamp/Train/define_networks.py, 'ladmm' -> script),
         script, shrink, K, T, kwargs (dict for build_LADMM), A, xt, yt, xv, yv,
         lr, refinements, iv1, maxit, better_wait, batch_size (only if do_training takes it), seed
out.npz: init_<var>, final_<var>, idx (every minibatch index vector drawn, in order),
         log (the script's own log string), done

The original files are imported unchanged. The only intervention is numpy.random.choice,
replaced by a seeded RandomState.choice that also records the indices, so that the
PyTorch trainer can be fed exactly the same minibatches (np.savez is wrapped to keep the
parameters saved after every stage). Optional `gm_theta` replaces the GM initial value
returned by define_shrinkage.get_shrinkage_function (symmetry-breaking experiment only).
"""
import importlib.util
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parent.parent / "reference"
sys.path.insert(0, str(HERE))
import tf1_shim  # noqa: F401,E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import tensorflow as tf  # noqa: E402

inp = dict(np.load(sys.argv[1], allow_pickle=True))
out_path = Path(sys.argv[2])
wd = out_path.parent
files = {}
for nm, key, var in (("tc", "xt", "x"), ("tm", "yt", "y"), ("vc", "xv", "x"), ("vm", "yv", "y")):
    files[nm] = str(wd / f"{nm}.mat")
    sio.savemat(files[nm], {var: inp[key]})

net = str(inp["net"])
A = inp["A"]
T = int(inp["T"])
refs = tuple(np.asarray(inp["refinements"]).tolist())
if net == "lamp":
    sys.path.insert(0, str(REF / "gmlamp" / "Train"))
    import define_networks as mod  # noqa: E402
    if "gm_theta" in inp:   # optional: replace the GM initial value (symmetry-breaking test)
        import define_shrinkage
        _gsf = define_shrinkage.get_shrinkage_function
        define_shrinkage.get_shrinkage_function = lambda name, K: (_gsf(name, K)[0], np.asarray(inp["gm_theta"]))
    layers, y_ = mod.build_LAMP(A, T=T, shrink=str(inp["shrink"]), untied=True, K=int(inp["K"]))
    stages, x_ = mod.setup_training(A, layers=layers, trinit=float(inp["lr"]), refinements=refs)
else:
    spec = importlib.util.spec_from_file_location("ladmm_ref", str(inp["script"]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    layers, y_ = mod.build_LADMM(A, T=T, **inp["kwargs"].item())
    stages, x_ = mod.setup_training(layers=layers, trinit=float(inp["lr"]), refinements=refs)

# initial values (deterministic initializers -> identical in do_training's own session)
s0 = tf.Session()
s0.run(tf.global_variables_initializer())
init = {v.name[:-2]: s0.run(v) for v in tf.trainable_variables()}
s0.close()

rs = np.random.RandomState(int(inp["seed"]))
drawn = []


def _choice(a, size=None, replace=True, p=None):
    r = rs.choice(a, size=size, replace=replace, p=p)
    drawn.append(np.array(r, dtype=np.int64))
    return r


np.random.choice = _choice

# save_trainable_vars() calls np.savez(savefile, ...) after every stage: keep a copy of each
# snapshot (parameters after every stage) for stage-by-stage comparison
snapshots = []
_savez = np.savez


def _savez_rec(file, *args, **kwds):
    snapshots.append({k[:-2]: np.array(v) for k, v in kwds.items() if k.endswith(":0")})
    return _savez(file, *args, **kwds)


np.savez = _savez_rec

kw = dict(iv1=int(inp["iv1"]), maxit=int(inp["maxit"]), better_wait=int(inp["better_wait"]))
if "batch_size" in inspect.signature(mod.do_training).parameters:
    kw["batch_size"] = int(inp["batch_size"])
savefile = str(wd / "trained.npz")
mod.do_training(x_=x_, y_=y_, training_stages=stages, savefile=savefile, tc=files["tc"], tm=files["tm"],
                vc=files["vc"], vm=files["vm"], **kw)

np.savez = _savez
saved = dict(np.load(savefile, allow_pickle=True))
out = {f"init_{k}": v for k, v in init.items()}
for s, snap in enumerate(snapshots):
    for k, v in snap.items():
        out[f"snap{s}_{k}"] = v
out["n_snapshots"] = len(snapshots)
for k, v in saved.items():
    if k.endswith(":0"):
        out[f"final_{k[:-2]}"] = v
out["idx"] = np.stack(drawn) if drawn else np.zeros((0, 0), np.int64)
out["log"] = np.array(str(saved["log"]))
out["done"] = np.array([str(d) for d in saved["done"]])
np.savez(out_path, **out)
