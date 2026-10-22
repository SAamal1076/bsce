"""Reference run of an ORIGINAL LADMM TF training script (build_LADMM + setup_training).

usage: python ref_tf_ladmm.py in.npz out.npz
in.npz : script (path), A, y, x, T, kwargs (dict), lr, refinements, nsteps, test_stages, w_<var>
The script is imported as a module (its __main__ block does not run).
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tf1_shim  # noqa: F401
import numpy as np
import tensorflow as tf

inp = dict(np.load(sys.argv[1], allow_pickle=True))
script = str(inp["script"])
spec = importlib.util.spec_from_file_location("ladmm_ref", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

kw = inp["kwargs"].item()
A = inp["A"]
layers, y_ = mod.build_LADMM(A, T=int(inp["T"]), **kw)
stages, x_ = mod.setup_training(layers=layers, trinit=float(inp["lr"]),
                                refinements=tuple(inp["refinements"].tolist()))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tv = {v.name[:-2]: v for v in tf.trainable_variables()}
out = {"init_" + k: sess.run(v) for k, v in tv.items()}


ASSIGN_OPS, ASSIGN_FEED = [], {}
for _k, _v in inp.items():
    if _k.startswith("w_"):
        _var = tv[(prefix if "prefix" in globals() else "") + _k[2:]]
        _ph = tf.placeholder(_var.dtype.base_dtype, _var.shape)
        ASSIGN_OPS.append(tf.assign(_var, _ph))
        ASSIGN_FEED[_ph] = np.asarray(_v).astype(_var.dtype.base_dtype.as_numpy_dtype).reshape(_var.shape)


def assign_all():
    sess.run(ASSIGN_OPS, ASSIGN_FEED)


assign_all()
ydt = y_.dtype.as_numpy_dtype
fd = {y_: inp["y"].astype(ydt), x_: inp["x"].astype(ydt)}
outs = sess.run([l[1] for l in layers], fd)
for i, o in enumerate(outs):
    out[f"out_{i}"] = o
for si in inp["test_stages"].tolist():
    name, xhat_, loss_, nmse_, train_, var_list, var_all = stages[si][:7]
    gl = tf.gradients(loss_, list(var_all))
    pairs = [(v, g) for v, g in zip(var_all, gl) if g is not None]
    vals = sess.run([g for _, g in pairs], fd)
    for (v, _), g in zip(pairs, vals):
        out[f"grad_{si}_{v.name[:-2]}"] = g
    assign_all()
    for _ in range(int(inp["nsteps"])):
        sess.run(train_, fd)
    for v in var_all:
        out[f"after_{si}_{v.name[:-2]}"] = sess.run(v)
    assign_all()
out["stage_names"] = np.array([s[0] for s in stages])
out["stage_vars"] = np.array([",".join(v.name[:-2] for v in (s[5] if len(s[5]) else s[6])) for s in stages])
np.savez(sys.argv[2], **out)
