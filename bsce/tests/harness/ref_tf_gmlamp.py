"""Reference run of the ORIGINAL GM-LAMP/LAMP TF training graph (define_networks.py).

usage: python ref_tf_gmlamp.py in.npz out.npz
in.npz : A, y, x, shrink, K, T, lr, refinements, nsteps, test_layers, w_<varname>
out.npz: out_<i> (layer outputs), grad_<stage>_<var>, after_<stage>_<var>, names
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tf1_shim  # noqa: F401  (installs tf1 as 'tensorflow')
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(HERE.parent.parent / "reference" / "gmlamp" / "Train"))
import define_networks  # original file, unchanged

inp = dict(np.load(sys.argv[1], allow_pickle=True))
A = inp["A"].astype(np.float64)
shrink = str(inp["shrink"])
K, T = int(inp["K"]), int(inp["T"])
mode = str(inp["mode"])          # 'forward' (no training ops) or 'train'
layers, y_ = define_networks.build_LAMP(A, T=T, shrink=shrink, untied=True, K=K)
if mode == "train":
    stages, x_ = define_networks.setup_training(A, layers=layers, trinit=float(inp["lr"]),
                                                refinements=tuple(inp["refinements"].tolist()))
else:
    stages, x_ = [], tf.placeholder(tf.complex128, (A.shape[1], None))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tv = {v.name[:-2]: v for v in tf.trainable_variables()}


ASSIGN_OPS, ASSIGN_FEED = [], {}
for _k, _v in inp.items():
    if _k.startswith("w_"):
        _var = tv[(prefix if "prefix" in globals() else "") + _k[2:]]
        _ph = tf.placeholder(_var.dtype.base_dtype, _var.shape)
        ASSIGN_OPS.append(tf.assign(_var, _ph))
        ASSIGN_FEED[_ph] = np.asarray(_v).astype(_var.dtype.base_dtype.as_numpy_dtype).reshape(_var.shape)


def assign_all():
    sess.run(ASSIGN_OPS, ASSIGN_FEED)


out = {}
assign_all()
fd = {y_: inp["y"], x_: inp["x"]}
outs = sess.run([l[1] for l in layers], fd)
for i, o in enumerate(outs):
    out[f"out_{i}"] = o
out["layer_names"] = np.array([l[0] for l in layers])
# gradients of each stage loss wrt its var_all + nsteps Adam steps of selected stages
for si in inp["test_stages"].tolist():
    name, xhat_, loss_, nmse_, train_, var_list, var_all, flag = stages[si]
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
    out[f"loss_{si}"] = sess.run(loss_, fd)
    assign_all()
out["stage_names"] = np.array([s[0] for s in stages])
out["stage_vars"] = np.array([",".join(v.name[:-2] for v in (s[5] if len(s[5]) else s[6])) for s in stages])
np.savez(sys.argv[2], **out)
