"""Reference runs of the ORIGINAL ALISTA-repo (LISTA, ALISTA) and Borgerding (LVAMP) TF code.

usage: python ref_tf_ista.py in.npz out.npz
in.npz : net ('LISTA'|'ALISTA'|'LVAMP'), refdir, A, y, x, T, lam, untied, W, percent,
         max_percent, init_lr, decay_rate, lr_decay, refinements, nsteps, w_<var>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tf1_shim  # noqa: F401
import numpy as np
import tensorflow as tf

inp = dict(np.load(sys.argv[1], allow_pickle=True))
net = str(inp["net"])
sys.path.insert(0, str(inp["refdir"]))
A = inp["A"]
M, N = A.shape
T = int(inp["T"])
f64 = "f64" in str(inp["refdir"])
dt = tf.float64 if f64 else tf.float32
y_ = tf.placeholder(dt, (M, None))
x_ = tf.placeholder(dt, (N, None))
out = {}

if net in ("LISTA", "ALISTA"):
    import utils.train  # original
    if net == "LISTA":
        from models.LISTA import LISTA
        model = LISTA(A, T, float(inp["lam"]), bool(inp["untied"]), False, "LISTA")
    else:
        from models.ALISTA import ALISTA
        model = ALISTA(A, T, float(inp["lam"]), inp["W"], float(inp["percent"]), float(inp["max_percent"]),
                       False, "ALISTA")
    stages = utils.train.setup_sc_training(model, y_, x_, y_, x_, None, float(inp["init_lr"]),
                                           float(inp["decay_rate"]), tuple(inp["lr_decay"].tolist()))
    xhs_ = model.inference(y_, None)[1:]
    st_list = [(s[0], s[1], s[5], s[6]) for s in stages]       # name, loss, op, var_list
    prefix = model._scope + "/"
else:
    from tools import networks, train

    class Prob:
        pass
    prob = Prob()
    prob.A = A
    prob.A_ = tf.constant(A, dtype=dt)
    prob.x_, prob.y_ = x_, y_
    layers = networks.build_LVAMP(prob, T=T, shrink="bg")
    stages = train.setup_training(layers, prob, trinit=float(inp["init_lr"]),
                                  refinements=tuple(inp["refinements"].tolist()))
    xhs_ = [l[1] for l in layers]
    allv = tf.trainable_variables()
    st_list = [(s[0], s[2], s[4], s[5] if len(s[5]) else tuple(v for v, g in zip(allv, tf.gradients(s[2], allv)) if g is not None)) for s in stages]
    prefix = ""

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tv = {v.name[:-2]: v for v in tf.trainable_variables()}
out.update({"init_" + k[len(prefix):]: sess.run(v) for k, v in tv.items()})


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
fd = {y_: inp["y"], x_: inp["x"]}
for i, o in enumerate(sess.run(xhs_, fd)):
    out[f"out_{i}"] = o
for si, (name, loss_, op_, var_list) in enumerate(st_list):
    gl = tf.gradients(loss_, list(var_list))
    pairs = [(v, g) for v, g in zip(var_list, gl) if g is not None]
    for (v, _), g in zip(pairs, sess.run([g for _, g in pairs], fd)):
        out[f"grad_{si}_{v.name[len(prefix):-2]}"] = g
    assign_all()
    for _ in range(int(inp["nsteps"])):
        sess.run(op_, fd)
    for v in var_list:
        out[f"after_{si}_{v.name[len(prefix):-2]}"] = sess.run(v)
    assign_all()
out["stage_names"] = np.array([s[0] for s in st_list])
out["stage_vars"] = np.array([",".join(v.name[len(prefix):-2] for v in s[3]) for s in st_list])
np.savez(sys.argv[2], **out)
