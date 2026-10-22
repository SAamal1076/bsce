"""Make TF-1.x code run unchanged on TF 2.x (graph mode) for reference runs.

``import tf1_shim`` before importing an original script installs a module named
``tensorflow`` that exposes tf.compat.v1 (placeholders, Session, tf.real, tf.conj, ...)
plus ``tf.contrib.distributions.percentile`` (TF-1.12 'nearest' interpolation), which the
ALISTA code needs. All ops run in float64 when the original code asks for it.
"""
import sys
import types

import tensorflow as _tf

_tf.compat.v1.disable_v2_behavior()
_v1 = _tf.compat.v1

tfmod = types.ModuleType("tensorflow")
for _k in dir(_v1):
    if not _k.startswith("__"):
        setattr(tfmod, _k, getattr(_v1, _k))
tfmod.compat = _tf.compat
tfmod.__version__ = _tf.__version__


def _percentile(x, q, axis=None, interpolation=None, keep_dims=False, validate_args=False, name=None):
    """tf.contrib.distributions.percentile with the default 'nearest' interpolation."""
    x = _v1.convert_to_tensor(x)
    if axis is None:
        y = _v1.reshape(x, [-1])
        ax = 0
    else:
        ax = axis
        y = x
    nd = len(y.shape)
    perm = [i for i in range(nd) if i != ax] + [ax]
    yt = _v1.transpose(y, perm)
    n = _v1.shape(yt)[-1]
    sorted_y, _ = _v1.nn.top_k(yt, k=n, sorted=True)            # descending
    frac = 1.0 - q / 100.0
    idx = _v1.cast(_v1.round(_v1.cast(n - 1, _v1.float64) * frac), _v1.int32)
    g = _v1.gather(sorted_y, idx, axis=nd - 1)
    if keep_dims:
        if axis is None:
            g = _v1.reshape(g, [1] * len(x.shape))
        else:
            g = _v1.expand_dims(g, ax)
    return g


_contrib = types.ModuleType("tensorflow.contrib")
_contrib.distributions = types.SimpleNamespace(percentile=_percentile)
_contrib.data = types.SimpleNamespace()
tfmod.contrib = _contrib
sys.modules["tensorflow"] = tfmod
