"""Model registry: build any learned estimator from a config dict."""
from .admmnet import ADMMNet
from .cadmm import CADMM
from .graph_ladmm import GraphLADMM
from .ladmm import LADMM
from .lamp import LAMP, gm_theta_init
from .lista import ALISTA, LISTA, alista_W
from .lvamp import LVAMP
from .nfladmm import NFLADMM
from .utrr import UTRR
from .ensemble import Ensemble

REGISTRY = {
    "LAMP": lambda A, c: LAMP(A, T=c.get("T", 8), shrink="soft", untied=c.get("untied", True),
                              precision=c.get("precision", "float32")),
    "GM-LAMP": lambda A, c: LAMP(A, T=c.get("T", 8), shrink="gm", untied=c.get("untied", True), K=c.get("K", 4),
                                 precision=c.get("precision", "float32"), gm_init=c.get("gm_init", "original"),
                                 gm_spread=c.get("gm_spread", 1e-3)),
    "LADMM": lambda A, c: LADMM(A, T=c.get("T", 6), operator=c.get("operator", "full"), untied=c.get("untied", True),
                                rank=c.get("rank", 16), N1=c.get("N1", 16), N2=c.get("N2", 16),
                                init_precision=c.get("init_precision"), prior_init=c.get("prior_init", 0.1),
                                threshold=c.get("threshold", "scalar"), group_kappa=c.get("group_kappa", 3.0),
                                group_win=c.get("group_win", 3), array=c.get("array", "UPA"),
                                data_step=c.get("data_step", "plain"), gm_K=c.get("gm_K", 4),
                                noise_level=c.get("noise_level", "median"), debias=c.get("debias", "none"),
                                precision=c.get("precision", "float32")),
    "LISTA": lambda A, c: LISTA(A, T=c.get("T", 16), lam=c.get("lam", 0.4), untied=c.get("untied", True),
                                complex_mode=c.get("complex_mode", True), precision=c.get("precision", "float32")),
    "ALISTA": lambda A, c: ALISTA(A, T=c.get("T", 16), lam=c.get("lam", 0.4), W=c.get("W"),
                                  percent=c.get("percent", 1.2), max_percent=c.get("max_percent", 13.0),
                                  complex_mode=c.get("complex_mode", True), precision=c.get("precision", "float32")),
    "LVAMP": lambda A, c: LVAMP(A, T=c.get("T", 6), complex_mode=c.get("complex_mode", True),
                                precision=c.get("precision", "float32"),
                                noise_model=c.get("noise_model", "white")),   # saved before v27: as trained (white)
    "NF-LADMM": lambda A, c: NFLADMM(A, T=c.get("T", 8), array=c.get("array", "ULA"), denoiser=c.get("denoiser", "conv"),
                                     share_cnn=c.get("share_cnn", True), thr0=c.get("thr0", 1.0),
                                     hidden=c.get("hidden", 16), kernel=c.get("kernel"), sigma=c.get("sigma", "known"),
                                     N1=c.get("N1", 16), N2=c.get("N2", 16), update=c.get("update", "dual"),
                                     slope_init=c.get("slope_init", 6.0), noise_model=c.get("noise_model", "colored"),
                                     white_step=c.get("white_step", "normalized"),
                                     precision=c.get("precision", "float32")),
    "UTRR": lambda A, c: UTRR(A, T=c.get("T", 30), K=c.get("K", 16), rcc=bool(c.get("rcc", True)),
                              bn=bool(c.get("bn", False)), rho=c.get("rho", 1.0), alpha=c.get("alpha", 0.1),
                              bn_momentum=c.get("bn_momentum", 0.1), precision=c.get("precision", "float32")),
    "Ensemble": lambda A, c: Ensemble(A, members=c["members"], precision=c.get("precision", "float32")),
    "Graph-LADMM": lambda A, c: GraphLADMM(A, T=c.get("T", 8), zstep=c.get("zstep", "gtv_gate"),
                                           xstep=c.get("xstep", "analytic"), array=c.get("array", "ULA"),
                                           N1=c.get("N1", 16), N2=c.get("N2", 16), sigma=c.get("sigma", "residual"),
                                           ring=bool(c.get("ring", True)), edge_phase=c.get("edge_phase", "data"),
                                           learn_phase=bool(c.get("learn_phase", False)), gm_K=c.get("gm_K", 4),
                                           lr_mult=c.get("lr_mult", 10.0), graph_pool=bool(c.get("graph_pool", True)),
                                           data_step=c.get("data_step", "whitened"),
                                           beam_prior=c.get("beam_prior", "none"),
                                           precision=c.get("precision", "float32")),
    "CADMM": lambda A, c: CADMM(A, T=c.get("T", 6), rho0=c.get("rho0", 1.0), precision=c.get("precision", "float32")),
    "ADMM-Net": lambda A, c: ADMMNet(A, T=c.get("T", 15), beta=c.get("beta", 0.1), rho=c.get("rho", 1.0),
                                     precision=c.get("precision", "float32")),
}


def build_model(cfg: dict, A, device=None):
    name = cfg["model"]
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name}; choose from {list(REGISTRY)}")
    m = REGISTRY[name](A, cfg)
    if device is not None:
        m.to(device)
    return m
