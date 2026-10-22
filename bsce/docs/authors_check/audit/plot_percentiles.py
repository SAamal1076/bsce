import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
res = json.load(open(sys.argv[1]))
keys = list(res)
fig, axs = plt.subplots(2, 4, figsize=(17, 7.2))
for ax, k in zip(axs.T.ravel(), keys):
    r = res[k]; sub = np.array(r["subsets"])
    ax.hist(sub, bins=50, color="#C9D6EA", edgecolor="#9DB0CF")
    ax.axvline(r["authors_val"], color="#B3412E", lw=2, label=f"authors' validation {r['authors_val']:.2f} ({r['authors_val_percentile']:.1f}th pct)")
    ax.axvline(r["test"], color="#2F62C8", lw=2, ls="--", label=f"test set {r['test']:.2f} ({r['test_percentile']:.1f}th pct)")
    ax.set_title(f"{k}, 10-20 dB network, 10 dB\nrandom 2000-sets of our data: {r['subsets_mean']:.2f} ± {r['subsets_sd']:.3f} dB", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_xlabel("NMSE (dB)", fontsize=8); ax.tick_params(labelsize=7.5)
fig.suptitle("Is the authors' validation set like a random 2000-channel set of our data? (5000 random 2000-channel sets drawn from 20,000 of our training + validation channels)", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=125)
