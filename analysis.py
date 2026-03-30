# This is for overlapping (layer,head)
# don't use red and blue
# add bradley terry later
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

def get_topk_heads(path, K):
    df = pd.read_csv(path)
    topk = df.head(K)[["layer", "head"]].copy()
    return set(map(tuple, topk.to_numpy()))

# ===== paths =====
dw_path = "mistral_dw_outputs/headwise_spearman_dw.csv"
cfs_path = "mistral_cfs_outputs/headwise_spearman_cfs.csv"

K = 32

# exact (layer, head) sets
dw_heads = get_topk_heads(dw_path, K)
cfs_heads = get_topk_heads(cfs_path, K)

shared_heads = dw_heads & cfs_heads

print("DW top-K heads:", len(dw_heads))
print("CFS top-K heads:", len(cfs_heads))
print("Shared exact heads:", len(shared_heads))
print("Shared heads:", sorted(shared_heads))

# ===== build 32x32 grid =====
# rows = layers 1..32, cols = heads 1..32
# code:
# 0 = none
# 1 = DW only
# 2 = CFS only
# 3 = overlap

grid = np.zeros((32, 32), dtype=int)

for layer, head in dw_heads:
    grid[layer - 1, head - 1] = 1

for layer, head in cfs_heads:
    if grid[layer - 1, head - 1] == 1:
        grid[layer - 1, head - 1] = 3
    else:
        grid[layer - 1, head - 1] = 2

# ===== colors =====
# 0 = white
# 1 = blue
# 2 = red
# 3 = purple (overlap)
cmap = ListedColormap([
    "white",        # none
    "#54A84C",      # DW only
    "#E45756",      # CFS only
    "#7A3E9D"       # overlap
])

norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

# ===== plot =====
plt.figure(figsize=(10, 9))
plt.imshow(grid, cmap=cmap, norm=norm, origin="lower", aspect="equal")

plt.xlabel("Head")
plt.ylabel("Layer")
plt.title(f"Exact Top-{K} Head Overlap (DW vs CFS)")

# ticks shown as 1..32
plt.xticks(ticks=np.arange(32), labels=np.arange(1, 33))
plt.yticks(ticks=np.arange(32), labels=np.arange(1, 33))

# optional faint grid lines for each cell
plt.gca().set_xticks(np.arange(-0.5, 32, 1), minor=True)
plt.gca().set_yticks(np.arange(-0.5, 32, 1), minor=True)
plt.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
plt.tick_params(which="minor", bottom=False, left=False)

# legend
legend_handles = [
    mpatches.Patch(color="#4C78A8", label="DW only"),
    mpatches.Patch(color="#E45756", label="CFS only"),
    mpatches.Patch(color="#7A3E9D", label="Overlap")
]
plt.legend(handles=legend_handles, loc="upper right")

plt.tight_layout()
plt.savefig("topk_exact_head_overlap_heatmap.png", dpi=300, bbox_inches="tight")
print("Saved to topk_exact_head_overlap_heatmap.png")