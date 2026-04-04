import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_topk_layer_counts(path, K):
    df = pd.read_csv(path)
    if "spearman_cv" in df.columns:
        df = df.sort_values("spearman_cv", ascending=False)
    topk = df.head(K)
    # count how many of the top-K heads fall in each layer
    layer_counts = topk.groupby("layer").size()
    return layer_counts

dw_path = "mistral_dw_outputs/headwise_spearman.csv"
cfs_path = "mistral_cfs_outputs/headwise_spearman.csv"
bt_path = "mistral_bt_outputs/headwise_spearman.csv"
K = 96

# get counts for each probe
dw_counts = get_topk_layer_counts(dw_path, K)
cfs_counts = get_topk_layer_counts(cfs_path, K)
bt_counts = get_topk_layer_counts(bt_path, K)

# union of all layers appearing in any file
all_layers = sorted(set(dw_counts.index) | set(cfs_counts.index) | set(bt_counts.index))

# build final dataset
layer_overlap = pd.DataFrame({
    "layer": all_layers,
    "DW": [dw_counts.get(layer, 0) for layer in all_layers],
    "CFS": [cfs_counts.get(layer, 0) for layer in all_layers],
    "BT": [bt_counts.get(layer, 0) for layer in all_layers],
})

# optional: sort by layer
layer_overlap = layer_overlap.sort_values("layer").reset_index(drop=True)

print(layer_overlap)


df_plot = layer_overlap.set_index("layer")[["DW", "CFS", "BT"]]

plt.figure(figsize=(6,8))
sns.heatmap(df_plot, annot=True, cmap="Greens")
plt.show()

plt.savefig("layer_overlap.png", dpi=300, bbox_inches="tight")  # ← save
plt.show()