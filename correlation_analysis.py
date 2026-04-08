import pandas as pd
from scipy.stats import spearmanr

bt_file = "data/gpt_bt_score.csv"
dw_file = "data/116th_Senate_DW.csv"
cfs_file = "data/116th_Senate_CFS_bio.csv"

bt = pd.read_csv(bt_file)
dw = pd.read_csv(dw_file)
cfs = pd.read_csv(cfs_file)

bt = bt[["bioname", "bt_score"]].copy()
dw = dw[["bioname", "nominate_dim1", "party_code"]].copy()
cfs = cfs[["bioname", "recipient.cfscore"]].copy()

bt["bioname"] = bt["bioname"].astype(str).str.strip()
dw["bioname"] = dw["bioname"].astype(str).str.strip()
cfs["bioname"] = cfs["bioname"].astype(str).str.strip()


bt_names = set(bt["bioname"])
dw_names = set(dw["bioname"])
cfs_names = set(cfs["bioname"])

only_bt = bt_names - dw_names
only_dw = dw_names - bt_names
only_cfs = cfs_names - bt_names

print("Only in BT:", len(only_bt))
print("Only in DW:", len(only_dw))
print("Only in CFS:", len(only_cfs))

df = pd.merge(bt, dw, on="bioname", how="inner")
df = pd.merge(df, cfs, on="bioname", how="inner")


print("\nMatched senators:", len(df))
print(df.head())

# partycode -> party
def map_party(code):
    if code == 200:
        return "R"
    elif code == 100:
        return "D"
    else:
        return "I"
df["party"] = df["party_code"].apply(map_party)

corr_bt_dw, p_bt_dw = spearmanr(df["bt_score"], df["nominate_dim1"])
corr_bt_cfs, p_bt_cfs = spearmanr(df["bt_score"], df["recipient.cfscore"])
corr_dw_cfs, p_dw_cfs = spearmanr(df["nominate_dim1"], df["recipient.cfscore"])

print("\n=== Overall Spearman ===")
print(f"BT vs. DW Correlation: {corr_bt_dw:.6f}")
print(f"BT vs. DW P-value: {p_bt_dw:.6g}")
print(f"BT vs. CFS Correlation: {corr_bt_cfs:.6f}")
print(f"BT vs. CFS P-value: {p_bt_cfs:.6g}")
print(f"DW vs. CFS Correlation: {corr_dw_cfs:.6f}")
print(f"DW vs. CFS P-value: {p_dw_cfs:.6g}")


# treat Independents as Democrats
df["party_group"] = df["party"].replace({"I": "D"})

df_R = df[df["party_group"] == "R"]
df_D = df[df["party_group"] == "D"]

print("\nCounts:")
print("Republicans:", len(df_R))
print("Democrats (incl I):", len(df_D))

# By party
corr_R_bt_dw, p_R_bt_dw = spearmanr(df_R["bt_score"], df_R["nominate_dim1"])
corr_D_bt_dw, p_D_bt_dw = spearmanr(df_D["bt_score"], df_D["nominate_dim1"])
corr_R_bt_cfs, p_R_bt_cfs = spearmanr(df_R["bt_score"], df_R["recipient.cfscore"])
corr_D_bt_cfs, p_D_bt_cfs = spearmanr(df_D["bt_score"], df_D["recipient.cfscore"])
corr_R_dw_cfs, p_R_dw_cfs = spearmanr(df_R["nominate_dim1"], df_R["recipient.cfscore"])
corr_D_dw_cfs, p_D_dw_cfs = spearmanr(df_D["nominate_dim1"], df_D["recipient.cfscore"])

print("\n=== Within-party Spearman ===")

print("\nRepublicans BT vs. DW:")
print(f"Correlation: {corr_R_bt_dw:.6f}")
print(f"P-value: {p_R_bt_dw:.6g}")

print("\nDemocrats BT vs. DW :")
print(f"Correlation: {corr_D_bt_dw:.6f}")
print(f"P-value: {p_D_bt_dw:.6g}")

print("--------------------------------------------")
print("\nRepublicans BT vs. CFS:")
print(f"Correlation: {corr_R_bt_cfs:.6f}")
print(f"P-value: {p_R_bt_cfs:.6g}")

print("\nDemocrats BT vs. CFS :")
print(f"Correlation: {corr_D_bt_cfs:.6f}")
print(f"P-value: {p_D_bt_cfs:.6g}")

print("--------------------------------------------")
print("\nRepublicans DW vs. CFS:")
print(f"Correlation: {corr_R_dw_cfs:.6f}")
print(f"P-value: {p_R_dw_cfs:.6g}")

print("\nDemocrats DW vs. CFS:")
print(f"Correlation: {corr_D_dw_cfs:.6f}")
print(f"P-value: {p_D_dw_cfs:.6g}")