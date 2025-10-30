import pandas as pd


df_colors = pd.read_csv("adrian_all_weighted_colors.csv")
df_dr2 = pd.read_csv("filtered_data_ebmv_cut_Dr2_table_after_bailer_jones.csv")

print("Length of df_colors: ", len(df_colors))
print("Length of df_dr2: ", len(df_dr2))

df_merged = pd.merge(df_colors, df_dr2, on='object_id', how='inner')
print(f"Memory usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
df_merged.to_csv("merged_df_w3w4_final_all_colors.csv", index=False)
print("Length of df_merged: ", len(df_merged))
