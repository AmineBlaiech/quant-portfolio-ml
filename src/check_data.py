import pandas as pd

df = pd.read_parquet("data/prices.parquet")
print(df.shape)
print(df.head())
print(df["ticker"].unique())
print(df["date"].min(), df["date"].max())