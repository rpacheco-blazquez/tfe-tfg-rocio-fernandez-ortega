import pandas as pd
df = pd.read_csv(r"ruta\alturanodos.csv", nrows=5)
print(df.head())
print(df.columns.tolist())