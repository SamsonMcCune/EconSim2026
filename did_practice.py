import pandas as pd

try:
    df = pd.read_csv("cardkrueger.csv")
    print(df.head())
except Exception as e:
    print("Error loading dataset:", e)