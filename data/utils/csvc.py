import pandas as pd

df = pd.read_csv(r'C:\Data\dataset\casual\auto-mpg\auto-mpg.data.mixed.max.3.categories.txt', delimiter='\t')
df.to_csv(r'C:\Data\dataset\casual\auto-mpg\auto-mpg.data.csv', index=False)
