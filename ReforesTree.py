import pandas as pd
import numpy as np

df = pd.read_csv('data/mapping/final_dataset.csv')

name = 'Manuel Macias RGB'
print(name)
# Site = Nestor Macias RGB
df_Nestor = df[df['site'] == f'{name}']

print(df_Nestor['AGB'].sum())
print(len(df_Nestor))
print(df_Nestor["carbon"].sum())

