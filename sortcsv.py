import pandas as pd
path = r'D:\School\ADMU\4Y\SEM 1\MATH 199.11\Final\GWAP.csv' # use your path

df = pd.read_csv(path, index_col=False, header=0, engine='python')

df = df[df['REGION_NAME'] == 'CLUZ']

df.to_csv('CLUZ_GWAP.csv', index=False)