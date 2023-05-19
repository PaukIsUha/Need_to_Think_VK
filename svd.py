import pandas as pd
import numpy as np
from io import StringIO

path = '/content/train.csv'
df = pd.read_csv(path, chunksize=1000);

all_df = pd.DataFrame()
check = 0

for chunk in df:
    apl = chunk[chunk['ego_id'] == 0]
    if not check:
      if len(apl):
        check = 1
      else:
        continue
    else:
      if not len(apl):
        break
    all_df = all_df.append(apl, ignore_index=True)

# all_df = all_df.sort_values(by=['t'])
all_df = all_df.dropna()
# mean_time = all_df['t'].mean()
# print(mean_time)
# all_df = all_df[all_df['t'] > mean_time]
them = all_df['v'].unique()
them = np.append(them, all_df['u'].unique()) 
them = np.unique(them) # unique nodes
macroses = {} # macroses for nodes
for i in range(len(np.unique(them))):
  macroses[them[i]] = i
N = len(them)
matrix_sm = np.zeros((N, N))
for i, row in all_df.iterrows():
    I = macroses[row['u']]
    J = macroses[row['v']]
    matrix_sm[I][J] = 1
    matrix_sm[J][I] = 1
matrix_sm
u, s, vh = np.linalg.svd(matrix_sm)
# u.shape, s.shape, vh.shape
non_liq = int(len(u) * 0.50)
cpu = u[0:-non_liq]
cpvh = []
for r in vh:
  cpvh.append(r[0:-non_liq])
# cps = s[0:-non_liq]
# cpvh = vh[0:-non_liq]
l = np.dot(cpu * s, cpvh)
len(l)
# print(len(all_df['u'].unique()))
# all_df[(all_df['u'] == 10) | (all_df['v'] == 10)]