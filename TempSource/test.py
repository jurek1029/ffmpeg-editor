import pandas as pd
#                             ---------|--------|--------------|--
df = pd.DataFrame({'second': [0,1,2,3,4,10,11,12,27,28,29,30,31,40]})
firstRowGard = pd.DataFrame({'second':-10},index =[0])
lastRowGard = pd.Series([df.second.iloc[-1]])
df = pd.concat([firstRowGard, df[:]]).reset_index(drop=True) 
df['dif'] = df.diff(1)
df = df[df['dif'] > 1]
df['lastSec'] = 0
df.iloc[:-1,-1] = df[1:].second - df[1:].dif
df.iloc[-1,-1] = lastRowGard
df['len'] = df.lastSec - df.second + 1
df = df[['second','lastSec','len']]
print(df)