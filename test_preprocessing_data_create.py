import pandas as pd
import os

cat = ['rock', 'scissor', 'paper']
data = [[0,0,0,0,0], [0,1,1,0,0], [1,1,1,1,1]]
# df = pd.DataFrame()

for i in range(len(cat)):
    temp0, temp1, temp2, temp3, temp4, temp5 = [], [], [], [], [], []
    for _ in range(300):
        temp0.append(cat[i])
        temp1.append(data[i][0])
        temp2.append(data[i][1])
        temp3.append(data[i][2])
        temp4.append(data[i][3])
        temp5.append(data[i][4])
    df = pd.DataFrame({'category':temp0, 't1': temp1, 't2': temp2, 't3': temp3, 't4': temp4, 't5': temp5})

    df.to_csv('./resources/{}_data.csv'.format(cat[i]), index=False)
    print('create {}'.format(cat[i]))