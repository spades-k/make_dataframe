import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
# 定义了3个DataFrame的框架
res = DataFrame(columns=('left_fingertip_x', 'left_fingertip_y', 'right_fingertip_x', 'right_fingertip_y', 'Image'))
df = DataFrame(columns=('left_fingertip_x', 'left_fingertip_y', 'right_fingertip_x', 'right_fingertip_y', 'Image'))
dfa = DataFrame(columns=('left_fingertip_x', 'left_fingertip_y', 'right_fingertip_x', 'right_fingertip_y', 'Image'))

# 定义遍历文件夹的地址和文件名列表
data_list = []
dirpath = os.getcwd()

# 遍历当前文件夹内的所有.txt，加入到文件名列表中
for (dirpath, dirnames, filenames) in os.walk(dirpath):
    for filename in filenames:
        if filename.endswith('.txt'):
            data_list.append(os.sep.join([dirpath, filename]))
print("the number of this files", len(data_list))

x = len(data_list)
BATCH_SIZE = 10   # 一个一个文件加的话太慢了,30个30个的加

for j in range(0, x, BATCH_SIZE):
    for single_txt in data_list[j:j+BATCH_SIZE]:    # 30个文件一次循环
        # 读取单个文件
        data_all = ''
        data = 0
        single_data = np.loadtxt(single_txt)
        # 提取文件中前12544个数保存成str格式加空格隔开
        for i in range(12544):
            data = single_data[i]
            data = data.astype(int)
            data = str(data)
            data_all = data_all + ' ' + data
        # 放入Image标签下  注意中括号  str型的数据加入标签的时候需要加[]
        res['Image'] = [data_all]
        # 后四个分别放入对应的标签
        res["left_fingertip_x"] = single_data[12544]
        res['left_fingertip_y'] = single_data[12545]
        res['right_fingertip_x'] = single_data[12546]
        res['right_fingertip_y'] = single_data[12547]
        # 组合成一个行数据，之后每个循环加一行
        df = df.append(res, ignore_index=True)
        # df = pd.merge(df, res, how='outer')
        print(single_txt, 'done')
    # 30行的数据之间进行组合
    dfa = pd.merge(dfa, df, how='outer')
    print('txt', j+BATCH_SIZE, 'done')
# 保存成.csv格式
dfa.to_csv('dftraining.csv', index=None)
print(dfa)
