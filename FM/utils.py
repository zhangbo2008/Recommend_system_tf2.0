'''
# Time   : 2020/10/15 10:04
# Author : junchaoli
# File   : utils.py
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_criteo_dataset(file_path, test_size=0.3):
    data = pd.read_csv(file_path)

    dense_features = ['I' + str(i) for i in range(1, 14)] # 特征的列名字
    sparse_features = ['C' + str(i) for i in range(1, 27)]

    #缺失值填充
    data[dense_features] = data[dense_features].fillna(0) # 这个后续修改成中位数填充才靠谱.
    data[sparse_features] = data[sparse_features].fillna('-1')

    #归一化
    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features]) # 把数据缩放到0-1之间.
    #Onehot编码
    data = pd.get_dummies(data) # 把那些取离散值的数据都one-hot编码了. 很有用的一种暴力特征方法.

    #数据集划分
    X = data.drop(['label'], axis=1).values #把label这一个列给扔掉.
    y = data['label'] # y 取1表示用户点击了. 0表示用户没有点击.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return (X_train, y_train), (X_test, y_test)