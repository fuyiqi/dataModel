#coding:utf8

# 西瓜书3.3 分类任务

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

SCATTER_TITLE = 'data 3.0a scatter'
SCATTER_XLABEL = 'density'
SCATTER_YLABEL = 'sugar ratio'




def get_raw_data():
    # 读取数据
    dataset = np.genfromtxt('ml_data/3.0a.csv', delimiter=",")
    # 数据去表头
    dataset = dataset[1:, :]
    return dataset

def draw_raw_scatter(X,Y):
    # 取出符合Y=0的X的第一列属性作x轴和第二列属性作y轴
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], marker='o', color='k', s=100, label='bad')
    # 取出符合Y=1的X的第一列属性作x轴和第二列属性作y轴
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='g', s=100, label='good')
    # 设置图表标题
    plt.title(SCATTER_TITLE)
    # 设置图表x轴名
    plt.xlabel(SCATTER_XLABEL)
    # 设置图表y轴名
    plt.ylabel(SCATTER_YLABEL)
    # 个性化图例定制
    plt.legend(loc='best')
    plt.show()

def train_with_LogisticRegression(X,Y):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.5, random_state=0)
    # 选择模型
    mymodel = LogisticRegression()
    # 数据带入模型，train出模型参数
    mymodel.fit(X_train, Y_train)
    # 使用测试集验证模型,得出预测值
    Y_predict = mymodel.predict(X_test)
    # 误差混淆矩阵
    # print(metrics.confusion_matrix(Y_test, Y_predict))
    # print(metrics.classification_report(Y_test, Y_predict))
    precision, recall, thresholds = metrics.precision_recall_curve(Y_test,Y_predict)
    return mymodel

def draw_predicted_scatter(X,Y):

    h = 0.001
    # 确定X的第一列属性值的边界
    x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    # 确定X的第二列属性值的边界
    x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    # 枚举X的第一列属性值
    x0_arr = np.arange(x0_min, x0_max, h)
    # 枚举X的第二列属性值
    x1_arr = np.arange(x1_min, x1_max, h)
    # meshgrid 用两个坐标轴构建网格
    x0, x1 = np.meshgrid(x0_arr,x1_arr)
    # ravel函数多维矩阵降为一维；np.c_按列叠加两个矩阵 np.c_(m,n),丰富原有的X
    X_enum = np.c_[x0.ravel(),x1.ravel()]
    # 将训练的模型拿过来预测
    Y_enum = train_with_LogisticRegression(X,Y).predict(X_enum)

    Y_enum = Y_enum.reshape(x0.shape)
    plt.contourf(x0, x1, Y_enum, cmap=pl.cm.Paired)
    plt.title(SCATTER_TITLE)
    plt.xlabel(SCATTER_XLABEL)
    plt.ylabel(SCATTER_YLABEL)
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], marker='o', color='k', s=100, label='bad')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='g', s=100, label='good')
    plt.legend(loc='best')
    plt.show()

def data_viusal():
    draw_raw_scatter(X,Y)
    draw_predicted_scatter(X,Y)

# =================== 对数几率回归 ===================
def sigmoid_train():
    pass

def sigmoid_function(x):
    return 1/(1+np.exp(x))








if __name__ == "__main__":
    dataset = get_raw_data()
    # 分离自变量
    X = dataset[:, 1:3]
    # 分离因变量
    Y = dataset[:, 3]
    # 获取自变量X的维度
    m, n = X.shape


