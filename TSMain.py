# coding:utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr,spearmanr,kendalltau


user_num = 142
item_num = 4500
period_num = 64
default_value = -1


# ===============================常用方法===============================

def matrix_print(m):
    row_num = len(m)
    col_num = len(m[0])
    for r in range(row_num):
        for c in range(col_num):
            print(m[r][c], end=",")
        print()


# 通过pandas加载数据
def get_data_by_df(filename):
    raw_data = pd.read_csv(filename, sep=" ", names=['User', 'Item', 'Period', 'Value'])
    # 排序
    df = raw_data.sort_values(by=['Period', 'User', 'Item'])
    return df


# 取出pandas中某时刻的用户-服务评分
def build_ui(ui_df):
    # 初始化矩阵大小
    ui_matrix = [[-default_value] * item_num for _ in range(user_num)]
    for row in ui_df.itertuples():
        ui_matrix[row.User][row.Item] = row.Value
    return ui_matrix


# 构建时间-用户-服务-评分的三维矩阵
def build_uit(df):
    uit = []
    for t in range(period_num):
        ui = build_ui(df[df['Period'] == t])
        uit.append(ui)
    return uit


# 保存三阶矩阵为txt
def save_3matrix(m3, filename):
    m3_np = np.array(m3)
    np.save(file=filename, arr=m3_np)


# ===============================处理数据===============================

def get_3matrix(filename):
    return np.load(filename)

rt_np = get_3matrix('data/rt.npy')

def draw_1u_on_i():
    p,u,i = rt_np.shape
    x= range(0,i)
    ui_0 = rt_np[0,:,:]
    for uu in range(0,u):
        y = ui_0[uu,:]
        plt.plot(x, y)
    plt.show()


def draw_1u_on_t():
    # 取出某个用户的随着时间对所有item的打分
    u_on_i_with_t = rt_np[:, 0, :]
    p, i = u_on_i_with_t.shape
    x = range(0,p)
    for t in x:
        y = u_on_i_with_t[:, t]
        plt.plot(x, y)
    plt.title("one user on one item with time")
    plt.xlabel("time period")
    plt.ylabel("qos value")
    plt.show()

def pearsonrSim(x,y):
    ans = pearsonr(x, y)[0]
    return ans

# 计算相似度矩阵
def get_simMatrix(m,type='row'):
    row_num,col_num = m.shape
    if type == 'row':
        row_sim_m = np.zeros((row_num, row_num))
        for r_m in range(0, row_num):
            for r_n in range(r_m + 1, row_num):
                x = m[r_m, :]
                y = m[r_n, :]
                row_sim_m[r_m][r_n] = pearsonrSim(x, y)
        return row_sim_m.T + row_sim_m
    else:
        if type == 'col':
            col_sim_m = np.zeros((col_num, col_num))
            for c_m in range(0,col_num):
                for c_n in range(c_m+1,col_num):
                    x = m[:,c_m]
                    y = m[:, c_n]
                    col_sim_m[c_m][c_n] = pearsonrSim(x, y)
            return col_sim_m+col_sim_m.T


def get_TopKGroup(sim_m,topK):
    topK_res = []
    for n in range(0,user_num):
        row = sim_m[n,:]
        top_k_idx = row.argsort()[::-1][0:topK]
        topK_res.append(top_k_idx)
    return topK_res


def ucf():
    r = rt_np[0, :, :]
    row_simMatrix = get_simMatrix(r, type='row')
    top_k = 5
    row_sim_group = get_TopKGroup(row_simMatrix, top_k)

if __name__ == "__main__":
    pass
    