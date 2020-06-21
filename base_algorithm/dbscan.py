import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# 获取两个点距离
def get_distance(point_a, point_b):
    sum = 0
    for a, b in zip(point_a, point_b):
        sum += (a - b) ** 2
    return sum ** 0.5
# 获取所有距离
def get_all_distance(data_set):
    distance = np.zeros([data_set.shape[0], data_set.shape[0]])
    for i in range(distance.shape[0]):
        for j in range(i, distance.shape[1]):
            distance[i][j] = distance[j][i] = get_distance(data_set[i], data_set[j])
    return distance
# 获取每个对象的势值
def get_s(distance, mi, thete):
    result = np.sum(mi*np.exp(-np.power(distance/thete, 2)), axis=1)
    return result
# 获取第MinPts个点的距离
def get_kmin_dis(distance, min_pts):
    return np.sort(distance)[min_pts]

def dbscan(data_set, e, min_pts):
    point_count = data_set.shape[0]  # 总共有多少点
    all_distance = get_all_distance(data_set)  # 所有点之间的距离
    core_dict = {}  # 核心点集合
    C = {}  # 类别集合
    C[0] = []  # 噪声
    not_choice = [i for i in range(point_count)]  # 没有选择的点
    # 寻找核心点，找出噪声
    for i in range(point_count):
        tmp = [j for j in range(point_count) if all_distance[i][j] <= e and j!=i]
        if len(tmp) >= min_pts:
            core_dict[i] = tmp
        elif len(tmp) == 0:
            C[0].append(i)
            not_choice.remove(i)
    k = 1
    old_core_dict = core_dict.copy()
    while len(core_dict) > 0:
        old_not_choice = not_choice.copy()
        core_key = list(core_dict.keys())
        core_choice = core_key[random.randint(0, len(core_key)-1)]  # 随机选择一个核心点
        queue = [core_choice]
        not_choice.remove(core_choice)
        while len(queue) > 0:  # 从当前队列里面找出核心点，并把周围的点加入当前类
            q = queue[0]
            queue.remove(q)
            if q in old_core_dict.keys():
                delte = [val for val in old_core_dict[q] if val in not_choice]  # Δ = N(q)∩Γ
                queue.extend(delte)
                not_choice = [val for val in not_choice if val not in delte]  # Γ = Γ\Δ
        k += 1
        C[k] = [val for val in old_not_choice if val not in not_choice]
        for x in C[k]:
            if x in core_dict.keys():
                del core_dict[x]
        draw(C, data_set)
    C[0].extend(not_choice)
    draw(C, data_set)
    return C

def dbscan_sp(data_set, min_pts, mi=1, thete=1):
    point_count = data_set.shape[0]  # 总共多少个点
    not_choice = [i for i in range(point_count)]  # 哪些点还没有被划分类别
    distance = get_all_distance(data_set)  # 任意两个点之间的距离[point_count, point_count]
    N0_705_thete = {}  # 在半径为3*thete*math.sqrt(2)的情况下的邻居有哪些
    A = {}  # 存储最后的类别
    A[0] = []  # 存放噪声
    for i in range(data_set.shape[0]):
        tmp = []
        for j in range(data_set.shape[0]):
            if distance[i][j] <= 3*thete*math.sqrt(2) and i != j:
                tmp.append(j)
        if len(tmp) == 0:
            A[0].append(i)
            not_choice.remove(i)
        N0_705_thete[i] = tmp
    k = 1  # 当前第几类
    shi_zhi = get_s(distance, mi, thete)  # 获取每个点的势值
    # 当还有没有选择的点
    while len(not_choice) > 0:
        # 在未被处理的点中选择势场最大的点
        tmp_shi_zhi = shi_zhi[not_choice]
        xp = [i for i in not_choice if shi_zhi[i] == np.max(tmp_shi_zhi)][0]
        A_tmp = [xp]
        index = N0_705_thete[xp]  # xp的所有邻居
        eps = 0.0  # 计算eps  xp的所有邻居的第min_pts近距离的平均
        for i in index:
            eps += np.sort(distance[i])[min_pts]
        eps = eps / len(index)
        deletef = 0.0  # 计算deletef
        for i in index:
            for j in index:
                deletef += abs(shi_zhi[i] - shi_zhi[j])
        deletef /= (len(index) * len(index))
        this_step_deal_count = 0
        this_step_deal_point = []
        while this_step_deal_count < len(A_tmp):
            ai = None
            for ta in A_tmp:
                if ta not in this_step_deal_point:
                    ai = ta
                    this_step_deal_count += 1
                    this_step_deal_point.append(ta)
                    break
            if ai is not None:
                len_s = len(A_tmp)
                N_eps_ai = [x for x in not_choice if distance[ai][x] <= eps and x != ai]
                for xq in N_eps_ai:
                    if xq not in A_tmp:
                        N_eps_xq = [x for x in not_choice if distance[xq][x] <= eps and x != xq]
                        jiao = np.intersect1d(N_eps_xq, A_tmp)
                        for xs in jiao:
                            if abs(shi_zhi[xq] - shi_zhi[xs]) <= 2.12*deletef:
                                A_tmp.append(xq)
                                break
                A_tmp = list(set(A_tmp))
                len_e = len(A_tmp)
                if len_e > len_s:  # 增加了点以后可能前面有些不符合的就符合了
                    this_step_deal_count = 0
                    this_step_deal_point = []
        if len(A_tmp) < min_pts:
            A[0].append(xp)
            not_choice.remove(xp)
        else:
            A[k] = A_tmp
            for i in A[k]:
                not_choice.remove(i)
            k += 1
        draw(A, data_set)
    return A

def draw(C, data_set):
    x_min, x_max = np.min(data_set[:, 0])-0.2, np.max(data_set[:, 0])+0.2
    y_min, y_max = np.min(data_set[:, 1])-0.2, np.max(data_set[:, 1])+0.2
    plt.cla()
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for j in range(len(datas)):
            X.append(data_set[datas[j]][0])
            Y.append(data_set[datas[j]][1])
        plt.scatter(X, Y, marker='o', color=color[i % len(color)], label=i)
    plt.legend(loc='upper right')
    plt.pause(0.1)


def get_thete(data_set):
    x = np.arange(0.01, 3, 0.01)
    y = np.zeros_like(x)
    all_distance = get_all_distance(data_set)
    x_i = None
    for i in range(len(x)):
        shi_neng_i = get_s(all_distance, 1, x[i])
        tmp = shi_neng_i / np.sum(shi_neng_i)
        y[i] = -np.sum(tmp*np.power(2,tmp))
        if x_i is not None:
            if y[i] < y[x_i]:
                x_i = i
        else:
            x_i = i
    print(x[x_i])
    plt.plot(x, y)
    plt.show()




if __name__ == '__main__':
    data_set = pd.read_csv("../data/dbscandata.txt").values
    plt.figure(figsize=(8, 6), dpi=80)
    plt.ion()
    dbscan(data_set, 0.30, 10)
    # dbscan_sp(data_set, 10, thete=1)
    plt.ioff()
    plt.show()

    get_thete(data_set)






