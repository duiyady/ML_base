import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def draw_fig(data, labels, center):
    # 清除原有图像
    plt.cla()
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(data.shape[0]):
        coo_X = [data[i][0]]
        coo_Y = [data[i][1]]
        plt.scatter(coo_X, coo_Y, marker='x', color=colValue[labels[i] % len(colValue)], label=labels[i])
    for i in range(center.shape[0]):
        coo_X = [center[i][0]]
        coo_Y = [center[i][1]]
        plt.scatter(coo_X, coo_Y, marker='o', color=colValue[i % len(colValue)], label=i)
    plt.pause(0.5)

def kmeans(data, k, item):
    index = np.random.randint(0, data.shape[0], k)
    center_now = data[index]
    last_center = center_now.copy()
    labels = np.zeros(data.shape[0], dtype=np.int)


    run = True
    while run and item > 0:
        # 计算点到每个中心点的距离
        for i in range(data.shape[0]):
            cha = data[i] - center_now
            ping = np.square(cha)
            juli = np.sum(ping, axis=1)
            labels[i] = np.argmin(juli)

        # 更新中心点
        for i in range(k):
            tmp = data[labels == i]
            if tmp.shape[0] != 0:
                center_now[i] = np.mean(tmp, axis=0)
        draw_fig(data, labels, center_now)
        item -= 1
        if (center_now == last_center).all():
            run = False
        last_center = center_now.copy()



if __name__ == "__main__":
    data = pd.read_csv("../data/KmeansData.txt").values
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)
    # 打开交互模式
    plt.ion()
    kmeans(data, 4, 100)
    # 关闭交互模式
    plt.ioff()
    plt.show()



