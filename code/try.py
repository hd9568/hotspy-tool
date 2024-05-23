import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from gensim.models.word2vec import Word2Vec
from constants import Constants
import torch
import random
import mymodel
import networkx as nx
if(__name__=="__main__"):
    # model = Word2Vec.load("lr_word.model")
    # embedding = model.wv.vectors.mean(axis=0)
    # words = 'call void @y_solve_()\l'.split(" ")
    # wordvecs = np.array([model.wv[word] if word in model.wv else embedding for word in words])
    # print(wordvecs.mean(0).tolist())
    # n = random.randint(0,1)
    # print(n)

    # g = nx.DiGraph()
    # # g.add_nodes_from(["A","B","C","D","E","F"])
    # g.add_edges_from([("A","B"),("B","C"),("C","D"),("E","D"),("F","E")])
    # g2 = nx.shortest_path(g.to_undirected(),'B').keys()
    # print(np.array(nx.adjacency_matrix(g).todense()).shape[0])

    # import matplotlib.pyplot as plt
    # import numpy as np

    # # 定义数据
    # x = np.array([i for i in range(100000,100010)])  # 取出10个随机数
    # y = np.array([x * (1 + (-2/3)**x)  for x in range(100000,100010)]) 
    # # 绘图
    # # 1. 确定画布
    # plt.figure(figsize=(8, 4))  # figsize:确定画布大小 

    # # 2. 绘图
    # plt.scatter(x,  # 横坐标
    #             y,  # 纵坐标
    #             c='red',  # 点的颜色
    #             label='function')  # 标签 即为点代表的意思
    # # 3.展示图形
    # plt.legend()  # 显示图例
    # plt.savefig("func.png")
    # import numpy as np
    # import matplotlib.ticker as ticker
    # import matplotlib.pyplot as plt
    # name_list = ('Apple', 'Orange', 'Banana', 'Pear', 'Mango')
    # value_list = np.random.randint(0, 99, size = len(name_list))
    # pos_list = np.arange(len(name_list))
    # ax = plt.axes()
    # ax.xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
    # ax.xaxis.set_major_formatter(ticker.FixedFormatter((name_list)))
    # plt.bar(pos_list, value_list, color = 'c', align = 'center')
    # plt.show()
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
    plt.rcParams['font.family'] = 'sans-serif'
    plt.figure(figsize=(6,6))#将画布设定为正方形，则绘制的饼图是正圆
    label=['第一','第二','第三']#定义饼图的标签，标签是列表
    explode=[0.01,0.2,0.01]#设定各项距离圆心n个半径
    #plt.pie(values[-1,3:6],explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
    values=[4,7,9]
    plt.pie(values,explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
    plt.title('2018年饼图')
    plt.savefig('./2018年饼图')