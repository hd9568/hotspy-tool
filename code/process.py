from graphviz import Digraph
import graphviz
import torch 
import sys
from constants import Constants
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from gensim.models.word2vec import Word2Vec
import random
import copy

ID = 0
class DotGraph:
    def __init__(self,path):
        self.path = path
        self.nodes = []
        self.node2label = {}
        self.label2node = {}
        self.node2vec = {}
        self.edges = []
        self.node_features=[] #[N, node_feature_dim], N是节点数
        self.edge_index=[] #[2, E]，E是边的数量，存储每条边的起点和终点索引
        self.matrix = []
        self.adj_matrix = []
        self.w2v = False


    def add_node(self, node_id, label):
        self.nodes.append(node_id)
        self.node2label[node_id] = label
        if(label not in self.label2node):
           self.label2node[label]=node_id
        else:
            print("ERROR: LABEL SAME!!!")
 
    def add_edge(self, source, target):
        self.edges.append((source, target))


    def adjust(self,w2v):
        self.w2v = w2v
        n = len(self.node2label)
        self.matrix = [[0]*n for _ in range(n)]
        self.node_features = [[0 for _ in range(Constants.WORD2VECTOR_DIM)] for _ in range(Constants.MAX_NODE_NUM)]

        m = len(self.edges)
        self.edge_index = [[],[]]
        id_to_index = {}
        for index, node_id in enumerate(self.nodes):
           id_to_index[node_id]=index

        for pair in self.node2label:
            if w2v:
                self.node2vec[pair]=sentence_to_vector(self.node2label[pair])
                self.node_features[id_to_index[pair]]=self.node2vec[pair]


        self.adj_matrix = np.zeros([Constants.MAX_NODE_NUM, Constants.MAX_NODE_NUM * Constants.MAX_EDGE_NUM * 2])
        # 填充邻接矩阵
        for source, target in self.edges:
            source_index = id_to_index[source]
            target_index = id_to_index[target]
            self.matrix[source_index][target_index] = 1   
            self.edge_index[0].append(source_index)
            self.edge_index[1].append(target_index)
            if(w2v):
                self.adj_matrix[target_index][source_index] =  1
                self.adj_matrix[source_index][Constants.MAX_EDGE_NUM*Constants.MAX_NODE_NUM+target_index] =  1

       # print(self.edge_index)
    
    def display(self):
        print(self.path+" :")
        print(len(self.node2label))
        print(len(self.edges))

    def random_copy(self):
        newGraph = DotGraph(self.path)
        newGraph.node2label = self.node2label
        newGraph.label2node = self.label2node
        shuffled_list = copy.deepcopy(self.nodes) 
        random.shuffle(shuffled_list)
        newGraph.nodes = shuffled_list

        newGraph.edges = self.edges
        newGraph.adjust(self.w2v)
        c = random.randint(1,100)
        return newGraph


def dot_to_graph(dot_path,w2v=True):

    with open(dot_path) as f:
        dot_graph = graphviz.Source(f.read(), format="dot")

    graph = DotGraph(dot_path)
     
    for node in dot_graph.source.splitlines():
       if "->" in node:  
           parts = node.split("->")
           if len(parts) >= 2:
               source = parts[0].split(":")[0].strip()
               target = None
               if('[' in parts[1]):
                 target = parts[1].split("[")[0].strip()   
               else:
                 target = parts[1].split(";")[0].strip()
               graph.add_edge(source, target)
           else:
               print("Unexpected format:", node)
       elif "[" in node:  
               node_id =node.split(" ")[0].strip()
               label = node.split("label=\"{")[1].split("}\"")[0].strip()  
               graph.add_node(node_id, label)
    return graph



def sentence_to_vector(sentence):
    words = sentence.split(" ")
    wordvecs = None
    wordvecs = np.array([Constants.W2VMODEL.wv[word] if word in Constants.W2VMODEL.wv else Constants.W2VEMBEDDING for word in words])
    return wordvecs.mean(0).tolist()

import networkx as nx
from collections import deque

def bfs(G,V):
    node2visited={}
    node2visited[V]=True

    nodes = []

    Q = deque()
    Q.append(V)
    while True:
        if len(Q) == 0:
            break

        node = Q.popleft()  # 

        node2visited[node] = True
        nodes.append(node)

        if(len(nodes)>=Constants.CG_MAX_NODE_NUM):
            break
 
        for n in nx.neighbors(G, node):
            # 注意，如果n已经在队列里面，则不予重复添加。
            if (n not in node2visited):
                Q.append(n)
    return nodes

class Function:
    def __init__(self, program,name, cfg, cgs, label,num=0):
        self.program = program
        self.name = name  # 函数名
        self.cfg = cfg    # 控制流图（Control Flow Graph）
        self.cgs = cgs      # 调用图（Call Graph）
        self.label = label  # 标签，代表是否为热点函数
        self.num = num
 
        ## CG CUTTING BY NETWORKX
        self.cg_matrix = 0


    def __str__(self):
        return f"Function Name: {self.name}, Label: {self.label}"
    
    def cg_cutting(self):
        graph = self.cgs[0]
 
        node = graph.label2node[self.name]

        nGraph = nx.DiGraph()
        nGraph.add_nodes_from(graph.node2label.keys())
        nGraph.add_edges_from(graph.edges)

        if(len(nGraph.nodes) > Constants.CG_MAX_NODE_NUM):
           nGraph = nGraph.subgraph(bfs(nGraph.to_undirected(),node))
        self.cg_matrix = np.array(nx.adjacency_matrix(nGraph).todense())
 
    def copy(self):
        num = self.num + 1
        newFunc = Function(self.program,self.name, self.cfg.random_copy(), [self.cgs[0].random_copy()], self.label,num)
        newFunc.cg_cutting()
        return newFunc

name2graph = {}
cfg_nodeNum=[]
cfg_edgeNum=[]
cg_nodeNum=[]
cg_edgeNum=[]

def dot2graph(dot,w2v=False):
     if(dot in name2graph.keys()):
         return name2graph[dot]
     else:
         name2graph[dot]= dot_to_graph(dot,w2v)
         return name2graph[dot]

def func_is_valid(function,ignore):
    if(ignore == False):
        return len(function.cfg.node2label) <= Constants.MAX_NODE_NUM and len(function.cfg.edges) <= Constants.MAX_EDGE_NUM
    
    return ((len(function.cfg.node2label) <= Constants.MAX_NODE_NUM and
           len(function.cfg.edges) <= Constants.MAX_EDGE_NUM and 
           len(function.cfg.node2label) >= 2)) or (len(function.cfg.node2label) ==1 and function.label=='1')

def get_functions(args, w2v=True, if_ignore=True):
    """
    获取并处理训练或测试数据集中的函数
    :param args: 命令行参数对象
    :param w2v: 是否使用word2vec
    :param if_ignore: 是否忽略无效函数
    :return: 处理后的函数列表
    """
    json_file_path = args.train_json_path
    # 打开并读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    functions = [Function(item['program'], item['name'], item['cfg'], item['cgs'], item['label']) for item in data]
    print("All functions have been obtained! All graphs are being parsed:")

    newFunctions = []
    ignore = 0
    ignore1 = 0
    oneNode = 0
    oneNode_hot = 0

    for function in functions:
        function.cfg = dot_to_graph(function.cfg, w2v)  # bert!!!
        function.cgs[0] = dot2graph(function.cgs[0])

        if len(function.cfg.node2label) == 1:
            oneNode += 1
            if int(function.label) == 1:
                oneNode_hot += 1

        if func_is_valid(function, if_ignore):
            function.cfg.adjust(w2v)
            function.cgs[0].adjust(w2v=False)
            function.cg_cutting()

            # 根据标签决定是否需要复制
            if function.label == '1':
                for _ in range(args.hotcopy):
                    newFunctions.append(function.copy())
            else:
                for _ in range(args.nohotcopy):
                    newFunctions.append(function.copy())
        else:
            ignore += 1
            if int(function.label) == 1:
                ignore1 += 1

    print(f"The number of function which is only with one node is {oneNode}, {oneNode_hot} of that is Hot")
    print(f"{ignore} is ignored, {ignore1} of which is hot")
    print(f"dataloader is done! the final number is {len(newFunctions)}")

    for f in newFunctions:
        cfg_nodeNum.append(len(f.cfg.node2label))
        cfg_edgeNum.append(len(f.cfg.edges))

    random.shuffle(newFunctions)

    hot = sum(1 for f in newFunctions if f.label == "1")
    Constants.TRAIN_HOT_RATE = hot / len(newFunctions)
    return newFunctions


def get_test_functions(args, w2v=True, if_ignore=True):
    """
    获取并处理测试数据集中的函数
    :param args: 命令行参数对象
    :param w2v: 是否使用word2vec
    :param if_ignore: 是否忽略无效函数
    :return: 处理后的函数列表
    """
    json_file_path = args.test_json_path


    # 打开并读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    functions = [Function(item['program'], item['name'], item['cfg'], item['cgs'], item['label']) for item in data]
    print("All functions have been obtained! All graphs are being parsed:")

    newFunctions = []
    ignore = 0
    ignore_is_hot = 0
    big = 0
    big_hot = 0

    for function in functions:
        function.cfg = dot_to_graph(function.cfg, w2v)  # bert!!!
        function.cgs[0] = dot2graph(function.cgs[0])

        if len(function.cfg.node2label) <= Constants.MAX_NODE_NUM and len(function.cfg.edges) <= Constants.MAX_EDGE_NUM:
            function.cfg.adjust(w2v)
            function.cgs[0].adjust(w2v=False)
            function.cg_cutting()
            newFunctions.append(function)
        else:
            ignore += 1
            if function.label == "1":
                ignore_is_hot += 1
            if len(function.cfg.node2label) >= 500:
                big += 1
                if function.label == "1":
                    big_hot += 1

        cfg_nodeNum.append(len(function.cfg.node2label))
        cfg_edgeNum.append(len(function.cfg.edges))

    random.shuffle(newFunctions)
    Constants.IGNORE = ignore
    Constants.IGNORE_HOT = ignore_is_hot
    hot = sum(1 for f in newFunctions if f.label == "1")
    Constants.TEST_HOT_RATE = hot / len(newFunctions)
    return newFunctions


def print_max(a,n):
    a.sort(reverse=True)
    for i in range(n):
        print(a[i],end=' ')
    print('')


import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def data_analysis():
    functions = get_test_functions(sys.argv[1],w2v=False,if_ignore=False)
    for cg in name2graph.values():
        cg_nodeNum.append(len(cg.node2label))
        cg_edgeNum.append(len(cg.edges))
    cfgNodeNum = np.array(cfg_nodeNum)
    cfgEdgeNum = np.array(cfg_edgeNum)
    cgNodeNum = np.array(cg_nodeNum)
    cgEdgeNum = np.array(cg_edgeNum)

    print("cfg is "+str(len(cfg_nodeNum)))
    print("cg is "+str(len(cg_nodeNum)))
        
    name_list = ('1', '[2,9]', '[10,19]', '[20,99]', '[100,499]','≥500')
    value_list = [0,0,0,0,0,0]
    for i in cfg_nodeNum:
        if(i == 1):
            value_list[0]+=1
        elif(i>=2 and i<=9):
            value_list[1]+=1
        elif(i>9 and i<=19):
            value_list[2]+=1
        elif(i>19 and i<=99):
            value_list[3]+=1
        elif(i>99 and i<=499):
            value_list[4]+=1
        else:
            value_list[5]+=1
    value_list = np.array(value_list)
    pos_list = np.arange(len(name_list))
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((name_list)))
    plt.bar(pos_list, value_list, color = 'c', align = 'center')
    plt.savefig("cfg_distribution_before.png")
    return value_list



def pretrain(args):
    data = []
    functions = get_functions(args,False)
    for func in functions:
        for label in func.cfg.node2label.values():
            data.append(label.split(" "))
    functions = get_functions(args,False)
    for func in functions:
        for label in func.cfg.node2label.values():
            data.append(label.split(" "))
    
    model = Word2Vec(data,vector_size=args.WORD2VECTOR_DIM,workers=20,epochs=10,min_count=0)
    model.save("lr_word.model")


if(__name__=="__main__"):
    values  = data_analysis()
    print(Constants.TRAIN_HOT_RATE)
    print(Constants.TEST_HOT_RATE)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8)) 
    label=['n=1','2≤n≤9','10≤n≤19','20≤n≤99','100≤n≤499','n≥500'] 
    explode=[0.1,0.1,0.1,0.1,0.1,0.3] 
    plt.pie(values,explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
    plt.savefig('./before.png')