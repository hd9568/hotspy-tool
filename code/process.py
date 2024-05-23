from graphviz import Digraph
import graphviz
import torch 
import sys
sys.path.append('/home/hd/gp/code/')
sys.path.append('/home/hd/gp')
from constants import Constants
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from gensim.models.word2vec import Word2Vec
import random
import copy

# model = Word2Vec.load("lr_word.model")
# embedding = model.wv.vectors.mean(axis=0)

ID = 0
# 用于表示图的类
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
        # if w2v:
        #    self.node2vec[node_id]=sentence_to_vector(label)
        # else:
        #    self.node2vec[node_id]=[0,0,0]

    def add_edge(self, source, target):
        self.edges.append((source, target))


    def adjust(self,w2v):
        self.w2v = w2v
        # 初始化邻接矩阵为0
        n = len(self.node2label)
        self.matrix = [[0]*n for _ in range(n)]
        self.node_features = [[0 for _ in range(Constants.WORD2VECTOR_DIM)] for _ in range(Constants.MAX_NODE_NUM)]

        m = len(self.edges)
        self.edge_index = [[],[]]
        # 节点ID到矩阵索引的映射
        id_to_index = {}
        for index, node_id in enumerate(self.nodes):
           id_to_index[node_id]=index
        # index_to_id = {index: node_id for index, node_id in enumerate(self.node2label)}
        # print(self.nodes)
        # print(random.shuffle(self.nodes))

        for pair in self.node2label:
            if w2v:
                self.node2vec[pair]=sentence_to_vector(self.node2label[pair])
                self.node_features[id_to_index[pair]]=self.node2vec[pair]


        self.adj_matrix = np.zeros([Constants.MAX_NODE_NUM, Constants.MAX_NODE_NUM * Constants.MAX_EDGE_NUM * 2])
        # 填充邻接矩阵
        for source, target in self.edges:
            source_index = id_to_index[source]
            target_index = id_to_index[target]
            self.matrix[source_index][target_index] = 1  # 假设为无向图，如果是有向图，则还需要添加反向边
            self.edge_index[0].append(source_index)
            self.edge_index[1].append(target_index)
            if(w2v):
                self.adj_matrix[target_index][source_index] =  1
                self.adj_matrix[source_index][Constants.MAX_EDGE_NUM*Constants.MAX_NODE_NUM+target_index] =  1

       # print(self.edge_index)
    
    def display(self):
        # print("node2label:")
        # for node_id, label in self.node2label.items():
        #     print(f"ID: {node_id}, Label: {label}")
        # print("Edges:")
        # for source, target in self.edges:
        #     print(f"{source} -> {target}")
        # print("adjacent:")
        # print(torch.tensor(self.matrix))
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
        # if(c == 37):
        #     print(self.matrix)
        #     print(newGraph.matrix)
        return newGraph





# 读取.dot文件并转换成图数据结构
def dot_to_graph(dot_path,w2v=True):
    # 加载.dot文件
    with open(dot_path) as f:
        dot_graph = graphviz.Source(f.read(), format="dot")

    # 创建图实例
    graph = DotGraph(dot_path)
     
    # 解析节点
    # 解析节点
    for node in dot_graph.source.splitlines():
       if "->" in node:  # 如果是边
           parts = node.split("->")
           if len(parts) >= 2:
               source = parts[0].split(":")[0].strip()
               target = None
               if('[' in parts[1]):
                 target = parts[1].split("[")[0].strip()  # 假设边没有标签，且目标后可能跟随属性
               else:
                 target = parts[1].split(";")[0].strip()
               graph.add_edge(source, target)
           else:
               print("Unexpected format:", node)
       elif "[" in node:  # 如果是节点
               node_id =node.split(" ")[0].strip()
               label = node.split("label=\"{")[1].split("}\"")[0].strip()  # 假设只有标签属性
               graph.add_node(node_id, label)
    return graph



def sentence_to_vector(sentence):
    # inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # outputs = model(**inputs)
    # # 获取句子的平均池化特征作为句子向量
    # sentence_vector = outputs.last_hidden_state.mean(dim=1)
    # return sentence_vector.tolist()
    words = sentence.split(" ")
    wordvecs = None
    # for word in words:
    #     if(wordvecs is None):
    #         try:
    #           wordvecs=model.wv[word]
    #         except Exception:
    #           wordvecs=embedding
    #         continue
    #     try:
    #        wordvecs=np.vstack((wordvecs,model.wv[word]))
    #     except Exception:
    #        wordvecs=np.vstack((wordvecs,embedding))
    wordvecs = np.array([Constants.W2VMODEL.wv[word] if word in Constants.W2VMODEL.wv else Constants.W2VEMBEDDING for word in words])
    return wordvecs.mean(0).tolist()

# 示例：将.dot文件转换成图数据结构并打印
# dot_path = '/home/hd/gp/trainingSet/bt_data/cfg_dot/.add_.dot'  # 替换成你的.dot文件路径
# graph = dot_to_graph(dot_path)
# graph.display()


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

        node = Q.popleft()  # 左边最头部含有上一轮父层次的节点

        node2visited[node] = True
        nodes.append(node)

        if(len(nodes)>=Constants.CG_MAX_NODE_NUM):
            break
 
        for n in nx.neighbors(G, node):
            # 注意，如果n已经在队列里面，则不予重复添加。
            if (n not in node2visited):
                Q.append(n)
    # print('search_path', search_path)
    # print('=====')
    # print('\n标准的networkx广度优先搜索:')
    # print(list(nx.bfs_tree(G, source=0)))
    return nodes

class Function:
    def __init__(self, program,name, cfg, cgs, label,num=0):
        self.program = program
        self.name = name  # 函数名
        self.cfg = cfg    # 控制流图（Control Flow Graph）
        self.cgs = cgs      # 调用图（Call Graph）
        self.label = label  # 标签，代表是否为热点函数
        self.num = num
        # self.cfg_mean_vector = cfg.

        ## CG CUTTING BY NETWORKX
        self.cg_matrix = 0


    def __str__(self):
        return f"Function Name: {self.name}, Label: {self.label}"
    
    def cg_cutting(self):
        graph = self.cgs[0]
        # print("Function: "+self.name,end=" ")
      #  print("Before Cutting: "+str(len(graph.node2label)),end=" ")
        node = graph.label2node[self.name]

        nGraph = nx.DiGraph()
        nGraph.add_nodes_from(graph.node2label.keys())
        nGraph.add_edges_from(graph.edges)

        # nodes =nx.shortest_path(nGraph.to_undirected(),node).keys()
        if(len(nGraph.nodes) > Constants.CG_MAX_NODE_NUM):
           nGraph = nGraph.subgraph(bfs(nGraph.to_undirected(),node))
        self.cg_matrix = np.array(nx.adjacency_matrix(nGraph).todense())
      #  print("After Cutting: "+str(self.cg_matrix.shape[0]))

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

def get_functions(mode,w2v=True,if_ignore=True):
    json_file_path = ''
    if(mode=="train" or mode == "trainandtest"):
        json_file_path = '/home/hd/gp/trainSet/train.json'
    elif(mode=="test"):
        json_file_path = '/home/hd/gp/testSet/test.json'
    elif(mode=="selfTest"):
        json_file_path = '/home/hd/gp/selfTestSet/selfTest.json'
    elif(mode=="all"):
        json_file_path = '/home/hd/gp/allSet/all.json'
    elif(mode=="final"):
        json_file_path='/home/hd/gp/finalSet'
    else:
        print("mode is wrong")



    # 打开并读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
       data = json.load(file)
    json_string = json.dumps(data, indent=4)
    #print(json_string)

    # 遍历解析后的列表，为每个元素创建一个Function对象
    functions = [Function(item['program'],item['name'], item['cfg'], item['cgs'], item['label']) for item in data]
    print("All functions have been obtained! All graphs are being parsed:")


    # # 现在functions是一个Function对象的列表
    # for func in functions:
    #   print(f"Name: {func.name}, CFG: {func.cfg}, CGS: {func.cgs}, Label: {func.label}")
    newFunctions = []    
    i = 0
    ignore = 0
    ignore1 = 0
    oneNode = 0
    oneNode_hot = 0

    import random

    for function in tqdm(functions):
        function.cfg = dot_to_graph(function.cfg,w2v)## bert!!!
        function.cgs[0]=dot2graph(function.cgs[0])
        flag = random.randint(0,1)
        if(len(function.cfg.node2label)==1):
                oneNode += 1
                if(int(function.label)==1):
                  oneNode_hot += 1
        
        if(func_is_valid(function,if_ignore)):
            function.cfg.adjust(w2v)
            function.cgs[0].adjust(w2v=False)
            function.cg_cutting()
            newFunctions.append(function)
            ## random copy funcs
            if(function.label == '1'):
                newFunctions.append(function.copy())
                newFunctions.append(function.copy())
                newFunctions.append(function.copy())
                newFunctions.append(function.copy())
                newFunctions.append(function.copy())
                newFunctions.append(function.copy())
                newFunctions.append(function.copy())
                newFunctions.append(function.copy())
                # newFunctions.append(function.copy())
                # newFunctions.append(function.copy())
            else:
                newFunctions.append(function.copy())
        else:
            ignore+=1
            if(int(function.label)==1):
                ignore1+=1
        i+=1
        ################
        # if(i<=2846):
        #     print(str(len(function.cfg.node2label))
        #           +" "+str(len(function.cfg.edges))+" "+str(len(function.cgs[0].node2label))+" "+str(len(function.cgs[0].edges)))
        ###############
        # if(i==20):
        #     break
    print("The number of function which is only with one node is "+str(oneNode)+" ,"+str(oneNode_hot)+" of that is Hot")
    print(str(ignore)+" is ignored, "+str(ignore1)+" of which is hot")
    print("dataloader is done! the final number is " + str(len(newFunctions)))
    # function = functions[0]
    # function.cfg = dot_graph.dot_to_graph(function.cfg)
    # for i in range(len(function.cgs)):
    #     function.cgs[i]=dot_graph.dot_to_graph(function.cgs[i])
    for f in newFunctions:
        cfg_nodeNum.append(len(f.cfg.node2label))
        cfg_edgeNum.append(len(f.cfg.edges))
    random.shuffle(newFunctions)

    hot = 0
    notHot = 0
    for f in newFunctions:
        if f.label == "1":
            hot+=1
    notHot = len(newFunctions)-hot
    if(mode=="train" or mode == "trainandtest"):
        Constants.TRAIN_HOT_RATE = hot/len(newFunctions)
    else:
        Constants.TEST_HOT_RATE = hot/len(newFunctions)

    return newFunctions



def get_test_functions(mode,w2v=True,if_ignore=True):
    json_file_path = ''
    if(mode=="train" or mode == "trainandtest"):
        json_file_path = '/home/hd/gp/trainSet/train.json'
    elif(mode=="test"):
        json_file_path = '/home/hd/gp/testSet/test.json'
    elif(mode=="selfTest"):
        json_file_path = '/home/hd/gp/selfTestSet/selfTest.json'
    elif(mode=="all"):
        json_file_path = '/home/hd/gp/allSet/all.json'
    elif(mode=="final"):
        json_file_path='/home/hd/gp/finalSet/finalTest.json'
    else:
        print("mode is wrong")



    # 打开并读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
       data = json.load(file)
    json_string = json.dumps(data, indent=4)
    #print(json_string)

    # 遍历解析后的列表，为每个元素创建一个Function对象
    functions = [Function(item['program'],item['name'], item['cfg'], item['cgs'], item['label']) for item in data]
    print("All functions have been obtained! All graphs are being parsed:")


    # # 现在functions是一个Function对象的列表
    # for func in functions:
    #   print(f"Name: {func.name}, CFG: {func.cfg}, CGS: {func.cgs}, Label: {func.label}")
    newFunctions = []    
    import random
    ignore  = 0
    ignore_is_hot = 0
    big = 0
    big_hot = 0

    for function in tqdm(functions):
        function.cfg = dot_to_graph(function.cfg,w2v)## bert!!!
        function.cgs[0]=dot2graph(function.cgs[0])

        if(len(function.cfg.node2label) <= Constants.MAX_NODE_NUM and len(function.cfg.edges) <= Constants.MAX_EDGE_NUM):
                function.cfg.adjust(w2v)
                function.cgs[0].adjust(w2v=False)
                function.cg_cutting()
                newFunctions.append(function)
        else:
                ignore+=1
                if(function.label=="1"):
                    ignore_is_hot+=1
                    # print("======================")
                    # print("nodes:")
                    # print(len(function.cfg.node2label))
                    # print("edges:")
                    # print(len(function.cfg.edges))
                if(len(function.cfg.node2label)>=500):
                    big+=1
                    if(function.label == "1"):
                        big_hot+=1
                pass

        ################
        cfg_nodeNum.append(len(function.cfg.node2label))
        cfg_edgeNum.append(len(function.cfg.edges))
    random.shuffle(newFunctions)
    print("big is "+str(big)+" ,"+str(big_hot)+" of which is hot!")
     
    print(str(ignore)+" is ignored, ",str(ignore_is_hot)+" of which is hot!")
    hot = 0
    notHot = 0
    for f in newFunctions:
        if f.label == "1":
            hot+=1
    notHot = len(newFunctions)-hot
    if(mode=="train" or mode == "trainandtest"):
        Constants.TRAIN_HOT_RATE = hot/len(newFunctions)
    else:
        Constants.TEST_HOT_RATE = hot/len(newFunctions)


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
    functions = get_functions(sys.argv[1],w2v=False,if_ignore=True)
    for cg in name2graph.values():
        cg_nodeNum.append(len(cg.node2label))
        cg_edgeNum.append(len(cg.edges))
    cfgNodeNum = np.array(cfg_nodeNum)
    cfgEdgeNum = np.array(cfg_edgeNum)
    cgNodeNum = np.array(cg_nodeNum)
    cgEdgeNum = np.array(cg_edgeNum)

    print("cfg is "+str(len(cfg_nodeNum)))
    print("cg is "+str(len(cg_nodeNum)))
        
    # style set 这里只是一些简单的style设置
    # sns.set_palette('deep', desat=.6)
    # sns.set_context(rc={'figure.figsize': (8, 5) } )
    # np.random.seed(1425)
    # figsize是常用的参数.
    # bins = 100

    # plt.hist(x = cfgNodeNum,bins=bins,edgecolor="blue",range=(0,10000))
    # plt.savefig(sys.argv[1]+"_cfg_node.png")
    # print("cfg node num")
    # print_max(cfg_nodeNum,15)


    # plt.hist(x = cfg_edgeNum,bins=bins,edgecolor="red",range=(0,10000))
    # plt.savefig(sys.argv[1]+"_cfg_edge.png")
    # print("cfg edge num!")
    # print_max(cfg_edgeNum,15)

    # plt.hist(x = cg_nodeNum,bins=bins,edgecolor="green",range=(0,2000))
    # plt.savefig(sys.argv[1]+"_cg_node.png")
    # print("cg node num")
    # print_max(cg_nodeNum,15)

    # plt.hist(x = cg_edgeNum,bins=bins,edgecolor="orange",range=(0,3000))
    # plt.savefig(sys.argv[1]+"_cg_edge.png")
    # print("cg edge num")
    # print_max(cg_edgeNum,15)
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
    plt.savefig("cfg_distribution_after.png")
    return value_list



def pretrain():
    data = []
    functions = get_functions("train",False)
    for func in functions:
        for label in func.cfg.node2label.values():
            data.append(label.split(" "))
    functions = get_functions("test",False)
    for func in functions:
        for label in func.cfg.node2label.values():
            data.append(label.split(" "))
    
    model = Word2Vec(data,vector_size=Constants.WORD2VECTOR_DIM,workers=20,epochs=10,min_count=0)
    model.save("lr_word.model")


if(__name__=="__main__"):
    # if(sys.argv[1] == "train"):
    #    functions = get_functions(sys.argv[1],w2v=False,if_ignore=True)
    # else:
    #    functions = get_test_functions(sys.argv[1],w2v=False)
    

    # print(Constants.TRAIN_HOT_RATE)
    # print(Constants.TEST_HOT_RATE)

    values  = data_analysis()
    print(Constants.TRAIN_HOT_RATE)
    print(Constants.TEST_HOT_RATE)
    # program2total = {}
    # for func in tqdm(functions):
    #         # 将数据移动到GPU或CPU
    #         input = func
    #         target = int(func.label)

    #         if(func.program in program2total):
    #             program2total[func.program] += 1
    #         else:
    #             program2total[func.program] = 1
    # for program in program2total.keys():
    #     print(program + " datasize: "+str(program2total[program])+";")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))#将画布设定为正方形，则绘制的饼图是正圆
    label=['n=1','2≤n≤9','10≤n≤19','20≤n≤99','100≤n≤499','n≥500']#定义饼图的标签，标签是列表
    explode=[0.1,0.1,0.1,0.1,0.1,0.3]#设定各项距离圆心n个半径
    #plt.pie(values[-1,3:6],explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
    plt.pie(values,explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
    plt.savefig('./after.png')

    #   func = functions[0]
    #   print(func.name)
    #   print(func.cgs[0].label2node[func.name])
    # data_analysis()
#       # get_all_words()
#       model = Word2Vec.load("lr_word.model")
#       embedding=[]
#       # embedding = np.average(model.wv.vectors)
#       n = len(model.wv)
#       feature = np.zeros(Constants.WORD2VECTOR_DIM,dtype="float")
#       for v in model.wv.vectors:
#           feature = np.add(feature,v)
#       print(np.divide(feature,n))

#       try:
#            embedding = model.wv['%584']
#       except Exception:
#            embedding = np.average(model.wv.vectors())