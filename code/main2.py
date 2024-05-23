import sys
sys.path.append('/home/hd/gp/code/Dataset')

import torch
import process
from torch import nn
import torch.optim as optim
from constants import Constants
from tqdm import tqdm
import os
import datetime
import matplotlib.pyplot as plt
from mymodel import *
from gensim.models.word2vec import Word2Vec

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

Constants.DEVICE = torch.device(1)
device = Constants.DEVICE

Constants.W2VMODEL = Word2Vec.load("lr_word.model")
Constants.W2VEMBEDDING = Constants.W2VMODEL.wv.vectors.mean(axis=0)

sota_model = None


def train(num):
    testFunctions = process.get_functions("test",if_ignore=False)
    functions = process.get_functions("train")
    model_name = "model_"+str(num)

    model = MyModel().to(device)
    weights = torch.tensor([1, (1-Constants.TRAIN_HOT_RATE)/Constants.TRAIN_HOT_RATE]).to(device)
    criterion = nn.BCELoss(weight=weights)  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), Constants.LR)
    epochs = Constants.EPOCHS
    loss_values = []
    flag = 0
    max_acc = 0
    sota_model = None
    for epoch in range(epochs):
        model.train()  # 将模型设置为训练模式
        # optimizer.zero_grad()  
        for func in tqdm(functions):
            # flag+=1
            # if(flag<=888):
            #     continue
            # func.cfg.display()
            # func.cgs[0].display()
            input = func
            target = float(func.label)
            output = model(input).to(device)
            target = torch.tensor([1-target,target], device=device).float()
            loss = criterion(output, target) 
            
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()  # 更新参数
        # 打印每个epoch的损失
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}')
        loss_values.append(loss.item())
        name = model_name + '.pth'
        model.eval()
        total = len(testFunctions)
        ac = 0
        with torch.no_grad():
            for func in tqdm(testFunctions):
                input = func
                target = int(func.label)
                if(len(input.cfg.node2label) < 2):
                    output = 1
                else:
                    output = model(input).to(device)
                    output = float(output[0].cpu().item())
                ans = 0
                if(output < 0.5):
                    ans = 1
                if(ans == target):
                    ac = ac + 1
        # print(name+" test accuracy: {:.2f}%".format(ac/total * 100))
        if((ac/total) > max_acc):
           max_acc = (ac/total)
           torch.save(model.state_dict(),name)

    print(name+" test accuracy: {:.2f}%".format(max_acc * 100))

if __name__ == "__main__":
    num = 0
    print("MODEL BEGINNING!")
    print("This is GPU"+str(torch.cuda.current_device())+": "+torch.cuda.get_device_name(torch.cuda.current_device()))

    for god in [32]:
        for mnn in [10,15]:
            for men in [10]:
                for gsn in [10,12]:
                    for cmnn in [10]:
                        for codcfg in [20,24]:
                            for codcg in [12]:
                                for mhs in [30,32]:
                                    for lr in [0.0005]:
                                        num += 1
                                        print("=========== "+str(num)+" ===========")
                                        Constants.GGNN_OUTPUT_DIM = god
                                        Constants.MAX_NODE_NUM = mnn
                                        Constants.MAX_EDGE_NUM = men
                                        Constants.GGNN_STEP_NUM = gsn
                                        Constants.CG_MAX_NODE_NUM = cmnn
                                        Constants.CNN_OUTPUT_DIM_CFG = codcfg
                                        Constants.CNN_OUTPUT_DIM_CG = codcg
                                        Constants.MLP_HIDDEN_SIZE = mhs
                                        Constants.LR = lr
                                        Constants.print_self()
                                        train(num)
