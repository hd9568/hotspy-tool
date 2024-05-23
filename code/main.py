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

def savePicture(loss_values,name):
    time_steps = range(len(loss_values))

    # 绘制损失曲线
    plt.plot(time_steps, loss_values, '-o')

    # 设置图表标题和坐标轴标签
    plt.title('Loss Curve')
    plt.xlabel('Time Steps')
    plt.ylabel('Loss')

    # 显示网格线
    plt.grid(True)

    # 保存图表为PNG格式
    plt.savefig(name+'Loss.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    print("MODEL BEGINNING!")
    # print(torch.cuda.get_device_name(0))
    # model = MyModel(Constants.BERT_BASE_UNCASED_DIM,Constants.MPNN_OUT_DIM,
    #                 Constants.CNN_INPUT_CHANNELS,Constants.CNN_COV_OUTPUT_CHANNELS,
    #                 Constants.CNN_OUTPUT_DIM,Constants.MLP_HIDDEN_SIZE).to(device)
    # functions = data_loader.get_functions()
    # print(len(functions))
    # #print(functions[0].cgs)
    # print(model(functions[0]))
    model_name =  sys.argv[2]

    testFunctions = []

    # print("Pretraining...")
    # process.pretrain()
    # print("Pretrain is ok!")

    Constants.W2VMODEL = Word2Vec.load("lr_word.model")
    Constants.W2VEMBEDDING = Constants.W2VMODEL.wv.vectors.mean(axis=0)


    if(sys.argv[1] == "trainandtest"):
        starttime = datetime.datetime.now()
        testFunctions = process.get_test_functions("test",if_ignore=False)
        endtime = datetime.datetime.now()
        print("Test dataloader's time is "+str((endtime - starttime).seconds/60)+" mins")
     
    if(sys.argv[1] =="train" or sys.argv[1] == "littleTrain" or sys.argv[1] == "trainandtest"):
        print("This is GPU"+str(torch.cuda.current_device())+": "+torch.cuda.get_device_name(torch.cuda.current_device()))

        starttime = datetime.datetime.now()
        with torch.no_grad():
          functions = process.get_functions(sys.argv[1])
        endtime = datetime.datetime.now()
        print("Train dataloader's time is "+str((endtime - starttime).seconds/60)+" mins")
        print("========== All training data has already been ready! ===========")

        model = MyModel().to(device)
        weights = None
        weights = torch.tensor([1, (1-Constants.TRAIN_HOT_RATE)/Constants.TRAIN_HOT_RATE]).to(device)
        criterion = nn.BCELoss(weight=weights)  # 二分类交叉熵损失
        optimizer = optim.Adam(model.parameters(), Constants.LR)
        epochs = Constants.EPOCHS
        train_loss_values = []
        valid_loss_values = []
        flag = 0
        print("TRAIN_HOT:",Constants.TRAIN_HOT_RATE)
        print("TEST_HOT",Constants.TEST_HOT_RATE)
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
                train_loss = criterion(output, target) 
                
                optimizer.zero_grad()  
                train_loss.backward()
                optimizer.step()  # 更新参数
            # 打印每个epoch的损失
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss.item():.5f}')
            train_loss_values.append(train_loss.item())

            name = model_name+"_"+str(epoch)+'.pth'
            # torch.save(model.state_dict(),model_name+str(epoch)+'.pth')
            tp,tn,fp,fn = 0,0,0,0
            if(sys.argv[1]=="trainandtest"):
                model.eval()
                total = len(testFunctions)
                ac = 0
                with torch.no_grad():
                    for func in tqdm(testFunctions):
                        # 将数据移动到GPU或CPU
                        input = func
                        target = int(func.label)
                        output = model(input).to(device)
                        output = float(output[0].cpu().item())
                        ans = 0
                        if(output < 0.5):
                            ans = 1
                        if(ans == target):
                            ac = ac + 1

                        if(ans == 1 and target == 1):
                            tp += 1
                        elif(ans == 0 and target == 0):
                            tn += 1
                        elif(ans == 0 and target == 1):
                            fn += 1
                        elif(ans == 1 and target == 0):
                            fp += 1

                print("| TN="+str(tn)+" FP="+str(fp)+" |")
                print("| FN="+str(fn)+" TP="+str(tp)+" |")

                print("test accuracy: {:.2f}%".format(ac/total * 100))
                if(tp + fp != 0):
                   print("precision: {:.2f}%".format(tp / (tp + fp) * 100))
                print("recall: {:.2f}%".format(tp / (tp + fn) * 100))
                print(name+" test accuracy: {:.2f}%".format(ac / total * 100))
                torch.save(model.state_dict(),name)

        savePicture(train_loss_values, model_name)     
        print("=============== training finished ===============")

    elif(sys.argv[1] == "test" or sys.argv[1] == "selfTest" or sys.argv[1]=="all"):
        model_name = sys.argv[2]
        model = MyModel().to(device)
        model.load_state_dict(torch.load(model_name+".pth"))
        model.eval()
        print("this is GPU"+str(torch.cuda.current_device())+": "+torch.cuda.get_device_name(torch.cuda.current_device()))
        starttime = datetime.datetime.now()
        functions = process.get_test_functions(sys.argv[1],if_ignore=False)
        endtime = datetime.datetime.now()
        print("test dataloader's time is "+str((endtime - starttime).seconds/60)+" mins")
        print("========== all test data has already been ready! ===========")
        total = len(functions)
        ac = 0
        tp,tn,fp,fn = 0,0,0,0
        program2total = {}
        program2acc = {}
        program2hot = {}
        with torch.no_grad():
            for func in tqdm(functions):
                # 将数据移动到GPU或CPU
                input = func
                target = int(func.label)

                if(func.program in program2total):
                    program2total[func.program] += 1
                else:
                    program2total[func.program] = 1
                    program2hot[func.program] = 0
                
                if(func.program not in program2acc):
                    program2acc[func.program] = 0
                
                if(target == 1):
                     program2hot[func.program] += 1

   
                output = model(input).to(device)
                output = float(output[0].cpu().item())
                ans = 0
                if(output < 0.5):
                    print(func.name)
                    ans = 1
                if(ans == target):
                    ac = ac + 1
                    program2acc[func.program] += 1
                
                if(ans == 1 and target == 1):
                    tp += 1
                elif(ans == 0 and target == 0):
                    tn += 1
                elif(ans == 0 and target == 1):
                    fn += 1
                elif(ans == 1 and target == 0):
                    fp += 1
        print("| TN="+str(tn)+" FP="+str(fp)+" |")
        print("| FN="+str(fn)+" TP="+str(tp)+" |")
       
        if(ac/total > 0.8):
            print("00000000000000000000000000000000000000000000000000000000")

        print("test accuracy: {:.2f}%".format(ac/total * 100))
 
        if(tp + fp != 0):
            print("precision: {:.2f}%".format(tp / (tp + fp) * 100))
        print("recall: {:.2f}%".format(tp/(tp+fn) * 100))
        print("f1 score: {:.3f}".format(2*tp/(2*tp+fn+fp)))

        for program in program2acc.keys():
            print(program + " datasize: "+str(program2total[program])+"; "+"hot: {:.4f}".format(program2hot[program]/program2total[program])+"; acc: "+"{:.4f}".format(program2acc[program]/program2total[program]))
    else:
        print("wrong mode!")

