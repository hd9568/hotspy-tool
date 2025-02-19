import sys
import torch
import process
from torch import nn
import torch.optim as optim
from constants import Constants
from tqdm import tqdm
import os
import datetime
import matplotlib.pyplot as plt
from myModel import *
from gensim.models.word2vec import Word2Vec
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def savePicture(loss_values, name):
    plt.plot(range(len(loss_values)), loss_values, '-o')
    plt.title('Loss Curve')
    plt.xlabel('Time Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(name + 'Loss.png', dpi=300, bbox_inches='tight')

def calculate_confusion_matrix(predictions, targets, ignore_count=0, ignore_hot_count=0):
    """
    计算混淆矩阵中的各个值:TN, FP, FN, TP
    :param predictions: 预测结果列表
    :param targets: 真实标签列表
    :return: TN, FP, FN, TP
    """
    tn, fp, fn, tp = 0, 0, 0, 0

    for pred, target in zip(predictions, targets):
        if pred == 1 and target == 1:
            tp += 1
        elif pred == 0 and target == 0:
            tn += 1
        elif pred == 0 and target == 1:
            fn += 1
        elif pred == 1 and target == 0:
            fp += 1

    # 考虑忽略的数据
    tn += Constants.IGNORE - Constants.IGNORE_HOT
    fn += Constants.IGNORE_HOT

    return tn, fp, fn, tp

def main():
    parser = argparse.ArgumentParser(description="Train and test the model.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'trainandtest', 'littleTrain', 'selfTest', 'all'], help='Mode of operation')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--device', type=int, default=1, help='GPU device index')
    parser.add_argument('--train_json_path', type=str, default='/home/hd/gp/trainSet/train.json', help='Path to training JSON file')
    parser.add_argument('--test_json_path', type=str, default='/home/hd/gp/testSet/test.json', help='Path to testing JSON file')
    parser.add_argument('--self_test_json_path', type=str, default='/home/hd/gp/selfTestSet/selfTest.json', help='Path to self-test JSON file')
    parser.add_argument('--all_json_path', type=str, default='/home/hd/gp/allSet/allTest.json', help='Path to all JSON file')
    parser.add_argument('--final_json_path', type=str, default='/home/hd/gp/finalSet/finalTest.json', help='Path to final JSON file')
    parser.add_argument('--hotcopy', type=int, default=5, help='Number of copies for hotspot')
    parser.add_argument('--nohotcopy', type=int, default=1, help='Number of copies for non-hotspot')

    args = parser.parse_args()

    Constants.EPOCHS = args.epochs
    Constants.LR = args.lr
    Constants.DEVICE = torch.device(args.device)
    device = Constants.DEVICE

    print("MODEL BEGINNING!")
    model_name = args.model_name
    testFunctions = []

    Constants.W2VMODEL = Word2Vec.load("lr_word.model")
    Constants.W2VEMBEDDING = Constants.W2VMODEL.wv.vectors.mean(axis=0)

    if args.mode == "trainandtest":
        starttime = datetime.datetime.now()
        testFunctions = process.get_test_functions(args, if_ignore=False)
        endtime = datetime.datetime.now()
        print(f"Test dataloader's time is {(endtime - starttime).seconds / 60:.2f} mins")

    if args.mode in ["train", "littleTrain", "trainandtest"]:
        print(f"This is GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        starttime = datetime.datetime.now()
        with torch.no_grad():
            functions = process.get_functions(args)
        endtime = datetime.datetime.now()
        print(f"Train dataloader's time is {(endtime - starttime).seconds / 60:.2f} mins")
        print("========== All training data has already been ready! ===========")

        model = MyModel().to(device)
        weights = torch.tensor([1, (1 - Constants.TRAIN_HOT_RATE) / Constants.TRAIN_HOT_RATE]).to(device)
        criterion = nn.BCELoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), Constants.LR)
        train_loss_values = []

        for epoch in range(Constants.EPOCHS):
            model.train()
            for func in tqdm(functions):
                input = func
                target = float(func.label)
                output = model(input).to(device)
                target = torch.tensor([1 - target, target], device=device).float()
                train_loss = criterion(output, target)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{Constants.EPOCHS}], Loss: {train_loss.item():.5f}')
            train_loss_values.append(train_loss.item())

            name = f"./model/{model_name}_{epoch}.pth"
            torch.save(model.state_dict(), name)

            if args.mode == "trainandtest":
                model.eval()
                total = len(testFunctions) + Constants.IGNORE
                ac = Constants.IGNORE - Constants.IGNORE_HOT
                predictions, targets = [], []

                with torch.no_grad():
                    for func in tqdm(testFunctions):
                        input = func
                        target = int(func.label)
                        output = model(input).to(device)
                        output = float(output[0].cpu().item())
                        ans = 0 if output < 0.5 else 1

                        if ans == target:
                            ac += 1

                        predictions.append(ans)
                        targets.append(target)

                tn, fp, fn, tp = calculate_confusion_matrix(predictions, targets, Constants.IGNORE, Constants.IGNORE_HOT)
                print(f"| TN={tn} FP={fp} |")
                print(f"| FN={fn} TP={tp} |")
                print(f"test accuracy: {(ac / total * 100):.2f}%")
                if tp + fp != 0:
                    print(f"precision: {(tp / (tp + fp) * 100):.2f}%")
                print(f"recall: {(tp / (tp + fn) * 100):.2f}%")
                print(f"{name} test accuracy: {(ac / total * 100):.2f}%")
                torch.save(model.state_dict(), name)

        savePicture(train_loss_values, model_name)
        print("=============== training finished ===============")

    elif args.mode in ["test", "selfTest", "all"]:
        model_name = args.model_name
        model = MyModel().to(device)
        model.load_state_dict(torch.load(model_name + ".pth"))
        model.eval()
        print(f"This is GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        starttime = datetime.datetime.now()
        functions = process.get_test_functions(args.mode, if_ignore=False)
        endtime = datetime.datetime.now()
        print(f"Test dataloader's time is {(endtime - starttime).seconds / 60:.2f} mins")
        print("========== All test data has already been ready! ===========")

        total = len(functions) + Constants.IGNORE
        ac = Constants.IGNORE - Constants.IGNORE_HOT
        predictions, targets = [], []
        program2total = {}
        program2acc = {}
        program2hot = {}

        with torch.no_grad():
            for func in tqdm(functions):
                input = func
                target = int(func.label)

                if func.program in program2total:
                    program2total[func.program] += 1
                else:
                    program2total[func.program] = 1
                    program2hot[func.program] = 0

                if func.program not in program2acc:
                    program2acc[func.program] = 0

                if target == 1:
                    program2hot[func.program] += 1

                output = model(input).to(device)
                output = float(output[0].cpu().item())
                ans = 0 if output < 0.5 else 1

                if ans == target:
                    ac += 1
                    program2acc[func.program] += 1

                predictions.append(ans)
                targets.append(target)

        tn, fp, fn, tp = calculate_confusion_matrix(predictions, targets, Constants.IGNORE, Constants.IGNORE_HOT)

        print(f"| TN={tn} FP={fp} |")
        print(f"| FN={fn} TP={tp} |")
        print(f"test accuracy: {(ac / total * 100):.2f}%")

        if tp + fp != 0:
            print(f"precision: {(tp / (tp + fp) * 100):.2f}%")
        print(f"recall: {(tp / (tp + fn) * 100):.2f}%")
        print(f"f1 score: {(2 * tp / (2 * tp + fn + fp)):.3f}")

    else:
        print("Wrong mode!")

if __name__ == "__main__":
    main()