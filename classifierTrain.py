import os
import xml.etree.ElementTree as et

import torch
import random
# from tagger import find_verb
from Module import RobertaTemporalModel,CUDAorCPU
from Module_test import Xmldataset
# CUDAorCPU = torch.device("cuda")

# test_path = ["ClassifierTrainData/TE3-Silver-data-5/XIN_ENG_20061130.0405.tml",
#              "ClassifierTrainData/TE3-Silver-data-0/AFP_ENG_19970401.0006.tml"
#              ]

data = os.listdir("ClassifierTrainData")
data_file_path = []
for i in data:
    if "TE3" in i:
        i_path = os.path.join("ClassifierTrainData", i)
        for j in os.listdir(i_path):
            path = os.path.join(i_path, j)
            data_file_path.append(path)

data_file_path2=[]
path1 = os.path.join("ClassifierTrainData","TBAQ-cleaned","AQUAINT")
path2 = os.path.join("ClassifierTrainData","TBAQ-cleaned","TimeBank")
for i in os.listdir(path1):
    data_file_path2.append(os.path.join(path1,i))
for i in os.listdir(path2):
    data_file_path2.append(os.path.join(path2,i))
random.shuffle(data_file_path)
random.shuffle(data_file_path2)
data_file_path = data_file_path[:200]+data_file_path2[:250]
random.shuffle(data_file_path)
# data_file_path = ["ClassifierTrainData/TE3-Silver-data-0/AFP_ENG_19970409.0172.tml"]
model = RobertaTemporalModel()
#读取模型
model.load_state_dict(torch.load("classifier.pth"))

model = model.to(CUDAorCPU)
xmldataset = Xmldataset(data_file_path)
label_dict = {
    "BEFORE":0,
    "IBEFORE":0,
    "AFTER":1,
    "IAFTER":1,
    "INCLUDES":2,
    "IS_INCLUDED":2,
    "SIMULTANEOUS":2,
    "IDENTITY":2
}
#计算方差
# criterion = torch.nn.MSELoss()
#计算交叉熵
criterion = torch.nn.CrossEntropyLoss()

#优化器
optimizer = torch.optim.Adam(model.parameters(),lr=0.000001)
#训练
index = 0
epoch = 8
answer = 0
acc = 0
sum_answer = 0
for i in range(epoch):
    random.shuffle(data_file_path)
    random.shuffle(data_file_path2)
    data_file_path = data_file_path[:250] + data_file_path2[:250]
    random.shuffle(data_file_path)
    while True:
        index += 1
        # if index<10000:
        #     continue
        optimizer.zero_grad()
        data = xmldataset.__getitem__()

        if data is None or data=="OVER":
            break
        if data == "DataERR":
            continue
        S, s2_start, s2_end, s4_start, s4_end, label=data

        if label not in ["BEFORE","AFTER","INCLUDES",
                         "IS_INCLUDED","SIMULTANEOUS","IDENTITY","IAFTER","IBEFORE"]:
            # print(label,"may be ERR")
            continue
        label = torch.tensor(label_dict[label]).unsqueeze(0)
        # a = torch.zeros(5)
        # a[label_dict[label]]=1
        out = model(S,s2_start,s2_end,s4_start,s4_end)
        if out == None:
            continue
        loss = criterion(out.cpu(),label)
        loss.backward()
        optimizer.step()
        sum_answer+=1
        if torch.argmax(out[0]).item()>=0.7:
            answer += 1
            if torch.argmax(out[0]).item()==label.item():
                acc+=1
            else:
                acc=acc+1-1
        if index%100==0:
            print(index,loss.item())
            print(acc,"/",answer)
            print(label,out[0].tolist())
            print("accuracy:",acc/answer)
            print("filter",answer/sum_answer)
            acc=0
            answer=0
            sum_answer=0
            print("")

#保存模型
# torch.save(model.state_dict(),"classifier_attention.pth")
torch.save(model.state_dict(),"classifier.pth")
# 已执行至60000

print("over")





