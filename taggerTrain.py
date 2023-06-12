import torch.nn as nn
import torch
from transformers.modeling_roberta import RobertaModel
from transformers import RobertaTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import os
from Module import Extraction,Mydataset

vocab_file = 'model/vocab.json'
merges_file = 'model/merges.txt'
tokenizer = RobertaTokenizer(vocab_file, merges_file)

bert = RobertaModel.from_pretrained('model')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')


model = Extraction()
#读取.pt
model.load_state_dict(torch.load("Extraction.pth"))
model = model.cuda(0)

# datapath = ["AQ_Platinum_all.csv",
#             "TB_remaining_147docs.csv",
#             "TBDense_all_new.csv"]
datapath = os.listdir("TaggerTrainData")
datasets = Mydataset(datapath)
# dataloader = DataLoader(datasets,batch_size=8)

# datasets.__getitem__(0)
# loss_fun=nn.BCEWithLogitsLoss()
# loss_fun = nn.CrossEntropyLoss()
loss_fun = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(),lr=0.0001)
acc = 0
sum_acc=0

epochs = 3
for epoch in range(epochs):
    for i in range(datasets.__len__()):
        sentence,label = datasets.__getitem__(i)
        if label == None:
            continue
        out = model(sentence)
        label = torch.tensor(label).unsqueeze(0).unsqueeze(2).float()
        out = out.cpu()
        loss = loss_fun(out,label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # print(loss.item())

        out = out.cpu()
        label = label.cpu()


        indices = torch.nonzero(out>0.5)
        sum_acc += torch.nonzero(label>0).shape[0]
        out = out.view(-1)
        a = torch.zeros(out.size())
        a[torch.nonzero(out > 0.5)] = 1
        acc += sum(label.view(-1)*a).item()
        if i%100==0:
            print(i,"loss:",loss.item())
            print("acc:",acc/sum_acc)
            print("{}/{}".format(acc,sum_acc))
            print()
            acc=0
            sum_acc=0
    # print("epoch {}:".format(epoch),loss.item())
    torch.save(model.state_dict(),"Extraction_all.pth")



