import os

import torch
from Module import RobertaTemporalModel,CUDAorCPU,RobertaTokenizer
from tagger import find_verb
import pandas as pd

def get_info():
    sum_label = 0
    before = 0
    after = 0
    simultaneous = 0
    with open("FinalData_Finshed.txt")as f:
        for i in f.readlines():
            sum_label+=int(i.split(",")[1])
            before+=int(i.split(",")[2])
            after+=int(i.split(",")[3])
            simultaneous+=int(i.split(",")[4][:-1])
        #打印所有结果
        print("sum_label:",sum_label)
        print("before:",before)
        print("after:",after)
        print("simultaneous:",simultaneous)
get_info()
def wash(text):
    text = text.split("\n\n")
    a = []
    for i in text:
        if len(i)<5:
            continue
        if i[0] == "@":
            break
        a.append(i)
    text = "\n\n".join(a)
    return text
tokenizer = RobertaTokenizer.from_pretrained('model')

model = RobertaTemporalModel()
#读取模型
model.load_state_dict(torch.load("classifier.pth"))
model = model.to(CUDAorCPU)
model.eval()
label_dict = {
    0:"before",
    1:"after",
    2:"simultaneous"
}

file_path = os.path.join("DataFromCNN","stories")
data_list = os.listdir(file_path)[:500]
FILE_SKIP = []
confidence = {
    0:0.9,
    1:0.7,
    2:0.9,
}
append_new = []
with open("FinalData_Finshed.txt")as f:
    for i in f.readlines():
        FILE_SKIP.append(i.split(",")[0])
for FILE in data_list:
    if FILE in FILE_SKIP:
        continue

    file = os.path.join(file_path,FILE)
    # print(file)
    with open(file,"r")as f:
        text = f.read()
    # 删去文本中存在的@部分的文本
    text = wash(text)
    text_list = text.split("\n\n")
    verb_pos = []

    for S_index in range(len(text_list)):
        tokens,indices,sentence_token = find_verb(text_list[S_index],confidence=0.4,val="test")
        # print(tokens,indices)
        last_indice = -2
        for token,indice in zip(tokens.split(),indices):
            if indice-last_indice==1:
                continue
            last_indice = indice
            tokenlen = len(token)+1
            s1 = tokenizer.decode(sentence_token[:indice]).replace("<s>","").replace("</s>","")
            pos = len(s1)
            verb_pos.append((S_index,pos,tokenlen))
    for index,s2_pos in zip(range(len(verb_pos)),verb_pos):
        for s4_pos in verb_pos[index+1:]:
            append_dict = {}
            s2 = text_list[s2_pos[0]][s2_pos[1]:s2_pos[1]+s2_pos[2]]
            s4 = text_list[s4_pos[0]][s4_pos[1]:s4_pos[1]+s4_pos[2]]
            s1 = text_list[s2_pos[0]][:s2_pos[1]]
            s3 = text_list[s2_pos[0]][s2_pos[1]+s2_pos[2]:]+\
                 "".join(text_list[s2_pos[0]+1:s4_pos[0]])+\
                 text_list[s4_pos[0]][:s4_pos[1]]
            s5 = text_list[s4_pos[0]][s4_pos[1]+s4_pos[2]:]

            append_dict["S1"]=s1
            append_dict["S2"]=s2
            append_dict["S3"]=s3
            append_dict["S4"]=s4
            append_dict["S5"]=s5
            append_dict["FILE"] = FILE.split(".")[0]

            S = s1+s2+s3+s4+s5
            if len(S)>=300:
                break
            s1 = tokenizer(s1, return_tensors="pt")["input_ids"][0][:-1]
            s2 = tokenizer(s2, return_tensors="pt")["input_ids"][0][1:-1]
            s3 = tokenizer(s3, return_tensors="pt")["input_ids"][0][1:-1]
            s4 = tokenizer(s4, return_tensors="pt")["input_ids"][0][1:-1]
            S = tokenizer(S, return_tensors="pt")

            s2_start = len(s1)
            s2_end = s2_start + len(s2)
            s4_start = s2_end + len(s3)
            s4_end = s4_start + len(s4)

            out = model(S,s2_start,s2_end,s4_start,s4_end)
            # print(out)
            if out is None:
                continue
            if max(out[0]).item()>=confidence[torch.argmax(out[0]).item()]:
                #判断最大对应的标签
                append_dict["label"]=label_dict[torch.argmax(out[0]).item()]
                append_dict["confidence"]=max(out[0]).item()
                append_new.append(append_dict)
    info = pd.DataFrame(append_new)
    save_path = os.path.join("GenerateData","{}.csv".format(FILE.split(".")[0]))
    info.to_csv((save_path),index=False)
    append_new = []
    if len(info)==0:
        print(FILE,"Empty")
        continue
    with open("FinalData_Finshed.txt","a")as f:
        f.write("{},{},{},{},{}\n".format(FILE,len(info),
                                            info["label"].tolist().count("before"),
                                            info["label"].tolist().count("after"),
                                            info["label"].tolist().count("simultaneous")
                                        ))
        print(FILE,"Done")


# model.eval()
# with torch.no_grad():
#     model.requires_grad = False






