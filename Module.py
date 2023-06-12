import torch
import torch.nn as nn

from transformers.modeling_roberta import RobertaModel
from transformers import RobertaTokenizer
import pandas as pd
from torch.utils.data import Dataset

import xml.etree.ElementTree as et

CUDAorCPU = torch.device("cuda")

class Xmldataset():
    def __init__(self,file_list):
        self.file_list = file_list
        self.index_data = 0
        self.index_tlink = 0
        self.readnew = True
        self.tokenizer = RobertaTokenizer.from_pretrained('model')
    def __getitem__(self):
        if self.readnew:
            if self.index_data==len(self.file_list):
                return None
            self.read()
            self.readnew = False
            self.index_data+=1
            self.index_tlink=0

        if len(self.tlinks)==0:
            self.readnew = True
            return "DataERR"
        if self.index_tlink==self.len-1:
            self.readnew=True
        self.index_tlink += 1
        S = "".join(self.data_text)
        if len(S)>2000:
            return "DataERR"
        S = self.tokenizer(S,return_tensors="pt")
        tlink = self.tlinks[self.index_tlink-1]
        if len(tlink["relatedToEventInstance"])>6:
            tlink["relatedToEventInstance"] = "ei"+str(int(tlink["relatedToEventInstance"][6:]))
        try:
            s4_pos = self.event_eid[tlink["relatedToEventInstance"]] * 2 + 1
        except:
            self.readnew = True
            return "DataERR"
        s2_pos = self.event_eid[tlink["eventInstanceID"]]*2+1
        s2 = " " + self.data_text[s2_pos]
        s4 = " " + self.data_text[s4_pos]
        s1 = "".join(self.data_text[:s2_pos])
        s3 = "".join(self.data_text[s2_pos+1:s4_pos])
        s5 = "".join(self.data_text[s4_pos+1:])
        s1 = self.tokenizer(s1,return_tensors="pt")["input_ids"][0][:-1]
        s2 = self.tokenizer(s2,return_tensors="pt")["input_ids"][0][1:-1]
        s3 = self.tokenizer(s3,return_tensors="pt")["input_ids"][0][1:-1]
        s4 = self.tokenizer(s4,return_tensors="pt")["input_ids"][0][1:-1]
        s2_start = len(s1)-1
        s2_end = s2_start+len(s2)
        s4_start = s2_end+len(s3)-1
        s4_end = s4_start+len(s4)
        if s2_end > s4_start:
            return "DataERR"
        return S,s2_start,s2_end,s4_start,s4_end,tlink["relType"]

    def read(self):
        root = et.parse(self.file_list[self.index_data])
        root = root.getroot()
        text = root.find("TEXT")
        # dataText.replace("\n\n","")
        event_eid = {}
        pos = 0
        # 每个事件中的eid对应的位置索引
        for event in text:
            if event.tag == "EVENT":
                event_eid["ei"+event.attrib["eid"][1:]] = pos
            if event.tag == "TIMEX3":
                event_eid["ti"+event.attrib["tid"][1:]] = pos
            pos += 1
        # 对每个事件进行特殊字符的标注，以便split
        for event in text:
            event.text = "@#@" + event.text + "@#@"
        data_text = "".join([i for i in text.itertext()])
        data_text = data_text.split("@#@")
        # 获取TLINK标签下事件与事件的时序关系
        tlink_tags = root.findall("TLINK")
        tlinks = []
        for i in tlink_tags:
            if "relatedToEventInstance" in i.attrib:
                tlinks.append(i.attrib)
        self.tlinks = tlinks
        self.data_text = data_text
        self.event_eid = event_eid
        self.len = len(tlinks)
    def __len__(self):
        return len(self.file_list)

class RobertaTemporalModel(nn.Module):
    def __init__(self):
        bert = RobertaModel.from_pretrained('model')
        super(RobertaTemporalModel, self).__init__()
        for i in bert.parameters():
            i.requires_grad = False
        self.roberta = bert
        self.fc = nn.Linear(768*5,3)
        self.relu = nn.ReLU()
        self.tokenizer = RobertaTokenizer.from_pretrained('model')
        self.bn = nn.BatchNorm1d(1)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.wordattention_s1 = nn.Sequential(
            nn.Linear(768, 768),
            Transpose(1,2),
            nn.Conv1d(768, 1, kernel_size=1)
        )
        self.wordattention_s3 = nn.Sequential(
            nn.Linear(768, 768),
            Transpose(1, 2),
            nn.Conv1d(768, 1, kernel_size=1)
        )
        self.wordattention_s5 = nn.Sequential(
            nn.Linear(768, 768),
            Transpose(1, 2),
            nn.Conv1d(768, 1, kernel_size=1)
        )
        # self.test=nn.Linear(768,768)
        # self.conv1 = nn.Conv1d(768, 1, kernel_size=1,padding=0)
    def forward(self,S,s2_start,s2_end,s4_start,s4_end):
        S["input_ids"] = S["input_ids"].to(CUDAorCPU)
        S["attention_mask"] = S["attention_mask"].to(CUDAorCPU)

        embed = self.roberta(**S)[0]
        # embed = embed.permute(1,0,2)
        embed = self.relu(embed)
        s1 = embed[:,:s2_start,:]
        s2 = embed[:,s2_start:s2_end,:]
        s3 = embed[:,s2_end:s4_start,:]
        s4 = embed[:,s4_start:s4_end,:]
        s5 = embed[:,s4_end:,:]
        if s2_start<2 or s4_end>len(S["input_ids"][0])-2:
            return None
        #word attention
        try:
            s1 = torch.mm(self.wordattention_s1(s1)[0],s1[0]).unsqueeze(0)
            s3 = torch.mm(self.wordattention_s3(s3)[0],s3[0]).unsqueeze(0)
            s5 = torch.mm(self.wordattention_s5(s5)[0],s5[0]).unsqueeze(0)
            s2 = torch.mean(s2,dim=1).unsqueeze(0)
            s4 = torch.mean(s4,dim=1).unsqueeze(0)
        except:
            pass

        sentence = torch.cat([s1,s2,s3,s4,s5],dim=2)
        sentence = self.bn(sentence)
        out = self.fc(sentence[0])
        if torch.any(torch.isnan(out)):
            return None
        # out = self.sigmoid(out)
        out = self.softmax(out)
        return out


class Extraction(nn.Module):
    def __init__(self):
        bert = RobertaModel.from_pretrained('model')
        super(Extraction, self).__init__()
        for i in bert.parameters():
            i.requires_grad = False

        self.roberta = bert
        self.fc = nn.Linear(768,1)
        self.relu = nn.ReLU()
    def forward(self,sentences):
        sentences["input_ids"] = sentences["input_ids"].cuda(0)
        sentences["attention_mask"] = sentences["attention_mask"].cuda(0)
        embed = self.roberta(**sentences)[0]
        # embed = embed.permute(1,0,2)
        out = self.fc(embed)
        out = self.relu(out)

        return out

class Mydataset(Dataset):
    def __init__(self, path):
        data = []
        for i in path:
            data.append(pd.DataFrame(pd.read_csv(i)))
        self.df = pd.concat(data,axis=0,join="inner").reset_index(drop=True)
        self.tokenizer = RobertaTokenizer.from_pretrained('model')

    def __getitem__(self, index):
        # before = self.tokenizer(self.df["before"][index])["input_ids"][:-1]
        # after  = self.tokenizer(self.df["after"][index])["input_ids"][1:]
        #一个句子可能有多个verb
        # print("111")
        index += 3
        verbs   = [
            self.tokenizer(" " + self.df["verb"][index - 1])["input_ids"][1:-1],
            self.tokenizer(" " + self.df["verb"][index - 2])["input_ids"][1:-1],
            self.tokenizer(" " + self.df["verb"][index - 3])["input_ids"][1:-1],
            self.tokenizer(" "+self.df["verb"][index])["input_ids"][1:-1],
            self.tokenizer(" "+self.df["verb"][index+1])["input_ids"][1:-1],
            self.tokenizer(" "+self.df["verb"][index+2])["input_ids"][1:-1],
            self.tokenizer(" " + self.df["verb"][index + 3])["input_ids"][1:-1],
            self.tokenizer(" " + self.df["verb"][index + 4])["input_ids"][1:-1],
        ]
        try:
            sentence = self.df["before"][index]+" "+self.df["verb"][index]+" "+self.df["after"][index]
        except:
            sentence = " " + self.df["verb"][index] + " " + self.df["after"][index]
        sentence = self.tokenizer(sentence,return_tensors='pt')
        # 确定verb的位置
        label = [0 for i in range(sentence["input_ids"].shape[1])]
        sentence_list = sentence["input_ids"][0].tolist()
        for verb in verbs:
            if len(verb)>=3:
                return None,None
            if verb[0] in sentence_list:
                start = sentence_list.index(verb[0])
                for i in range(start,start+len(verb)):
                    label[i] = 1
        return sentence,label
    def __len__(self):
        return len(self.df)-7

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)