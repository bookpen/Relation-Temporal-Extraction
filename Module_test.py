from transformers import RobertaTokenizer
import xml.etree.ElementTree as et

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
            self.readnew = False
            self.index_data+=1
            self.index_tlink=0
        IS_OVER = self.read()
        if IS_OVER=="OVER":
            return "OVER"
        if len(self.tlinks)==0:
            self.readnew = True
            return "DataERR"
        if self.index_tlink==self.len-1:
            self.readnew=True
        self.index_tlink += 1
        text = self.text.__copy__()
        tlink = self.tlinks[self.index_tlink-1]
        if len(tlink["relatedToEventInstance"]) > 6:
            tlink["relatedToEventInstance"] = "ei" + str(int(tlink["relatedToEventInstance"][6:]))

        src_e = tlink["eventInstanceID"]
        tgt_e = tlink["relatedToEventInstance"]
        if int(src_e[2:])>int(tgt_e[2:]):
            src_e,tgt_e = tgt_e,src_e
        if int(src_e[2:])==int(tgt_e[2:]):
            return "DataERR"
        for event in text:
            if "eid" not in event.attrib:
                continue
            if event.attrib["eid"] == src_e.replace("i",""):
                event.text = "@#@"+event.text+"@#@"
            if event.attrib["eid"] == tgt_e.replace("i", ""):
                event.text = "#@#" + event.text + "#@#"
        text = "".join([i for i in text.itertext()])
        first_split = "@#@"
        second_split = "#@#"
        if "@#@" not in text or "#@#" not in text:
            return "DataERR"
        if text.index("@#@") > text.index("#@#"):
            first_split="#@#"
            second_split="@#@"
        text_split = text.split(first_split)
        s2 = " "+text_split[1]
        s1 = text_split[0].split("\n\n")[-1][:-1]

        text_split = text_split[2].split(second_split)
        s4 = " " + text_split[1]
        s3 = text_split[0]
        s5 = text_split[2].split("\n\n")[0]
        if len(s1) * len(s2) * len(s3) * len(s4)*len(s5)==0:
            return "DataERR"

        S = s1+s2+s3+s4+s5
        # print(text)
        # 进行标注
        if len(S)>2200:
            return "DataERR"
        # S = self.tokenizer(S,return_tensors="pt")["input_ids"][0]
        S = self.tokenizer(S,return_tensors="pt")

        s1 = self.tokenizer(s1,return_tensors="pt")["input_ids"][0][:-1]
        s2 = self.tokenizer(s2,return_tensors="pt")["input_ids"][0][1:-1]
        s3 = self.tokenizer(s3,return_tensors="pt")["input_ids"][0][1:-1]
        s4 = self.tokenizer(s4,return_tensors="pt")["input_ids"][0][1:-1]

        s2_start = len(s1)
        s2_end = s2_start+len(s2)
        s4_start = s2_end+len(s3)
        s4_end = s4_start+len(s4)
        if s2_end >= s4_start:
            return "DataERR"
        return S,s2_start,s2_end,s4_start,s4_end,tlink["relType"]

    def read(self):
        try:
            root = et.parse(self.file_list[self.index_data])
        except:
            return "OVER"
        root = root.getroot()
        text = root.find("TEXT")
        # 每个事件中的eid对应的位置索引
        # 对每个事件进行特殊字符的标注，以便split

        # 获取TLINK标签下事件与事件的时序关系
        tlink_tags = root.findall("TLINK")
        tlinks = []
        for i in tlink_tags:
            if "relatedToEventInstance" in i.attrib and "eventInstanceID" in i.attrib:
                tlinks.append(i.attrib)
        self.tlinks = tlinks
        self.text = text
        self.len = len(tlinks)
    def __len__(self):
        return len(self.file_list)