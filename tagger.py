from Module import Extraction,Mydataset

import torch
from transformers import RobertaTokenizer
import pandas as pd
import os


vocab_file = 'model/vocab.json'
merges_file = 'model/merges.txt'
tokenizer = RobertaTokenizer(vocab_file, merges_file)

text = "you replace this to that"
encoded_input = tokenizer(text, return_tensors='pt')

model = Extraction()
# model.load_state_dict(torch.load("Extraction.pth"))
model.load_state_dict(torch.load("Extraction_all.pth"))
model = model.cuda()

def find_verb(text,confidence=0.5,val=None):
    if val=="test":
        model.eval()
    encoded_input = tokenizer(text, return_tensors='pt')
    # encoded_input["input_ids"] = encoded_input["input_ids"]
    # encoded_input["attention_mask"] = encoded_input["attention_mask"].cuda(0)
    out = model(encoded_input)
    out = out.view(-1)
    indices = torch.nonzero(out>confidence)
    a = encoded_input["input_ids"][0][indices]
    return tokenizer.decode(a),indices.cpu(),encoded_input["input_ids"][0].cpu()


# a="""
# The world is facing a great challenge due to the outbreak of COVID-19. This pandemic has affected everyone's daily life and routine. People are required to wear masks, maintain social distance, and follow strict hygiene practices. Many businesses have shut down, and millions have lost their jobs. The education sector has also been impacted as schools and universities have closed their doors. However, in these testing times, people have come together, and different communities have shown great resilience and solidarity. Frontline workers like doctors, nurses, and medical staff are working tirelessly to control the spread of the virus. It is essential that we continue to follow guidelines and work collectively to overcome this global crisis.
# """
# with open("interview.story")as f:
#     data = f.read()
# print(test(a,confidence=0.5))

def verb_position(verbs):
    position = []
    exsit = []
    for i in verbs:
        position.append(exsit.count(i))
        exsit.append(i)
    return position

def generate_data(file_paths:list):
    info_dict = {"bodytxt":[],"before": [], "after": [], "verb": [], "eiid": [], "docname": []}
    for i in file_paths:
        eiid = 1
        with open("{}".format(i),"r")as f:
            data = f.read()

        data = data.split("\n")
        data_=[]
        for j in data:
            if len(j)>5:
                data_.append(j)
        for sentence in data_:
            verbs = find_verb(sentence,0.2)
            verbs = verbs.split()
            position = verb_position(verbs)
            for pos,verb in zip(range(len(verbs)),verbs):
                before,after = "".join(sentence.split(verb)[:position[pos]+1]),\
                               "".join(sentence.split(verb)[position[pos]+1:])

                info_dict["bodytxt"].append(sentence)
                info_dict["before"].append(before)
                info_dict["after"].append(after)
                info_dict["verb"].append(verb)
                info_dict["eiid"].append(eiid)
                info_dict["docname"].append(i.split(".")[0])
                eiid += 1
    info_dict = pd.DataFrame(info_dict)
    info_dict.to_csv("data.csv",index=False)
if __name__ == "__main__":
    file_path = os.listdir("DataFromCNN_test")
    # file_path = ["interview.story","other.story"]
    generate_data(file_path)






