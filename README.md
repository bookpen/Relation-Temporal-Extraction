# Relation Temporal Extraction

## Introducation

在Amazon AI《Severing the Edge Between Before and After Neural》这篇论文中，会使用一个自训练生成的银数据集silver data。生成silver data分为三步

1. 准备一个原始文本的数据集，该数据集选自CNN
2. 训练一个事件标记器tagger，对于一个原始文本，tagger可以标注出文本中的事件
3. 训练一个事件对分类器classifier，将标注好事件的文本中的事件两两组对，classifier可以判断每个事件对的时序关系，比如before，after，simultaneous。



## How to use

### Necessary Download
你需要下载一个Roberta预训练模型文件，MATRES数据集，以及Tempeval3的数据集
MATRES：https://github.com/qiangning/MATRES 新建并存放到TaggerTrainData文件夹下
Tempeval3：https://github.com/qiangning/StructTempRel-EMNLP17/tree/master/data/TempEval3/Training 新建并存放到ClassifierTrainData文件夹下
Roberta：https://huggingface.co/roberta-base/tree/main 新建并存放到model文件夹下

### Train a tagger

在**taggerTrain.py**中，使用到的训练集来自于MATRES，存放到了TaggerTrainData中，最后训练的模型保存在Extraction_all.pth。

### Train a classifier

在**classifierTrain.py**中，使用到的训练集来自于Tempeval3。该训练集可以分为两大部分，**TBAQ**和**TE3-Silver-data**，两个数据集由于各有其特点，为了可以训练出一个好的效果，决定将两个训练集均衡使用，也就是一个训练epoch的数据集一半来自于**TBAQ**，另一半来自于**TE3-Silver-data**。

训练后的模型保存保存在了classifier.pth

### Generate silver data

以上两步为训练过程，在项目中已经训练好了这两个模型了，可以直接运行classifier.py文件生成银数据集。

原始文本在**DataFromCNN/story**下，总计92579个文章。每篇文章都会生成一个CSV文件并存储到**GenerateData**目录下，命名互相对应上。一片文章的数据集格式如下。

```
S1	S2	S3	S4	S5	FILE	label	confidence

S2,S4：为事件
S1,S3,S5：为事件之间间隔的部分
FILE：文章名
label：事件之间的时序关系
confidence：置信度，来自模型的输出结果
```

为了更加能更直接使用这个数据集，所以就不使用原来MATRES数据集的格式了，也无需eiid。



## Other Files

### FinalData_Finshed.txt

**FinalData_Finshed.txt**文件存储一些文章的名字，在执行程序的时候会忽略这些文章。而在执行程序的的时候，每用完一篇文章，该文章的名字也会写入该文件中，以便下次执行程序时忽略该文章。同时该文件还会存储每个文章提取出来的标签总数与个类别的数目，其格式如下

```
FILE	SUMofLabel	Before	After	simultaneous
```

### argment.py

该文件存储了一些classifier的执行参数

cover_all：为True则不按照FinalData_Finshed.txt忽略文章，而是全部文章重新生成。

max_input_length：为事件对所处语境的句子的最大长度，该值越大则越有可能出现事件之间跨度很大的情况，该值不能超过2200。

confidence：置信度，与生成数据集标签有关，对应标签的置信度越高，则该标签出现频率就越低。详情可参考开发日志q7



## Further

目前该项目还有需要改进的地方

1. 可以尝试对Roberta的预训练模型进行一个简单训练。
2. 标记器的仅训练了标记动词事件，可以尝试训练标记名词事件。















