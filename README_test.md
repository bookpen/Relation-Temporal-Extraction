# Relation Temporal Extraction

## Introducation
在Amazon AI《Severing the Edge Between Before and After Neural》这篇论文中，会使用一个自训练的数据集。该部分分为三步
1. 准备一个原始文本的数据集，该数据集选自CNN
2. 训练一个事件标记器tagger，对于一个原始文本，tagger可以标注出文本中的事件
3. 训练一个事件对分类器classifier，将标注好事件的文本中的事件两两组对，classifier可以判断每个事件对的时序关系，比如before，after，simultaneous。

## 

