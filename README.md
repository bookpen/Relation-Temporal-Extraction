q0:关于eiid？

A：在 TempEval-3 任务中，“eiid”是指事件信息标识符（Event Instance ID），用于唯一标识数据集中的每个事件实例。每个事件实例都具有一个唯一的eiid。

在 TempEval-3 数据集中，eiid的格式如下：`<fileid>_ei<number>`，其中 “fileid” 表示所属文档（文件）的标识符，而 “number” 则表示该事件实例在文档内的序号。

例如，在 TempEval-3 中，如果我们看到一个eiid为“APW_ENG_20030305.0061:e2_38”的事件实例，它意味着该事件实例属于“APW_ENG_20030305.0061”这个文档，并且是该文档中的第38个事件实例。

eiid的唯一性使得可以在分析过程中跟踪和识别不同的事件实例，帮助NLP研究人员进行准确的语义角色标注、时间信息提取等任务。



q1:为什么eventid会有缺少的部分

A：因为原来的文本中（参考TempEval-3中的）是存在名词性时间的，但是在Matres中删去了，所以有空隙。如图中标签的pos属性	

![image-20230603173600532](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20230603173600532.png)



q2：为什么eventid和eiid对不上号

A：因为在文中的标签并不是按数字顺序排下来的，eventid可能存在“插队”现象，导致被插队的eventid会比eiid大。

His presence on American soil is necessary to <EVENT eid="e86" class="OCCURRENCE">dispel</EVENT> doubts about his fitness as a father _ <EVENT eid="e5" class="OCCURRENCE">raised</EVENT> maliciously by Elian's Miami

但是在label文件中，却又是按照原来的eiid标注的，所以这两个id就没有对上号



q3：MATRE的数据集难以使用，（训练baseline RoBERTa temporal model用）

A：更换TempEval-3 



q4：输入句子太长，Roberta会出错

A：阈值挺高的，似乎没有影响

A：在Roberta tempral model中由于文本过长会出现问题，大约在字符数超过2200左右就会报错，



q5:池化部分是如何进行的

有两个注意力



q6:评分系统

对于个预测结果，都是有网络输出经过softmax得出的，其值都在0~1之间，选择出最大值作为待选标签，如果其大于某个置信度，才可以被入选。例如程序中的置信度可以为：0,1,2分别为三个输出结果，before，after，simultaneous，其值代表置信度

```
confidence = {
    0:0.9,
    1:0.5,
    2:0.8,
}
```

### 示例网站来自

[StructTempRel-EMNLP17/NYT20000406.0002.tml at master · qiangning/StructTempRel-EMNLP17 (github.com)](https://github.com/qiangning/StructTempRel-EMNLP17/blob/master/data/TempEval3/Training/TBAQ-cleaned/AQUAINT/NYT20000406.0002.tml#L23)
