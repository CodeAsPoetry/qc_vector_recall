# Query-Docs向量召回优化
## 目标

1. 在保证泛化性的前提下尽量提升向量召回3分结果recall@20（候选为整个开发集）的效果

2. 需要解决向量量化（float32转int8）的效果，量化方案可自选

3. 需要解决query和doc表征空间统一的问题，提升其可聚类性（如下图，从表征看c好于b好于a）

   ![可聚性说明](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/images/pic_1.png)

4. 向量维度<=1024（越小越好）
5. 模型不限制，可以做数据增强

## 数据

附件中提供3000条问句、问句对应doc、标注，按需拆解为训练集和测试集

## 要求

1. 提供必要的方案说明及训练&测试代码
2. 提供开发集测试效果

## 调研

1. [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/pdf/2210.07316)

2. [Piccolo2: General Text Embedding with Multi-task Hybrid Loss Training](https://arxiv.org/html/2405.06932v1)

3. [Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/pdf/2308.03281)

4. [Matryoshka Representation Learning](https://arxiv.org/pdf/2205.13147)

5. https://huggingface.co/sensenova/piccolo-large-zh-v2

6. https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct

7. [InfoNCE loss](https://zhuanlan.zhihu.com/p/506544456)

8. [CoSENT](https://spaces.ac.cn/archives/9341)

9. https://www.github.com/wangyuxinwhy/uniem

10. https://huggingface.co/moka-ai/m3e-large

11. https://huggingface.co/spaces/mteb/leaderboard

    ![Chinese mteb leaderboard](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/images/pic_2.png)

## 数据分析

1. 原始数据：**3000** 个 query，一共有 **23530** 个 doc

2. 在 query 粒度上去重，将一样 query 的多条数据下对应所有 doc 合并，对每个 query 下的所有 doc 进行去重，一共有 **2991** 个query，**23410** 个 doc

3. 将所有的 23410 个 doc 统一去重，最终得到 **19582** 个 doc

4. query 对应的候选 doc 数目**最少是6，最多是16，平均 7.8**

5. 19582 个候选文档，长度分布：**最短24，最长4293，平均454.3，大量集中在 256～512 区间内，有 16532 个**

   ![Doc Length Distribution](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/images/pic_3.png)

6. 数据分析脚本(包含去重、统计、绘图等)，代码链接：[工作流脚本](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/query_content_vector_recall.ipynb) 代码中的“数据分析”部分

## 切分数据集

1. held out 测试集，随机从 2991 条 query 中切分出 100 条 query ，将100条 query 对应的 doc 全部拿来，但如果 doc 同时还是作为训练集其他 query 的候选 doc ，则在训练集中该doc予以保留；
2. held in 测试集，随机从 2891 (2991-100) 条 query 中选 100 条 query ，各取一条1分doc和3分doc；
3. 除上之外，剩余的是训练集，2891 条 query，其中有 100 条query是held in，各自对应的doc，如果held in取出，则进行剔除。
4. 数据切分脚本(包含切分数据集、验证切分是否正确、落盘文件化等)，代码链接：[工作流脚本](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/query_content_vector_recall.ipynb) 代码中的“切分数据集”部分
5. 数据集：
   1. 训练集：[train_dev_data.jsonl](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/train_dev_data.jsonl)
   2. held in 测试集：[held_in_eval.jsonl](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/held_in_eval.jsonl)
   3. held out 测试集：[held_out_eval.jsonl](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/held_out_eval.jsonl)

## 现成模型能力边界探查

###gte-Qwen2-7B-instruct

1. 中文 C-MTEB 榜单第二名
2. 按照 huggerface 给出的推理脚本，将 **2991** 个query，**23410** 个 doc 送进去进行向量化，
   1. 所有 query 的 json文件：[queies_list.json](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/queies_list.json)
   2. 所有 doc 的 json文件：[docs_list.json](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/docs_list.json)
3. [模型推理脚本](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/get_embedding_by_gte.py)
   1. 切分 batch 进行推理，按照batch保存npy文件
   2. 整体合并成大的 npy 文件
4. 最终得到的嵌入向量压缩包
   1. 所有 query 向量嵌入表示矩阵(行索引和queies_list.json的文件行号一一对应)，矩阵shape为(2981，3584)：[querise向量npy文件](https://github.com/CodeAsPoetry/qc_vector_recall/tree/main/query_embeddings)
   2. 所有 doc 向量嵌入表示矩阵(行索引和docs_list.json的文件行号一一对应)，矩阵shape为(19582，3584)：[docs向量npy文件](https://github.com/CodeAsPoetry/qc_vector_recall/tree/main/document_embeddings)
5. 算分，计算recall@20的指标
   1. held out 测试集，100条query，逐 query 计算 recall@20，整体在 query 粒度上算平均，**0.1667**
   2. held out 测试集，根据数据集中规定的某条 query 对应的多个候选doc的范围，获取相应匹配分，把其中所有1分的算匹配平均值，所有3分的算匹配平均值，3分集合匹配平均分大于1分集合匹配平均分，这种情形占比 **94%**
   3. held in 测试集，根据数据集中规定的某条 query 各自对应1个3分，和1个1分，3分比1分的匹配平均分大的情形，占比**86%**
6. 探查结论：
   1. gte-Qwen2-7B-instruct 是可以针对某个query，找出更匹配的文档，**但对于该数据集，召回某条query的相关doc，容易受到其他query的候选doc影响**，而且，目标是recall@20(候选为整个开发集)
7. BadCase 分析
   1. query：
   2. 错召回 doc：

### Qwen-72b-Chat

1. 借助 Prompt 工程，对数据进行改造，适配 Qwen-72b-Chat 的任务范式，代码见：[工作流脚本](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/query_content_vector_recall.ipynb)中的“prompt改造”

   1. 模版一

      ```tex
      template_prompt_1 = '''下面有一个来自用户的查询query和一个候选文档doc：
      ===
      用户查询query：{}
      候选文档doc：{}
      ===
      请判断该文档doc能不能作为此查询query的相关内容返回给用户。建议从文本相关度进行判断，请直接给出答案，回答“能”还是“不能”，不要输出其他内容。'''
      ```

   2. 模版二

      ```tex
      template_prompt_2 = '''下面有一个来自用户的查询query和两个候选文档doc1和doc2：
      ===
      用户查询query：{}
      候选文档一doc1：{}
      候选文档二doc2：{}
      ===
      请判断哪个候选文档更与此查询query相关，建议从文本相关度进行判断，请直接给出更相关的答案，回答“doc1”还是“doc2，不要输出其他内容”。'''
      ```

2. 对 held out 进行模版一改造，对 held out 进行模版一、模版二改造

   1. 对应测试集文件中的 prompt_1、prompt_2 字段值
   2. 按照huggerface上 Qwen-72b-Chat 的推理服务，获取推理结果
      1. [推理脚本](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/get_qwen2_70b_predict.py)
      2. held in 测试集，Qwen-72b-Chat 推理结果，[held_in_eval_output.jsonl](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/held_in_eval_output.jsonl)
      3. held out 测试集，Qwen-72b-Chat 推理结果，[held_out_eval_output.jsonl](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/held_out_eval_output.jsonl)
   3. 实验结果
      1. held in 上，逐条计算每个 query 在对应给出的 doc 候选范围中，三分doc的查准和查全，再在 query 粒度上平均。最终，3分 doc 的**查准0.995** ，**查全0.79**；**pair_wise 比较的准确率 100%**
      2. held in 上，逐条计算每个 query 在对应给出的 doc 候选范围中，三分doc的查准和查全，再在 query 粒度上平均。最终，3分 doc 的**查准0.9902** ，**查全0.7433**；

3. 探查结论：

   1. Qwen-72b-Chat 是可以针对某个query，非常精准地找出更匹配的文档，但对于该数据集，在召回某条query的相关doc，即使限定到数据集中该条 query 对应的平均7.8条小范围候选 docs 集合中，召回也不算很高，平均 0.75 左右

## 建模优化方案

### 训练代码参考

https://www.github.com/wangyuxinwhy/uniem


### 尽快拿结果(优先级提高)

**采用 M3E 系列模型**

1. M3E-base：https://huggingface.co/moka-ai/m3e-base
2. 按照 uniem 支持三种任务，构建数据：
   1. Pair 句对样本(text, text_pos)
   2. ScoredPair 带有分数的句对样本(text, text_pos, label=1.0), (text, text_neg, label=0.0)
   3. Triplet 句子三元组，(text, text_pos, text_neg)

3. 获取 M3E-base 模型推理向量，统计recall@20
   1. held in 测试集上，recall@20 为 **0.625**；held out 测试集上，recall@20 为 **0.6**

4. 进行 baseline 实验
   1. 训练集所有样本构造 Pair(text, text_pos) 句对样本，获取每个query和对应的label为3分的样本对
   2. 训练集所有样本构造 ScoredPair(text, text_pos, label=1.0), (text, text_neg, label=0.0) 带有分数的句对样本，获取每个query和对应的label为1分和3分的样本对
   3. 训练集所有样本构造 Triplet(text, text_pos, text_neg) 句子三元组样本，针对每个query的每条1分doc和3分doc，加上query，两两组对
   4. M3E-base 上，进行上面三个数据集的 finetune，每个数据集过1个epoch，三个epoch的学习率默认 5e-5
      1. 第1个epoch，过了 Pair 句对，held in 测试集上，recall@20 为 **0.94**；held out 测试集上，recall@20 为 **0.853**
      2. 第2个epoch，过了 ScoredPair 句对，held in 测试集上，recall@20 为 **0.92**；held out 测试集上，recall@20 为 **0.83**
      3. 第3个epoch，过了 Triplet 三元组，held in 测试集上，recall@20 为 **0.97**；held out 测试集上，recall@20 为 **0.788**
   5. 结论：
      1. held in 指标逐渐上升，held out 指标却大幅下滑，上述策略存在过拟合，每个 query 和 doc 都见了太多次
      2. ScoredPair 任务貌似并不好
         1. 从训练loss上看，ScoredPair的loss大过 Pair 、Triplet 任务loss约 1个数量级，在训练集上，达到4～5之间，在验证集上，达到5～6之间；
         2. 从数据分析上，1分doc和3分doc，可能更针对具体query，也就是说，来自不同的query对应的都是 1 分的多个doc，它们并不一定真正属于一类，可能并不具有统一的分值意义，对于模型并不友好
         3. 三个任务，学习率都是 5e-5 不太合理，这三个任务彼此之间有关联的，学好Pair任务，显然可以辅助Triplet任务

5. 进行模型优化
   1. 随机挑选训练集中的 2/3 样本，构造 Pair(text, text_pos) 句对样本，获取每个query和对应的label为3分的样本对
      1. 训练集：[finetune_m3e_5371_pair_train.json](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/data_v2/finetune_m3e_5371_pair_train.json)
      2. 验证集：[finetune_m3e_100_pair_valid_new_1.json](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/data_v2/finetune_m3e_100_pair_valid_new_1.json)
   2. 训练集剩下的 1/3 样本，构造Triplet(text, text_pos, text_neg) 句子三元组样本
      1. 训练集：[finetune_m3e_13215_triplet_train.json](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/data_v2/finetune_m3e_13215_triplet_train.json)
      2. 验证集：[finetune_m3e_100_triplet_valid_new_1.json](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/data_v2/finetune_m3e_100_triplet_valid_new_1.json)
   3. M3E-base 上，进行上面两个数据集的 finetune，考虑已采取控制每个query和doc出现的频次，可以多过epoch，每个数据集过3个epoch
      1. [finetune 脚本](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/finetune_m3e.py)
      1. Pair 任务，学习率 5e-5，3个epoch，[训练log](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/finetune_m3e_base_pair_3epoch_data_v2.log)
      2. Triplet 任务，学习率 1e-5，3个epoch，[训练log](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/finetune_m3e_base_triplet_3epoch_data_v2.log)
      3. 最终获取模型的嵌入向量
         1. [queries向量](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/query_embeddings_m3e_triplet_3_data_v2.tar.gz)，向量维度 768
         2. [docs向量](https://github.com/CodeAsPoetry/qc_vector_recall/blob/main/document_embeddings_m3e_triplet_3_data_v2.tar.gz)，向量维度 768
      4. 指标，模型在 held in 测试集上，recall@20 为 **0.99**；held out 测试集上，recall@20 为 **0.862**

6. 回顾目标
   1. held out 测试集中，query 是独立于整个训练验证的，在整个开发集上 recall@20 指标达到 0.862，说明模型是在一定的泛化性的前提下尽量提升向量召回3分结果
   2. 需要解决向量量化（float32转int8）的效果，量化方案可自选
   3. 任务二在一定程度上从侧面反映query和doc表征空间统一的问题，在进行模型探索上，有一定建设
   4. 目前向量维度 768，小于1024（越小越好）
   5. 出于时间受限，模型暂未进行数据增强，若进行，指标将会进一步得到提高

### 进一步思路(优先级往后排一下)

**采用 gte-Qwen2-7B-instruct 基座，将切分出来的训练集，按照 Piccolo2 这篇论文的思路，利用多任务混合loss的思路，并采用“俄罗斯套娃”的学习方式，进行训练优化。**

1. gte-Qwen2-7B-instruct 基座：采用了文章提到的 improved contrastive loss ，经过预训练和微调两个阶段的对比学习得到的
2. Piccolo2 中，针对 Retrieval and Reranking 任务，利用 standard InfoNCE loss ；针对 STS and PairClassification 任务，利用 cosent loss ；针对 Classification and Clustering ，利用分类 $L_{cls}$ 损失，形式和 InfoNCE loss 一致，但没有 batch-in negatives，因为很容易在一个batch内有属于同一类的多条数据，造成错误。
3. Piccolo2 中利用了 Dimension Scaling up 和 MRL Training 机制，尽管在增大维度没有预期提升收益，以及不管是否采用“ MRL Training(俄罗斯套娃机制)”，性能也几乎没变。**但采用 MRL Training 机制，在维度上鲁棒，即与单维训练相比，它能够支持灵活的维度长度，而不会牺牲性能。这符合我们目标中的第 4 点**
4. 如何利用 2891 条 query ，万量级的doc，构造多任务？
   1. 我们**本身就是一个  Retrieval(检索) 任务**，利用Piccolo2中的 “standard InfoNCE loss ”或者gte-Qwen2-7B-instruct中的“ improved contrastive loss ”
   2. Qwen-72b-Chat 探查时，发现针对该数据集，进行pair_wise 比较的准确率 100%，**利用prompt工程对多个匹配pair对比较**，构造出 PairClassification 任务，比如“前提是(query1，doc1)和(query2，doc2)都是对应匹配的，输出(query1，doc1)匹配程度和(query2，doc2)的匹配程度那个更大”，这样就可以采用 cosent loss 
   3. 根据 query 和相关匹配的 doc ，类似目标第3点的图示，构建聚类任务，利用分类 $L_{cls}$ 损失

