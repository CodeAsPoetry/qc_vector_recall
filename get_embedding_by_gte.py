import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import json
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
model_dir = "/online/pangchengjie.pcj/models/gte_Qwen2-7B-instruct/iic/gte_Qwen2-7B-instruct"

model = SentenceTransformer(model_dir)
# In case you want to reduce the maximum length:
model.max_seq_length = 8192

queries = []
with open("queies_list.json", "r", encoding="utf-8") as f_1:
    lines = f_1.readlines()
    for line in lines:
        line = line.strip()
        query = json.loads(line)["query"]
        queries.append(query)

documents = []
with open("docs_list.json", "r", encoding="utf-8") as f_2:
    lines = f_2.readlines()
    for line in lines:
        line = line.strip()
        doc = json.loads(line)["doc"]
        documents.append(doc)

print(len(queries), len(documents))
'''
batch_size = 32
q_batchs_num = len(queries)//batch_size + 1

for i in range(q_batchs_num):
    if i == q_batchs_num-1:
        batch_queries = queries[batch_size*i:]
    else:
        batch_queries = queries[batch_size*i:batch_size*(i+1)]
    query_embeddings = model.encode(batch_queries, prompt_name="query")
    print("query_embeddings", type(query_embeddings), query_embeddings.shape)
    np.save("query_embeddings/query_embeddings_{}.npy".format(str(i)), query_embeddings)
'''
batch_size = 8
d_batchs_num = len(documents)//batch_size + 1
for i in range(d_batchs_num):
    if i == d_batchs_num-1:
        batch_docs = documents[batch_size*i:]
    else:
        batch_docs = documents[batch_size*i:batch_size*(i+1)]
    document_embeddings = model.encode(batch_docs)
    print("document_embeddings", type(document_embeddings), document_embeddings.shape)
    np.save("document_embeddings/document_embeddings_{}.npy".format(str(i)), document_embeddings)

print("finish!!!")
# [[70.00668334960938, 8.184843063354492], [14.62419319152832, 77.71407318115234]]
