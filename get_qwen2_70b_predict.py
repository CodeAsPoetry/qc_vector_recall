import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
print(torch.cuda.device_count())
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/Users/codeaspoetry/models/qwen-72b-chat", revision='master', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/Users/codeaspoetry/models/qwen-72b-chat", device_map="auto", trust_remote_code=True, fp16=True).eval()


import json
from tqdm import tqdm

with open("held_out_eval.jsonl", "r", encoding="utf-8") as f, open("held_out_eval_output.jsonl", "w", encoding="utf-8") as fw:
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        sample = json.loads(line)
        docs = sample['docs']
        new_docs = []
        for doc in docs:   
            query = doc["prompt_1"]
            response, history = model.chat(tokenizer, query, history=None)
            doc["prompt_1_output"] = response
            new_docs.append(doc)
        
        sample['new_docs'] = new_docs
        
        if "prompt_2" in sample.keys():
            query = sample["prompt_2"]
            response, history = model.chat(tokenizer, query, history=None)
            sample['prompt_2_output'] = response
            
        
        fw.write(json.dumps(sample, ensure_ascii=False))
        fw.write("\n")
