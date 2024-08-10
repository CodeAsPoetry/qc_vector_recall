import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
print(torch.cuda.device_count())
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
#import torch
# device = torch.device("cuda:2,5,7")



# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("/chatgpt_nas/33.59.0.139_backup/online/duyiyang.dyy/models/qwen-72b-chat", revision='master', trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-72B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
model = AutoModelForCausalLM.from_pretrained("/chatgpt_nas/33.59.0.139_backup/online/duyiyang.dyy/models/qwen-72b-chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-72B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
# model = AutoModelForCausalLM.from_pretrained("/online/duyiyang.dyy/models/qwen-72b-chat", revision='master', device_map="auto", trust_remote_code=True).eval()
# NOTE: The above line would require at least 144GB memory in total

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-72B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# 第一轮对话 1st dialogue turn
# response, history = model.chat(tokenizer, "你好", history=None)
# print(response)

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
