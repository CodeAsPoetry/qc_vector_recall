# coding="utf-8"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from uniem.finetuner import FineTuner
from datasets import load_dataset

# model_dir = "/home/pangchengjie.pcj/models/m3e-base/AI-ModelScope/m3e-base"
# train_file_path = "/home/pangchengjie.pcj/ths/data_v2/finetune_m3e_5371_pair_train.json"
# valid_file_path = "/home/pangchengjie.pcj/ths/data_v2/finetune_m3e_100_pair_valid_new_1.json"

# model_dir = "/home/pangchengjie.pcj/ths/data_v1/finetuned-model-pair-1-3epoch/model"
# train_file_path = "/home/pangchengjie.pcj/ths/data_v1/finetune_m3e_7549_scored_pair_train.json"
# valid_file_path = "/home/pangchengjie.pcj/ths/data_v1/finetune_m3e_100_scored_pair_valid_new.json"

model_dir = "/home/pangchengjie.pcj/ths/data_v2/finetuned-model-pair-1-3epoch/model"
train_file_path = "/home/pangchengjie.pcj/ths/data_v2/finetune_m3e_13215_triplet_train.json"
valid_file_path = "/home/pangchengjie.pcj/ths/data_v2/finetune_m3e_100_triplet_valid_new_1.json"



dataset = load_dataset('json', data_files={'train': train_file_path, 'validation': valid_file_path}, cache_dir='/online/pangchengjie.pcj/datasets/ths')
finetuner = FineTuner.from_pretrained(model_dir, dataset=dataset)
fintuned_model = finetuner.run(epochs=3, lr=1e-5, output_dir='data_v2/finetuned-model-triplet-2-3epoch')
