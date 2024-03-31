import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
# Load peft config for pre-trained checkpoint etc.
peft_model_id = "/home/clouduser/fengly/models/HIT-SCIR/Chinese-Mixtral-8x7B"
config = PeftConfig.from_pretrained(peft_model_id)

model_path = "/home/clouduser/fengly/models/HIT-SCIR/Chinese-Mixtral-8x7B"

# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")
model.eval()

## 设置batch
batch_num = 4

import json
test_path = "/home/clouduser/fengly/DeepSeek-MoE/test0.json"
with open(test_path, "r") as f:
    content = json.load(f)
    
input_data = [i["query"] for i in content]
labels = [i["response"] for i in content]

wrong_cases =[]
true_num = 0
for idx in range(0, len(input_data), batch_num):
    input_ids = tokenizer.batch_encode_plus(input_data[idx:idx+batch_num], return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=False, top_p=0.8)
    prediction_list = tokenizer.batch_decode(outputs)
    assert t_n+len(wrong_case)==len(input_data[idx:idx+batch_num])
    true_num+=t_n
    wrong_cases.extend(wrong_case)
print("right num:", true_num)
print("all num:", len(input_data))
print("rate:", true_num/len(input_data)*100)
print(wrong_cases)