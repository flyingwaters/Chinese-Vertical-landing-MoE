import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

peft_exist = True
# Load peft config for pre-trained checkpoint etc.
peft_model_id = "/home/clouduser/fengly/Chinese-Mixtral-8x7B/lora_rank_8_out/checkpoint-600/adapter_model"

if peft_exist:
    config = PeftConfig.from_pretrained(peft_model_id)

model_path = "/home/clouduser/fengly/models/HIT-SCIR/Chinese-Mixtral-8x7B"
# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2",trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# Load the Lora model
if peft_exist:
    model = PeftModel.from_pretrained(model, peft_model_id, trust_remote_code=True, device_map="auto")
model.eval()

import json
test_path = "/home/clouduser/fengly/DeepSeek-MoE/test0.json"
with open(test_path, "r") as f:
    content = json.load(f)

input_data = [i["query"] for i in content]
print("example:", input_data[-1])
labels = [i["response"] for i in content]

batch_num = 32

def post_process(answer):
    answer = answer.replace("<|EOT|>", "")
    answer = answer.replace("<s>", "")
    answer = answer.replace("</s>", "")
    answer = answer.replace("\n", "")
    res = answer.split("答案是:") 
    return res[0], res[-1]

def acc_caculate(A,B):
    """ acc rate
    A is groundtruth B is prediction
    """
    wrong_case = []
    true_n = 0
    for data_id in range(len(A)):
        if A[data_id].replace(" ", "") in B[data_id][-1].replace(" ", ""):
            true_n+=1
        else:
            wrong_case.append((A[data_id], B[data_id]))
    return (true_n, wrong_case)

wrong_cases =[]
true_num = 0
for idx in range(0, len(input_data), batch_num):
    input_ids = tokenizer.batch_encode_plus(input_data[idx:idx+batch_num], return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.8, temperature=0.1)
    prediction_list = tokenizer.batch_decode(outputs)
    prediction_list = [post_process(i) for i in prediction_list]
    t_n, wrong_case = acc_caculate(labels[idx:idx+batch_num], prediction_list)
    true_num+=t_n
    wrong_cases.extend(wrong_case)

print(true_num/len(input_data) * 100)
print(wrong_cases)
print(len(wrong_cases))