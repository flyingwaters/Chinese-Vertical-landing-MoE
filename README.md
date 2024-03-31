<div align="center">
    <h1>
        Chinese-Vertical-landing-MoE
    </h1>
</div>

<div align=center>
    <img width="400" height="300" src="./pictures/logo.png"/>
</div>

## 🚀 介绍

本项目基于[Chinese-mixtral-8*7b](https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B)而来, 针对垂直行业数据做了tokens 增加算法和合并tokenizer的功能，支持在垂直行业的文本语料上做词表扩充后的增量预训练，以及指令微调（instruction tuning）的训练。Chinese-mixtral-8*7b 是哈工大检索信息中心基于Mistral发布的模型[Mixtral-8x7B](https://mistral.ai/news/mixtral-of-experts/)进行了中文扩词表增量预训练，希望进一步促进中文自然语言处理社区对MoE模型的研究。我们扩充后的词表显著提高了模型对中文的编解码效率，并通过大规模开源语料对扩词表模型进行增量预训练，使模型具备了强大的中文生成和理解能力。

目前开源内容：

- Chinese-Mixtral-8*7B的代码
- 合并词表和alp词表合并算法
- 指令微调的代码
- 词表合并和去重的算法

未来开源内容
- 基于groq 框架的一键高效部署
- 垂直领域的数据的爬取，过滤和清洗框架
- 集成量化压缩框架


> 请注意，Chinese-Mixtral-8x7B仍然可能生成包含事实性错误的误导性回复或包含偏见/歧视的有害内容，请谨慎鉴别和使用生成的内容，请勿将生成的有害内容传播至互联网。

## 📥 模型下载

Chinese-Mixtral-8*7b 项目使用QLoRA进行训练，LoRA权重与合并权重后的模型分别开源，您可以根据自己的需求选择下载：

|             模型名称             | 模型大小  |                                     下载地址                                      |                                                         备注                                                          |
|:----------------------------:|:-----:|:-----------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|
|     Chinese-Mixtral-8x7B     | 88GB  |     [HuggingFace](https://huggingface.co/HIT-SCIR/Chinese-Mixtral-8x7B)<br>[ModelScope](https://modelscope.cn/models/HIT-SCIR/Chinese-Mixtral-8x7B/summary)     |                                                  中文扩词表完整模型，可以直接使用                                                   |
| Chinese-Mixtral-8x7B-adapter | 2.7GB | [HuggingFace](https://huggingface.co/HIT-SCIR/Chinese-Mixtral-8x7B-adapter) | LoRA权重，需要与原版Mixtral-8x7B进行合并才可以使用，合并脚本请参考[这里](https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930) |

## 💻 模型推理

Chinese-Mixtral-8x7B支持完整的Mixtral-8x7B模型生态，包括使用`vLLM`、`Flash Attention 2`进行加速，使用`bitsandbytes`进行模型量化等。以下是使用Chinese-Mixtral-8x7B进行推理的代码示例。

如inference.py 中所示，我们的推理方法

使用Flash Attention 2：
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
# Load peft config for pre-trained checkpoint etc.
peft_model_id = "/xxx/models/HIT-SCIR/Chinese-Mixtral-8x7B"
config = PeftConfig.from_pretrained(peft_model_id)

model_path = "/xxx/models/HIT-SCIR/Chinese-Mixtral-8x7B"

# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")
model.eval()
```

使用4bit量化：
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HIT-SCIR/Chinese-Mixtral-8x7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

text = "我的名字是"
inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

请注意，Chinese-Mixtral-8x7B为基座模型，没有经过指令微调，因此指令遵循能力有限。您可以参考[微调](#微调)一节对模型进行微调。

## 📈 模型性能

### 模型综合能力

