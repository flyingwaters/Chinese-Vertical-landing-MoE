import os
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# parser = argparse.ArgumentParser() # 创建一个ArgumentParser对象
# parser.add_argument('--llama_tokenizer_dir', default=r'L:/20230902_Llama1/llama-7b-hf', type=str, required=True) # 添加参数
# parser.add_argument('--chinese_sp_model_file', default='./chinese_sp.model', type=str) # 添加参数
# args = parser.parse_args() # 解析参数
# llama_tokenizer_dir = args.llama_tokenizer_dir # 这里是LLaMA tokenizer的路径
# chinese_sp_model_file = args.chinese_sp_model_file # 这里是Chinese tokenizer的路径

llama_tokenizer_dir = r'L:/20230902_Llama1/llama-7b-hf'  # 这里是LLaMA tokenizer的路径
chinese_sp_model_file = r'./chinese_sp.model'  # 这里是Chinese tokenizer的路径

# 加载tokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)  # 加载LLaMA tokenizer
chinese_sp_model = spm.SentencePieceProcessor()  # 定义Chinese tokenizer
chinese_sp_model.Load(chinese_sp_model_file)  # 加载Chinese tokenizer

llama_spm = sp_pb2_model.ModelProto()  # 定义LLaMA tokenizer的sentencepiece model
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())  # 从LLaMA tokenizer中加载sentencepiece model
chinese_spm = sp_pb2_model.ModelProto()  # 定义Chinese tokenizer的sentencepiece model
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())  # 从Chinese tokenizer中加载sentencepiece model

# 输出tokens的信息
print(len(llama_tokenizer), len(chinese_sp_model))  # 两个tokenizer的词表大小；输出为32000、20000
print(llama_tokenizer.all_special_tokens)  # LLaMA tokenizer的special tokens；输出为['']
print(llama_tokenizer.all_special_ids)  # LLaMA tokenizer的special tokens对应的id；输出为[0]
print(llama_tokenizer.special_tokens_map)  # LLaMA tokenizer的special tokens；输出为{'bos_token': '', 'eos_token': '', 'unk_token': ''}


# 将Chinese tokenizer的词表添加到LLaMA tokenizer中（合并过程）
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)  # LLaMA tokenizer的词表
print(len(llama_spm_tokens_set))  # LLaMA tokenizer的词表大小；输出为32000
print(f"Before:{len(llama_spm_tokens_set)}")  # LLaMA tokenizer的词表大小；输出为Before:32000
for p in chinese_spm.pieces:  # 遍历Chinese tokenizer的词表
    piece = p.piece  # Chinese tokenizer的词
    if piece not in llama_spm_tokens_set:  # 如果Chinese tokenizer的词不在LLaMA tokenizer的词表中
        new_p = sp_pb2_model.ModelProto().SentencePiece()  # 创建一个新的sentencepiece
        new_p.piece = piece  # 设置sentencepiece的词
        new_p.score = 0  # 设置sentencepiece的score
        llama_spm.pieces.append(new_p)  # 将sentencepiece添加到LLaMA tokenizer的词表中
print(f"New model pieces: {len(llama_spm.pieces)}")  # LLaMA tokenizer的词表大小；输出为New model pieces: 49953


# 保存LLaMA tokenizer
output_sp_dir = 'merged_tokenizer_sp'  # 这里是保存LLaMA tokenizer的路径
output_hf_dir = 'merged_tokenizer_hf'  # 这里是保存Chinese-LLaMA tokenizer的路径
os.makedirs(output_sp_dir, exist_ok=True)  # 创建保存LLaMA tokenizer的文件夹
with open(output_sp_dir + '/chinese_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/chinese_llama.model')  # 创建LLaMA tokenizer
tokenizer.save_pretrained(output_hf_dir)  # 保存Chinese-LLaMA tokenizer
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")  # 保存Chinese-LLaMA tokenizer

# 测试tokenizer
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)  # LLaMA tokenizer
chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)  # Chinese-LLaMA tokenizer
print(tokenizer.all_special_tokens)  # LLaMA tokenizer的special tokens；输出为['<s>', '</s>', '<unk>']
print(tokenizer.all_special_ids)  # LLaMA tokenizer的special tokens对应的id；输出为[0, 1, 2]
print(tokenizer.special_tokens_map)  # LLaMA tokenizer的special tokens；输出为{'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}
text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
The primary use of LLaMA is research on large language models, including'''
print("Test text:\n", text)  # 测试文本
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")  # 测试LLaMA tokenizer
# 输出结果
# Tokenized by LLaMA tokenizer:['▁', '白', '日', '<0xE4>', '<0xBE>', '<0x9D>', '山', '<0xE5>', '<0xB0>', '<0xBD>', '，', '黄', '河', '入', '海', '流', '。', '<0xE6>', '<0xAC>', '<0xB2>', '<0xE7>', '<0xA9>', '<0xB7>', '千', '里', '目', '，', '更', '上', '一', '<0xE5>', '<0xB1>', '<0x82>', '<0xE6>', '<0xA5>', '<0xBC>', '。', '<0x0A>', 'The', '▁primary', '▁use', '▁of', '▁L', 'La', 'MA', '▁is', '▁research', '▁on', '▁large', '▁language', '▁models', ',', '▁including']
print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")  # 测试Chinese-LLaMA tokenizer
# 输出结果
# Tokenized by Chinese-LLaMA tokenizer:['▁白', '日', '依', '山', '尽', '，', '黄河', '入', '海', '流', '。', '欲', '穷', '千里', '目', '，', '更', '上', '一层', '楼', '。', '<0x0A>', 'The', '▁primary', '▁use', '▁of', '▁L', 'La', 'MA', '▁is', '▁research', '▁on', '▁large', '▁language', '▁models', ',', '▁including']