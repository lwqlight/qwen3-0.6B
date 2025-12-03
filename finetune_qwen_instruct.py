import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# --- 1. 模型加载 ---
model_dir = "./qwen3-0.6B" # 本地路径

if not os.path.exists(model_dir):
    raise FileNotFoundError(f"错误：找不到模型文件夹 '{model_dir}'")

print(f"正在加载本地模型: {model_dir} ...")

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

# --- 2. 数据集加载 ---
data_file = "./data/smarthome_data/train_complex.json"

if not os.path.exists(data_file):
    raise FileNotFoundError(f"错误：找不到数据集文件 '{data_file}'")

dataset = load_dataset("json", data_files=data_file, split="train")

# --- 3. 数据预处理 (关键修改：手动合并列) ---
# 我们不依赖 Trainer 的 formatting_func，而是直接把数据清洗好
def process_data_to_text(example):
    # 1. 拼接输入
    instruction = example['instruction']
    input_text = example['input']
    output_text = example['output']
    
    if input_text:
        user_content = f"任务：{instruction}\n指令：{input_text}"
    else:
        user_content = instruction

    # 2. 构造 Messages
    messages = [
        {"role": "system", "content": "你是一个智能家居中控助手，请将用户的自然语言指令转换为JSON格式的控制代码。"},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text}
    ]
    
    # 3. 转换为最终训练文本
    # 注意：这里我们生成好文本，放入 'text' 列
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

print("正在预处理数据集...")
# map 函数会应用到每一行数据
dataset = dataset.map(process_data_to_text)

# --- 4. 关键步骤：只保留 text 列，删除其他列 ---
# 这样 Trainer 就不会试图去训练 'instruction' 这种字符串列了，从而解决报错
columns_to_remove = ["instruction", "input", "output"]
# 检查一下列是否存在，防止报错
existing_columns = [col for col in columns_to_remove if col in dataset.column_names]
dataset = dataset.remove_columns(existing_columns)

print("处理后的数据示例:", dataset[0]['text'])

# --- 5. LoRA 配置 ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# --- 6. 训练参数 ---
output_dir = "./finetune_model/qwen3_0.6B_smarthome_mutil_instruct"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    remove_unused_columns=True, # 恢复为 True（默认值），因为我们已经手动删除了无用列
    dataloader_pin_memory=False # 加上这个有时候能避免一些显存碎片问题
)

# --- 7. Trainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=512,  # 512长度足够 长任务使用1024
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text", # 明确告诉 Trainer 训练哪一列
    packing=False,
)

print("开始指令微调...")
trainer.train()

# --- 8. 保存 ---
print(f"训练完成，正在保存最终模型至 {output_dir} ...")
trainer.save_model(output_dir)
print("全部完成！")