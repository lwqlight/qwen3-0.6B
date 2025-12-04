import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 配置路径
base_model_path = "./qwen3-0.6B"                  # 原始基座
lora_path = "./finetune_model/qwen3_0.6B_smarthome_mutil_instruct" # 你的微调结果
merged_save_path = "./finetune_model/qwen3_0.6B_merged_fp16"     # 合并后的保存路径

print("正在加载基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print("正在加载 LoRA 并合并...")
# 加载 LoRA
model = PeftModel.from_pretrained(base_model, lora_path)
# 核心步骤：将 LoRA 权重加回到基座参数中
model = model.merge_and_unload()

print(f"正在保存合并模型至 {merged_save_path} ...")
model.save_pretrained(merged_save_path)
tokenizer.save_pretrained(merged_save_path)

print("✅ 模型合并完成！")