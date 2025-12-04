import pandas as pd
import torch
import json
import re
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =================配置区域=================
# 输入文件：只包含 'Input_Text' 一列的 Excel
INPUT_EXCEL = "./data/test/raw_inputs.xlsx"       
# 输出文件：模型填好数据的 Excel
OUTPUT_EXCEL = "./data/test/human_data_labeled.xlsx" 

# 模型路径配置
BASE_MODEL_PATH = "./qwen3-0.6B"
# 指向你的 LoRA 权重路径
LORA_PATH = "./finetune_model/qwen3_0.6B_smarthome_mutil_instruct" 

# 统一指令 (必须与训练时一致)
UNIFIED_INSTRUCTION = "智能家居中控：提取用户指令中的实体与意图，输出标准的JSON控制代码。"
# =========================================

# --- 1. 加载模型 (PyTorch 原生) ---
print("🚀 正在加载 PyTorch 模型...")
try:
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 加载基座
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 加载 LoRA
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval() # 切换到评估模式
    print("✅ 模型加载成功！")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# --- 2. 核心：推理函数 ---
def predict_labels(user_input):
    # 构造 Prompt
    user_content = f"任务：{UNIFIED_INSTRUCTION}\n指令：{user_input}"
    
    messages = [
        {"role": "system", "content": "你是一个智能家居中控助手，请将用户的自然语言指令转换为JSON格式的控制代码。"},
        {"role": "user", "content": user_content}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False 
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    model_inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512, # 足够容纳多任务指令
            temperature=0.1,    # 低温保证格式稳定
            top_p=0.9,
            do_sample=True
        )

    # 解析输出 (去除输入的 prompt)
    input_len = model_inputs.input_ids.shape[1]
    new_tokens = generated_ids[0][input_len:]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return content

# --- 3. 辅助：JSON 提取器 ---
def extract_json(text):
    try:
        # 优先找列表 [...]
        match_list = re.search(r'\[.*\]', text, re.DOTALL)
        if match_list: 
            return json.loads(match_list.group())
        
        # 再找对象 {...}
        match_dict = re.search(r'\{.*\}', text, re.DOTALL)
        if match_dict: 
            # 统一转为列表返回，方便后续处理
            return [json.loads(match_dict.group())] 
    except:
        pass
    return None

# --- 4. 主业务逻辑：读取 Excel -> 标注 -> 保存 ---
def auto_label():
    print(f"📂 读取原始数据文件: {INPUT_EXCEL}")
    if not os.path.exists(INPUT_EXCEL):
        print("❌ 找不到输入文件，请先创建一个名为 raw_inputs.xlsx 的表格，包含 'Input_Text' 列。")
        return

    try:
        df = pd.read_excel(INPUT_EXCEL)
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    # 准备结果容器
    labeled_rows = []
    total = len(df)
    
    print(f"⚡ 开始自动标注，共 {total} 条数据...")

    for index, row in df.iterrows():
        input_text = str(row['Input_Text']).strip()
        # 跳过空行
        if not input_text or input_text.lower() == 'nan':
            continue

        print(f"[{index+1}/{total}] 处理: {input_text}")

        # --- 调用模型推理 ---
        raw_output = predict_labels(input_text)
        
        # --- 提取 JSON ---
        json_data = extract_json(raw_output)

        if json_data:
            # 如果解析成功，遍历结果（兼容多任务指令生成多行）
            for item in json_data:
                labeled_rows.append({
                    "Input_Text": input_text,
                    "Target": item.get("target", ""),
                    "Action": item.get("action", ""),
                    "Value": item.get("value", "")
                })
        else:
            # 如果解析失败，填入原始内容，标记为待人工检查
            print(f"  ⚠️ 模型输出格式异常，需人工填写: {raw_output}")
            labeled_rows.append({
                "Input_Text": input_text,
                "Target": "MANUAL_CHECK", # 标记关键词
                "Action": raw_output,     # 把原始输出填进去参考
                "Value": ""
            })

    # --- 保存结果 ---
    print("💾 正在保存结果...")
    result_df = pd.DataFrame(labeled_rows)
    result_df.to_excel(OUTPUT_EXCEL, index=False)
    
    print("\n" + "="*50)
    print(f"✅ 自动标注完成！")
    print(f"📂 结果文件: {OUTPUT_EXCEL}")
    print("⚠️  下一步：请务必打开表格进行人工校验，修正 'MANUAL_CHECK' 及错误的标注！")
    print("="*50)

if __name__ == "__main__":
    auto_label()