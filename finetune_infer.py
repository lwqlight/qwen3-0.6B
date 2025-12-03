import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 1. 路径配置 ---
base_model_path = "./qwen3-0.6B"                  # 本地基座路径
lora_path = "./finetune_model/qwen3_0.6B_smarthome_instruct"     # 你的微调权重路径

print("正在加载模型...")
# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 加载基座
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载 LoRA
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# --- 2. 核心：解析函数 (结合了你的参考逻辑和正则提取) ---
def parse_output(generated_ids, input_len):
    # 获取仅生成的这一部分 token id
    output_ids = generated_ids[0][input_len:].tolist()
    
    # 尝试寻找 </think> 的结束符 (Token ID: 151668)
    # 这是一个特殊的 Token，专门用于标记思考结束
    think_end_token = 151668 
    
    try:
        # 从后往前找，找到 </think> 的位置
        # output_ids[::-1] 是倒序，index 找的是倒序的索引
        reversed_ids = output_ids[::-1]
        index_from_end = reversed_ids.index(think_end_token)
        split_index = len(output_ids) - index_from_end
    except ValueError:
        # 如果没找到 </think>，说明模型可能没输出思考过程，直接从头开始
        split_index = 0

    # 解码思考部分（可选，调试用）
    # thinking_content = tokenizer.decode(output_ids[:split_index], skip_special_tokens=True)
    
    # 解码正文部分 (这是我们真正要的)
    content = tokenizer.decode(output_ids[split_index:], skip_special_tokens=True).strip()
    
    return content

# --- 3. 辅助：JSON 提取器 (双重保险) ---
def extract_json(text):
    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass

    # 2. 如果失败，使用正则寻找第一个 { 和最后一个 } 之间的内容
    # 这样可以忽略掉 <think> 标签残留或者 Markdown 代码块 (```json ... ```)
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except:
        pass
        
    return None

# --- 4. 推理主函数 ---
def predict(instruction, input_text=""):
    # 构造输入
    if input_text:
        user_content = f"任务：{instruction}\n指令：{input_text}"
    else:
        user_content = instruction

    messages = [
        {"role": "system", "content": "你是一个智能家居中控助手，请将用户的自然语言指令转换为JSON格式的控制代码。"},
        {"role": "user", "content": user_content}
    ]

    # Qwen3 推荐开启 enable_thinking=True 以保持模板一致性，虽然我们微调时可能没用到
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.1, # 低温保证 JSON 格式稳定
            top_p=0.9,
            do_sample=True
        )

    # 调用上面的解析函数
    raw_content = parse_output(generated_ids, model_inputs.input_ids.shape[1])
    
    return raw_content

# --- 5. 测试 ---
print("\n=== 开始智能家居指令测试 (修正版) ===\n")

test_cases = [
    {"instruction": "解析用户的智能家居指令。", "input": "帮我把卧室的吸顶灯关掉。"},
    {"instruction": "解析用户的智能家居指令。", "input": "把书房的灯带调成暖光。"},
    {"instruction": "识别用户意图并控制空调。", "input": "我觉得客厅有点太热了。"},
    {"instruction": "根据用户语言切换智能家居场景模式。", "input": "嘿Siri，我们要看电影了。"}
]

for i, case in enumerate(test_cases):
    print(f"--- 测试案例 {i+1} ---")
    print(f"用户指令: {case['input']}")
    
    # 获取原始文本结果（去掉了 <think>）
    result_text = predict(case['instruction'], case['input'])
    print(f"清洗后的模型输出: {result_text}")
    
    # 提取 JSON 对象
    json_obj = extract_json(result_text)
    
    if json_obj:
        print("JSON 解析: ✅ 成功")
        print(f"结构化数据: {json_obj}")
        print(f"执行动作: 目标=[{json_obj.get('target', 'N/A')}] 动作=[{json_obj.get('action')}]")
    else:
        print("JSON 解析: ❌ 失败")
    
    print("-" * 30)