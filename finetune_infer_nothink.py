import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# --- 1. 路径配置 ---
base_model_path = "./qwen3-0.6B"
#  LoRA 微调后保存模型的路径
lora_path = "./finetune_model/qwen3_0.6B_smarthome_mutil_instruct" 

# --- 2. 全局统一指令 ---
UNIFIED_INSTRUCTION = "智能家居中控：提取用户指令中的实体与意图，输出标准的JSON控制代码。"

print("正在加载模型...")
# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 加载基座 (使用 bfloat16 或 float16)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",  # gpu->auto cpu->'cpu'
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载 LoRA
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# --- 3. 核心：解析函数  ---
# 关闭了 thinking，不需要再找 </think> 标签了，直接解码即可
def parse_output(generated_ids, input_len):
    # 1. 截取模型新生成的 token
    new_tokens = generated_ids[0][input_len:]
    
    # 2. 直接解码
    content = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return content

# --- 4. 辅助：JSON 提取器 (兼容 List 和 Dict) ---
def extract_json(text):
    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass

    # 2. 正则提取 (优先找列表，再找对象)
    try:
        # 寻找列表 [...]
        match_list = re.search(r'\[.*\]', text, re.DOTALL)
        if match_list:
            return json.loads(match_list.group())
        
        # 寻找对象 {...}
        match_dict = re.search(r'\{.*\}', text, re.DOTALL)
        if match_dict:
            return json.loads(match_dict.group())
    except:
        pass
        
    return None

# --- 5. 推理主函数 (添加了 enable_thinking=False) ---
def predict(user_input):
    if user_input:
        user_content = f"任务：{UNIFIED_INSTRUCTION}\n指令：{user_input}"
    else:
        user_content = UNIFIED_INSTRUCTION

    messages = [
        {"role": "system", "content": "你是一个智能家居中控助手，请将用户的自然语言指令转换为JSON格式的控制代码。"},
        {"role": "user", "content": user_content}
    ]

    # === 关键修改 ===
    try:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False  # 显式关闭思考模式
        )
    except TypeError:
        # 兼容性处理：如果当前 tokenizer 版本不支持 enable_thinking 参数，则回退
        # (通常意味着该模型版本本身就不带 thinking 功能，或者 transformers 版本较老)
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    model_inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024, # 增加长度，防止多任务输出被截断
            temperature=0.1, # 保持低温，增加 JSON 稳定性
            top_p=0.9,
            do_sample=True
        )

    return parse_output(generated_ids, model_inputs.input_ids.shape[1])

# --- 6. 测试部分 ---
print("\n=== 智能家居中控测试 (Think模式已关闭) ===\n")

test_cases = [
    "帮我把卧室的吸顶灯关掉。",
    "把书房的灯带调成暖光，顺便调暗一点。",
    "我觉得客厅有点太冷了。",
    "打开卧室空调，然后把窗帘拉上。",
    "我要出门了，把客厅灯关了，关闭空调，另外让扫地机器人去打扫厨房。"
]

for i, input_text in enumerate(test_cases):
    print(f"--- 测试案例 {i+1} ---")
    print(f"用户指令: {input_text}")
    
    # 推理
    start_time = time.time()
    result_text = predict(input_text)
    print(f"模型输出结果为: {result_text}")
    end_time = time.time()
    print(f"测试案例 {i+1} 的模型推理时间为: {end_time - start_time:.2f} 秒")
    
    # 打印原始输出，确认没有 <think> 标签了
    # print(f"DEBUG(原始输出): {result_text}") 
    
    # 提取 JSON
    json_data = extract_json(result_text)
    
    if json_data is not None:
        print("JSON 解析: ✅ 成功")
        
        # 归一化为列表
        if isinstance(json_data, dict):
            task_list = [json_data]
        elif isinstance(json_data, list):
            task_list = json_data
        else:
            task_list = []

        print(f"解析出 {len(task_list)} 个动作:")
        for idx, task in enumerate(task_list):
            target = task.get('target', 'N/A')
            action = task.get('action', 'N/A')
            value = task.get('value', 'N/A')
            print(f"  [动作 {idx+1}] 设备: {target:<15} | 操作: {action:<12} | 参数: {value}")
            
    else:
        print(f"JSON 解析: ❌ 失败")
        print(f"失败输出内容: {result_text}")
    
    print("-" * 50)