from llama_cpp import Llama
import json
import re
import time

# --- 配置 ---
#  gguf 文件
model_path = "./finetune_model/qwen3_0.6B_q4_k_m.gguf" 
UNIFIED_INSTRUCTION = "智能家居中控：提取用户指令中的实体与意图，输出标准的JSON控制代码。"

# --- 加载模型 ---
# n_ctx: 上下文长度，设为 1024 或 2048
# n_gpu_layers: 如果有显卡，设为 -1 表示全扔进显卡；如果是纯 CPU，设为 0
print("正在加载 GGUF 模型...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048, 
    n_gpu_layers=0, # 0表示纯CPU运行，适合树莓派
    verbose=False   # 关闭底层日志
)

# --- 辅助函数 (保持之前的正则逻辑) ---
def extract_json(text):
    try:
        match_list = re.search(r'\[.*\]', text, re.DOTALL)
        if match_list: return json.loads(match_list.group())
        match_dict = re.search(r'\{.*\}', text, re.DOTALL)
        if match_dict: return json.loads(match_dict.group())
    except:
        pass
    return None

def predict(user_input):
    # 构造 Prompt (格式必须符合 Qwen 的 ChatML)
    # GGUF 不会自动帮你加 <|im_start|>，你需要手动拼接字符串，或者使用 llama-cpp 的 chat 接口
    
    messages = [
        {"role": "system", "content": "你是一个智能家居中控助手，请将用户的自然语言指令转换为JSON格式的控制代码。"},
        {"role": "user", "content": f"任务：{UNIFIED_INSTRUCTION}\n指令：{user_input}"}
    ]
    start_time = time.time()
    # 使用 llama-cpp-python 内置的 chat 完成接口
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.1,
        top_p=0.9
    )
    end_time = time.time()
    print(f"推理时间: {end_time - start_time:.2f} 秒")
    
    # 提取内容
    return output['choices'][0]['message']['content']

# --- 测试 ---
print("\n=== GGUF 端侧推理测试 ===\n")
user_text = "把客厅灯关了，顺便打开空调"
print(f"指令: {user_text}")

result = predict(user_text)
print(f"原始输出: {result}")

json_data = extract_json(result)
if json_data:
    print(f"JSON 解析成功: {json_data}")
else:
    print("JSON 解析失败")