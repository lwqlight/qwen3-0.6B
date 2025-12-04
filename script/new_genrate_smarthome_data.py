import json
import random

# --- 全局统一指令 (保持不变，增加鲁棒性) ---
UNIFIED_INSTRUCTION = "智能家居中控：提取用户指令中的实体与意图，输出标准的JSON控制代码。"

# 定义数据总量 (为了保证覆盖度，建议稍微多一点)
BASE_SAMPLES = 200  # 基础样本数
CCT_SAMPLES = 400   # 色温+亮度样本数
COMPLEX_SAMPLES = 300 # 复杂多任务样本数
# TOTAL_SAMPLES = 1000 
dataset = []

# --- 资源池 ---
rooms = ["客厅", "卧室", "书房", "厨房", "儿童房", "浴室", "阳台", "玄关"]
devices = ["吸顶灯", "筒灯", "灯带", "落地灯", "台灯"]
colors = ["红色", "蓝色", "绿色", "暖光", "冷光", "中性光", "紫色"]

# --- 辅助函数：生成单个原子指令 (Atomic Command) ---
# 返回: (text_str, json_dict)
def get_atomic_command():
    task_type = random.choice(["light", "ac", "curtain", "cleaner", "music"])
    room = random.choice(rooms)
    
    # 1. 灯光开关
    if task_type == "light":
        device = random.choice(devices)
        state = random.choice(["打开", "关闭", "灭掉", "点亮"])
        val = "ON" if state in ["打开", "点亮"] else "OFF"
        return (
            f"把{room}的{device}{state}",
            {"target": f"{room}_{device}", "action": "turn", "value": val}
        )
    
    # 2. 空调控制
    elif task_type == "ac":
        temp = random.randint(18, 28)
        return (
            f"{room}空调调到{temp}度",
            {"target": f"{room}_ac", "action": "set_temp", "value": temp}
        )
    
    # 3. 窗帘
    elif task_type == "curtain":
        act_zh, act_en = random.choice([("拉开", "open"), ("关上", "close"), ("停一下", "stop")])
        return (
            f"{act_zh}{room}的窗帘",
            {"target": f"{room}_curtain", "action": act_en}
        )
        
    # 4. 扫地机器人 (新增设备类型)
    elif task_type == "cleaner":
        return (
            f"让扫地机器人去打扫{room}",
            {"target": "robot_cleaner", "action": "clean_area", "value": room}
        )

    # 5. 背景音乐 (新增设备类型)
    elif task_type == "music":
        style = random.choice(["爵士乐", "轻音乐", "摇滚", "白噪音"])
        return (
            f"{room}播放点{style}",
            {"target": f"{room}_speaker", "action": "play_music", "value": style}
        )
    
    return None

# ==========================================
# 核心生成逻辑
# ==========================================

# --- 第一部分：基础原子能力 (100条) ---
# 保持基础能力的稳固
for _ in range(BASE_SAMPLES):
    txt, json_obj = get_atomic_command()
    dataset.append({
        "instruction": UNIFIED_INSTRUCTION,
        "input": txt + "。",
        "output": json.dumps(json_obj, ensure_ascii=False)
    })

# --- 第二部分：[重点] 色温与亮度复合控制 (100条) ---
# 需求：控制灯的色温以及亮度
# 逻辑：这是一条自然语言对应两个动作 (set_color + set_brightness)
brightness_words = [("调亮一点", "up"), ("调暗一点", "down"), ("最亮", "100"), ("微亮", "20")]

for _ in range(CCT_SAMPLES):
    room = random.choice(rooms)
    device = random.choice(devices)
    color = random.choice(colors)
    bright_txt, bright_val = random.choice(brightness_words)
    
    # 组合语料： "把卧室吸顶灯调成暖光，并且调暗一点"
    connectors = ["，同时", "，并且", "，而且", "，顺便", "，再", "，还要"]
    input_text = f"把{room}的{device}调成{color}{random.choice(connectors)}{bright_txt}。"
    
    # 输出 JSON 列表：包含两个动作
    output_obj = [
        {"target": f"{room}_{device}", "action": "set_color", "value": color},
        {"target": f"{room}_{device}", "action": "set_brightness", "value": bright_val}
    ]
    
    dataset.append({
        "instruction": UNIFIED_INSTRUCTION,
        "input": input_text,
        "output": json.dumps(output_obj, ensure_ascii=False)
    })

# --- 第三部分：[高难度] 2-5 个任务的复杂长指令 (200条) ---
# 逻辑：随机抽取 N 个任务，拼接句子，合并列表
connectors = ["，然后", "，接着", "，", "，还要", "，别忘了", "，同时"]

for _ in range(COMPLEX_SAMPLES):
    # 随机决定任务数量：2 到 5 个
    num_tasks = random.randint(2, 5)
    
    combined_text = ""
    combined_json = []
    
    # 生成 N 个不重复的任务 (防止逻辑冲突，这里简单处理)
    # 实际逻辑：循环 N 次，每次取一个原子指令
    for i in range(num_tasks):
        txt, json_obj = get_atomic_command()
        
        # 拼接文本
        if i == 0:
            combined_text += txt
        else:
            # 随机选一个连接词
            combined_text += f"{random.choice(connectors)}{txt}"
            
        # 收集 JSON (如果是单个对象就 append，如果是列表就 extend，这里 atomic 都是单个)
        combined_json.append(json_obj)
    
    combined_text += "。"
    
    dataset.append({
        "instruction": UNIFIED_INSTRUCTION,
        "input": combined_text,
        "output": json.dumps(combined_json, ensure_ascii=False)
    })

# --- 保存与验证 ---
random.shuffle(dataset)
file_path = "./data/smarthome_data/train_complex_1000.json"
import os
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"✅ 高难度数据集生成完毕：{file_path}")
print(f"✅ 数据总量：{len(dataset)} 条")
print("\n--- 示例：色温+亮度 ---")
cct_sample = next(d for d in dataset if "set_color" in d['output'] and "set_brightness" in d['output'])
print(f"Input: {cct_sample['input']}")
print(f"Output: {cct_sample['output']}")

print("\n--- 示例：多任务 (3个以上) ---")
# 找一个 output 是长列表的样本
multi_sample = next(d for d in dataset if d['output'].count('{') >= 3)
print(f"Input: {multi_sample['input']}")
print(f"Output: {multi_sample['output']}")