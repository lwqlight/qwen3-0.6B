import json
import random

# 定义生成数据的总数
TOTAL_SAMPLES = 200
dataset = []

# --- 辅助列表：用于随机组合生成多样化数据 ---
rooms = ["客厅", "卧室", "书房", "厨房", "儿童房", "浴室", "阳台"]
colors = ["红色", "蓝色", "绿色", "暖光", "冷光", "白色", "紫色"]
devices = ["灯", "吸顶灯", "筒灯", "灯带"]

# --- 1. 灯光控制 (80条) ---
# 目标：提取地点、设备、操作、参数
for _ in range(80):
    room = random.choice(rooms)
    device = random.choice(devices)
    
    # 随机选择一种操作类型
    action_type = random.choice(["switch", "color", "brightness"])
    
    if action_type == "switch":
        state = random.choice(["打开", "关闭", "开一下", "关掉"])
        cmd_val = "ON" if "开" in state else "OFF"
        instruction = f"帮我把{room}的{device}{state}。"
        # 输出为伪代码/JSON格式，方便程序解析
        output = json.dumps({"target": f"{room}_{device}", "action": "turn", "value": cmd_val}, ensure_ascii=False)
        
    elif action_type == "color":
        color = random.choice(colors)
        instruction = f"把{room}的{device}调成{color}。"
        output = json.dumps({"target": f"{room}_{device}", "action": "set_color", "value": color}, ensure_ascii=False)
        
    elif action_type == "brightness":
        level = random.randint(10, 100)
        instruction = f"{room}的{device}亮度调到{level}%。"
        output = json.dumps({"target": f"{room}_{device}", "action": "set_brightness", "value": level}, ensure_ascii=False)

    dataset.append({
        "instruction": "解析用户的智能家居指令。",
        "input": instruction,
        "output": output
    })

# --- 2. 空调与环境控制 (60条) ---
for _ in range(60):
    room = random.choice(rooms)
    temp = random.randint(18, 28)
    
    # 模式1：直接调温
    if random.random() > 0.5:
        instruction = f"{room}空调温度设为{temp}度。"
        output = json.dumps({"target": f"{room}_ac", "action": "set_temp", "value": temp}, ensure_ascii=False)
    # 模式2：模糊指令（需要推理）
    else:
        diff = random.choice(["热", "冷"])
        instruction = f"我觉得{room}有点{diff}。"
        action = "temp_down" if diff == "热" else "temp_up" # 热就降温，冷就升温
        output = json.dumps({"target": f"{room}_ac", "action": action, "value": "auto"}, ensure_ascii=False)

    dataset.append({
        "instruction": "识别用户意图并控制空调。",
        "input": instruction,
        "output": output
    })

# --- 3. 窗帘与安防 (30条) ---
curtain_actions = [("打开", "open"), ("拉上", "close"), ("暂停", "stop")]
for _ in range(30):
    room = random.choice(rooms)
    act_zh, act_en = random.choice(curtain_actions)
    
    dataset.append({
        "instruction": "控制家中的窗帘设备。",
        "input": f"麻烦{act_zh}一下{room}的窗帘。",
        "output": json.dumps({"target": f"{room}_curtain", "action": act_en}, ensure_ascii=False)
    })

# --- 4. 场景模式 (30条) ---
scenes = [
    ("我要睡觉了", "sleep_mode"),
    ("我们要看电影", "movie_mode"),
    ("我出门了", "away_mode"),
    ("家里来客人了", "guest_mode"),
    ("起床了", "wakeup_mode")
]
for _ in range(30):
    txt, mode = random.choice(scenes)
    # 增加一些口语变体
    prefix = random.choice(["", "嘿Siri，", "小爱，", "管家，"]) 
    
    dataset.append({
        "instruction": "根据用户语言切换智能家居场景模式。",
        "input": f"{prefix}{txt}",
        "output": json.dumps({"action": "activate_scene", "scene": mode}, ensure_ascii=False)
    })

# 打乱并保存
random.shuffle(dataset)

file_path = "train_smarthome.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"智能家居数据集生成完毕：{file_path}")
print(f"示例数据：{dataset[0]}")