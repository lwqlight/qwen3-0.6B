import json
import random
import pandas as pd
import os

# --- 全局配置 ---
UNIFIED_INSTRUCTION = "智能家居中控：提取用户指令中的实体与意图，输出标准的JSON控制代码。"
EXCEL_FILE_PATH = "human_data.xlsx" # 你的Excel文件路径
OUTPUT_FILE_PATH = "./data/smarthome_data/train_hybrid.json"

# ==========================================
# 模块 1: 处理人工 Excel 数据 (核心新增)  需要先使用auto_excel_label.py将真实用户给的指令excel生成训练数据格式
# ==========================================
def load_human_data_from_excel(file_path):
    print(f"📂 正在读取人工标注数据: {file_path} ...")
    
    if not os.path.exists(file_path):
        print("⚠️ 未找到Excel文件，跳过人工数据加载。")
        return []

    # 读取 Excel
    try:
        df = pd.read_excel(file_path)
        # 确保列名去空格
        df.columns = [c.strip() for c in df.columns]
        
        # 检查必要列是否存在
        required_cols = ['Input_Text', 'Target', 'Action', 'Value']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Excel 缺少必要列，请确保表头包含: {required_cols}")

        human_dataset = []
        
        # --- 核心逻辑：按“用户指令”分组 ---
        # 这样处理可以将多行（多任务）合并为同一个 Input 对应的 Output 列表
        grouped = df.groupby('Input_Text')

        for input_text, group in grouped:
            action_list = []
            
            for _, row in group.iterrows():
                # 处理 Value 的类型：如果是数字，保持数字；如果是字符串，保持字符串
                val = row['Value']
                # 尝试转为 int，如果失败则保持原样 (处理 '26.0' 变成 26)
                try:
                    if float(val).is_integer():
                        val = int(val)
                except:
                    pass # 不是数字，保持原样 (如 'ON', 'warm')

                action_obj = {
                    "target": str(row['Target']).strip(),
                    "action": str(row['Action']).strip(),
                    "value": val
                }
                action_list.append(action_obj)

            # 构造训练样本
            # 如果列表只有一个动作，且你希望它是对象而不是列表，可以解包
            # 但为了统一性，建议全部保持为 List 结构，或者根据长度判断
            # 这里为了兼容之前的逻辑：单个动作存 dict，多个动作存 list
            if len(action_list) == 1:
                final_output = action_list[0]
            else:
                final_output = action_list

            human_dataset.append({
                "instruction": UNIFIED_INSTRUCTION,
                "input": str(input_text).strip(),
                "output": json.dumps(final_output, ensure_ascii=False)
            })
            
        print(f"✅ 成功加载人工数据: {len(human_dataset)} 条 (原始行数: {len(df)})")
        return human_dataset

    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return []

# ==========================================
# 模块 2: 生成机器合成数据 (保留你原本的逻辑)
# ==========================================
def generate_synthetic_data():
    print("🤖 正在生成机器合成数据...")
    dataset = []
    
    # 定义数据量
    BASE_SAMPLES = 200
    CCT_SAMPLES = 400
    COMPLEX_SAMPLES = 300
    
    rooms = ["客厅", "卧室", "书房", "厨房", "儿童房", "浴室", "阳台", "玄关"]
    devices = ["吸顶灯", "筒灯", "灯带", "落地灯", "台灯"]
    colors = ["红色", "蓝色", "绿色", "暖光", "冷光", "中性光", "紫色"]

    def get_atomic_command():
        task_type = random.choice(["light", "ac", "curtain", "cleaner", "music"])
        room = random.choice(rooms)
        
        if task_type == "light":
            device = random.choice(devices)
            state = random.choice(["打开", "关闭", "灭掉", "点亮"])
            val = "ON" if state in ["打开", "点亮"] else "OFF"
            return (f"把{room}的{device}{state}", {"target": f"{room}_{device}", "action": "turn", "value": val})
        elif task_type == "ac":
            temp = random.randint(18, 28)
            return (f"{room}空调调到{temp}度", {"target": f"{room}_ac", "action": "set_temp", "value": temp})
        elif task_type == "curtain":
            act_zh, act_en = random.choice([("拉开", "open"), ("关上", "close"), ("停一下", "stop")])
            return (f"{act_zh}{room}的窗帘", {"target": f"{room}_curtain", "action": act_en})
        elif task_type == "cleaner":
            return (f"让扫地机器人去打扫{room}", {"target": "robot_cleaner", "action": "clean_area", "value": room})
        elif task_type == "music":
            style = random.choice(["爵士乐", "轻音乐", "摇滚", "白噪音"])
            return (f"{room}播放点{style}", {"target": f"{room}_speaker", "action": "play_music", "value": style})
        return None

    # 1. 基础原子能力
    for _ in range(BASE_SAMPLES):
        txt, json_obj = get_atomic_command()
        dataset.append({
            "instruction": UNIFIED_INSTRUCTION,
            "input": txt + "。",
            "output": json.dumps(json_obj, ensure_ascii=False)
        })

    # 2. 色温与亮度复合
    brightness_words = [("调亮一点", "up"), ("调暗一点", "down"), ("最亮", 100), ("微亮", 20)]
    for _ in range(CCT_SAMPLES):
        room = random.choice(rooms)
        device = random.choice(devices)
        color = random.choice(colors)
        bright_txt, bright_val = random.choice(brightness_words)
        connectors = ["，同时", "，并且", "，而且", "，顺便", "，再", "，还要"]
        input_text = f"把{room}的{device}调成{color}{random.choice(connectors)}{bright_txt}。"
        output_obj = [
            {"target": f"{room}_{device}", "action": "set_color", "value": color},
            {"target": f"{room}_{device}", "action": "set_brightness", "value": bright_val}
        ]
        dataset.append({
            "instruction": UNIFIED_INSTRUCTION,
            "input": input_text,
            "output": json.dumps(output_obj, ensure_ascii=False)
        })

    # 3. 复杂多任务
    connectors = ["，然后", "，接着", "，", "，还要", "，别忘了", "，同时"]
    for _ in range(COMPLEX_SAMPLES):
        num_tasks = random.randint(2, 5)
        combined_text = ""
        combined_json = []
        for i in range(num_tasks):
            txt, json_obj = get_atomic_command()
            if i == 0:
                combined_text += txt
            else:
                combined_text += f"{random.choice(connectors)}{txt}"
            combined_json.append(json_obj)
        combined_text += "。"
        dataset.append({
            "instruction": UNIFIED_INSTRUCTION,
            "input": combined_text,
            "output": json.dumps(combined_json, ensure_ascii=False)
        })
    
    return dataset

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 获取机器数据
    synthetic_data = generate_synthetic_data()
    
    # 2. 获取人工数据
    human_data = load_human_data_from_excel(EXCEL_FILE_PATH)
    
    # 3. 合并数据
    final_dataset = synthetic_data + human_data
    
    # 4. 打乱顺序
    random.shuffle(final_dataset)
    
    # 5. 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print("\n" + "="*40)
    print(f"🎉 混合数据集生成完毕！")
    print(f"📊 机器样本: {len(synthetic_data)}")
    print(f"📊 人工样本: {len(human_data)}")
    print(f"🚀 总样本数: {len(final_dataset)}")
    print(f"💾 保存路径: {OUTPUT_FILE_PATH}")
    print("="*40)
    
    # 打印一条人工数据示例（如果有）
    if human_data:
        print("\n🔍 人工数据示例:")
        print(json.dumps(human_data[0], ensure_ascii=False, indent=2))