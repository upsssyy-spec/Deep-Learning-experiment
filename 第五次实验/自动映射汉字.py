import os

# ==================== 配置你的数据集路径 ====================
DATASET_PATH = "./data/train"  # 改成你的训练集路径！


def generate_character_map(dataset_path):
    """自动扫描数据集，生成汉字映射表"""

    # 获取所有类别文件夹
    folders = sorted([f for f in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, f))])

    print(f"📊 发现 {len(folders)} 个类别文件夹\n")

    character_map = {}

    print("=" * 50)
    print("正在自动生成映射表...")
    print("=" * 50)

    for i, folder_name in enumerate(folders):
        # 方法1：文件夹名就是数字（00000, 00001...）
        try:
            class_id = int(folder_name)
        except:
            # 方法2：文件夹名带前导零（00008 → 8）
            try:
                class_id = int(folder_name.lstrip('0')) if folder_name != '00000' else 0
            except:
                class_id = i

        # 方法3：尝试解码GBK编码（CASIA标准数据集格式）
        try:
            # CASIA数据集常用：文件夹名是GBK十六进制，如 "B0A1" → "啊"
            if len(folder_name) == 4 and all(c in '0123456789ABCDEFabcdef' for c in folder_name):
                gbk_bytes = bytes.fromhex(folder_name)
                char = gbk_bytes.decode('gbk')
            # 方法4：文件夹名本身就是汉字
            elif len(folder_name) == 1 and '\u4e00' <= folder_name <= '\u9fff':
                char = folder_name
            # 方法5：文件夹名是拼音，需要手动对应（这里先留空）
            else:
                char = f"请手动设置({folder_name})"
        except:
            char = f"类别{class_id}"

        character_map[class_id] = char
        print(f"{class_id:3d} → {char}")

    print("\n" + "=" * 50)
    print("✅ 生成完成！复制下面的代码到你的GUI中：")
    print("=" * 50)
    print()

    # 输出Python格式的映射表
    print("CHARACTER_MAP = {")
    for i in range(0, len(character_map), 10):
        line = "    "
        for j in range(10):
            idx = i + j
            if idx in character_map:
                line += f"{idx}: \"{character_map[idx]}\", "
        print(line)
    print("}")

    return character_map


# ==================== 运行生成 ====================
if __name__ == "__main__":
    generate_character_map(DATASET_PATH)