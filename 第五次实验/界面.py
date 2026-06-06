import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# ==================== 汉字映射表 ====================
CHARACTER_MAP = {
    0: "一", 1: "丁", 2: "七", 3: "万", 4: "丈", 5: "三", 6: "上", 7: "下", 8: "不", 9: "与",
    10: "丑", 11: "专", 12: "且", 13: "世", 14: "丘", 15: "丙", 16: "业", 17: "丛", 18: "东", 19: "丝",
    20: "丢", 21: "两", 22: "严", 23: "丧", 24: "个", 25: "丫", 26: "中", 27: "丰", 28: "串", 29: "临",
    30: "丸", 31: "丹", 32: "为", 33: "主", 34: "丽", 35: "举", 36: "乃", 37: "久", 38: "么", 39: "义",
    40: "之", 41: "乌", 42: "乍", 43: "乎", 44: "乏", 45: "乐", 46: "乒", 47: "乓", 48: "乔", 49: "乖",
    50: "乘", 51: "乙", 52: "九", 53: "乞", 54: "也", 55: "习", 56: "乡", 57: "书", 58: "买", 59: "乱",
    60: "乳", 61: "乾", 62: "了", 63: "予", 64: "争", 65: "事", 66: "二", 67: "于", 68: "亏", 69: "云",
    70: "互", 71: "五", 72: "井", 73: "亚", 74: "些", 75: "亡", 76: "亢", 77: "交", 78: "亥", 79: "亦",
    80: "产", 81: "亨", 82: "亩", 83: "享", 84: "京", 85: "亭", 86: "亮", 87: "亲", 88: "人", 89: "亿",
    90: "什", 91: "仁", 92: "仅", 93: "仆", 94: "仇", 95: "今", 96: "介", 97: "仍", 98: "从", 99: "仑"
}


# ==================== 网络结构 ====================
class MYNET(nn.Module):
    def __init__(self):
        super(MYNET, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2704)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== 预测工具类 ====================
class CharacterPredictor:
    def __init__(self, model_path='./tmp/model.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MYNET().to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("✅ 模型加载成功！")
        else:
            print(f"❌ 模型不存在：{model_path}")

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def predict(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)

                class_id = prediction.item()
                char = CHARACTER_MAP.get(class_id, f"未知({class_id})")
                return char, class_id, confidence.item() * 100
        except Exception as e:
            raise Exception(f"预测失败：{str(e)}")


# ==================== 修复版GUI（按钮固定可见）====================
class PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("汉字识别系统")
        self.root.geometry("1000x700")  # 增加窗口高度
        self.root.resizable(True, True)

        self.predictor = CharacterPredictor()
        self.current_img_path = None

        self.create_ui()

    def create_ui(self):
        # ========== 顶部标题 ==========
        title_frame = tk.Frame(self.root, pady=15)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="🔤 手写汉字识别系统", font=("微软雅黑", 22, "bold")).pack()

        # ========== 主体：左右分栏 ==========
        main_frame = tk.Frame(self.root, padx=20, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------- 左侧：识别窗口（按钮固定在顶部）----------
        left_frame = tk.LabelFrame(main_frame, text="📷 识别窗口", font=("微软雅黑", 12, "bold"), padx=15, pady=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # ✅ 【修复】按钮放在最顶部，永远可见！
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Button(btn_frame, text="📁 选择图片", command=self.open_image,
                  font=("微软雅黑", 12), width=15, bg="#4285F4", fg="white", padx=10, pady=8).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🔍 开始识别", command=self.do_predict,
                  font=("微软雅黑", 12), width=15, bg="#34A853", fg="white", padx=10, pady=8).pack(side=tk.LEFT, padx=5)

        # 图片预览区（在按钮下方）
        self.img_label = tk.Label(left_frame, text="请选择图片\n\n支持 PNG、JPG、BMP 格式",
                                  bg="#f5f5f5", font=("微软雅黑", 12), fg="gray",
                                  width=35, height=18, relief=tk.SUNKEN)
        self.img_label.pack(fill=tk.BOTH, expand=True, pady=5)

        # ---------- 右侧：预测结果窗口 ----------
        right_frame = tk.LabelFrame(main_frame, text="📊 预测结果", font=("微软雅黑", 12, "bold"), padx=15, pady=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # 印刷体汉字显示
        result_char_frame = tk.Frame(right_frame, bg="white", relief=tk.SUNKEN, height=300)
        result_char_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        result_char_frame.pack_propagate(False)  # 固定高度

        self.result_char_label = tk.Label(result_char_frame, text="?",
                                          font=("宋体", 120, "bold"), bg="white", fg="#1a1a1a")
        self.result_char_label.pack(expand=True)

        tk.Label(result_char_frame, text="↑ 印刷体对照", font=("微软雅黑", 10), fg="gray", bg="white").pack(pady=5)

        # 详细信息
        info_frame = tk.Frame(right_frame)
        info_frame.pack(fill=tk.X, pady=10)

        tk.Label(info_frame, text="类别编号:", font=("微软雅黑", 11)).grid(row=0, column=0, sticky="e", padx=5, pady=3)
        self.class_id_label = tk.Label(info_frame, text="-", font=("微软雅黑", 11, "bold"), fg="#2196F3")
        self.class_id_label.grid(row=0, column=1, sticky="w", padx=5, pady=3)

        tk.Label(info_frame, text="置信度:", font=("微软雅黑", 11)).grid(row=1, column=0, sticky="e", padx=5, pady=3)
        self.confidence_label = tk.Label(info_frame, text="-", font=("微软雅黑", 11, "bold"), fg="#4CAF50")
        self.confidence_label.grid(row=1, column=1, sticky="w", padx=5, pady=3)

        # 状态提示
        self.status_label = tk.Label(right_frame, text="等待输入...", font=("微软雅黑", 10), fg="gray")
        self.status_label.pack(pady=10)

        # ========== 底部提示 ==========
        footer_frame = tk.Frame(self.root, pady=10)
        footer_frame.pack(fill=tk.X)
        tk.Label(footer_frame, text="💡 提示：点击「选择图片」上传手写汉字，再点击「开始识别」",
                 font=("微软雅黑", 10), fg="gray").pack()

    def open_image(self):
        path = filedialog.askopenfilename(
            title="选择手写汉字图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")]
        )

        if path:
            self.current_img_path = path

            img = Image.open(path)
            img.thumbnail((350, 350))
            photo = ImageTk.PhotoImage(img)

            self.img_label.config(image=photo, text="")
            self.img_label.image = photo

            self.result_char_label.config(text="?")
            self.class_id_label.config(text="-")
            self.confidence_label.config(text="-")
            self.status_label.config(text="图片已加载，点击「开始识别」", fg="blue")

    def do_predict(self):
        if not self.current_img_path:
            messagebox.showwarning("提示", "请先选择一张图片！")
            return

        try:
            char, class_id, confidence = self.predictor.predict(self.current_img_path)

            self.result_char_label.config(text=char)
            self.class_id_label.config(text=str(class_id))
            self.confidence_label.config(text=f"{confidence:.2f}%")

            if confidence >= 80:
                self.confidence_label.config(fg="#4CAF50")
            elif confidence >= 50:
                self.confidence_label.config(fg="#FF9800")
            else:
                self.confidence_label.config(fg="#f44336")

            self.status_label.config(text="✅ 识别完成！", fg="green")

        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.status_label.config(text="❌ 识别失败", fg="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorGUI(root)

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()