# Direction Detector (方向检测器)

一个基于 PyTorch 和 ResNet18 的轻量级模型，用于检测图像中物体的旋转方向（轴向）。该项目通过对透明背景的 PNG 图片进行随机旋转和背景合成，自动生成训练数据并训练模型。

## 🌟 功能特点

- **自动数据增强**：只需准备少量透明背景 PNG 图片，程序会自动进行旋转、缩放、随机背景颜色填充及颜色抖动。
- **高性能训练**：支持 CUDA 加速，并使用混合精度训练（AMP）以提升速度。
- **无向轴向预测**：模型预测范围为 0-180 度，适用于对称物体的轴向识别（采用 $2\theta$ 映射算法解决角度跳变问题）。
- **交互式测试**：提供 `test.py` 脚本，可以实时输入角度旋转图片并查看模型预测结果。

## 🛠️ 安装与环境配置

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行包管理。

1. **克隆仓库**
   ```bash
   git clone https://github.com/huangdihd/direction_detector.git
   cd direction_detector
   ```

2. **安装依赖**
   如果你已经安装了 `uv`：
   ```bash
   uv sync
   ```
   或者使用普通的 `pip`：
   ```bash
   pip install torch torchvision numpy pillow tqdm scikit-learn pandas
   ```

## 📂 项目结构

```text
direction_detector/
├── train_images/       # 训练用图片目录 (存放透明背景 .png)
├── test_images/        # 测试用图片目录
├── train.py            # 训练脚本
├── test.py             # 交互式测试脚本
├── orientation_model.pth # 训练好的权重文件 (默认已忽略，需自行训练或上传)
├── pyproject.toml      # 项目配置文件
└── README.md           # 项目说明
```

## 🚀 使用方法

### 1. 准备图片
在 `train_images/` 和 `test_images/` 文件夹中放入透明背景的 `.png` 图片（例如小车、工具、图标等）。

### 2. 训练模型
运行训练脚本，模型会根据 `train_images` 中的图片自动生成 10,000 个样本进行训练。
```bash
python train.py
```
- 训练过程中会自动保存 Loss 达标的模型。
- 你可以随时按 `Ctrl+C` 中断训练，程序会自动保存当前最新的模型。

### 3. 测试模型
运行测试脚本来验证模型效果（命令行交互）：
```bash
python test.py
```
- 程序会列出 `test_images` 中的图片，你可以输入编号选择图片。
- 输入一个旋转角度（如 `45`），程序会展示模型预测的轴向度数。

### 4. 启动 WebUI (推荐)
本项目提供了一个基于 Gradio 的图形化界面，支持上传图片、滑动调整角度，并实时在图像上绘制模型预测的轴线：
```bash
python app.py
```
运行后，在浏览器中打开 `http://127.0.0.1:7860` 即可体验。

## 🔬 技术细节

- **模型架构**：Backbone 使用 ResNet18，全连接层经过自定义修改以适配回归任务。
- **标签映射**：为了解决 0 度和 180 度之间的不连续性，模型预测的是 $\sin(2\theta)$ 和 $\cos(2\theta)$。
- **损失函数**：使用余弦相似度作为基础的损失衡量方式，确保预测向量与目标标签方向一致。

## 📝 许可证

MIT License
