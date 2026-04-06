import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from train import OrientationNet

# 1. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OrientationNet().to(device)

try:
    # 尝试加载预训练权重
    model.load_state_dict(torch.load('orientation_model.pth', map_location=device, weights_only=True))
    model.eval()
    model_status = "✅ 模型已成功加载 (orientation_model.pth)"
except FileNotFoundError:
    model_status = "⚠️ 未找到 orientation_model.pth，目前使用的是未经训练的初始模型。请先运行 train.py 进行训练。"
except Exception as e:
    model_status = f"❌ 模型加载失败: {e}"

# 2. 推理与可视化逻辑
def process_and_predict(image, angle):
    if image is None:
        return None, "请上传一张包含透明背景的PNG图片。"

    # 将图片转换为 RGBA 并调整大小
    img = image.convert('RGBA').resize((200, 200))
    
    # 模拟实际环境：旋转图片并粘贴到随机颜色的背景上
    from PIL.Image import Resampling
    import random
    
    rotated_obj = img.rotate(angle, expand=True, resample=Resampling.BICUBIC)
    # 为了UI美观，使用稍微柔和的随机颜色
    random_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    canvas_size = int(max(img.size) * 1.5)
    final_img = Image.new("RGB", (canvas_size, canvas_size), random_color)
    
    paste_x = (canvas_size - rotated_obj.size[0]) // 2
    paste_y = (canvas_size - rotated_obj.size[1]) // 2
    final_img.paste(rotated_obj, (paste_x, paste_y), rotated_obj)
    
    # 图像预处理 (去除了训练时的随机颜色抖动和灰度化)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(final_img).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        
    pred_sin = output[0, 0].cpu()
    pred_cos = output[0, 1].cpu()
    
    # 将网络输出的 2θ 转换回真实的角度
    angle_rad_2theta = torch.atan2(pred_sin, pred_cos)
    angle_deg_2theta = torch.rad2deg(angle_rad_2theta)
    
    axis_deg = angle_deg_2theta.item() / 2.0
    if axis_deg < 0:
        axis_deg += 180.0
        
    final_deg = axis_deg % 180.0
    
    # 在图片上绘制预测的轴线 (可视化)
    draw = ImageDraw.Draw(final_img)
    center_x, center_y = canvas_size // 2, canvas_size // 2
    line_length = canvas_size // 2.5
    
    # 计算直线的终点 (因为是无向轴，画一条穿过中心的线)
    rad = np.radians(final_deg)
    dx = line_length * np.cos(rad)
    dy = line_length * np.sin(rad)  # PIL中Y轴向下，但角度计算习惯上是逆时针
    
    # 画一条红色的预测线，代表模型的无向轴向预测
    draw.line(
        [(center_x - dx, center_y + dy), (center_x + dx, center_y - dy)], 
        fill="red", width=5
    )

    result_text = f"**设定旋转角度**: {angle}°\n\n**模型预测轴向**: <span style='color:red; font-size:1.5em;'>{final_deg:.2f}°</span>\n\n*(注意：红线代表模型预测的对称轴方向，范围为 0°~180°)*"
    
    return final_img, result_text

# 3. 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 🧭 Direction Detector (方向检测器)")
    gr.Markdown(f"**系统状态**: {model_status}")
    gr.Markdown("上传一张透明背景的物体图片，调整旋转角度，模型将实时预测其轴向（红线表示）。")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传透明背景图片 (PNG)", image_mode="RGBA")
            angle_slider = gr.Slider(minimum=0, maximum=360, step=1, value=0, label="模拟旋转角度 (°)")
            predict_btn = gr.Button("开始预测", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(type="pil", label="模型视角与预测结果")
            output_text = gr.Markdown(label="预测数据")
            
    predict_btn.click(
        fn=process_and_predict,
        inputs=[input_image, angle_slider],
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    # 在 launch 中设置 title, theme 并开启 share=True 以便远程访问
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True,
        title="Direction Detector WebUI",
        theme=gr.themes.Soft()
    )
