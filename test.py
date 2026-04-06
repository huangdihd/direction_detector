import os
import torch
from PIL import Image
from train import OrientationNet, get_rotated_img

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型并加载权重
    print("正在加载模型...")
    model = OrientationNet().to(device)
    try:
        model.load_state_dict(torch.load('orientation_model.pth', weights_only=True))
    except FileNotFoundError:
        print("❌ 找不到 orientation_model.pth，请先运行 train.py 进行训练！")
        exit()

    model.eval()
    print("✅ 模型加载完毕！")

    # 2. 读取测试图片列表
    IMAGE_FOLDER = 'test_images'
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.png')]

    if not image_files:
        print(f"❌ {IMAGE_FOLDER} 文件夹中没有找到任何 png 图片！")
        exit()

    print("\n--- 🎯 方向检测测试 ---")
    while True:
        try:
            # 打印可用的图片菜单
            print("\n可用图片:")
            for idx, f in enumerate(image_files):
                print(f"[{idx}] {f}")

            # 选择图片
            img_input = input("\n请选择图片编号 (输入 'q' 退出): ")
            if img_input.lower() == 'q': break

            img_idx = int(img_input)
            selected_file = image_files[img_idx]
            object_img = Image.open(os.path.join(IMAGE_FOLDER, selected_file)).convert('RGBA').resize((200, 200))

            # 输入角度
            angle_input = input(f"请输入你想将 '{selected_file}' 旋转的角度: ")
            if angle_input.lower() == 'q': break
            test_angle = float(angle_input)

            # 生成测试图并推理
            test_img, _ = get_rotated_img(object_img, test_angle)
            test_img = test_img.unsqueeze(0).to(device)

            # === 在 test.py 中修改 ===

            with torch.no_grad():
                output = model(test_img)

            pred_sin = output[0, 0].cpu()
            pred_cos = output[0, 1].cpu()

            # 1. 算出来的结果是 2θ (范围是从 -180度 到 180度)
            angle_rad_2theta = torch.atan2(pred_sin, pred_cos)
            angle_deg_2theta = torch.rad2deg(angle_rad_2theta)

            # 2. 除以 2 还原回我们真正的物体轴向
            axis_deg = angle_deg_2theta.item() / 2.0

            # 3. 把负数角度转成正数，并将其限制在 0 ~ 180 度之间
            if axis_deg < 0:
                axis_deg += 180.0

            # 加上保险的取模运算（确保绝对在 0-180 之间）
            final_deg = axis_deg % 180.0

            print(f"=====================================")
            print(f"🖼️ 当前测试图片: {selected_file}")
            print(f"📐 预测的无向轴向: {final_deg:.2f}° (代表该倾斜率上的直线)")
            print(f"=====================================")

        except (ValueError, IndexError):
            print("⚠️ 输入无效，请输入正确的编号或数字。")
        except Exception as e:
            print(f"⚠️ 发生错误: {e}")