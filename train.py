import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# --- 1. 模型定义 ---
class OrientationNet(nn.Module):
    def __init__(self):
        super(OrientationNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.backbone(x)


# --- 2. 数据处理与增强逻辑 ---
def get_rotated_img(object_img, angle):
    from PIL.Image import Resampling
    rotated_obj = object_img.rotate(angle, expand=True, resample=Resampling.BICUBIC)

    random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    canvas_size = int(max(object_img.size) * 1.5)
    final_img = Image.new("RGB", (canvas_size, canvas_size), random_color)

    paste_x = (canvas_size - rotated_obj.size[0]) // 2
    paste_y = (canvas_size - rotated_obj.size[1]) // 2
    final_img.paste(rotated_obj, (paste_x, paste_y), rotated_obj)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomGrayscale(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(final_img)
    rad = np.radians(angle * 2)
    label = torch.tensor([np.sin(rad), np.cos(rad)], dtype=torch.float32)

    return img_tensor, label


# --- 3. 多图 Dataset 定义 ---
class MultiImageDirectionDataset(Dataset):
    def __init__(self, image_folder, total_size=2000):
        self.total_size = total_size
        self.images = []

        # 遍历文件夹，加载所有 png 图片
        print(f"正在从 {image_folder} 加载图片...")
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(".png"):
                path = os.path.join(image_folder, filename)
                img = Image.open(path).convert('RGBA').resize((200, 200))
                self.images.append(img)

        if not self.images:
            raise ValueError(f"错误：在 {image_folder} 文件夹中没有找到任何 .png 图片！")
        print(f"成功加载了 {len(self.images)} 张基础图片。")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # 随机从图库中抽取一张图片
        base_img = random.choice(self.images)
        angle = random.uniform(0, 360)
        return get_rotated_img(base_img, angle)


# --- 4. 训练主循环 ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrientationNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    IMAGE_FOLDER = 'train_images'
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
        print(f"已创建 {IMAGE_FOLDER} 文件夹，请放入你的透明背景 png 图片后重新运行。")
        exit()

    dataloader = DataLoader(
        MultiImageDirectionDataset(IMAGE_FOLDER, total_size=10000),
        batch_size=384,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    scaler = GradScaler('cuda')
    epoch = 1

    consecutive_good_epochs = 0
    target_loss = 0.005
    patience = 2

    print(f"🚀 开始在 {device} 上训练... (按 Ctrl+C 可随时安全退出并保存模型)")

    # 【新增】初始化 TensorBoard
    writer = SummaryWriter('runs/direction_detector')
    loss_history = []

    # 【新增】使用 try 块包裹整个训练循环
    try:
        while True:
            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

            epoch_loss_sum = 0.0
            batch_count = 0

            for images, labels in pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with autocast('cuda'):
                    outputs = model(images)
                    outputs = F.normalize(outputs, p=2, dim=1)
                    loss = 1.0 - (outputs * labels).sum(dim=1).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss_sum += loss.item()
                batch_count += 1

                pbar.set_postfix(avg_loss=f"{epoch_loss_sum / batch_count:.4f}")

                # 【新增】每个 batch 记录到 TensorBoard
                global_step = (epoch - 1) * len(dataloader) + batch_count
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            avg_epoch_loss = epoch_loss_sum / batch_count
            loss_history.append(avg_epoch_loss)

            # 【新增】每个 epoch 记录一次平均 loss
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)

            if avg_epoch_loss < target_loss:
                consecutive_good_epochs += 1
                print(f"🌟 本轮平均 Loss: {avg_epoch_loss:.4f}。连续达标: {consecutive_good_epochs}/{patience}")
            else:
                if consecutive_good_epochs > 0:
                    print(f"⚠️ 本轮平均 Loss: {avg_epoch_loss:.4f}。Loss 反弹，连续达标计数清零...")
                consecutive_good_epochs = 0

            if consecutive_good_epochs >= patience:
                print(f"\n✅ 已连续 {patience} 轮平均 Loss 低于 {target_loss}！")
                torch.save(model.state_dict(), 'orientation_model.pth')
                print("💾 模型已成功保存为 orientation_model.pth")
                break

            if epoch >= 200:
                print(f"\n⚠️ 达到最大训练轮数 (200)，强制结束训练。")
                torch.save(model.state_dict(), 'orientation_model.pth')
                print("💾 模型已成功保存为 orientation_model.pth")
                break

            epoch += 1

    # 【新增】拦截 Ctrl+C 中断信号
    except KeyboardInterrupt:
        print("\n\n🛑 接收到中止信号 (Ctrl+C)！正在紧急保存当前模型...")
        torch.save(model.state_dict(), 'orientation_model.pth')
        print("💾 保存成功！模型已保存为 orientation_model.pth")

    finally:
        # 【新增】训练结束或中断时，关闭 TensorBoard 并绘制 Loss 曲线图
        writer.close()
        if loss_history:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig('loss_curve.png')
            print("📈 已将 Loss 曲线保存为 loss_curve.png")