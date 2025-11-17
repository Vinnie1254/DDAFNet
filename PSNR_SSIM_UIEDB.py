import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS
from torch_fidelity import calculate_metrics
import pandas as pd

# =========================
# 路径设置
# =========================
gt_folder = '/home/gxy/PycharmProjects/Model_main_water/datasets/UIEDB/test/target'
pred_folder = '/home/gxy/PycharmProjects/Model_main_water/results/OURNet_water/UIEDB_test_visual_1020'

# =========================
# 图像预处理
# =========================
to_tensor = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LPIPS模型（AlexNet版本）
lpips_fn = LPIPS(net='alex').to(device)

# =========================
# 记录指标
# =========================
psnr_list, ssim_list, lpips_list, names_list = [], [], [], []

# =========================
# 逐图像计算 PSNR / SSIM / LPIPS
# =========================
gt_images = sorted(os.listdir(gt_folder))
pred_images = sorted(os.listdir(pred_folder))


print(f"GT图像与预测图像一一对应")
for name in tqdm(gt_images, desc="计算各项指标中..."):
    gt_path = os.path.join(gt_folder, name)
    pred_path = os.path.join(pred_folder, name)
    if not os.path.exists(pred_path):
        print(f" 跳过 {name}, 因为预测文件夹中未找到对应文件。")
        continue

    # 打开图像并转换为 RGB
    gt_img = Image.open(gt_path).convert('RGB')
    pred_img = Image.open(pred_path).convert('RGB')

    gt_np = np.array(gt_img).astype(np.float32)
    pred_np = np.array(pred_img).astype(np.float32)

    # --- PSNR (RGB)
    psnr_val = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
    psnr_list.append(psnr_val)

    # --- SSIM (RGB)
    ssim_val = structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255)
    ssim_list.append(ssim_val)

    # --- LPIPS
    gt_t = to_tensor(gt_img).unsqueeze(0).to(device)
    pred_t = to_tensor(pred_img).unsqueeze(0).to(device)
    lpips_val = lpips_fn(gt_t, pred_t).mean().item()
    lpips_list.append(lpips_val)

    names_list.append(name)

# =========================
# FID 计算
# =========================
print("\n计算FID中...")
metrics = calculate_metrics(
    input1=pred_folder,
    input2=gt_folder,
    cuda=torch.cuda.is_available(),
    isc=False, fid=True, kid=False,
    verbose=False
)
fid_val = metrics['frechet_inception_distance']

# =========================
# 保存详细结果（每张图像）
# =========================
df = pd.DataFrame({
    'Image': names_list,
    'PSNR': psnr_list,
    'SSIM': ssim_list,
    'LPIPS': lpips_list
})

csv_path = os.path.join(pred_folder, "metrics_each_image.csv")
df.to_csv(csv_path, index=False)

# =========================
# 输出平均结果
# =========================
mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)
mean_lpips = np.mean(lpips_list)

print("\n========== 平均指标结果 ==========")
print(f"平均 PSNR: {mean_psnr:.4f}")
print(f"平均 SSIM: {mean_ssim:.4f}")
print(f"平均 LPIPS: {mean_lpips:.4f}")
print(f"FID: {fid_val:.4f}")

# =========================
# 保存到 txt 文件
# =========================
save_path = os.path.join(pred_folder, "metrics_results.txt")
with open(save_path, "w") as f:
    f.write(f"平均 PSNR: {mean_psnr:.4f}\n")
    f.write(f"平均 SSIM: {mean_ssim:.4f}\n")
    f.write(f"平均 LPIPS: {mean_lpips:.4f}\n")
    f.write(f"FID: {fid_val:.4f}\n")

print(f"\n平均结果已保存至: {save_path}")
print(f" 每张图像详细结果已保存至: {csv_path}")
