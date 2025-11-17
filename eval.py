import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from data import test_dataloader
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def _eval(model, args):


    state_dict = torch.load(args.test_model, map_location='cpu')
    model.load_state_dict(state_dict['model'])


    device = torch.device('cpu')
    model.to(device)
    model.eval()


    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    to_pil_image = transforms.ToPILImage()


    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            start_time = time.time()
            pred = model(input_img)[2]
            elapsed = time.time() - start_time

            pred_clip = torch.clamp(pred, 0, 1)

            # 保存图像
            if args.save_image:
                os.makedirs(args.result_dir, exist_ok=True)
                save_name = os.path.join(args.result_dir, name[0])
                pred_image = to_pil_image(pred_clip.squeeze(0).cpu())
                pred_image.save(save_name)

            print(f"[{iter_idx + 1}/{len(dataloader)}] 已完成: {name[0]}  用时: {elapsed:.4f}s")

    print(" 测试全部完成！")


