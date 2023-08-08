import os
from glob import glob
import numpy as np
import lpips
import torch
import subprocess
import sys
from tqdm import tqdm


def calculate_fid(src, dst):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform == "win32":
        num_workers = 0
    else:
        num_workers = 8
    subprocess.run(
        f"python -m pytorch_fid  {src} {dst} --device {device} --batch-size 8 --num-workers {num_workers}",
    )


def calculate_lpips(src, dst):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_list = []
    for i, j in tqdm(zip(glob(src + "/*"), glob(dst + "/*")), total=len(glob(src + "/*"))):
        img1 = lpips.im2tensor(lpips.load_image(i)).to(device)
        img2 = lpips.im2tensor(lpips.load_image(j)).to(device)
        lpips_list.append(loss_fn_alex(img1, img2).item())
    print(f"LPIPS:{np.mean(lpips_list)}")


if __name__ == "__main__":
    src = "../0/GT"
    dst = "../0/Out"
    calculate_fid(src, dst)
    calculate_lpips(src, dst)
