import argparse
from cleanfid import fid
from core.base_dataset import BaseDataset
from models.metric import inception_score
import numpy as np
import glob
import os
from skimage.measure import compare_psnr, compare_ssim
from PIL import Image
from rich.progress import track


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str,
                        help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str,
                        help='Generate images directory')

    ''' parser configs '''
    args = parser.parse_args()

    psnr = []
    ssim = []
    for gt_path, out_path in track(zip(glob.glob(os.path.join(args.src, "*")), glob.glob(os.path.join(args.dst, "*"))), total=685):
        gt = np.array(Image.open(gt_path))
        out = np.array(Image.open(out_path))
        _psnr = compare_psnr(gt, out)
        _ssim = ssim = compare_ssim(
            gt, out, multichannel=True, gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
        psnr += [_psnr]
        ssim += [_ssim]
    psnr = sum(psnr)/len(psnr)
    ssim = sum(ssim)/len(ssim)
    print(
        f'PSNR: {psnr}\n',
        f'SSIM: {ssim}\n',
    )
    fid_score = fid.compute_fid(args.src, args.dst)
    is_mean, is_std = inception_score(
        BaseDataset(args.dst),
        cuda=True,
        batch_size=8,
        resize=True,
        splits=10,
    )
    print(
        f'FID: {fid_score}\n',
        f'IS: {is_mean} {is_std}\n',
    )
