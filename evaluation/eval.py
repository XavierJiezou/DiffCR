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

<<<<<<< HEAD
=======
<<<<<<< HEAD
    # psnr = []
    # ssim = []
    # for gt_path, out_path in track(zip(glob.glob(os.path.join(args.src, "*")), glob.glob(os.path.join(args.dst, "*"))), total=685):
    #     gt = np.array(Image.open(gt_path))
    #     out = np.array(Image.open(out_path))
    #     _psnr = compare_psnr(gt, out)
    #     _ssim = ssim = compare_ssim(
    #         gt, out, multichannel=True, gaussian_weights=True, use_sample_covariance=False, sigma=1.5)
    #     psnr += [_psnr]
    #     ssim += [_ssim]
    # psnr = sum(psnr)/len(psnr)
    # ssim = sum(ssim)/len(ssim)
    # print(
    #     f'PSNR: {psnr}\n',
    #     f'SSIM: {ssim}\n',
    # )
=======
>>>>>>> a13ebef0541ec6fe26f52d5598a109d848a51b9c
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
<<<<<<< HEAD
=======
>>>>>>> ea6fb03699555768d2adcb4bc12908f82098912d
>>>>>>> a13ebef0541ec6fe26f52d5598a109d848a51b9c
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

"""
1. Palette-single-v1.0[epoch=9000]
- ↓PSNR: 14.62888054092963 
- ↑SSIM: 0.3244409937711537
- ↑FID: 103.76912140940459 
- ↑IS: 3.008023500263284 0.37180917089147886 
2. Palette-single-v2.0[epoch=9000]
- ↑PSNR: 9.25222585999201
- ↑SSIM: 0.44265318616227906
- ↓FID: 153.20687384326067
- ↑IS: 2.8962933749979887 0.23164383408662623
3. PMAA-single[训练100个epoch]
- ↑PSNR:
- ↑SSIM: 
- ↓FID: 
- ↑IS: 
4. Palette-multiple[TODO]
- ↑PSNR:
- ↑SSIM: 
- ↓FID: 
- ↑IS: 
5. Palette-multiple[epoch=3000+dpm-solver]
<<<<<<< HEAD
=======
<<<<<<< HEAD
- PSNR: 12.076057469841748
- SSIM: 0.6256649852628741
6. Palette-multiple[epoch=4000+dpm-solver]
PSNR: 12.070057176915736
SSIM: 0.6076268317926176
FID: 89.40558513039605
IS: 2.6435228768332335 0.48625733502371776
6. Palette-multiple[epoch=3000+ddim1000]
- PSNR: 11.962246446296197
- SSIM: 0.6039557910557187
=======
>>>>>>> a13ebef0541ec6fe26f52d5598a109d848a51b9c
- ↑PSNR: 12.076057469841748
- ↑SSIM: 0.6256649852628741
- ↓FID: 
- ↑IS: 
6. Palette-multiple[epoch=3000+ddim1000]
PSNR: 11.962246446296197
SSIM: 0.6039557910557187
<<<<<<< HEAD
=======
>>>>>>> ea6fb03699555768d2adcb4bc12908f82098912d
>>>>>>> a13ebef0541ec6fe26f52d5598a109d848a51b9c
"""
