<div align="center">
<h1 align="center">DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal from Optical Satellite Images</h1>
<p align="center">This repository is the official PyTorch implementation of the TGRS 2024 paper DiffCR.</p>

[![arXiv Paper](https://img.shields.io/badge/arXiv-2308.04417-B31B1B)](https://arxiv.org/abs/2308.04417)
[![Project Page](https://img.shields.io/badge/Project%20Page-DiffCR-blue)](https://xavierjiezou.github.io/DiffCR/)
[![HugginngFace Models](https://img.shields.io/badge/🤗HugginngFace-Models-orange)](https://huggingface.co/XavierJiezou/diffcr-models)
[![HugginngFace Datasets](https://img.shields.io/badge/🤗HugginngFace-Datasets-orange)](https://huggingface.co/datasets/XavierJiezou/diffcr-datasets)

![DiffCR](image/README/diffcr.jpg)

</div>

## News

- [2023/07/30] Code release.
- [2023/07/16] PMAA got accepted by ECAI 2023.
- [2023/03/29] PMAA is on arXiv now.

## Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

To download datasets:

- Sen2_MTC_Old: [multipleImage.tar.gz](https://doi.org/10.7910/DVN/BSETKZ)

- Sen2_MTC_New: [CTGAN.zip](https://drive.google.com/file/d/1-hDX9ezWZI2OtiaGbE8RrKJkN1X-ZO1P/view?usp=share_link)

## Training

To train the models in the paper, run these commands:

```train
python run.py -p train -c config/ours_sigmoid.json
```

<!-- >📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Testing

To test the pre-trained models in the paper, run these commands:

```bash
python run.py -p test -c config/ours_sigmoid.json
```

## Evaluation

To evaluate my models on two datasets, run:

```bash
python evaluation/eval.py -s [ground-truth image path] -d [predicted-sample image path]
```

<!-- >📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below). -->

<!-- ## Pre-trained Models

You can download pretrained models here:

- Our awesome model trained on Sen2_MTC_Old: [diffcr_old.pth](/pretrained/diffcr_old.pth)
- Our awesome model trained on Sen2_MTC_New: [diffcr_new.pth](/pretrained/diffcr_new.pth) -->

<!-- >📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. -->

## Citation 

If you use our code or models in your research, please cite with:

```
@ARTICLE{diffcr,
  author={Zou, Xuechao and Li, Kai and Xing, Junliang and Zhang, Yu and Wang, Shiying and Jin, Lei and Tao, Pin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal From Optical Satellite Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
}
```

## Acknowledgments

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=Janspiry&repo=Palette-Image-to-Image-Diffusion-Models)](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=openai&repo=guided-diffusion)](https://github.com/openai/guided-diffusion)
