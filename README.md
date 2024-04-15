# Source-free UDA for Panoramic Semantic Segmentation
### (360SFUDA & 360SFUDA++)
[Semantics, Distortion, and Style Matter: Towards Source-free UDA for Panoramic Segmentation](https://arxiv.org/pdf/2403.12505), CVPR 2024.
<img width="612" alt="image" src="https://github.com/zhengxuJosh/360SFUDA/assets/49426295/6962a1a0-7fb8-49ce-a895-2d9bb1b1449b">

[360SFUDA++: Towards Source-free UDA for Panoramic Segmentation with Reliable Prototypical Adaptation](), Arxiv 2024. 

## Update
[04/2024], pre-trained weights and evalutation code on C-to-D scenario for 360SFUDA++ are released.


## Environments
```
pip install -r requirements.txt
```

## Data Preparation
Used Datasets: 
[Cityscapes](https://www.cityscapes-dataset.com/) / [DensePASS](https://github.com/chma1024/DensePASS) / [SynPASS](https://drive.google.com/file/d/1u-5J13CD6MXpWB53apB-L6kZ3hK1JR77/view?usp=sharing) / [Stanford2D3D](https://arxiv.org/abs/1702.01105)

```
datasets/
├── cityscapes
│   ├── gtFine
│   └── leftImg8bit
├── Stanford2D3D
│   ├── area_1
│   ├── area_2
│   ├── area_3
│   ├── area_4
│   ├── area_5a
│   ├── area_5b
│   └── area_6
├── SynPASS
│   ├── img
│   │   ├── cloud
│   │   ├── fog
│   │   ├── rain
│   │   └── sun
│   └── semantic
│       ├── cloud
│       ├── fog
│       ├── rain
│       └── sun
├── DensePASS
│   ├── gtFine
│   └── leftImg8bit
```
## Pre-trained Weights of 360SFUDA++
| BackBone  | C-to-D | Weights |
|--------|--------|--------|
| SegFormer-B1 | 50.19 | [model](https://drive.google.com/file/d/1OjIS5txbyy2JJZ8_hVQVS0dosJJ7T75S/view?usp=drive_link) |  
| SegFormer-B2 | 52.99 | [model](https://drive.google.com/file/d/1g0EvvpYDEBWaTKynKbcUK8OPkfqVHFHk/view?usp=drive_link) |

## Evaluation
Download the pretrained weights in C-to-D scenarios from GoogleDirve.

```
python val_city.py
```
## References
We appreciate the previous open-source works: [Trans4PASS](https://github.com/jamycheung/Trans4PASS) / [SegFormer](https://github.com/NVlabs/SegFormer)

## Citations
If you are interested in this work, please cite the following works:
```
@article{zheng2024semantics,
  title={Semantics, Distortion, and Style Matter: Towards Source-free UDA for Panoramic Segmentation},
  author={Zheng, Xu and Zhou, Pengyuan and Vasilakos, Athanasios and Wang, Lin},
  journal={arXiv preprint arXiv:2403.12505},
  year={2024}
}
```
