# Source-free UDA for Panoramic Semantic Segmentation
### (360SFUDA & 360SFUDA++)
### Semantics, Distortion, and Style Matter: Towards Source-free UDA for Panoramic Segmentation, CVPR 2024. [pdf](https://arxiv.org/pdf/2403.12505)
<img width="612" alt="image" src="https://github.com/zhengxuJosh/360SFUDA/assets/49426295/6962a1a0-7fb8-49ce-a895-2d9bb1b1449b">

### 360SFUDA++: Towards Source-free UDA for Panoramic Segmentation with Reliable Prototypical Adaptation, Arxiv 2024. [pdf]()
<img width="783" alt="image" src="https://github.com/zhengxuJosh/360SFUDA/assets/49426295/f9837259-0033-4b59-be6a-f4381c3f4eb4">

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
| BackBone  | Outdoor C-to-D | Indoor Spin-to-Span |
|--------|--------|--------|
| SegFormer-B1 | 50.19 [model](https://drive.google.com/file/d/1OjIS5txbyy2JJZ8_hVQVS0dosJJ7T75S/view?usp=drive_link) |  |
| SegFormer-B2 | | 68.84 [model](https://drive.google.com/file/d/1IlsHVsInhPzu3c8qdyKnlQpEKXkzcP-J/view?usp=drive_link) |

