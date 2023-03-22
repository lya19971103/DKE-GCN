# DKE-GCN

This is the implementation of Decoupled Knowledge Emedded Graph Convolutional Network for Skeleton-based Action Recognition, which has been submitted to ICCV2023. We have verified the effectiveness of our distillation paradigm in CTR-GCN and MS-G3D.

To emphasize, this is only a preliminary version, which will be further supplemented in a few weeks.

Currently, we only provide some pre-trained models based on CTR-GCN to ensure authenticity.  We are sorting out the project and uploading it within several weeks, including the all code and more pre-trained models.



# Overall Architecture

![](https://github.com/lya19971103/DKE-GCN/blob/main/OA.png)

# Dependencies

+ Python >= 3.6
+ PyTorch >= 1.2.0
+ PyYAML, tqdm, tensorboardX
+ Some dependencies required by the teacher model.

# Data Preparation

There are 4 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- Kinetics 400 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120 

The same as CTR-GCN

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract  to `./data/nturgbd_raw`

```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

3. Generating the skeleton data:

```
cd ./data/ntu # or cd ./data/ntu120
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

# Pre-trained Model

We only provide some pre-trained DKE-GCN  based on teacher CTR-GCN and student S8 .  More pre-trained models based on teacher MS-G3D will be uploaded  within several weeks.  

Link: https://pan.baidu.com/s/1dQhKbaS-rp5pNteS9H8ERg 
Password: uyvh

The student S8 is the shrinking version of [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) . You can test our distilled student  by :

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

