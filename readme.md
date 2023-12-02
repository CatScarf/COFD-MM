# Readme

## Required packages

```
pytorch >= 1.13  # Older versions of pytorch do not support stable sorting of argsort.
```

## Datasets

You can get the dataset from the following link and extract it to the directory of the code.

https://drive.google.com/file/d/1dihV0_uT2mpUACs7it7Zif7EUE2CGxmU/view?usp=sharing

## Run

```bash
# Make sure the dataset is properly downloaded and unpacked before running.
python train.py
```

## Cite

```text
@inproceedings{10.1145/3581783.3613780,
author = {Hu, Jinzhang and Hu, Ruimin and Wang, Zheng and Li, Dengshi and Wu, Junhang and Ren, Lingfei and Zang, Yilong and Huang, Zijun and Wang, Mei},
title = {Collaborative Fraud Detection: How Collaboration Impacts Fraud Detection},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3613780},
doi = {10.1145/3581783.3613780},
abstract = {Collaborative fraud has become increasingly serious in telecom and social networks, but is hard to detect by traditional fraud detection methods. In this paper, we find a significant positive correlation between the increase of collaborative fraud and the degraded detection performance of traditional techniques, implying that those fraudsters that are difficult to detect with traditional methods are often collaborative in their fraudulent behavior. As we know, multiple objects may contact a single target object over a period of time. We define multiple objects with the same contact target as generalized objects, and their social behaviors can be combined and processed as the social behaviors of one object. We propose Fraud Detection Model based on Second-order and Collaborative Relationship Mining (COFD), exploring new research avenues for collaborative fraud detection. Our code and data are released at https://github.com/CatScarf/COFD-MM https://github.com/CatScarf/COFD-MM.},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {8891–8899},
numpages = {9},
keywords = {collaborative fraud, graph neural networks, fraud detection},
location = {, Ottawa ON, Canada, },
series = {MM '23}
}
```
