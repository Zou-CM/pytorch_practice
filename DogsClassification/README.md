#### 介绍

该项目借用了kaggle上的Dog Breed Identification比赛上的数据集，做了个小的分类实战，算是入个pytorch的门，kaggle链接：[https://www.kaggle.com/c/dog-breed-identification](https://www.kaggle.com/c/dog-breed-identification)

#### 项目的完整目录

- checkpoints
  - model.pth
- dataset
  - \__init__.py
  - MyDataset.py
- dev
- train
- labels.csv
- sample_submission.csv
- config.py
- main.py
- utils.py
- README.md

#### 使用方法

1. 下载好数据集，解压到最外层目录
2. 运行utils.py对数据进行分组和重命名
3. 运行main.py