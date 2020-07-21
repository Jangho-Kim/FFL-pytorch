# Feature Fusion for Online Mutual Knowledge Distillation

This repository is the official implementation of [Feature Fusion for Online Mutual Knowledge Distillation (FFL)](https://arxiv.org/abs/1904.09058). 
The source code is for reproducing the results of Table 1 of the original paper.


## Requirements

To install requirements using [environment.yml](environment.yml) refer to the [documentation.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)


## Training

[train_FFL.py](train_FFL.py) is the code for training FFL  **with Mutual Knowledge Distillation (MKD)**. To train the model(s) in the paper, run this command:

```train
#The results from the original paper can be reproducd by running : 
python train_FFL.py  --lr 0.1 --cu_num 0 --depth 32
```


## Citation
Please refer to the following citation if this repository is useful for your research.

### Bibtex:

```
@article{kim2019feature,
  title={Feature fusion for online mutual knowledge distillation},
  author={Kim, Jangho and Hyun, Minsung and Chung, Inseop and Kwak, Nojun},
  journal={arXiv preprint arXiv:1904.09058},
  year={2019}
}
```

