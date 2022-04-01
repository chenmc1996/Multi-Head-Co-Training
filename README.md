# Semi-Supervised Learning with Multi-Head Co-Training (PyTorch)

<img src="src/diagram.png">


## Experiments

First, please make sure your pytorch version is above 1.6.
Then run the train.py, such as

```
$ python train.py --num_labels 4000 --save_name cifar10_4000 --dataset cifar10 --overwrite --data_dir path-to-your-data
```
## Reference
Most of codes in this repository are modified from:
* ["https://github.com/google-research/fixmatch"](https://github.com/google-research/fixmatch),
* ["https://github.com/LeeDoYup/FixMatch-pytorch"](https://github.com/LeeDoYup/FixMatch-pytorch),
* ["https://github.com/ildoonet/pytorch-randaugment"](https://github.com/ildoonet/pytorch-randaugment).
