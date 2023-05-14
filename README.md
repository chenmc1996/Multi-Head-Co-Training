# Multi-Head Co-Training (PyTorch)
<p align="center">
<img src="src/diagram.png" width="500">
</p>


## Experiments

First, please make sure your pytorch version is above 1.6.
Then run the train.py, such as

```
$ python train.py --num_labels 4000 --save_name cifar10_4000 --dataset cifar10 --overwrite --data_dir path-to-your-data
```

## Requirements

- Python >= 3.6
- PyTorch >= 1.6
- CUDA
- Numpy

### Results on semi-supervised learning benchmarks
- Test Accuracy(%) on CIFAR10

|\# labels         |250    |1000    |4000    |
|-------------------|-------|-------|-------|
|Multi-Head Co-Training   |4.98±0.30  | 4.74±0.16 |3.84±0.09  |
  
### Results on open-set semi-supervised learning benchmarks
- Test Accuracy(%) on CIFAR10 with only 60% know classes

|\# labels         |50    |100    |400    |
|-------------------|-------|-------|-------|
|Multi-Head Co-Training   |5.8±0.9  | 5.3±0.9 |4.4±0.9  |

## Reference
Part of codes in this repository are modified from:
* ["https://github.com/google-research/fixmatch"](https://github.com/google-research/fixmatch),
* ["https://github.com/LeeDoYup/FixMatch-pytorch"](https://github.com/LeeDoYup/FixMatch-pytorch),
* ["https://github.com/ildoonet/pytorch-randaugment"](https://github.com/ildoonet/pytorch-randaugment).
