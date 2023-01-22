# Training using train_general.py
The script is used for training on CIFAR10/100 or Imagenet dataset. 

---
## General
- For single GPU/[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) simply use
  `python train_general.py <args>`
- For [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)
  training I use [torchrun launcher](https://pytorch.org/docs/stable/elastic/run.html) 
  like so:
  ````shell
  torchrun --standalone --nnodes=1 --nproc_per_node=<num gpus (usually 4)> <train_general.py script_path> <args>
  ````
- As default, no orthogonal regularization is used. To run with inter/intra add `+experiment=GSO_<inter/intra>` 
  to the cmd.
  
- You can diable checkpoint saving by running with checkpoint_off=True
  
## Examples
Resnet101 training on ImageNet using with intra-group orthogonalization, distributed:
```shell
torchrun --standalone --nnodes=1 --nproc_per_node=4 train_general.py model=resnet101_imnet.yaml +experiment=GSO_intra
```
resnet110 on CIFAR100, single GPU, inter-group orthogonalization 
(note that the model argument is ommited since it's the default):
```shell
python train_general.py data=cifar100 +experiment=GSO_inter
```
 

