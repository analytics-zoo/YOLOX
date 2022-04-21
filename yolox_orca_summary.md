# 1.Modify to CPU version
To get the CPU version, we need to replace `cuda` with `cpu`.  

For example:  
`x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()`  
modify to:  
`x = torch.ones(1, 3, test_size[0], test_size[1]).cpu()`  

And yolox use `DataPrefetcher` to load the dataloader. But `DataPrefetcher` use cuda stream to get the stream. So we need to replace `DataPrefetcher` and find another way to load the data.

## The main modification classes as follows:
tools/train.py  
yolox/core/launch.py  
yolox/core/trainer.py  
yolox/evaluators/coco_evaluator.py  
yolox/exp/yolox_base.py  
yolox/models/yolo_head.py  

# 2. Run orca distributed in yolox
## three issues:
### 1. Orca do not support preprocess inputs before training in forward: 
https://github.com/intel-analytics/BigDL/issues/4410 

### 2. yolox does not require loss creator when running on orca: 
https://github.com/intel-analytics/BigDL/issues/4412 

### 3. Orca only supports basic metrics, some customized metrics do not support: 
https://github.com/intel-analytics/BigDL/issues/4414

### 4. Orca do not support redirect sys output in Ray Mode
Yolox redirect system out in logger.py as follows:
```python
def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger
```
But Orca do not support it when use torch_distributed as you backend


# 3. Problem encountered
### (1). yolox does not support COCO128
Using yolox training COCO128 dataset will result in some errors, such as getting incorrect output from evaluate function. 
So we need to train on the full COCO dataset. The full COCO dataset has three folders: annotations, train2017, val2017. There are 118287 images in train2017 and 5000 images in val2017. We need use full COCO dataset instead of mini dataset

### (2). some problems in train_loader
cause by :  
in `exp/yolox_base.py`，use `train_loader = DataLoader(self.dataset, **dataloader_kwargs)` to get the DataLoader. This method assigns `train_loader.batch_sampler.batch_size` a value, but if we use orca, What orca actually used is `train_loader.batch_size`.   
errors:
```bash
  File "/home/ryan/anaconda3/envs/yolox/lib/python3.7/site-packages/bigdl/orca/learn/pytorch/pytorch_spark_estimator.py", line 134, in fit
    assert batch_size is None and data.batch_size > 0, "When using PyTorch Dataloader as " \
           │                      │    └ None
           │                      └ <yolox.data.dataloading.DataLoader object at 0x7f7da8878390>
           └ None

TypeError: '>' not supported between instances of 'NoneType' and 'int'
Stopping orca context
```
So we need to give `train_loader.batch_size` a value. In dataloading.py，add `self.batch_size = self.batch_sampler.batch_size`

### (3). In spark and torch_distributed backend
In spark and torch_distributed backend, we must use function as parameters instead of concrete value. So we must implement functions such as `model_creator(config),optimizer_creator(model, config),train_loader_creator(config, batch_size),test_loader_creator(config, batch_size)`
```python
def model_creator(config):
    args = config['args']
    exp = get_exp(args.exp_file,args.name)
    model = exp.get_model()
    is_distributed = config['is_distributed']
    if is_distributed:
        model = DDP(model, device_ids=[get_local_rank()], broadcast_buffers=False)
    return model  
```

### (4) orca's performance is slower than CPU
cause: orca doesn't use all the core in this computer, so you must set the cores in `init_orca_context(cores="*")`

# 4. Summary of results
## use GPU to train original yolox

```bash
1 epoch:

Average forward time: 7.77 ms, Average NMS time: 0.66 ms, Average inference time: 8.43 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.004

2022-03-31 19:53:52 | INFO     | yolox.core.trainer:351 - Save weights to ./YOLOX_outputs/yolox_s
2022-03-31 19:53:52 | INFO     | yolox.core.trainer:351 - Save weights to ./YOLOX_outputs/yolox_s
2022-03-31 19:53:52 | INFO     | yolox.core.trainer:195 - Training of experiment is done and the best AP is 0.01

--------------------------------------------
2 eopch
Average forward time: 7.79 ms, Average NMS time: 0.57 ms, Average inference time: 8.36 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.008
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.020
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.004
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.019
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.033
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.036
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.007
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.025
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.050

 ---------------------------------------
 3 epoch

 Average forward time: 7.77 ms, Average NMS time: 0.56 ms, Average inference time: 8.33 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.026
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.020
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.026
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.035
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.052
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.081
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.084
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.108

2022-03-31 21:53:29 | INFO     | yolox.core.trainer:351 - Save weights to ./YOLOX_outputs/yolox_s
2022-03-31 21:53:29 | INFO     | yolox.core.trainer:351 - Save weights to ./YOLOX_outputs/yolox_s
2022-03-31 21:53:29 | INFO     | yolox.core.trainer:195 - Training of experiment is done and the best AP is 2.61
```

## use CPU to train COCO on yolox (4 epoch--26h)
### problem： AP is lower than training on GPU, only 0.01, The same problem arises with training on orca. 
The guess is that yolox itself does not support training using the CPU 

issue link: 

https://github.com/Megvii-BaseDetection/YOLOX/issues/1202

result：
```bash
Evaluate annotation type *bbox*2022-04-01 17:35:46 | INFO     | yolox.core.trainer:328 - 
Average forward time: 42.98 ms, Average NMS time: 0.69 ms, Average inference time: 43.67 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001

2022-04-01 17:35:46 | INFO     | yolox.core.trainer:338 - Save weights to ./YOLOX_outputs/yolox_s
2022-04-01 17:35:46 | INFO     | yolox.core.trainer:338 - Save weights to ./YOLOX_outputs/yolox_s
2022-04-01 17:35:47 | INFO     | yolox.core.trainer:182 - Training of experiment is done and the best AP is 0.01

```

