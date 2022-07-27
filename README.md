# ShuffleNetV2Plus (size=Small)

## ImageNet training with PyTorch

This implements training of ShuffleNetV1 on the ImageNet dataset, mainly modified from [Github](https://github.com/pytorch/examples/tree/master/imagenet).

## ShuffleNetV2Plus Detail

Base version of the model from [the paper author's code on Github](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2%2B).
The training script is adapted from [the ShuffleNetV2 script on Gitee](https://gitee.com/ascend/modelzoo/tree/master/built-in/PyTorch/Official/cv/image_classification/Shufflenetv2_for_PyTorch).

## Requirements

- pytorch_ascend, apex_ascend, tochvision
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

## Training
一、训练流程：
        
单卡训练流程：

    1.安装环境
    2.修改参数device_id（单卡训练所使用的device id），为训练配置device_id，比如device_id=0
    3.开始训练
        bash ./test/train_full_1p.sh  --data_path=数据集路径    # 精度训练
        bash ./test/train_performance_1p.sh  --data_path=数据集路径  # 性能训练


​    
多卡训练流程

    1.安装环境
    2.修改参数device_id_list（多卡训练所使用的device id列表），为训练配置device_id，例如device_id=0,1,2,3,4,5,6,7
    3.执行train_full_8p.sh开始训练
        bash ./test/train_full_8p.sh  --data_path=数据集路径         # 精度训练
        bash ./test/train_performance_8p.sh  --data_path=数据集路径  # 性能训练

二、测试结果
    
训练日志路径：网络脚本test下output文件夹内。例如：

      test/output/devie_id/train_${device_id}.log           # 训练脚本原生日志
      test/output/devie_id/ShuffleNetV1_bs8192_8p_perf.log  # 8p性能训练结果日志
      test/output/devie_id/ShuffleNetV1_bs8192_8p_acc.log  # 8p精度训练结果日志

训练模型：训练生成的模型默认会写入到和test文件同一目录下。当训练正常结束时，checkpoint.pth.tar为最终结果。



## ShufflenetV2Plus training result

| Acc@1    | FPS       | Npu_nums| Epochs   | Type     |
| :------: | :------:  | :------ | :------: | :------: |
| 73.132   | 6306      | 8       | 360      | O2       |

备注：由于模型开发中发现NPU上clamp算子反向错误，以上结果为使用自行编写的clamp函数训练获得。见blocks.py中的注释掉的函数clamp。



```
# ModelArts训练
# 正确输出了pth、onnx模型文件
bash run_train.sh [/path/to/code/in/obs] 'code/modelarts/train_start.py' '/tmp/log/training.log' --data_url=[path/to/data/in/obs] --train_url=[/path/to/output/in/obs] --epochs=1 --batch-size=4
# 验收结果： OK 
# 备注： 目标输出结果无误

# 模型转换
cd /infer/convert
bash convert_om.sh ../data/model/shufflenetv2plus_npu.onnx ../data/model/shufflenetv2plus_npu

# SDK推理
# 运行SDK推理：
cd /infer/sdk
bash run.sh ../data/input/imagenet/val/ sdk_pred_result.txt
# 测试推理精度：
python3 ../util/task_metric.py sdk_pred_result.txt ../data/config/val_label.txt sdk_pred_result.acc.json 5
# 验收结果： OK 
# 备注： 输出了正确的推理结果、推理精度达标
# MxBase推理
# 编译可执行程序：
cd /infer/mxbase
bash build.sh
# 预处理数据集：
python3 ../util/preprocess.py ../data/input/imagenet/val/ binfile
# 运行推理程序：
./build/shufflenetv2plus ./binfile/
# 测试推理精度：
python3 ../util/task_metric.py mx_pred_result.txt ../data/config/val_label.txt sdk_pred_result.acc.json 5
# 验收结果： OK 
# 备注： 输出了正确的推理结果、推理精度达标
```