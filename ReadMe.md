# 星系分类模型算法部分

Update: 2024/05/13

Author: weishirui

### 文件夹结构
```
galaxy_classification
    |—— dataset/    # 数据集
    |   |—— Galaxy10_DECals_predict/    # 基于Galaxy10 DECals的测试数据集，包含十张图片，一类一张，用于测试
    |—— model/  # 模型
    |   |—— zoobot/ # 基于zoobot v2框架的模型
    |       |—— convnext_nano/  # encoder模型文件
    |       |—— finetune_model/ # 微调后的模型权重文件
    |—— outputs/    # 输出结果保存
    |—— lightning_logs/ # 日志（自动生成）
    |—— galaxy_classification_main.py   # 主函数，测试全流程
```
---
### Quick Start
#### 环境配置

- zoobot的安装
ref: https://github.com/mwalmsley/zoobot?tab=readme-ov-file#installation
```
# 进入本项目文件夹
cd galaxy_classification 

# 将zoobot下载至本项目文件夹中
git clone https://github.com/mwalmsley/zoobot.git 

# 无gpu版本安装，将会下载zoobot相关依赖【相关库列表可查看https://github.com/mwalmsley/zoobot/blob/main/setup.py】
pip install -e "zoobot[pytorch-cpu]" --extra-index-url https://download.pytorch.org/whl/cpu 
```
- 若上面装完了应该能启动了，启动还缺什么库，就另外安装吧...
```
pip install -r requirements.txt
```
#### Run
- 前提：需要魔改zoobot的库函数以使其能够跑起来：）否则会尝试连接huggingface，然后报443

文件路径：本项目文件夹下的zoobot库（刚才下载的）：./zoobot/zoobot/pytorch/training/finetune.py
```
代码第125-128行：
if name is not None:
    assert encoder is None, 'Cannot pass both name and encoder to use'
    self.encoder = timm.create_model(name, num_classes=0, pretrained=True)
    self.encoder_dim = self.encoder.num_features
```
改为：
```
if name is not None:
    assert encoder is None, 'Cannot pass both name and encoder to use'
    # 修改这！！！
    # self.encoder = timm.create_model(name, num_classes=0, pretrained=True)
    self.encoder = timm.create_model('convnext_nano', num_classes=0,pretrained=True, pretrained_cfg_overlay=dict(file='/root/demo/model/zoobot/convnext_nano/pytorch_model.bin')) # 路径请找到本项目文件夹中的./model/zoobot/convnext_nano/pytorch_model.bin，然后使用绝对路径
    self.encoder_dim = self.encoder.num_features
```

- 主函数：galaxy_classification_main.py 

输入：修改输入参数
- 图片路径（image_path），使用./dataset/Galaxy10_DECals_predict中的图片，一次只分类一张图片
- 模型权重路径（weight_path），使用./model/zoobot/finetune_model/FinetuneableZoobotClassifier.ckpt，目前只有这个模型

输出：
- predict_res：对于输入的图片的分类结果
- predict_probability：该分类结果的置信度
- result_dict：所有10类结果的置信度（字典形式）
```
if __name__ == '__main__':
    image_path = './dataset/Galaxy10_DECals_predict/Barred Spiral_1.png'
    model_path = './model/zoobot/finetune_model/FinetuneableZoobotClassifier.ckpt'
    predict_res,predict_probability,result_dict = galaxy_classify_zoobot(image_path,model_path)
    print("==============================")
    print("galaxy type:",predict_res)
    print("classfication confidence:",str(predict_probability))
    print("all types results:")
    print(result_dict)
```
- 运行
```
python galaxy_classification_main.py
```
- 输出结果：
```
==============================
galaxy type: Barred Spiral
classfication confidence: 0.99669313
all types results:
{'Disturbed': 0.0012529207, 'Merging': 6.805567e-05, 'Round Smooth': 0.0001473881, 'In-between Round Smooth': 9.510765e-05, 'Cigar Shaped Smooth': 3.2831576e-05, 'Barred Spiral': 0.99669313, 'Unbarred Tight Spiral': 9.466443e-05, 'Unbarred Loose Spiral': 0.0005237365, 'Edge-on without Bulge': 0.00012735139, 'Edge-on with Bulge': 0.00096479757}
```