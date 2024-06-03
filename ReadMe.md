# 星系分类模型算法和搜索部分

Update: 2024/06/03

Author: weishirui

### 文件夹结构
```
galaxy_classification
    |—— dataset/    # 数据集
    |   |—— Galaxy10_DECals_predict/    # 基于Galaxy10 DECals的测试数据集，包含十张图片，一类一张，用于测试
    |   |—— galaxy_catalog/ # 基于SDSS星表的星系坐标，用于测试搜索功能
    |—— model/  # 模型
    |   |—— zoobot/ # 基于zoobot v2框架的模型
    |       |—— convnext_nano/  # encoder模型文件
    |       |—— finetune_model/ # 微调后的模型权重文件
    |—— outputs/    # 输出结果保存
    |—— lightning_logs/ # 日志（自动生成）
    |—— search_tool/ # 搜索功能
    |       |—— aladin.html # Aladin GUI交互界面 & 获取图像的按钮，用于单个星系的图像搜索以及下载
    |       |—— main.py # FastAPI：获取前端返回的Aladin视图信息，下载对应图像并返回Simbad查询得到的单个星系信息
    |       |—— search_tool.py # 搜索功能包含的主要函数：图像URL生成、图像下载、信息查询
    |       |—— test.py # 基于表格数据的多星系图像下载测试脚本
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
- 另外可能需要安装的库
```
pip install -r requirements.txt
```
#### Debug
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
- Warning解决：如遇到以下Warning：
```
GZDESI/GZRings/GZCD not available from galaxy_datasets.pytorch.datasets - skipping
```
可通过修改安装环境中的galaxy_datasets库文件来解决。

首先找到你的安装环境下的galaxy_datasets包，例如`/root/miniconda3/lib/python3.10/site-packages/galaxy_datasets/`，再找到`galaxy_datasets/pytorch/__init__.py`文件。
将代码3-7行注释掉，即
```
from galaxy_datasets.pytorch.datasets import GZCandels, GZDecals5, GZHubble, GZ2, Tidal

try:
    from galaxy_datasets.pytorch.datasets import GZDesi, GZRings, GZH2O
except ImportError:
    # not using logging in case config still required
    print('GZDESI/GZRings/GZCD not available from galaxy_datasets.pytorch.datasets - skipping')    
```
改为
```
from galaxy_datasets.pytorch.datasets import GZCandels, GZDecals5, GZHubble, GZ2, Tidal

# try:
#     from galaxy_datasets.pytorch.datasets import GZDesi, GZRings, GZH2O
# except ImportError:
#     # not using logging in case config still required
#     print('GZDESI/GZRings/GZCD not available from galaxy_datasets.pytorch.datasets - skipping')    
```
可解决该问题。

问题参考：https://github.com/mwalmsley/galaxy-datasets/issues/26

#### Classification
**星系分类算法主函数：galaxy_classification_main.py**

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
#### Search
搜索功能前端：search_tool/aladin.html

搜索功能后端：search_tool/main.py(FastAPI) & search_tool.py (function)

基于表格文件的多星系搜索测试脚本：test.py

- 后端启动：
```
cd search_tool/
uvicorn main:app --reload
```
- **单个星系搜索**：

页面：Aladin GUI初始化中心应为一个螺旋星系，可随意选择可视天区。

点击"Get Image"按钮，将会自动下载对应天区的图像，并保存至自动新建的images/文件夹下。【图像路径设置：search_tool.py: image_dir = './images/'】若图像成功下载并保存，将打印输出"Saving Image Successfully!"，并以JSONResponse的形式返回该图像的信息。
例如：
```
Response from backend: 

    "ra": 53.40190832999999,
    "dec": -36.14065833000001,
    "fov": 0.5,
    "size": 400,
    "surveyid": "ivo://CDS/P/DESI-Legacy-Surveys/DR10/color",
    "info_table": [
        {
            "ra": 53.38474166666666,
            "dec": -36.11139166666667,
            "main_id": "6dFGS gJ033332.3-360641",
            "rvz_redshift": 0.0051368870660515415,
            "otype": "Sy1",
            "morph_type": "",
            "galdim_majaxis": null
        },
        {
            "ra": 53.36006747623708,
            "dec": -36.15888581177,
            "main_id": "Gaia DR3 4859918261801064704",
            "rvz_redshift": null,
            "otype": "G",
            "morph_type": "",
            "galdim_majaxis": 0.05190116539597511
        }
    ]
}
```
其中"info_table"为根据RA&DEC坐标以0.05deg半径在SIMBAD中查询的结果（若查询到多个天体，则返回<=5个结果）。查询得到的星系信息包括：RA, DEC, main_id, rvz_redshift（红移）, otype（天体类型：星系G）, morph_type（若有，则为星系的形态）, galdim_majaxis（星系主轴的角尺寸大小）。

- **多星系搜索**

测试脚本test.py，只用于流程演示。
```
cd search_tool
python test.py
```

输入：星表文件，要求至少包括每个待标注天体的RA&DEC信息，可加上FOV、图片尺寸、巡天信息

输出：下载并保存星表中每个待标注天体的图像，保存路径同`单星系搜索`。不在SIMBAD中搜索信息。





