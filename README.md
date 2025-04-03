

此存储库包含 MRNM（多模态特征和关系预测网络）的实现，这是一种用于历史文档布局分析的模型。MRNM 将多模态特征（视觉、文本和空间）与关系预测网络相结合，以检测布局元素并预测阅读顺序。

模型架构 MRNM 由三个主要部分组成：
多模态网络 ：提取并融合视觉、文本和空间特征
区域检测 ：执行布局元素检测和分类
关系预测网络 ：使用图神经网络预测读取顺序

数据集下载地址如下：
链接: https://pan.baidu.com/s/1yRo5O0Ehu2SOcTcWZizmuA 提取码: u4ik 

目录结构如下
MRNM_layout_analysis/
│
├── data_loader/               
│   ├── __init__.py
│   ├── page_data.py           
│   ├── dataset.py             
│   └── transforms.py          
│
├── models/                    
│   ├── __init__.py            # 主模型MRNM实现
│   ├── backbone/              # 基础网络
│   │   ├── __init__.py
│   │   ├── cnn.py             # CNN视觉特征提取
│   │   └── transformer.py     # Transformer相关模块
│   ├── multimodal/            # 多模态特征提取
│   │   ├── __init__.py
│   │   ├── visual.py          # 视觉特征提取
│   │   ├── text.py            # 文本特征提取
│   │   ├── spatial.py         # 空间特征提取
│   │   └── fusion.py          # 多模态融合
│   └── heads/                 # 任务头
│       ├── __init__.py
│       ├── rpn.py             # 区域建议网络
│       ├── classifier.py      # 分类头
│       └── relation_prediction.py  # 关系预测网络
│
├── utils/                     
│   ├── __init__.py
│   ├── losses.py              # 损失函数
│   ├── metrics.py             # 评估指标
│   ├── attention.py           # 注意力机制实现
│   ├── geometry.py            # 几何变换工具
│   └── edge_features.py       # 边特征提取
│
├── configs/                   
│   ├── default.yaml           # 默认配置
│   ├── train.yaml             # 训练专用配置
│   └── model.yaml             # 模型结构配置
│
├── scripts/                  
│   ├── train.py             
│   ├── test.py               
│   └── visualize.py          
│
├── tools/                     
│   ├── convert_data.py        
│   ├── create_annotations.py  
│   └── evaluate_results.py    

