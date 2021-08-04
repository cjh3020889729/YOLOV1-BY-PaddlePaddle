# YOLOV1-BY-PaddlePaddle
使用PaddlePaddle复现YOLOV1算法框架，以及训练和评估流程——正在抓紧更新中

> 以上部分功能代码还未实现, 未在以下注明的，还在跟近中

> 后期复现完成，会跟进讲解视频，帮助大家学习理解

1. 特征提取网络完成 -- 2021-8-3(Net.py)
2. YOLO损失完成 -- 2021-8-3(yolo_loss.py)
3. YOLO检测网络完成 -- 2021-8-3(yolov1.py)
4. 数据加载设计未完成 -- important -- 2021-8-4(部分:训练数据加载已完成——待添加验证数据加载，以及备用的预测数据加载)(dataset.py)
5. 训练方法未实现 -- second
6. 评估方法未实现 -- 2021-8-3(部分)utils.py -- final
7. 数据加载的预处理方法 -- important -- 2021-8-4(部分:Resize实现，其余逐步更新，会在进行训练前，完成本部分)(transforms.py)

something to do:
    获取用于评估的结果的API已完成，但评估的mAP指标计算未实现，整体的评估方法还未整体实现
    训练数据加载已完成，但还未实现验证数据的加载——备用预测加载，是为了方便批量预测


