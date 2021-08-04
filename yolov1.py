# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-08-03 17:12:43
# @Last Modified by:   红白黑
# @Last Modified time: 2021-08-04 18:22:34
import paddle 
from paddle import nn
from paddle.nn import functional as F

import numpy as np
from utils import visualize_det, xywh2xyxy, get_evaluate_result
from net import Net, Ascension_Layer, OutLinaer_Layer

class YOLOV1(nn.Layer):
    '''YOLOV1目标检测网络的实现
        Args:
            img_size: 输入图片大小
            in_channels: 输入图片通道数
            ncls: 模型检测的类别数
            nb: 每个cell对应的bounding box数
        Funcs:
            encode: 编码YOLOV1网络的输出Tensor
            decode: 解码经过nms后的YOLOV1输出结果，得到一个List结果，[Batch, Num_bbox, 6(cls_id+cls_score+xyxy)]
            nms: nms后处理, 得到最后输出的索引(当前采用numpy筛选，未采用paddle原生实现)
        Infos:
            执行YOLOV1模型的训练与预测输出
            训练/预测数据输入: Tensor--H,W,in_channels
            训练数据输出: Tensor -- N,S,S,B*2+NCLS
            预测数据输出: List -- N,B,6(CLS_ID+CLS_SCORE+XYXY)
    '''
    def __init__(self, img_size=448, in_channels=3, ncls=20, nb=2):
        super(YOLOV1, self).__init__()
        self.img_size=img_size # 图片大小
        self.in_channels=in_channels # 输入数据的通道数
        self.ncls=ncls # 分类数
        self.nb=nb # 每个cell预测的bounding box数
        
        # 骨干特征提取网络
        self.backbone=Net(in_channels=in_channels)
        # 特征提升网络
        self.ascension_layer=Ascension_Layer(in_channels=self.backbone.out_channels)
        # 全连接输出网络
        self.outLinaer_layer=OutLinaer_Layer(in_feature_size=self.ascension_layer.out_channels,
                                             out_feature_size=nb*5+ncls)
    
    def forward(self, inputs):
        x=self.backbone(inputs)
        x=self.ascension_layer(x)
        x=self.outLinaer_layer(x) # N, S, S, (B*5+NCLS)

        if self.training is False:
            print('Predicting--')
            pxywh, pobj, pcls=self.encode(x)
            batch_keep=self.nms(pxywh, pobj, pcls, threshold=0.5, use_confidence=False)
            batch_result=self.decode(batch_keep, pxywh, pobj, pcls)
            return batch_result # 每一个batch中保留的结果数量可以不相同
        
        return x
    
    def encode(self, inputs):
        '''对YOLOV1输出的数据进行编码
            Args:
                inputs:  [list]YOLOV1的输出数据: N, S, S, (B*5+NCLS)
            Return:
                *arg: [list]预测的结果[pxywh, pobj, pcls]
        '''
        pxywh=[]
        pobj=[]
        for i in range(self.nb):
            xywh=inputs[:, :, :, i*5:i*5+4]
            obj=inputs[:, :, :, i*5+5]

            pxywh.append(xywh)
            pobj.append(obj)
        pxywh=paddle.stack(pxywh, axis=3) # N, S, S, NB, XYWH
        pobj=paddle.stack(pobj, axis=3) # N, S, S, NB
        pcls=inputs[:, :, :, self.nb*5:] # N, S, S, NCLS

        N, S, S, B, _=pxywh.shape
        H, W=paddle.meshgrid(paddle.arange(S), paddle.arange(S))
        xy_offset=paddle.stack([H, W], axis=-1)  # 获取xy的偏移值阵列
        xy_offset=paddle.stack([xy_offset]*2, axis=-2).unsqueeze(0)
        
        stride=self.img_size/S  # 获取每一个像素从原图像到当前feature map的缩放步长
        pxywh[:,:,:,:, :2]=paddle.clip((pxywh[:,:,:,:, :2]+xy_offset) * stride, min=0., max=self.img_size-1.) # 获取xy真实值
        pxywh[:,:,:,:, 2:]=paddle.clip(pxywh[:,:,:,:, 2:] * self.img_size, min=0., max=self.img_size) # 获取wh的真实值
        
        return [pxywh, pobj, pcls]
    
    def nms(self, pxywh, pobj, pcls, threshold=0.5, use_confidence=False):
        '''nms模型后处理(用于encode操作后)
            Args:
                pxywh: [Tensor]预测的坐标: N, S, S, B, 4
                pobj: [Tensor]预测的obj概率: N, S, S, B
                pcls: [Tensor]预测的cls概率: N, S, S, 20
                threshold: [float]nms处理的阈值: [0., 1.)
                use_confidence: [bool]obj得分使用置信度计算方式确定
            Return:
                batch_keep: [list]nms处理后，每个batch中保留的数据的index集合
        '''
        N, S, S, B, _=pxywh.shape

        # to be one dim
        xywh=paddle.reshape(pxywh.clone(), (N,S*S*B,4)).numpy()
        objs=paddle.reshape(pobj.clone(), (N,S*S*B)).numpy()
        clses=paddle.reshape(pcls.clone(), (N,S*S,20)).numpy()

        # 根据obj得分确定nms的排序
        if use_confidence is False:
            sorted_obj=np.argsort(-objs, axis=-1) # 排序 -- 升序，取反再排序排序==得到降序的index返回结果
        else:
            max_cls_score=np.max(clses, axis=-1)
            objs*=max_cls_score  # 分类置信度
            sorted_obj=np.argsort(-objs, axis=-1) # 再排序
        
        # 批量保存结果
        batch_keep=[]
        for i in range(N):
            sorted_indexs=sorted_obj[i]
            keep=[]
            while len(sorted_indexs)>0:
                idx=sorted_indexs[0]
                keep.append(idx) # 保存当前结果
                sorted_indexs=np.delete(sorted_indexs, 0, axis=0) # delete the best score index

                # 当前pred_box坐标信息
                x, y, w, h=xywh[i, idx] # 获取当前sorted_obj中得分最高的一个bbox位置
                x1=x-w/2. # left up
                y1=x-w/2.
                x2=x+w/2. # right down
                y2=x+w/2.
                # 其余pred_box坐标信息
                other_xywh=xywh[i, sorted_indexs]  # 取指定的index下的数据
                other_x1=other_xywh[:, 0] - other_xywh[:, 2]/2.
                other_y1=other_xywh[:, 1] - other_xywh[:, 3]/2.
                other_x2=other_xywh[:, 0] + other_xywh[:, 2]/2.
                other_y2=other_xywh[:, 1] + other_xywh[:, 3]/2.
                # 两者交集区域
                iniou_x1=np.clip(other_x1, a_min=x1, a_max=x2)
                iniou_y1=np.clip(other_y1, a_min=y1, a_max=y2)
                iniou_x2=np.clip(other_x2, a_min=x1, a_max=x2)
                iniou_y2=np.clip(other_y2, a_min=y1, a_max=y2)
                iniou_w=np.clip(iniou_x2-iniou_x1, a_min=0., a_max=self.img_size)
                iniou_h=np.clip(iniou_y2-iniou_y1, a_min=0., a_max=self.img_size)
                iniou=iniou_w*iniou_h
                # 各自的区域
                _area=w*h
                other_area=other_xywh[:, 2]*other_xywh[:, 3]
                # 各自的iou
                iou=iniou/(other_area+_area-iniou)

                # 查询小于阈值保留的mask
                keep_mask=iou>threshold
                sorted_indexs=sorted_indexs[keep_mask] # 更新待nms的index集合

                if len(sorted_indexs)==1:
                    keep.append(sorted_indexs[0])
                    break
            batch_keep.append(keep)
    
        return batch_keep
    
    def decode(self, batch_keep, pxywh, pobj, pcls, use_confidence=False):
        '''根据nms结果解码YOLOV1的输出 -- N, 6(cls_id+cls_score+xyxy)
            Args:
                batch_keep: [list]Batch下每个Batch待保留数据的index: N, S*S*B
                pxywh: [Tensor]模型输出的xywh参数值: N, S, S, B, 4
                pobj: [Tensor]模型输出的obj概率: N, S, S, B
                pcls: [Tensor]模型输出的cls概率: N, S, S, 20
                use_confidence: [bool]类别得分使用置信度计算方式确定
            Return:
                batch_results: [list]多个batch的结果: N, KEEP_NUM, 6(CLS_ID+CLS_SCORE+XYXY)
        '''
        N, S, S, B, _=pxywh.shape

        # to be one dim
        xywh=paddle.reshape(pxywh.clone(), (N,S*S*B,4)).numpy()
        objs=paddle.reshape(pobj.clone(), (N,S*S*B)).numpy()

        clses=pcls.clone().numpy()
        clses=clses[:, :, :, np.newaxis, :]
        clses=np.repeat(clses, self.nb, axis=-1).reshape(N,S*S*B,self.ncls)

        batch_results=[]
        for i in range(N):
            result_xywh=xywh[i, batch_keep[i], :].reshape(-1, 4)
            result_xyxy=xywh2xyxy(result_xywh, img_size=self.img_size)
            result_objs=objs[i, batch_keep[i]].reshape(-1, 1)
            
            result_clses=clses[i, batch_keep[i], :].reshape(-1, self.ncls)
            result_clses_id=np.argmax(result_clses, axis=-1).reshape(-1, 1) # index
            result_clses_score=np.asarray([result_clses[idx, index] for idx, index in enumerate(result_clses_id[:, 0])]).reshape(-1, 1)
            if use_confidence is True:
                result_clses_score*=result_objs
            
            result=np.concatenate([result_clses_id, result_clses_score, result_xyxy], axis=-1).astype(np.float32)
            batch_results.append(result)
        return batch_results


# 测试代码
if __name__ == "__main__":
    img_shape=[448, 448, 3]
    data_shape=[1, img_shape[2], img_shape[0], img_shape[1]]
    print('Input data shape: ', data_shape)

    import numpy as np
    data=np.random.randint(0, 256, data_shape).astype(np.float32)/255.
    data=paddle.to_tensor(data)

    model=YOLOV1(img_size=448, in_channels=3, ncls=20, nb=2)

    model.eval()
    y_pred=model(data)
    
    if model.training is False:
        print(y_pred[0].shape)
        print(get_evaluate_result(y_pred)) # 获取用于验证计算的结果形式
    else:
        print(y_pred.shape)
    
