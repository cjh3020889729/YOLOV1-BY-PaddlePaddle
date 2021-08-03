# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-08-02 23:30:58
# @Last Modified by:   红白黑
# @Last Modified time: 2021-08-03 20:14:14
import paddle 
from paddle import nn
from paddle.nn import functional as F

class YOLOV1Loss(nn.Layer):
    '''YOLOV1损失类
        Args:
            nb: [int]每个cell预测bounding box数量
            reduction: [str:'none', 'sum', 'mean']损失统计方式，类似MSELoss中的reduction描述
            coord: [float]物体存在且最佳匹配时，计算坐标损失的权重系数
            noobj: [float]物体不存在且最佳匹配时，计算置信度损失的权重系数
        Funcs:
            encode: 编码输入数据
            compute_iou: 计算每个真实框与预测框的iou
        Infos:
            执行YOLOV1的损失计算
            输入预测结果: Tensor--N, S, S, (B*5+NCLS)
            输入真实结果: Tensor--N, S, S, (B*5+NCLS)
            输出损失: Tensor
    '''
    def __init__(self, nb=2, reduction='none', coord=5., noobj=.5):
        super(YOLOV1Loss, self).__init__()
        self.reduction=reduction
        self.nb=nb

        self.coord=coord # 物体存在当前cell下的最佳匹配bounding box时的坐标计算权重系数
        self.noobj=noobj # 物体不存在当前cell下的最佳匹配bounding box时的置信度计算权重系数

        self.mse_loss=nn.MSELoss(reduction='none') # 损失计算均为平方损失
    

    def forward(self, preds, targets):
        '''计算YOLOV1的损失
            Args:
                preds: N, S, S, (b*5 + ncls)
                targets: N, S, S, (b*5 + ncls)
            Retrun:
                YOLOLoss
        '''
        # YOLOV1数据编码，分离参数集
        # (预测的x、y、w、h)pxywh: N, S, S, NB, 4
        # (预测落在该cell位置的物体概率)pobj: N, S, S, NB
        # (该cell的类别预测概率)pcls: N, S, S, 20
        pxywh, pobj, pcls=self.encode(preds)
        txywh, tobj, tcls=self.encode(targets)
        # print('xywh shape: ', pxywh.shape)
        # print('obj shape: ', pxywh.shape)
        # print('cls shape: ', pxywh.shape)


        # 1.计算的iou值 -- 为选择最佳拟合的bounding box做出选择参考，最大iou
        # iou: N, S, S, NB
        iou=self.compute_iou(pxywh, txywh)
        iou.stop_gradient=True  # min=0
        iou_max=iou.max(axis=3) # N, S, S, 1
        iou_max=paddle.stack([iou_max]*self.nb, axis=-1) # N, S, S, NB
        print('iou_max shape: ', iou.shape)


        # 2.确定最佳bounding box的mask -- 即iou max
        # two bounding box都等于0，则还是为False
        # 不等于0，取最大；如果相等，都取True  -- 这里的mask的True对应非最佳Iou的mask
        no_best_box_mask=paddle.less_than(iou.squeeze(), iou_max)  # just 遵循大于0时，且两个bounding box预测位置不相同
        no_best_box_mask=paddle.cast(no_best_box_mask, dtype=iou_max.dtype)
        # 重叠最好的框的mask -- 取没重叠好的mask的相反结果得到重叠最好的结果
        best_box_mask=paddle.ones_like(no_best_box_mask) - no_best_box_mask # 为了筛选每个cell中两个box哪一个为最佳box，所以shape为:N, S, S, NB
        print('best box mask shape: ', best_box_mask.shape)


        # 3.确定预测的置信度(obj or confidence)
        # pobj: N, S, S, NB
        # iou: N, S, S, NB
        pobj*=iou # 论文所指的训练时的置信度计算
        loss_obj=self.mse_loss(pobj, tobj) # 开始计算confidence的平方损失
        print('obj loss shape: ', loss_obj.shape)


        # 4.位置损失
        # loss_xy: N, S, S, NB, 1
        loss_xy=self.mse_loss(pxywh[:, :, :, :, :2], txywh[:, :, :, :, :2])
        loss_xy=paddle.sum(loss_xy, axis=-1)
        # loss_wh: N, S, S, NB, 1
        # sqrt to fit paper
        loss_wh=self.mse_loss(paddle.sqrt(pxywh[:, :, :, :, :2]), paddle.sqrt(txywh[:, :, :, :, :2]))
        loss_wh=paddle.sum(loss_wh, axis=-1)
        # loss_xywh: N, S, S, NB, 1
        loss_xywh=loss_xy+loss_wh
        print('xywh loss shape: ', loss_xywh.shape)


        # 5.物体所在的位置的掩码，即当前cell中是否存在object
        # 不存在物体的掩码noobj_mask
        # first: N, S, S, NB
        noobj_mask=paddle.less_than(tobj, paddle.ones_like(tobj)) # bool 类型的数据不支持切片索引
        # final: N, S, S, NB
        noobj_mask=paddle.cast(noobj_mask, dtype=pobj.dtype)
        # 根据noobj_mask推出obj_mask: N, S, S, NB
        obj_mask=paddle.ones_like(noobj_mask) - noobj_mask
        print('obj mask shape: ', obj_mask.shape)

        
        # 6.类别损失: N, S, S, 1
        loss_cls=self.mse_loss(pcls, tcls)
        loss_cls=paddle.sum(loss_cls, axis=-1).unsqueeze(-1)
        print('cls loss shape: ', loss_cls.shape)

        
        # 根据mask获取有效的损失
        # loss_xywh: N, S, S, NB
        # best_box_mask: N, S, S, NB
        loss_xywh*=best_box_mask*self.coord
        # loss_cls: N, S, S, 1
        # obj_mask: N, S, S, 1
        loss_cls*=obj_mask
        # loss_obj: N, S, S, NB, 1
        # no_best_box_mask: N, S, S, 1
        loss_pos_obj=loss_obj*best_box_mask
        loss_neg_obj=loss_obj*no_best_box_mask

        # total loss : by add
        loss=loss_xywh+loss_cls+loss_pos_obj+loss_neg_obj

        if self.reduction == 'sum': # 求和损失
            loss=paddle.sum(loss)
            return loss
        elif self.reduction == 'mean': # 平均损失
            loss=paddle.mean(loss)
            return loss

        return loss # 直接返回


    def encode(self, inputs):
        '''对输入的数据进行编码
            Args:
                inputs:  [list]YOLOV1的输出数据: N, S, S, (B*5+NCLS)
            Return:
                *arg: [list]预测的结果[pxywh, pobj, pcls]
        '''
        xywh=[] # 拆分后的xywh值
        obj=[] # 拆分后的obj值
        for i in range(self.nb): # 遍历bounding box个数，获取每一个bounding box的xywh与obj值
            _xywh=inputs[:, :, :, i*5:i*5+4]
            _obj=inputs[:, :, :, i*5+5]

            xywh.append(_xywh) # 把每一个提取结果保存
            obj.append(_obj)
        pxywh=paddle.stack(xywh, axis=3) # 堆叠得到: N, S, S, NB, XYWH(4)
        pobj=paddle.stack(obj, axis=3)   # 堆叠得到: N, S, S, NB
        pcls=inputs[:, :, :, self.nb*5:] # N, S, S, NCLS(20)
        
        return [pxywh, pobj, pcls]


    def compute_iou(self, pbox, tbox):
        '''计算预测bbox与真实bbox之间的iou值
            Args:
                pbox: [Tensor]N, S, S, B, XYWH(4)
                tbox: [Tensor]N, S, S, B, XYWH(4)
            Return:
                iou: [Tensor]N, S, S, B
        '''
        # 避免计算过程中，影响计算图，在无梯度计算的环境中执行iou计算
        with paddle.no_grad():
            # 预测bbox的坐标信息
            px1=pbox[:, :, :, :, 0] - pbox[:, :, :, :, 2] / 2. # left up
            py1=pbox[:, :, :, :, 1] - pbox[:, :, :, :, 3] / 2.
            px2=pbox[:, :, :, :, 0] + pbox[:, :, :, :, 2] / 2. # right down
            py2=pbox[:, :, :, :, 1] + pbox[:, :, :, :, 3] / 2.
            # 真实bbox的坐标信息
            tx1=tbox[:, :, :, :, 0] - tbox[:, :, :, :, 2] / 2. # left up
            ty1=tbox[:, :, :, :, 1] - tbox[:, :, :, :, 3] / 2.
            tx2=tbox[:, :, :, :, 0] + tbox[:, :, :, :, 2] / 2. # right down
            ty2=tbox[:, :, :, :, 1] + tbox[:, :, :, :, 3] / 2.
            # 交并集区域的坐标 -- 多个bounding box参与计算
            iniou_x1=paddle.clip(px1, min=tx1, max=tx2)  # 裁剪到真实框区域，保证重叠区域的面积为: (0-真实框面积)
            iniou_y1=paddle.clip(py1, min=ty1, max=ty2)
            iniou_x2=paddle.clip(px2, min=tx1, max=tx2)
            iniou_y2=paddle.clip(py2, min=ty1, max=ty2)
            iniou_w=paddle.clip(iniou_x2 - iniou_x1, min=paddle.zeros_like(iniou_x2)) # 交并集区域的宽
            iniou_h=paddle.clip(iniou_y2 - iniou_y1, min=paddle.zeros_like(iniou_y2)) # 交并集区域的高
            iniou=iniou_w*iniou_h # 交并集区域的面积
            # 各自bbox的自身面积
            pbox_area=paddle.clip(px2 - px1, min=paddle.zeros_like(px2)) * paddle.clip(py2 - py1, min=paddle.zeros_like(py2))
            tbox_area=paddle.clip(tx2 - tx1, min=paddle.zeros_like(tx2)) * paddle.clip(ty2 - ty1, min=paddle.zeros_like(ty2))
            
            # 计算交并集区域的占比
            iou=iniou / (pbox_area+tbox_area-iniou)
            
        return iou



# 测试代码
if __name__ == "__main__":
    data_shape=[2, 7, 7, 30]
    print('Input data shape: ', data_shape)

    import numpy as np
    pred_xywh=np.random.rand(2, 7, 7, 2*5)
    pred_xywh[0, :, :, 5]=1.
    pred_xywh[1, :, :, 9]=1.
    pred_cls=np.zeros((2, 7, 7, 20))
    pred_cls[0, :, :, 5]=1.
    pred_cls[1, :, :, 1]=1.
    pred_data=np.concatenate([pred_xywh, pred_cls], axis=-1)
    # print(pred_data.shape) # Input data shape

    target_xywh=np.random.rand(2, 7, 7, 2*5)
    target_xywh[0, :, :, 9]=1.
    target_xywh[1, :, :, 9]=1.
    target_cls=np.zeros((2, 7, 7, 20))
    target_cls[0, :, :, 5]=1.
    target_cls[1, :, :, 2]=1.
    target_data=np.concatenate([target_xywh, target_cls], axis=-1)
    # print(target_data.shape) # Model output data shape

    pred_data=paddle.to_tensor(pred_data)
    pred_data.stop_gradient=False
    target_data=paddle.to_tensor(target_data)

    loss_func=YOLOV1Loss(nb=2, reduction='mean')

    # loss_func(pred_data, target_data)

    loss=loss_func(pred_data, target_data)
    print(loss.shape)

    loss.backward()

    print(pred_data.grad.shape)
    
