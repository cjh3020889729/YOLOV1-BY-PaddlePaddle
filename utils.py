# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-08-03 17:11:44
# @Last Modified by:   红白黑
# @Last Modified time: 2021-08-04 18:19:50
import cv2
import numpy as np
from numpy.lib.arraysetops import isin


def get_evaluate_result(inputs):
    '''获取评估的结果表示(inputs来自模型预测的输出)
        Args:
            inputs: [list]内含batch中每张图片的预测结果: N, B, 6(CLS_ID+CLS_SCORE+XYXY)
        Return:
            eval_results: [list]用于验证的结果格式: N, (dict) -- dict:{cls_id:[cls_score+xyxy]}
    '''
    N=len(inputs)
    eval_results=[]
    for i in range(N): # 遍历batch并将每一个结果进行解析
        find_result_set=inputs[i]
        eval_result={}
        for j in range(len(find_result_set)):
            cls_id=int(find_result_set[j, 0].item())
            cls_score=find_result_set[j, 1].item()
            xyxy=find_result_set[j, 2:]
            if cls_id in eval_result.keys():
                eval_result[cls_id].append([cls_score]+xyxy.tolist())
            else:
                eval_result[cls_id]=[]
                eval_result[cls_id].append([cls_score]+xyxy.tolist())
        eval_results.append(eval_result)
    return eval_results # list中每一个元素都是一张图片的所有预测结果


def xywh2xyxy(xywh, img_size=448):
    '''将xywh格式的坐标数据转换为xyxy格式
        Args:
            xywh: [np.ndarray]: N,4(XYWH)
        Return:
            xyxy: [np.ndarray]: N,4(XYXY)
    '''
    assert isinstance(xywh, np.ndarray),\
        'Func xywh2xyxy Error: Please make sure the xywh data type(now is {0}) is np.ndarray.'.format(type(xywh))
    assert img_size>0,\
        'Func xywh2xyxy Error: Please make sure the img_size(now is {0}) > 0.'.format(img_size)

    x1=np.clip(xywh[:, 0] - xywh[:, 2]/2., a_min=0., a_max=img_size-1.)[:, np.newaxis]
    y1=np.clip(xywh[:, 1] - xywh[:, 3]/2., a_min=0., a_max=img_size-1.)[:, np.newaxis]
    x2=np.clip(xywh[:, 0] + xywh[:, 2]/2., a_min=0., a_max=img_size-1.)[:, np.newaxis]
    y2=np.clip(xywh[:, 1] + xywh[:, 3]/2., a_min=0., a_max=img_size-1.)[:, np.newaxis]

    xyxy=np.concatenate([x1, y1, x2, y2], axis=-1)
    return xyxy


def visualize_det(img, bboxs, mode='xywh'):
    '''对预测结果进行可视化（单张图片绘制矩形框）
        Args:
            img: [np.ndarray]输入图片数据: H, W, 3
            bboxs: [np.ndarray]待绘制框的坐标信息: N, 6(cls_id+cls_score+xywh)
            mode: [str:'xywh', 'xyxy']输入坐标模式
        Return:
            img: [np.ndarray]返回图片数据: H, W, 3
    '''
    assert mode in ['xywh', 'xyxy'],\
        "Func visualize_det Error: Please make sure the bboxs data input mode(now is {0}) is in ['xywh', 'xyxy'] or None.".format(mode)
    assert isinstance(img, np.ndarray),\
        "Func visualize_det Error: Please make sure the img data type(now is {0}) is ndarray.".format(type(img))
    assert img.shape[2] <= 4,\
        "Func visualize_det Error: Please make sure the img data dim 2:  dim_size(now is {0}) <= 4.".format(img.shape[2])
    assert isinstance(bboxs, np.ndarray),\
        "Func visualize_det Error: Please make sure the bboxs data type(now is {0}) is ndarray.".format(type(bboxs))
    assert bboxs.ndim == 2,\
        "Func visualize_det Error: Please make sure the bboxs.ndim: ndim(now is {0}) == 2.".format(bboxs.ndim)
    cls_id=bboxs[:, 0]
    cls_score=bboxs[:, 1]
    bboxs=bboxs[:, 2:]
    for i in range(len(bboxs)):
        if mode=='xywh':
            x1=int(np.clip(bboxs[i, 0]-bboxs[i, 2]/2., a_min=0., a_max=img.shape[0]-1.))
            y1=int(np.clip(bboxs[i, 1]-bboxs[i, 3]/2., a_min=0., a_max=img.shape[1]-1.))
            x2=int(np.clip(bboxs[i, 0]+bboxs[i, 2]/2., a_min=0., a_max=img.shape[0]-1.))
            y2=int(np.clip(bboxs[i, 1]+bboxs[i, 3]/2., a_min=0., a_max=img.shape[1]-1.))
        elif mode=='xyxy':
            x1=int(bboxs[i, 0])
            y1=int(bboxs[i, 1])
            x2=int(bboxs[i, 2])
            y2=int(bboxs[i, 3])
        img=cv2.rectangle(img, (x1, y1), (x2, y2), color=0, thickness=1)
        cv2.putText(img, "cls_id:{0}-{1:.6f}".format(cls_id[i], cls_score[i]), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), thickness=1)
    return img
