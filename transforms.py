# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-08-04 18:12:59
# @Last Modified by:   红白黑
# @Last Modified time: 2021-08-04 18:15:41
import cv2
import numpy as np
from PIL import Image


class Resize(object):
    '''图像缩放
        Args:
            target_size: [int]缩放的目标大小
    '''
    def __init__(self, target_size=448):
        self.target_size=target_size

    def _apply_img(self, img):
        '''对图片的操作
            Args:
                img: [np.ndarray]图像数据: (h, w, c)/(h, w)
            Return:
                img_map: [np.ndarray]缩放后的图像数据: (h, w, c)/(h, w)
                img_scale: [float]xy统一缩放的比例
                offset_x, offset_y: [int]缩放过程中，原图填充的偏移值——也是边界框缩放后需要调整xy偏移值
        '''
        if len(img.shape)<3:
            img_map=np.zeros((self.target_size, self.target_size)).astype(np.float32)
            img_map.fill(255.)
        elif len(img.shape)==3:
            img_map=np.zeros((self.target_size, self.target_size, 3)).astype(np.float32)
            img_map.fill(255.)
        h,w=img.shape[0:2]
        dist_=max(h, w)
        img_scale=self.target_size/dist_
        img=cv2.resize(img, None, fx=img_scale, fy=img_scale)
        
        offset_x=(self.target_size-img.shape[1])//2
        offset_y=(self.target_size-img.shape[0])//2
        img_map[offset_y:(self.target_size-offset_y), offset_x:(self.target_size-offset_x)]=img
        return img_map, [img_scale, [offset_x, offset_y]]

    def _apply_bbox(self, bboxs, scale, offsets):
        '''对边界框的操作
            Args:
                bboxs: [np.ndarray]边界框数据: B, 4(xyxy/xywh)
                scale: [float]xy统一缩放的比例
                offsets: [list:offset_x, offset_y]边界框缩放后需要调整xy偏移值
            Return:
                bboxs: [np.ndarray]缩放调整后的边界框数据: B, 4(xyxy/xywh)
        '''
        offset_x, offset_y=offsets
        bboxs[:, 0::2] = bboxs[:, 0::2] * scale + offset_x
        bboxs[:, 1::2] = bboxs[:, 1::2] * scale + offset_y
        bboxs[:, 0::2] = np.clip(bboxs[:, 0::2], 0., self.target_size)
        bboxs[:, 1::2] = np.clip(bboxs[:, 1::2], 0., self.target_size)
        return bboxs
        
    
    def __call__(self, img, bbox):
        '''执行缩放
            Args:
                img: [np.ndarray]图片数据: (h, w, c)/(h, w)
                bbox: [np.ndarray]边界框数据: B, 4(xyxy/xywh)
            Return:
                new_img: [np.ndarray]缩放调整后的图片数据: (h, w, c)/(h, w)
                new_bboxs： [np.ndarray]缩放调整后的边界框数据: B, 4(xyxy/xywh)
        '''
        new_img, [img_scale, [offset_x, offset_y]] = self._apply_img(img)  # 执行图片缩放
        new_bboxs = self._apply_bbox(bbox, img_scale, [offset_x, offset_y]) # 执行边界框缩放
        return new_img, new_bboxs
