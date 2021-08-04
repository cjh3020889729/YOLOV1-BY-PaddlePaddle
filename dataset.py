# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-08-03 17:11:25
# @Last Modified by:   红白黑
# @Last Modified time: 2021-08-04 18:15:38
from paddle.io import Dataset

import xml.etree.ElementTree as ET
import os.path as osp
import numpy as np
import cv2

from transforms import Resize


# 返回的边界框信息为模型输出的格式，方便计算损失: 7, 7, 30
class YOLOV1_TrainVOCDataset(Dataset):
    '''获取YOLOV1数据集
        Args:
            dataset_dir: [str]数据集的根目录: eg: data/pascalvoc
            train_txt: [str]训练数据的索引txt文件路径: eg: data/pascalvoc/train.txt
                    txt内容示意:（目录结构: data/pascalvoc/VOCdevkit）
                        VOCdevkit/VOC2012/JPEGImages/2011_000837.jpg	VOCdevkit/VOC2012/Annotations/2011_000837.xml
            label_txt: [str]训练数据中类别情况的txt文件路径: eg: data/pascalvoc/label_list.txt
                    txt内容示意:
                        bird
                        car
            feature_size: [int]YOLOV1最后输出的特征图大小: h=w
            anchor_num: [int]YOLOV1最后输出的每一个cell预测的bounding box数
            class_num: [int]YOLOV1检测的物体类别数
            transforms: [object]YOLOV1数据加载的预处理方法
                    现支持方法如下:
                        1. Resize: 缩放图片以及真实框到目标大小: 策略--以长边缩放到目标大小，短边空余部分用255填充
    '''
    def __init__(self, 
                 dataset_dir,
                 train_txt,
                 label_txt,
                 feature_size=7,
                 anchor_num=2,
                 class_num=20,
                 transforms=None):
        super(YOLOV1_TrainVOCDataset, self).__init__()
        self.dataset_dir=dataset_dir
        self.train_txt=train_txt
        self.label_txt=label_txt
        self.feature_size=feature_size
        self.anchor_num=anchor_num
        self.class_num=class_num
        self.transforms=transforms
        
        self.cname2cid=self.get_cname2cid_dict(label_txt) # 获取类别到id的映射字典

        self.data_files=[]  # 保存索引信息 -- 即每一个样本的文件路径: [img_file, xml_file]
        with open(train_txt, 'r') as f: # 读取txt文件
            file_list=f.readlines()
            for line in file_list:
                img_file, xml_file=line.strip().split()
                img_file=osp.join(dataset_dir, img_file)
                xml_file=osp.join(dataset_dir, xml_file)
                if not osp.exists(img_file): # 判断是否存在该图片，不存在就丢弃该索引信息
                    continue
                if not osp.exists(xml_file): # 判断是否存在该标注，不存在就丢弃该索引信息
                    continue
                self.data_files.append([img_file, xml_file]) # xml与img均存在，保存索引信息
        self.lens=len(self.data_files) # 获取总的有效数据的长度
        self.img_size=0 # 数据预处理后的图片大小 -- 在encode中起作用，对真实框大小进行限制


    def __getitem__(self, index):
        '''获取指定index的样本
            Args:
                index: [int]索引
            Return:
                img: [np.ndarray]预处理之后的图像数据
                yolov1_box_map: [np.ndarray]映射到YOLOV1输出特征图的数据格式: feature_size, feature_size, anchor_num*5+class_num
        '''
        img_file, xml_file=self.data_files[index]
        bboxs, clses=self.parse_voc(xml_file) # 解析xml数据，得到真实框以及对应的类别
        img=cv2.imread(img_file)[:, :, ::-1]  # BGR TO RGB
        if self.transforms is not None:
            img, bboxs=self.transforms(img, bboxs) # 图像预处理
        if self.img_size == 0:
            self.img_size=img.shape[0] # 获取预处理后的目标图像大小
        yolov1_box_map=self.encode(bboxs, clses) # 将得到的真实框转换为YOLOV1输出的特征图格式数据，方便计算损失进行优化
        return img, yolov1_box_map

    
    def __len__(self):
        '''返回数据dataset的样本数量
            Return:
                lens: [int]样本数量
        '''
        return self.lens


    def get_cname2cid_dict(self, label_txt):
        # 查看标签
        cname2cid={}
        with open(label_txt, 'r') as f:
            label_id=0
            for cname in f.readlines():
                if cname.strip()=='background': # 跳过background，因为YOLOV1不需要背景单独作为类别
                    continue
                cname2cid[cname.strip()]=label_id
                label_id+=1
        return cname2cid


    def parse_voc(self, xml_path): # 一份xml，对应一张图片，对应一个样本数
        '''解析VOC的标注文件xml
            Args:
                xml_path: [str]xml文件的路径
            Return:
                bboxs: [np.ndarray]标注的ground truth box信息: B,5(cls+xyxy)
        '''
        tree=ET.parse(xml_path) # 解析xml
        size=tree.find('size') # 获取图片大小
        img_w, img_h=int(size.find('width').text), int(size.find('height').text)
        objs=tree.findall('object') # 获取所有的物体对象

        bboxs=[] # 保存所有的真实框坐标数据: xyxy
        cls=[] # 保存每一个框的类别
        for obj in objs:
            # get data
            cname=obj.find('name').text # 取物体的类别名
            cid=self.cname2cid[cname] # 获取类别名对应的类别id
            box=obj.find('bndbox') # 取物体的边界框文本对象
            x1=min(max(int(box.find('xmin').text), 0), img_w-1.0)  # 获取在原始大小上的边界框坐标
            y1=min(max(int(box.find('ymin').text), 0), img_h-1.0)
            x2=min(max(int(box.find('xmax').text), 0), img_w-1.0)
            y2=min(max(int(box.find('ymax').text), 0), img_h-1.0)
            bboxs.append([x1, y1, x2, y2])
            cls.append([cid])
        return np.asarray(bboxs), np.asarray(cls)
    

    def encode(self, bboxs, clses):
        '''将直接解析后的bounding box数据解析为YOLOV1的数据格式
            Args:
                bboxs: [np.ndarray]标注的ground truth box信息: B,5(cls+xyxy)
            Return:
                yolov1_box_map: [np.ndarray]映射到YOLOV1输出特征图的数据格式: feature_size, feature_size, anchor_num*5+class_num
        '''
        # YOLOV1输出格式的特征图模板
        yolov1_box_map=np.zeros((self.feature_size, self.feature_size, (self.anchor_num*5+self.class_num))).astype(np.float32)
        
        N=bboxs.shape[0] # 当前图片中的真实框个数
        for i in range(N):
            cls_id=clses[i][0] # 获取框的类别
            x1, y1, x2, y2=bboxs[i] # 获取框的坐标信息
            w=min(max(x2-x1, 0), self.img_size)  # 原始大小上ground truth box的宽高
            h=min(max(y2-y1, 0), self.img_size)
            center_x=x1+w/2. #  # 原始大小上ground truth box的中心点坐标
            center_y=y1+h/2.
            
            w/=self.img_size*1.  # (按照原论文的意思)缩放ground truth box的宽高到: 0-1
            h/=self.img_size*1.
            x=int(center_x // (self.img_size // self.feature_size)) # 计算中心点缩放到feature_size大小后所在的cell坐标: x,y
            y=int(center_y // (self.img_size // self.feature_size)) # 计算缩放步长: (self.img_size // self.feature_size)
            
            # 将框的信息填充到特征图模板中
            yolov1_box_map[x, y,self.anchor_num*5+cls_id]=1. # 将ground truth box所属cell对应的类别设置为1.，其余类别值不动，保持为零
            for i in range(self.anchor_num): # 为当前cell的每一个bounding box, 配置ground truth box的数据信息
                offset_x=center_x / (self.img_size // self.feature_size) - x # 位于cell上的一个偏移值: 0-1
                offset_y=center_y / (self.img_size // self.feature_size) - y
                yolov1_box_map[x, y,i*5:i*5+5]=[offset_x, offset_y, w, h, 1.]  # 修改cell中的bounding box数据: xywh, obj
        return yolov1_box_map



if __name__=="__main__":
    train_dataset=YOLOV1_TrainVOCDataset(dataset_dir='data/pascalvoc', 
                                    train_txt='data/pascalvoc/train.txt', 
                                    label_txt='data/pascalvoc/label_list',
                                    feature_size=7, anchor_num=2, class_num=20, 
                                    transforms=Resize())
    img, yolov1_box_map=train_dataset[0]

    from PIL import Image
    # img.shape
    Image.fromarray(img.astype(np.uint8))




