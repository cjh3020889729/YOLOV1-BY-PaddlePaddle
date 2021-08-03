# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-08-03 17:09:26
# @Last Modified by:   红白黑
# @Last Modified time: 2021-08-03 17:09:54

# -*- coding: utf-8 -*-
# @Author: 红白黑
# @Date:   2021-07-30 14:29:25
# @Last Modified by:   红白黑
# @Last Modified time: 2021-08-03 15:27:37
import paddle 
from paddle import nn
from paddle.nn import functional as F


LEAKY_SLOPE=0.1 # leaky_relu的转换率
DROPOUT_RATIO=0.5 # 丢弃层的丢弃率

class BaseConv(nn.Layer):
    '''YOLOV1网络的基本卷积块
        Args:
            in_channels: [int]输入通道数
            out_channels: [int]输出通道数
            kernel_size: [int]卷积核大小
            stride: [int]步长大小
            padding: [int]填充大小
        Return:(forward)
            x: [Tensor]最终预测输出: N, S, S, out_channels
    '''
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(BaseConv, self).__init__()
        self.conv=nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn=nn.BatchNorm2D(out_channels)
        self.leaky_relu=nn.LeakyReLU(negative_slope=LEAKY_SLOPE)

    def forward(self, inputs):
        x=self.conv(inputs)
        x=self.bn(x)
        x=self.leaky_relu(x)
        return x


class ReduceBlock(nn.Layer):
    '''YOLOV1骨干网络的特征提取块
        Args:
            in_channels: [int]输入通道数
            out_channels: [int]输出通道数
            num_block: [int]当前内嵌的block数量
            max_pool: [bool]是否在当前block中启用池化
        Return:(forward)
            x: [Tensor]最终预测输出: N, S, S, out_channels
    '''
    def __init__(self, in_channels, out_channels, num_block=1, max_pool=False):
        super(ReduceBlock, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_block=num_block
        self.max_pool=max_pool

        inchannels=in_channels
        outchannels=out_channels//2  # first input outchannels
        block_layers=[]
        for i in range(num_block):
            block_layers.append(
                nn.Sequential(
                    BaseConv(in_channels=inchannels, out_channels=outchannels, kernel_size=1),
                    BaseConv(in_channels=outchannels, out_channels=out_channels, kernel_size=3, padding=1)
                )
            )
            inchannels=out_channels  # next block use new inchannels, from out_channels
        self.blocks=nn.Sequential(*block_layers)

        if max_pool is True:
            self.max_pool_layers=nn.MaxPool2D(kernel_size=2, stride=2)
        
    def forward(self, inputs):
        x=self.blocks(inputs)
        if self.max_pool is True:
            x=self.max_pool_layers(x)
        return x


class Front_layer(nn.Layer):
    '''YOLOV1的前层特征提取网络
        Args:
            in_channels: [int]输入通道数
        Return:(forward)
            x: [Tensor]最终预测输出: N, S, S, 192
    '''
    def __init__(self, in_channels):
        super(Front_layer, self).__init__()
        # padding 3 make output scale is 1/4
        self.conv1=BaseConv(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.max_pool1=nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2=BaseConv(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.max_pool2=nn.MaxPool2D(kernel_size=2, stride=2)

        self.in_channels=in_channels
        self.out_channels=192
    
    def forward(self, inputs):
        x=self.conv1(inputs)
        x=self.max_pool1(x)
        x=self.conv2(x)
        x=self.max_pool2(x)
        return x

class Net(nn.Layer):
    '''YOLOV1的骨干特征提取网络
        Args:
            in_channels: [int]输入通道数
        Return:(forward)
            x: [Tensor]最终预测输出: N, S, S, 1024
    '''
    def __init__(self, in_channels=3):
        super(Net, self).__init__()
        self.front_layer=Front_layer(in_channels=in_channels)

        self.block_num_config=[1, 1, 4, 1, 2]
        self.block_outchannel_config=[256, 512, 512, 1024, 1024]
        self.block_maxpool_config=[False, True, False, True, False]

        net_blocks=[]
        inchannel=self.front_layer.out_channels
        for i in range(len(self.block_num_config)):
            outchannel=self.block_outchannel_config[i]
            num_block=self.block_num_config[i]
            block_maxpool=self.block_maxpool_config[i]
            net_blocks.append(
                ReduceBlock(in_channels=inchannel, out_channels=outchannel,
                            num_block=num_block, max_pool=block_maxpool)
            )
            inchannel=outchannel  # next block inchannel be current outchannel
        self.net_blocks=nn.Sequential(*net_blocks)
        
        self.out_channels=inchannel  # the net outchannel be final block outchannel
        
    def forward(self, inputs):
        x=self.front_layer(inputs)
        x=self.net_blocks(x)
        return x

class Ascension_Layer(nn.Layer):
    '''YOLOV1的骨干特征提升网络
        Args:
            in_channels: [int]输入通道数
        Return:(forward)
            x: [Tensor]最终预测输出: N, S, S, 1024
    '''
    def __init__(self, in_channels):
        super(Ascension_Layer, self).__init__()
        self.conv1=BaseConv(in_channels=in_channels, out_channels=1024, kernel_size=3, padding=1)
        self.conv2=BaseConv(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.conv3=BaseConv(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv4=BaseConv(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        self.out_channels=1024
        
    def forward(self, inputs):
        x=self.conv1(inputs)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.transpose((0, 2, 3, 1))  # trans dim
        return x


class OutLinaer_Layer(nn.Layer):
    '''YOLOV1输出的线性网络
        Args:
            in_feature_size: [int]输入特征数
            out_feature_size: [int]输出特征数
        Return:(forward)
            x: [Tensor]最终预测输出: N, S, S, out_feature_size
    '''
    def __init__(self, in_feature_size, out_feature_size):
        super(OutLinaer_Layer, self).__init__()
        self.linear1=nn.Linear(in_features=in_feature_size, out_features=4096)
        self.linear2=nn.Linear(in_features=4096, out_features=out_feature_size)

        self.dropout=nn.Dropout(p=DROPOUT_RATIO)
        self.leaky_relu=nn.LeakyReLU(negative_slope=LEAKY_SLOPE)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, inputs):
        x=self.linear1(inputs)
        x=self.leaky_relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.sigmoid(x)
        return x



# if __name__ == "__main__":
#     img_shape=[448, 448, 3]
#     data_shape=[1, img_shape[2], img_shape[0], img_shape[1]]
#     print('Input data shape: ', data_shape)

#     import numpy as np
#     data=np.random.randint(0, 256, data_shape).astype(np.float32)/255.
#     data=paddle.to_tensor(data)

#     model=Net()

#     y_pred=model(data)
    
#     print('Model output shape: ', y_pred.shape)



