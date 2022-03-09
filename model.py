import torch.nn as nn


class MyModel(nn.Module):
    """
    自定义自己的模型架构
    """

    def __init__(self, loss_fnc):
        """
        1. 初始化相关参数；
        2. 声明所需网络层
        """
        self.loss_fnc = loss_fnc

    def forward(self, *_input):
        """
        前向传播网络架构
        :param _input: 输入训练数据
        :return: 预测值pred和loss
        """