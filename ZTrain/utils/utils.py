import random
import numpy as np
import torch


def set_seed(seed):
    """
    固定随机种子，使训练结果一致
    :param seed: 种子值
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
