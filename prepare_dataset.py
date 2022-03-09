from torch.utils.data import Dataset


class PrepareDataset:
    def __init__(self, file):
        """
        读取文件
        :param file:
        >>> with open(file, 'r') as f:
        >>>     self.data = f.read()
        """

    def process(self):
        """
        处理数据 ————> 模型输入
        :return: object
            the data input to your model
        """


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx].process()
