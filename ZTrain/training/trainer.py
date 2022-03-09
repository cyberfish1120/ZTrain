import torch
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler=None,
                 metric=None,
                 device='cpu'
                 ):
        """
        统一训练框架
        :param model: 待训练模型
        :param optimizer: 优化器
        :param scheduler: 动态学习率
        :param metric: 评价指标
        :param device: 模型训练位置：cpu or gpu
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fnc = loss_fnc
        self.scheduler = scheduler
        self.metric = metric
        self.device = device

    def fit(self,
            train_data,
            dev_data=None,
            batch_size=32,
            epochs=10,
            callbacks=None
            ):
        """
        训练+（验证）
        :param train_data: 训练集数据
        :param dev_data: 验证集数据（可选）
        :param batch_size: batch size
        :param epochs: 迭代轮数
        :param callbacks: 回调函数
        :return: 训练（验证）相关log
        """
        self.callbacks = callbacks
        model = self.model

        callbacks.on_train_begin()
        try:
            for epoch in range(epochs):
                callbacks.on_epoch_begin(epoch)
                train_logs = self.train_step(train_data)
                print(f'Epoch: {epoch}/{self.args.epochs} train info: {train_logs}')

                if dev_data:
                    valid_logs = self.evaluate(dev_data)
                    print(f'Epoch: {epoch}/{self.args.epochs} dev info: {train_logs}')

                callbacks.on_epoch_end()

                if model.stop_training:
                    print(f"Early Stopping at Epoch {epoch}")
                    break

        finally:
            callbacks.on_train_end()

        return self

    def train_step(self, dataloader):
        """
        完成一轮（整个训练集）训练
        :param dataloader: DataLoader 训练集数据
        :return: 训练相关logs
        """
        device = self.device
        model = self.model
        optimizer = self.optimizer
        callbacks = self.callbacks

        model.train()
        for batch_data in tqdm(dataloader):
            callbacks.on_train_batch_begin()
            _, loss = model(batch_data.to(device))
            loss.backward()
            optimizer.step()
            model.zero_grad()

            callbacks.on_train_batch_end()

    def evaluate(self, test_data):
        device = self.device
        model = self.model
        metric = self.metric
        callbacks = self.callbacks
        preds, labels = [], []

        model.eval()
        for data, label in tqdm(test_data):
            callbacks.on_test_batch_begin()

            pred, _ = model(data.to(device))
            preds.append(pred)
            labels.append(label)
        result = metric(preds, labels)
        return result
