# 💡 ZTrain: 基于Pytorch的一个模块化的训练、验证、测试框架

提供一个模块化，易于扩展的训练、验证、测试框架和API，和一些常用的NLP任务示例

---

请注意: *ZTrain*仍然处于早期阶段，API很可能会继续变化。


# 🚀 安装

请确保你预先安装了[PyTorch](https://pytorch.org) 和 [Transformers](https://huggingface.co/docs/transformers/index).

```bash
# 即将推出
pip install -U ZTrain
```

或者

```bash
# 目前推荐
git clone https://github.com/cyberfish1120/ZTrain.git && cd ZTrain
pip install -e . --verbose
```
“-e”表示“可编辑”模式，这样你就不用每次修改都重新安装了

# ⚡ 快速上手

## 一个简单的例子

```python
import torch

from ZTrain.training import Trainer
from ZTrain.training.callbacks import EarlyStopping

from model import MyModel
from metric import metric
from config import arg_parse
from prepare_dataset import PrepareDataset, MyDataset

args = arg_parse()

train_data_raw = PrepareDataset(file=None)

train_data = MyDataset(train_data_raw)

dev_data_raw = PrepareDataset(file=None)
test_data_raw = PrepareDataset(file=None)

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

trainer = Trainer(model, optimizer=optimizer, scheduler=scheduler, metric=metric, device=args.device)
callbacks = EarlyStopping(monitor='loss', patience=3)
trainer.fit(train_data, dev_data_raw, args.batch_size, args.epochs, callbacks=callbacks)
trainer.evaluate(test_data_raw)
# trainer.predict(test_data_raw)
```

# 👀 更新
