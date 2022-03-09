import torch

from ZTrain.training import Trainer

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
callbacks = []
trainer.fit(train_data, dev_data_raw, args.batch_size, args.epochs, callbacks=callbacks)
trainer.evaluate(test_data_raw)
# trainer.predict(test_data_raw)
