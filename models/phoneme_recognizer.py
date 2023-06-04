import pytorch_lightning as pl

from typing import Any

from models.mixins import *


class LossMetric(Metric):
    def __init__(self, torch_module, target_type=torch.float32):
        super().__init__()
        self.loss = torch_module
        self.target_type = target_type
        self.add_state('loss_value', default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.loss_value = self.loss(preds.squeeze(), target.to(self.target_type))

    def compute(self):
        return self.loss_value

class AudioPreprocessorCallback(pl.Callback):
    def __init__(self, transform: nn.Module, device: str):
        self.transform = transform
        self.transform.to(device)

    def on_train_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ):
        data, labels = batch
        data = self.transform(data)
        return data, labels

    def on_validation_batch_start(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any, batch_idx: int,
            dataloader_idx: int
    ):
        data, labels = batch
        data = self.transform(data)
        return data, labels

    def on_test_batch_start(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int
    ):
        data, labels = batch
        data = self.transform(data)
        return data, labels


class PhonemeRecognizer(ClassifierMixin, pl.LightningModule):
    def __init__(self,
                 acoustic_model: torch.nn.Module,
                 model_params: dict,
                 lr: float = 3e-2,
                 num_of_classes: int = 2,
                 loss_criterion=nn.NLLLoss(),
                 target_type=torch.float32
                 ):
        #
        super().__init__()
        self.target_type = target_type
        self.loss_criterion = loss_criterion
        self.num_of_classes = num_of_classes
        if self.num_of_classes == 2:
            self.task = 'binary'
        else:
            self.task = 'multiclass'
        self.save_hyperparameters(ignore=['loss_criterion', 'target_type', 'num_of_classes'])
        self._init_metrics()
        self.log_grad_and_acts_every_n_steps = 10
        self.acoustic_model = acoustic_model(**model_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.acoustic_model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        data, target = batch
        output = self.acoustic_model(data)
        loss = self.loss_criterion(output.squeeze(), target.to(self.target_type))

        return {'loss': loss, 'preds': output.squeeze(), 'targets': target}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.acoustic_model(data)
        loss = self.loss_criterion(output.squeeze(), target.to(self.target_type))

        return {'loss': loss, 'preds': output.squeeze(), 'targets': target}

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.hparams.lr)
        return optimizer
