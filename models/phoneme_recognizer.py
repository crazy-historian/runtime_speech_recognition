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


# class ClassifierModuleMixin(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#
#     def _set_metrics(self):
#         metrics = MetricCollection({
#             'loss': LossMetric(self.loss_criterion, self.target_type),
#             'accuracy': Accuracy(average='macro', num_classes=self.num_classes, dist_sync_on_step=True),
#             'precision': Precision(average='macro', num_classes=self.num_classes, dist_sync_on_step=True),
#             'recall': Recall(average='macro', num_classes=self.num_classes, dist_sync_on_step=True),
#             'f1': F1Score(average='macro', num_classes=self.num_classes, dist_sync_on_step=True)
#         })
#         self.train_metrics = metrics.clone(prefix='train/')
#         self.val_metrics = metrics.clone(prefix='val/')
#         self.test_metrics = metrics.clone(prefix='test/')
#         self.checkpoint_metrics = metrics.clone(prefix='checkpoint/')
#
#     def training_step_end(self, outputs):
#         preds, targets = outputs['preds'], outputs['targets']
#         self.train_metrics(preds, targets)
#         self.log_dict(self.train_metrics,
#                       prog_bar=False,
#                       logger=True, on_epoch=True,
#                       batch_size=outputs['preds'].shape[0])
#         return outputs['loss']
#
#     def validation_step_end(self, outputs):
#         preds, targets = outputs['preds'], outputs['targets']
#         self.val_metrics(preds, targets)
#         self.checkpoint_metrics(preds, targets)
#         self.log_dict(self.val_metrics, prog_bar=False, logger=True, on_epoch=True,
#                       batch_size=outputs['preds'].shape[0])
#         self.log_dict(self.checkpoint_metrics, prog_bar=False, logger=True, on_epoch=True,
#                       batch_size=outputs['preds'].shape[0])
#         return outputs['loss']
#
#     def test_step_end(self, outputs):
#         preds, targets = outputs['preds'], outputs['targets']
#         self.test_metrics(preds, targets)
#         self.log_dict(
#             self.test_metrics,
#             prog_bar=False,
#             logger=True,
#             on_epoch=True,
#             batch_size=outputs['preds'].shape[0])
#         return outputs['loss']
#
#     def on_after_backward(self, trainer=None, pl_module=None, optimizer=None):
#         if self.trainer.global_step % self.log_grad_and_acts_every_n_steps == 0:
#             for name, params in self.named_parameters():
#                 if "weight" not in name or params.grad is None:
#                     continue
#                 self.log('train_meta_info/variance/gradients/{}'.format(name),
#                          torch.std(params.grad.data.view(-1).detach()), on_epoch=True, on_step=True, prog_bar=False,
#                          logger=True)
#                 self.log('train_meta_info/mean/gradients/{}'.format(name),
#                          torch.mean(params.grad.data.view(-1).detach()), on_epoch=True, on_step=True, prog_bar=False,
#                          logger=True)
#                 if hasattr(self, 'logger') and self.logger is not None:
#                     self.logger.experiment.add_histogram(tag=f'grads/{name}',
#                                                          values=params.grad.data.view(-1).detach(),
#                                                          global_step=self.trainer.global_step)
#
#     def on_train_epoch_end(self, trainer=None, pl_module=None):
#         for name, params in self.named_parameters():
#             if "weight" not in name:
#                 continue
#             if hasattr(self, 'logger') and self.logger is not None:
#                 self.logger.experiment.add_histogram(tag=f'weights/{name}',
#                                                      values=params.data.view(-1).detach(),
#                                                      global_step=self.trainer.global_step)


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
                 loss_criterion=nn.NLLLoss(),
                 target_type=torch.float32
                 ):
        #
        super().__init__()
        self.target_type = target_type
        self.loss_criterion = loss_criterion
        self.save_hyperparameters(ignore=['loss_criterion', 'target_type'])
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
