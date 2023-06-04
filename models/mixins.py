import torch
from torch import nn
from torchmetrics import (MetricCollection, MeanAbsoluteError,
                          MeanAbsolutePercentageError,
                          MeanSquaredError, R2Score,
                          Accuracy, Precision, Recall, F1Score)
from torchmetrics import Metric


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


class ClassifierMetricsMixin:
    def _init_metrics(self):
        dist_sync_on_step = True
        metrics = MetricCollection({
            'loss': LossMetric(self.loss_criterion, self.target_type),
            'accuracy': Accuracy(dist_sync_on_step=dist_sync_on_step, task=self.task, num_classes=self.num_of_classes),
            'precision': Precision(dist_sync_on_step=dist_sync_on_step, task=self.task, num_classes=self.num_of_classes),
            'recall': Recall(dist_sync_on_step=dist_sync_on_step, task=self.task, num_classes=self.num_of_classes),
            'f1': F1Score(dist_sync_on_step=dist_sync_on_step, task=self.task, num_classes=self.num_of_classes)
            })
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        self.checkpoint_metrics = metrics.clone(prefix='val_')


class ValStepMixin:
    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.acoustic_model(data)
        loss = self.loss_criterion(output.squeeze(), target.to(self.target_type))

        return {'loss': loss, 'preds': output.squeeze(), 'targets': target}


class TestStepMixin:
    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.acoustic_model(data)
        loss = self.loss_criterion(output.squeeze(), target.to(self.target_type))

        return {'loss': loss, 'preds': output.squeeze(), 'targets': target}


class ValTestStepMixin(ValStepMixin, TestStepMixin):
    ...


class TrainStepEndMixin:
    def training_step_end(self, outputs):
        preds, targets = outputs['preds'], outputs['targets']
        self.train_metrics(preds, targets)
        self.log_dict(self.train_metrics,
                      prog_bar=False,
                      logger=True, on_epoch=True,
                      batch_size=outputs['preds'].shape[0])
        return outputs['loss']


class ValStepEndMixin:
    def validation_step_end(self, outputs):
        preds, targets = outputs['preds'], outputs['targets']
        self.val_metrics(preds, targets)
        self.checkpoint_metrics(preds, targets)
        self.log_dict(self.val_metrics, prog_bar=False, logger=True, on_epoch=True,
                      batch_size=outputs['preds'].shape[0])
        self.log_dict(self.checkpoint_metrics, prog_bar=False, logger=True, on_epoch=True,
                      batch_size=outputs['preds'].shape[0])
        return outputs['loss']


class TestStepEndMixin:
    def test_step_end(self, outputs):
        preds, targets = outputs['preds'], outputs['targets']
        self.test_metrics(preds, targets)
        self.log_dict(
            self.test_metrics,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            batch_size=outputs['preds'].shape[0])
        return outputs['loss']


class TrainValTestEndMixin(TrainStepEndMixin, ValStepEndMixin, TestStepEndMixin):
    ...


class LogWeightsMixin:
    def on_train_epoch_end(self, trainer=None, pl_module=None):
        for name, params in self.named_parameters():
            if "weight" in name:
                self.log_weights(name, params)

    def log_weights(self, name, params):
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.experiment.add_histogram(tag=f'weights/{name}',
                                                 values=params.data.view(-1).detach(),
                                                 global_step=self.trainer.global_step)


class LogGradientsMixin:
    def on_after_backward(self, trainer=None, pl_module=None, optimizer=None):  # todo: another hook?
        if self.trainer.global_step % self.log_grad_and_acts_every_n_steps == 0:
            for name, params in self.named_parameters():
                if "weight" in name or params.grad is not None:
                    self.log_gradients(name, params)

    def log_gradients(self, name, params):
        self.log(f'train_meta_info/variance/gradients/{name}',
                 torch.std(params.grad.data.view(-1).detach()), on_epoch=True, on_step=True, prog_bar=False,
                 logger=True)
        self.log(f'train_meta_info/mean/gradients/{name}',
                 torch.mean(params.grad.data.view(-1).detach()), on_epoch=True, on_step=True, prog_bar=False,
                 logger=True)
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.experiment.add_histogram(tag=f'grads/{name}',
                                                 values=params.grad.data.view(-1).detach(),
                                                 global_step=self.trainer.global_step)


class LogWeightsAndGradientsMixin(LogWeightsMixin, LogGradientsMixin):
    ...


class TensorBoardLoggerMixin(LogWeightsAndGradientsMixin):
    ...


class ClassifierMixin(ClassifierMetricsMixin, ValTestStepMixin, TrainValTestEndMixin):
    def __init__(self):
        super().__init__()
        # self._init_metrics()


class ClassifierTensorBoardMixin(ClassifierMixin, TensorBoardLoggerMixin):
    def __init__(self):
        super().__init__()
