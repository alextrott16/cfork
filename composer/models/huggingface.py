from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional, List

from torch import nn
from torchmetrics import Metric
from torchmetrics.collections import MetricCollection
from composer.models.base import ComposerModel

if TYPE_CHECKING:
    import transformers

__all__ = ["HuggingFaceModel"]


class HuggingFaceModel(ComposerModel):
    """
    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`):  You can  use your own models defined as
        `torch.nn.Module` as long as they work the same way as the 🤗 Transformers models.
        metrics (list[Metric], optional): list of torchmetrics to apply to the output of `validate`.

    Example:

    .. testcode::

        import transformers
        from composer.models import HuggingFaceModel

        hf_model = transformers.AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model = HuggingFaceModel(hf_model)
    """

    def __init__(self,
                 model: Union[transformers.PreTrainedModel, nn.Module],
                 metrics: Optional[List[Metric]] = None) -> None:
        super().__init__()
        self.model = model

        metric_collection = MetricCollection(metrics)
        self.train_metrics = metric_collection.clone(prefix='train_')
        self.valid_metrics = metric_collection.clone(prefix='val_')

    def forward(self, batch):
        output = self.model(**batch)
        return output

    def loss(self, outputs, batch):
        return outputs['loss']

    def validate(self, batch):
        labels = batch.pop('labels')
        output = self.forward(batch)
        output = output['logits']
        return output, labels

    def metrics(self, train: bool = False):
        return self.train_metrics if train else self.valid_metrics
