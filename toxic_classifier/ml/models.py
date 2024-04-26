from torch import Tensor, nn
from transformers import BatchEncoding

from toxic_classifier.ml.adapters import Adapter
from toxic_classifier.ml.backbone import Backbone
from toxic_classifier.ml.head import Head


class Model(nn.Module):
    pass


class BinaryTextClassifier(Model):
    def __call__(self, backbone: Backbone, head: Head, adapter: Adapter | None = None) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.adapter = adapter


def forward(self, encodings: BatchEncoding) -> Tensor:
    features = self.backbone(encodings)
    if self.adapter:
        features = self.adapter(features)
    output = self.head(features)
    return output
