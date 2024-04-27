from torch import Tensor, nn
from transformers import BatchEncoding
from typeguard import typechecked

from toxic_classifier.ml.adapters import Adapter
from toxic_classifier.ml.backbone import Backbone
from toxic_classifier.ml.head import Head
from toxic_classifier.ml.transformations import Transformation


class Model(nn.Module):
    pass


class BinaryTextClassifier(Model):
    """This is used to define a binary text classifier."""

    @typechecked
    def __call__(self, backbone: Backbone, head: Head, adapter: Adapter | None = None) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.adapter = adapter

    @typechecked
    def forward(self, encodings: BatchEncoding) -> Tensor:
        """This is used to apply the backbone, the adapter and the head."""
        features = self.backbone(encodings)
        if self.adapter:
            features = self.adapter(features)
        output = self.head(features)
        return output

    @typechecked
    def get_transformation(self) -> Transformation:
        """This is used to get the transformation of the backbone."""
        return self.backbone.get_transformation()
