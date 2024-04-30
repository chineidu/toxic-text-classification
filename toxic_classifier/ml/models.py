from typing import Any

from torch import Tensor, nn
from transformers import BatchEncoding
from typeguard import typechecked

from toxic_classifier.ml.adapters import Adapter
from toxic_classifier.ml.backbone import BaseBackbone
from toxic_classifier.ml.head import Head
from toxic_classifier.ml.transformations import BaseTransformation


class Model(nn.Module):
    pass


class BinaryTextClassifier(Model):
    """This is used to define a binary text classifier."""

    @typechecked
    def __init__(
        self,
        backbone: BaseBackbone | Any,
        head: Head | Any,
        adapter: Adapter | None = None,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.adapter = adapter

    @typechecked
    def forward(self, encodings: BatchEncoding | Any) -> Tensor:
        """This is used to apply the backbone, the adapter and the head."""
        output: Tensor = self.backbone(encodings).pooler_output
        if self.adapter is not None:
            output = self.adapter(output)
        output = self.head(output).squeeze()
        return output

    @typechecked
    def get_transformation(self) -> BaseTransformation:
        """This is used to get the transformation of the backbone."""
        return self.backbone.get_transformation()
