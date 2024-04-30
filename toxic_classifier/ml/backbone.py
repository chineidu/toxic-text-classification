"""This is used to define the model without any head (classification, regression etc.)"""

from torch import nn
from transformers import AutoConfig, AutoModel, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling

from toxic_classifier.ml.transformations import BaseTransformation


class BaseBackbone(nn.Module):
    """Base class for model backbones. It's used to define models without heads."""

    def __init__(self, transformation: BaseTransformation) -> None:
        """This is used to initialize the backbone."""
        super().__init__()

        self.transformation = transformation

    def get_transformation(self) -> BaseTransformation:
        return self.transformation


class HuggingFaceBackbone(BaseBackbone):
    """Load a pretrained or random weights HuggingFace model as backbone."""

    def __init__(
        self,
        transformation: BaseTransformation,
        pretrained_model_name_or_path: str,
        pretrained: bool,
    ) -> None:
        super().__init__(transformation)

        self.backbone = self.get_backbone(pretrained_model_name_or_path, pretrained)

    def forward(self, encodings: BatchEncoding) -> BaseModelOutputWithPooling:
        """Every class that inherits from Backbone (nn.Module) should implement
        this forward pass."""
        output: BaseModelOutputWithPooling = self.backbone(**encodings)
        return output

    def get_backbone(self, pretrained_model_name_or_path: str, pretrained: bool) -> nn.Module:
        """Loads the pretrained/random weights transformer model."""
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        if pretrained:
            backbone_from_pretrained: nn.Module = AutoModel.from_pretrained(
                pretrained_model_name_or_path, config=config
            )
            return backbone_from_pretrained

        # Assign random weights
        backbone_from_config = AutoModel.from_config(config)
        return backbone_from_config
