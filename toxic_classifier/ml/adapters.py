"""Adapters in deep learning are parameter-efficient building blocks that enable modular
fine-tuning of pre-trained models for various tasks, offering flexibility and efficient training."""

from operator import attrgetter
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typeguard import typechecked


class Adapter(nn.Module):
    """Adapter module to wrap a backbone with adapter layers."""

    pass


class Normalization(nn.Module):
    @typechecked
    def __init__(self, p: float = 2.0) -> None:
        """This normalizes tensors along a specified dimension by a given norm. This is useful for
        preparing data (zero mean, unit variance) or processing activations in neural networks.
        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the normalization layer."""
        return F.normalize(x, p=self.p, dim=1)


class FCLayer(Adapter):
    @typechecked
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        activation_fn: nn.Module | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        order: str = "LABDN",
    ) -> None:
        """order (LABDN): Layout of layers (L)inear, (A)ctivation, (B)atch Norm, (D)ropout,
        (N)ormalization"""

        super().__init__()

        order = order.upper()
        # Add layers based on order
        # Linear layer
        layers: dict[str, tuple[nn.Module, ...]] = {
            "L": ("linear", nn.Linear(in_features, out_features, bias=bias))
        }

        if activation_fn is not None:
            # Activation layer
            layers["A"] = ("activation", activation_fn)

        if batch_norm:
            # Batch normalization layer
            layers["B"] = (
                "batch_norm",
                nn.BatchNorm1d(
                    out_features if order.index("L") < order.index("B") else in_features
                ),
            )

        if dropout:
            # Dropout layer
            layers["D"] = ("dropout", nn.Dropout(p=dropout))

        if "N" in order:
            # Normalization layer
            layers["N"] = ("normalization", Normalization())

        # Initialize and add layers
        self.layers = nn.Sequential()
        for layer_code in order:
            if layer_code in layers:
                name, layer = layers[layer_code]
                self.layers.add_module(name, layer)

    @typechecked
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through fully connected layers."""
        output: Tensor = self.layers(x)
        return output


class MLPLayer(Adapter):
    @typechecked
    def __init__(
        self,
        output_feature_sizes: list[int],
        biases: list[bool] | None = None,
        activation_fns: list[str | None] | None = None,
        dropout_drop_probs: list[float] | None = None,
        batch_norms: list[bool] | None = None,
        order: str = "LABDN",
        standardize_input: bool = True,
    ) -> None:
        """Multi Layers Perceptron"""
        super().__init__()

        self.output_feature_sizes = output_feature_sizes
        self.output_embedding_size = output_feature_sizes[-1]

        # Exclude the last layer (embedding size)
        num_of_layers = len(output_feature_sizes) - 1
        biases = [False] * num_of_layers if biases is None else biases
        activation_functions: list[str | None] = (
            [None] * num_of_layers if activation_fns is None else activation_fns
        )
        dropout_drop_probabilities: list[float] = (
            [0.0] * num_of_layers if dropout_drop_probs is None else dropout_drop_probs
        )
        batch_normalizations: list[bool] = (
            [False] * num_of_layers if batch_norms is None else batch_norms
        )

        assert (
            num_of_layers
            == len(biases)
            == len(activation_functions)
            == len(dropout_drop_probabilities)
        )
        self.adapter = nn.Sequential()
        if standardize_input:
            self.adapter.add_module(
                "standardize_input",
                nn.LayerNorm(normalized_shape=output_feature_sizes[0], elementwise_affine=False),
            )
        # Add the fully connected layers
        for layer_idx in range(num_of_layers):
            activation_fn = activation_functions[layer_idx]
            self.adapter.add_module(
                f"fc_layer_{layer_idx}",
                FCLayer(
                    in_features=output_feature_sizes[layer_idx],
                    out_features=output_feature_sizes[layer_idx + 1],
                    bias=biases[layer_idx],
                    activation_fn=(
                        getattr(nn, activation_fn)() if activation_fn is not None else None
                    ),
                    dropout=dropout_drop_probabilities[layer_idx],
                    batch_norm=batch_normalizations[layer_idx],
                    order=order,
                ),
            )

    @typechecked
    def forward(self, backbone_output: Tensor) -> Tensor:
        """Forward pass through MLP layers."""
        output = self.adapter(backbone_output)
        return output


class MLPWithPooling(Adapter):
    @typechecked
    def __init__(
        self,
        output_feature_sizes: list[int],
        biases: list[bool] | None = None,
        activation_fns: list[str | None] | None = None,
        dropout_drop_probs: list[float] | None = None,
        batch_norms: list[bool] | None = None,
        order: str = "LABDN",
        standardize_input: bool = True,
        pooling_method: str | None = None,
        output_attribute_to_use: (Literal["pooler_output", "last_hidden_state"] | None) = None,
    ) -> None:
        """Multi Layers Perceptron with Pooling"""
        super().__init__()

        self.output_feature_sizes = output_feature_sizes
        self.output_embedding_size = output_feature_sizes[-1]

        # Exclude the last layer (embedding size)
        num_of_layers = len(output_feature_sizes) - 1
        if num_of_layers > 0:
            self.projection = MLPLayer(
                output_feature_sizes=output_feature_sizes,
                biases=biases,
                activation_fns=activation_fns,
                dropout_drop_probs=dropout_drop_probs,
                batch_norms=batch_norms,
                order=order,
                standardize_input=standardize_input,
            )
        else:
            self.projection = nn.Identity()  # type: ignore

        if pooling_method == "mean_pooler":
            self.pooler = mean_pool_tokens
        elif pooling_method == "cls_pooler":
            self.pooler = cls_pool_tokens
        else:
            self.pooler = nn.Identity()

        if output_attribute_to_use is not None:
            self.get_output_tensor = attrgetter(output_attribute_to_use)
        else:
            self.get_output_tensor = nn.Identity()  # type: ignore

    @typechecked
    def forward(self, backbone_output: BaseModelOutputWithPooling) -> Tensor:
        """Forward pass through MLP layers and pooling."""
        output: Tensor = self.get_output_tensor(backbone_output)
        output = self.pooler(output)
        output = self.projection(output)
        assert isinstance(output, Tensor)
        return output


@typechecked
def mean_pool_tokens(tensor: Tensor) -> Tensor:
    """Average pool the token embeddings of the last hidden state."""
    dims: int = len(tensor.shape)
    if dims != 3:
        raise ValueError(f"Tokens pooling expects exactly 3 dimensional tensor, got: {dims}")
    return torch.mean(tensor, dim=1)


@typechecked
def cls_pool_tokens(tensor: Tensor) -> Tensor:
    """
    Pools the token embeddings by taking the first token (usually the [CLS] token) from
    the last hidden state.

    Params:
    -------
        tensor (torch.Tensor): The input tensor of shape (batch_size, sequence_length,
        embedding_size).

    Returns:
    --------
        torch.Tensor: The pooled tensor of shape (batch_size, embedding_size).
    """
    dims: int = len(tensor.shape)
    if dims != 3:
        raise ValueError(f"Tokens pooling expects exactly 3 dimensional tensor, got: {dims}")
    return tensor[:, 0, :]
