from __future__ import annotations

from dataclasses import dataclass
from math import prod, sqrt
from typing import Any, Iterator, Sequence

import jax
import jax.numpy as jnp


ParamTree = dict[str, Any]


def _normalize_init_type(init_type: str) -> str:
    normalized = str(init_type).strip().lower().replace("_", "")
    if normalized in {"glorot", "xavier"}:
        return "glorot"
    if normalized == "lecun":
        return "lecun"
    raise ValueError(f"Unsupported init_type '{init_type}'. Use Glorot or LeCun.")


def _linear_init(rng_key, in_dim, out_dim, init_type: str):
    init_name = _normalize_init_type(init_type)
    k_w, k_b = jax.random.split(rng_key)

    if init_name == "glorot":
        # Glorot/Xavier uniform — matches stax.Dense
        bound = sqrt(6.0 / float(in_dim + out_dim))
        kernel = jax.random.uniform(
            k_w,
            shape=(in_dim, out_dim),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        )
        # Zero bias — matches stax.Dense
        bias = jnp.zeros((out_dim,), dtype=jnp.float32)
        return {"kernel": kernel, "bias": bias}

    # LeCun uniform
    bound = 1.0 / sqrt(float(max(1, in_dim)))
    kernel = jax.random.uniform(
        k_w,
        shape=(in_dim, out_dim),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )
    bias = jax.random.uniform(
        k_b,
        shape=(out_dim,),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )
    return {"kernel": kernel, "bias": bias}

def _activation(name: str, x: jax.Array) -> jax.Array:
    if name == "identity":
        return x
    if name == "relu":
        return jax.nn.relu(x)
    if name == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"Unsupported activation '{name}'.")


@dataclass(frozen=True)
class JAXModel:
    architecture: str
    input_shape: tuple[int, ...]
    num_classes: int
    hidden_dim: int
    num_hidden_layers: int
    use_bias: bool
    activation_name: str
    init_type: str

    def init(self, rng_key: jax.Array) -> ParamTree:
        if self.architecture in {"mlp", "mlp_relu", "mlp_tanh"}:
            input_dim = int(prod(self.input_shape))
            hidden_dims = [self.hidden_dim for _ in range(self.num_hidden_layers)]
            layer_dims = [input_dim, *hidden_dims, self.num_classes]

            keys = jax.random.split(rng_key, num=max(1, len(layer_dims) - 1))
            hidden_layers: list[dict[str, jax.Array]] = []
            for layer_idx in range(len(layer_dims) - 2):
                hidden_layers.append(
                    _linear_init(
                        keys[layer_idx],
                        layer_dims[layer_idx],
                        layer_dims[layer_idx + 1],
                        self.init_type,
                    )
                )
            classifier = _linear_init(
                keys[len(layer_dims) - 2],
                layer_dims[-2],
                layer_dims[-1],
                self.init_type,
            )

            return {
                "hidden_layers": hidden_layers,
                "classifier": classifier,
            }

        raise ValueError(
            f"Unsupported model architecture '{self.architecture}'. "
            "JAX runner currently supports: mlp, mlp_relu, mlp_tanh."
        )

    def apply(
        self,
        params: ParamTree,
        x: jax.Array,
        *,
        return_intermediates: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        if self.architecture in {"mlp", "mlp_relu", "mlp_tanh"}:
            x_arr = jnp.asarray(x, dtype=jnp.float32)
            x_flat = jnp.reshape(x_arr, (x_arr.shape[0], -1))

            h = x_flat
            first_linear_pre: jax.Array | None = None
            first_activation_post: jax.Array | None = None

            for layer_idx, layer in enumerate(params["hidden_layers"]):
                z = h @ layer["kernel"]
                if self.use_bias:
                    z = z + layer["bias"]
                if layer_idx == 0:
                    first_linear_pre = z
                h = _activation(self.activation_name, z)
                if layer_idx == 0:
                    first_activation_post = h

            pre_classifier = h
            logits = pre_classifier @ params["classifier"]["kernel"]
            if self.use_bias:
                logits = logits + params["classifier"]["bias"]

            if not return_intermediates:
                return logits

            intermediates: dict[str, jax.Array] = {
                "input": x_flat,
                "pre_classifier": pre_classifier,
            }
            if first_linear_pre is not None:
                intermediates["first_linear_pre"] = first_linear_pre
            if first_activation_post is not None:
                intermediates["first_activation_post"] = first_activation_post
            return logits, intermediates

        raise ValueError(f"Unsupported model architecture '{self.architecture}'.")

    def iter_named_parameters(self, params: ParamTree) -> Iterator[tuple[str, jax.Array]]:
        if self.architecture in {"mlp", "mlp_relu", "mlp_tanh"}:
            for layer_idx, layer in enumerate(params["hidden_layers"]):
                yield (f"feature_extractor.{layer_idx}.weight", layer["kernel"])
                if self.use_bias:
                    yield (f"feature_extractor.{layer_idx}.bias", layer["bias"])
        else:
            raise ValueError(f"Unsupported model architecture '{self.architecture}'.")

        yield ("fc.weight", params["classifier"]["kernel"])
        if self.use_bias:
            yield ("fc.bias", params["classifier"]["bias"])

    def classifier_weight_matrix(self, params: ParamTree) -> jax.Array:
        # Store kernels as [in_dim, out_dim], but NC geometry expects rows by class.
        return jnp.transpose(params["classifier"]["kernel"], (1, 0))

    def classifier_bias_vector(self, params: ParamTree) -> jax.Array | None:
        if not self.use_bias:
            return None
        return params["classifier"]["bias"]


@dataclass(frozen=True)
class BuiltModel:
    model: JAXModel
    params: ParamTree


def build_model(
    model_cfg: dict,
    input_shape: Sequence[int],
    num_classes: int,
    random_seed: int,
) -> BuiltModel:
    architecture = str(model_cfg.get("architecture", "mlp")).strip().lower()
    use_bias = bool(model_cfg.get("use_bias", True))
    init_type = _normalize_init_type(model_cfg.get("init_type", "glorot"))

    if architecture == "mlp":
        activation_name = "identity"
    elif architecture == "mlp_relu":
        activation_name = "relu"
    elif architecture == "mlp_tanh":
        activation_name = "tanh"
    else:
        raise ValueError(f"Unsupported model architecture '{architecture}'.")

    model = JAXModel(
        architecture=architecture,
        input_shape=tuple(int(v) for v in input_shape),
        num_classes=int(num_classes),
        hidden_dim=int(model_cfg.get("nn_width", 10)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 1)),
        use_bias=use_bias,
        activation_name=activation_name,
        init_type=init_type,
    )

    params = model.init(jax.random.PRNGKey(int(random_seed)))
    return BuiltModel(model=model, params=params)