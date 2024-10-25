"""
Utilities for rendering pixel observations directly on an XLA device
"""

from typing import Optional, Dict
from brax import base
import flax
import jax.numpy as jnp


@flax.struct.dataclass
class PixelState(base.Base):
    pipeline_state: Optional[base.State]
    obs: jnp.ndarray
    pixels: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    key: Optional[jnp.ndarray]
    metrics: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)
    info: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)
