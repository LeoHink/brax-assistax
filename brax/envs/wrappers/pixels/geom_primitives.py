from typing import Optional
from flax import struct
import jax
import jax.numpy as jp
from brax import math
from brax.base import Base, Transform


@struct.dataclass
class Geometry(Base):
    """A surface or spatial volume with a shape and material properties.

    Attributes:
      link_idx: Link index to which this Geometry is attached
      transform: transform for the geometry frame relative to the link frame, or
        relative to the world frame in the case of unparented geometry
      friction: resistance encountered when sliding against another geometry
      elasticity: bounce/restitution encountered when hitting another geometry
      solver_params: (7,) solver parameters (reference, impedance)
    """

    link_idx: Optional[jax.Array]
    transform: Transform
    friction: jax.Array
    elasticity: jax.Array
    solver_params: jax.Array
