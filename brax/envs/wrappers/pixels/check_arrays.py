import numpy as np
import jax.numpy as jnp
import pickle

mesh_idx = 0
keys = ["vertices", "vertex_normals", "faces"]

numpy_arrays = {}
jax_arrays = {}

for k in keys:
    with open(
        f"/disk/scratch1/tmcinroe/data_assistax/mesh_{mesh_idx}_{k}_numpy.data", "rb"
    ) as f:
        data = pickle.load(f)

    print(f"numpy: {data}")

    with open(
        f"/disk/scratch1/tmcinroe/data_assistax/mesh_{mesh_idx}_{k}_jax.data", "rb"
    ) as f:
        data = pickle.load(f)

    print(f"jax: {data}")
    qqq
