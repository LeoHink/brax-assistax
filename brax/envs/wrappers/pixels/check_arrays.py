import numpy as np
import jax.numpy as jnp
import pickle

mesh_idx = 0
keys = ["vertices", "vertex_normals", "faces"]

numpy_arrays = {}
jax_arrays = {}


for mesh_idx in range(57):
    for k in keys:
        with open(
            f"/disk/scratch1/tmcinroe/data_assistax/mesh_{mesh_idx}_{k}_numpy.data",
            "rb",
        ) as f:
            numpy_data = pickle.load(f)

        with open(
            f"/disk/scratch1/tmcinroe/data_assistax/mesh_{mesh_idx}_{k}_jax.data", "rb"
        ) as f:
            jax_data = pickle.load(f)

        if not np.allclose(numpy_data, jax_data):
            print(f"{mesh_idx}: {k} // {(numpy_data - jax_data).sum()}")
