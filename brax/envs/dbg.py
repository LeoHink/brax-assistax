"""
We are attempting to replace Env.sys.geoms from brax.v1.base.System:
  - geoms: list of batched geoms grouped by type

A brax.v1.base.Geometry (for us: pixels.geom_primitives.Geometry)
Attributes needed:
    - link_idx: Link index to which this Geometry is attached
    - transform: transform for the geometry frame relative to the link frame, or
        relative to the world frame in the case of unparented geometry
    - friction: resistance encountered when sliding against another geometry
    - elasticity: bounce/restitution encountered when hitting another geometry
    - solver_params: (7,) solver parameters (reference, impedance)


I think we need to access mujoco.mjx._src.types.Data, as this changes dynamically every state. mjx_model does not
https://github.com/google-deepmind/mujoco/blob/f24de91cc9d6724b838bafc7883a0d463d7f76c3/mjx/mujoco/mjx/_src/types.py#L1145


Plane  (0):
    For plane we don't need to grab anything from the state nor sys. We use the _GROUND variable instantiated independently

Capsule (3):
    radius:
    length:
    tex: col.rgba[:3].reshape((1, 1, 3))



In original brax, mj_model = mujoco.MjModel.from_xml_string(xml) (https://github.com/trevormcinroe/pixel_brax/blob/main/brax/test_utils.py)
"""

from brax import envs
from mujoco.mjx._src.types import GeomType
import jax


from flax.struct import dataclass, field
from typing import List
import inspect
from PIL import Image
import numpy as np
from tqdm import tqdm
import imageio
import wandb
import os
import json

os.environ["WANDB_DIR"] = "./wandb"
os.environ["WANDB_CACHE_DIR"] = "./wandb"
os.environ["WANDB_CONFIG_DIR"] = "./wandb"
os.environ["WANDB_DATA_DIR"] = "./wandb"

# print(f"dir: {os.listdir('../../')}")

"""LOGGING"""
with open("../../wandb.txt", "r") as f:
    API_KEY = json.load(f)["api_key"]

wandb.init(
    project="testing-assistax",
    entity="trevor-mcinroe",
    name="pls",
)


env = envs.create(
    "scratchitch",
    batch_size=3,
    pixel_obs={
        "hw": 84,
        "frame_stack": 1,
        "return_float32": False,
        "cache_objects": True,
        "n_envs": 3,
    },
)
print(f"sys: {env.sys.mj_model.ngeom} // {env.sys.mj_model.nbody}")


key = jax.random.PRNGKey(0)
obs = env.reset(key)
frames = [np.array(obs.pixels[0])[None]]

keys = jax.random.split(key, obs.pixels.shape[0])
action = jax.random.uniform(key, shape=(obs.pixels.shape[0], env.action_size))
_step_fn = jax.jit(env.step)

import time
def _step_env_loop(carry, unused):
    key, obs, iterator, time_carry = carry
    begin = time.time()
    _, key = jax.random.split(key)
    action = jax.random.uniform(key, shape=(obs.pixels.shape[0], env.action_size))
    obs = _step_fn(keys, obs, action)
    time_carry += ((time.time() - begin) * iterator)
    return (key, obs, iterator + 1 - iterator, time_carry), ()

start = time.time()
print("Begin loop.")
(_, _, iterator, time_carry), _ = jax.lax.scan(_step_env_loop, (key, obs, 0, 0.0), (), length=100)
print(f"Took {time.time() - start} seconds")
print(f"{iterator} // {time_carry}")
qqq

print(f"clearing compile time...")
for _ in range(3):
    _step_fn(keys, obs, action)
    action = jax.random.uniform(key, shape=(obs.pixels.shape[0], env.action_size))
    obs = _step_fn(keys, obs, action)


for _ in tqdm(range(100)):
    _, key = jax.random.split(key)
    obs = _step_fn(keys, obs, action)
    frames.append(np.array(obs.pixels[0])[None])
    action = jax.random.uniform(key, shape=(obs.pixels.shape[0], env.action_size))


wandb.log(
    {"video": wandb.Video(np.concatenate(frames, 0).transpose(0, 3, 1, 2), fps=24)}
)
# qqq
# imageio.mimsave("./random_actions.mp4", frames, fps=10)
# qqq

for i in range(10):
    im = np.array(obs.pixels[i])  # .transpose(2, 0, 1)
    print(f"im: {im.shape} // {im.dtype}")
    im = Image.fromarray(im)
    im.save(f"./please_work_{i}.png")

qqq


# import inspect
#
# dataclass_instance = obs.pipeline_state
# members = inspect.getmembers(type(dataclass_instance))
# fields = list(dict(members)["__dataclass_fields__"].values())
# for v in fields:
#    print(f"{v.name} // {dataclass_instance[v.name].shape}")


qqqq
ru.render_pixels(env.sys, obs.pipeline_state)

print(f"done")
qqq


print(f"info: {obs.pipeline_state.geom_xpos.shape} // {obs.pipeline_state.q.shape}")
print(f"sys: {env.sys.mj_model.ngeom} // {env.sys.mj_model.nbody}")
# just a bunch of ints [0, 7, 4, 4, 2, ...]

# members = inspect.getmembers(type(obs.pipeline_state.xd))
# fields = list(dict(members)["__dataclass_fields__"].values())
# for v in fields:
#    print(f"{v.name} // {v.type}")
print(f"sys: {env.sys.mj_model.geom_type}")


# Array of floate
print(f"capsule radius: {env.sys.mj_model.geom_rbound.shape}")
print(f"capsule len:")

# sizes are *not* batched [106, 3]
# print(f"size: {env.sys.mj_model.geom_pos.shape}")
print("done.")
