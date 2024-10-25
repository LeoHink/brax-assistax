"""
Utilities for rendering pixel observations directly on an XLA device
"""

from typing import Optional, Dict
from brax import base
import flax
import jax.numpy as jnp
import jax


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


@jax.jit
def render_pixels(sys: brax.System, pipeline_states: brax.State):
  # (1) grab the cameras and the view targets. The camera object contains its own view target
  # the extra bit we grab with _get_targets() is only used to render shadows. Maybe we can remove?
  # print(f'Within render pixels')

  # The "current_frame" arg is meant to work for the "video distractor" case
  batched_camera = _get_cameras(sys, pipeline_states)
  # print(f'after batched_camera')
  batched_target = _get_targets(pipeline_states)
  # print(f'after batched_target')
  objs = _build_objects(sys)
  # print(f'after _build_objects')
  images = _render(objs, pipeline_states, batched_camera, batched_target)
  # print(f'after _render')
  # return None
  return images

# The groundplane used in the environment. This component is important for the locomoation
# tasks, as the agent needs to understand the contact between the plane and the rigid body
def grid(grid_size: int, color) -> jnp.ndarray:
  grid = jnp.zeros((grid_size, grid_size, 3), dtype=jnp.single)
  grid[:, :] = jnp.array(color) / 255.0
  grid[0] = jnp.zeros((grid_size, 3), dtype=jnp.single)
  # to reverse texture along y direction
  grid[:, -1] = jnp.zeros((grid_size, 3), dtype=jnp.single)
  return jnp.asarray(grid)

_GROUND: jnp.ndarray = grid(hw, [200, 200, 200])


# TODO; make every grom primitive congruent. That way we can vmap over the objects instead of looping over
#  them. This could grant us a large speedup.
class Obj(NamedTuple):
  """An object to be rendered in the scene.

  Assume the system is unchanged throughout the rendering.

  col is accessed from the batched geoms `sys.geoms`, representing one geom.
  """
  instance: Instance
  """An instance to be rendered in the scene, defined by jaxrenderer."""
  link_idx: int
  """col.link_idx if col.link_idx is not None else -1"""
  off: jnp.ndarray
  """col.transform.rot"""
  rot: jnp.ndarray
  """col.transform.rot"""

@jax.jit
 def _build_objects(sys: brax.System) -> list[Obj]:
   """
   Converts a brax System to a list of Obj.

   Args:
     sys:

   Returns:

   """
   objs: list[Obj] = []

   def take_i(obj, i):
     return jax.tree_map(lambda x: jnp.take(x, i, axis=0), obj)

   testing = []
   for batch in sys.geoms:
     num_geoms = len(batch.friction)
     inner = []
     for i in range(num_geoms):
       inner.append(take_i(batch, i))
     testing.append(inner)

   for geom in testing:
     for col in geom:
       tex = col.rgba[:3].reshape((1, 1, 3))
       # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
       specular_map = jax.lax.full(tex.shape[:2], 2.0)

       if isinstance(col, base.Capsule):
         half_height = col.length / 2
         model = create_capsule(
           radius=col.radius,
           half_height=half_height,
           up_axis=UpAxis.Z,
           diffuse_map=tex,
           specular_map=specular_map,
         )
       elif isinstance(col, base.Box):
         model = create_cube(
           half_extents=col.halfsize,
           diffuse_map=tex,
           texture_scaling=jnp.array(16.),
           specular_map=specular_map,
         )
       elif isinstance(col, base.Sphere):
         model = create_capsule(
           radius=col.radius,
           half_height=jnp.array(0.),
           up_axis=UpAxis.Z,
           diffuse_map=tex,
           specular_map=specular_map,
         )
       elif isinstance(col, base.Plane):
         tex = _GROUND
         model = create_cube(
           half_extents=jnp.array([1000.0, 1000.0, 0.0001]),
           diffuse_map=tex,
           texture_scaling=jnp.array(8192.),
           specular_map=specular_map,
         )
       elif isinstance(col, base.Convex):
         # convex objects are not visual
         continue
       elif isinstance(col, base.Mesh):
         tm = trimesh.Trimesh(vertices=col.vert, faces=col.face)
         model = RendererMesh.create(
           verts=tm.vertices,
           norms=tm.vertex_normals,
           uvs=jnp.zeros((tm.vertices.shape[0], 2), dtype=int),
           faces=tm.faces,
           diffuse_map=tex,
         )
       else:
         raise RuntimeError(f'unrecognized collider: {type(col)}')

       i: int = col.link_idx if col.link_idx is not None else -1
       instance = Instance(model=model)
       off = col.transform.pos
       rot = col.transform.rot
       obj = Obj(instance=instance, link_idx=i, off=off, rot=rot)
       objs.append(obj)
   return objs
