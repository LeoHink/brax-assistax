from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
import mujoco 
from mujoco import mj_id2name, mj_name2id
from enum import IntEnum
from mujoco.mjx._src.support import contact_force
import numpy as np

def contact_id(pipeline_state: State, id1: int, id2: int) -> int:
    """Returns the contact id between two geom ids."""
    mask = (pipeline_state.contact.geom == jp.array([id1, id2])) | (pipeline_state.contact.geom == jp.array([id2, id1])) 
    mask2 = jp.all(mask[0], axis=1)
    id = jp.where(mask2)  # this was missing in the original code my bad
    return id

class PushCoop(PipelineEnv):
    """PushCoop environment."""

    def __init__(
        self,
        ctrl_cost: float = 1e-6,
        dist_reward_weight: float = 0.1,
        dist_scale: float = 0.1,
        t_dist_weight: float = 0.1,
        t_contact_weight: float = 0.1,
        backend="mjx",
        reset_noise_scale=5e-3,
        **kwargs
    ):
        """Initializes the PushCoop environment.

        Args:
            ctrl_cost: Cost for control.
            dist_reward_weight: Weight for distance to target reward.
            dist_scale: Scale for distance.
            t_dist_weight: Weight for t distance.
            t_contact_weight: Weight for t contact distance.
        """
        self.path = epath.resource_path("brax") / "envs/assets/push_coop.xml"

        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        if backend == "mjx":
            self.sys = self.sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )                
        
        GEOM_IDX = mujoco.mjtObj.mjOBJ_GEOM
        BODY_IDX = mujoco.mjtObj.mjOBJ_BODY
        ACTUATOR_IDX = mujoco.mjtObj.mjOBJ_ACTUATOR
        SITE_IDX = mujoco.mjtObj.mjOBJ_SITE      

        self.panda1_actuator_ids = []
        self.panda2_actuator_ids = []   

        self.panda1_joint_id_start = 1
        self.panda2_joint_id_start = 8
        self.panda1_joint_id_end = 8   
        self.panda2_joint_id_end = 15

        for i in range(mjmodel.nu):
            actuator_name = mj_id2name(mjmodel, ACTUATOR_IDX, i)
            if "panda1" in actuator_name:
                self.panda1_actuator_ids.append(i)
            elif "panda2" in actuator_name:
                self.panda2_actuator_ids.append(i)
            else:
                raise ValueError(f"Unknown actuator name: {actuator_name}")
            
        self.panda1_pusher_body_idx = mj_name2id(mjmodel, BODY_IDX, "panda1_pusher")
        self.panda1_pusher_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "panda1_pusher_stick")
        self.panda1_pusher_point_idx = mj_name2id(mjmodel, SITE_IDX, "panda1_pusher_point")

        self.panda2_pusher_body_idx = mj_name2id(mjmodel, BODY_IDX, "panda2_pusher")
        self.panda2_pusher_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "panda2_pusher_stick")
        self.panda2_pusher_point_idx = mj_name2id(mjmodel, SITE_IDX, "panda2_pusher_point")

        self.t_shape_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "t_main")
        self.t_shape_geom_idx2 = mj_name2id(mjmodel, GEOM_IDX, "t_cross")
        self.t_shape_body_idx = mj_name2id(mjmodel, BODY_IDX, "t_object")

        self.obstacle1_idx = mj_name2id(mjmodel, GEOM_IDX, "obs1")
        self.obstacle2_idx = mj_name2id(mjmodel, GEOM_IDX, "obs2")
        self.obstacle3_idx = mj_name2id(mjmodel, GEOM_IDX, "obs3")
        self.obstacle4_idx = mj_name2id(mjmodel, GEOM_IDX, "obs4")
        self.obstacle5_idx = mj_name2id(mjmodel, GEOM_IDX, "obs5")
        self.obstacle6_idx = mj_name2id(mjmodel, GEOM_IDX, "obs6")

        self.table_top_idx = mj_name2id(mjmodel, GEOM_IDX, "table_top")

        self.panda1_sensor_idx = mj_name2id(mjmodel, GEOM_IDX, "panda1_touch")
        self.panda2_sensor_idx = mj_name2id(mjmodel, GEOM_IDX, "panda2_touch")

        # contact ids for T shape with floor
        self.contact_id_tmain = [0, 1, 2, 3]
        self.contact_id_tcross = [4, 5, 6, 7]

        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=self.sys, backend=backend, **kwargs)
        self._ctrl_cost = ctrl_cost
        self._dist_reward_weight = dist_reward_weight
        self._dist_scale = dist_scale
        self._t_dist_weight = t_dist_weight
        self._t_contact_weight = t_contact_weight
        self._reset_noise_scale = reset_noise_scale

        # TODO: implement this function
        self.target_pos = self._initialize_target_pos(self.table_top_idx)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment.

        Args:
            rng: Random number generator.

        Returns:
            State: The initial state of the environment.
        """
        rng_pos, rng_vel = jax.random.split(rng, 2)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        init_q = self.sys.mj_model.keyframe("init").qpos
        qpos = init_q + jax.random.uniform(
        rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)
        
        robo1_obs = self._get_robo1_obs(pipeline_state)
        robo2_obs = self._get_robo2_obs(pipeline_state)
        breakpoint()
        obs = jp.concatenate((
            robo1_obs["pusher_pos"],
            robo1_obs["pusher_rot"],
            robo1_obs["pusher_forces"][None], # expnd_dims
            robo1_obs["t_location"],
            robo1_obs["robo1_joint_angles"],
            robo2_obs["pusher_pos"],
            robo2_obs["pusher_rot"],
            robo2_obs["pusher_forces"][None],
            robo2_obs["t_location"],
            robo2_obs["robo2_joint_angles"]
        ))

        reward, done, zero = jp.zeros(3)

        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
            "reward_t_dist": zero,
        }

        info = {
            "dist_to_target": zero,
            "t_contact": zero,
            "t_dist": zero,
            "t_contact_id": zero,
            "t_contact_force": zero,
        }

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        ctrl_cost = -jp.sum(jp.square(action))
        robo1_obs = self._get_robo1_obs(pipeline_state)
        robo2_obs = self._get_robo2_obs(pipeline_state)
        obs = jp.concatenate((
            robo1_obs["pusher_pos"],
            robo1_obs["pusher_rot"],
            robo1_obs["pusher_forces"][None],
            robo1_obs["t_location"],
            robo1_obs["robo1_joint_angles"],
            robo2_obs["pusher_pos"],
            robo2_obs["pusher_rot"],
            robo2_obs["pusher_forces"][None],
            robo2_obs["t_location"],
            robo2_obs["robo2_joint_angles"]
        ))
        dist_target = self._get_dist_target(pipeline_state)
        
        target_dist_reward = jp.exp(-dist_target**2 / self._dist_scale) 
        dist1, dist2 = self._ee_dist_to_t(pipeline_state)
        dist1_reward = jp.exp(-dist1**2 / self._dist_scale)
        dist2_reward = jp.exp(-dist2**2 / self._dist_scale)
        done = self._get_t_floor_contact(pipeline_state) # add termination condition

        reward = self._dist_reward_weight * target_dist_reward + self._t_dist_weight * (dist1_reward + dist2_reward) + ctrl_cost

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_robo1_obs(self, pipeline_state: base.State) -> jax.Array:
        """Get the observation for robot 1."""
        pusher_pos = pipeline_state.site_xpos[self.panda1_pusher_geom_idx]
        pusher_rot = pipeline_state.xquat[self.panda1_pusher_body_idx]

        pusher_forces = pipeline_state.sensordata[self.panda1_sensor_idx]

        t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]

        robo1_joint_angles = pipeline_state.qpos[self.panda1_joint_id_start:self.panda1_joint_id_end]

        return {
            "pusher_pos": pusher_pos,
            "pusher_rot": pusher_rot,
            "pusher_forces": pusher_forces,
            "t_location": t_location,
            "robo1_joint_angles": robo1_joint_angles
        }
    
    def _get_robo2_obs(self, pipeline_state: base.State) -> jax.Array:
        """Get the observation for robot 2."""
        pusher_pos = pipeline_state.site_xpos[self.panda2_pusher_geom_idx]
        pusher_rot = pipeline_state.xquat[self.panda2_pusher_body_idx]
        
        pusher_forces = pipeline_state.sensordata[self.panda2_sensor_idx]

        t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]

        robo2_joint_angles = pipeline_state.qpos[self.panda2_joint_id_start:self.panda2_joint_id_end]

        return {
            "pusher_pos": pusher_pos,
            "pusher_rot": pusher_rot,
            "pusher_forces": pusher_forces,
            "t_location": t_location,
            "robo2_joint_angles": robo2_joint_angles
        }
    
    # TODO do set random target position as self
    def _initialize_target_pos(self, table_top_idx: int) -> jax.Array:
        """Initialize the target position."""
        table_top_pos = self.sys.mj_model.geom_pos[table_top_idx]
        table_top_size = self.sys.mj_model.geom_size[table_top_idx]
        table_top_height = table_top_pos[2] + table_top_size[2]

        # Set the target position to be above the table
        target_pos = jp.array([0.0, 0.0, table_top_height + 0.1])
        return target_pos
    
    def _get_dist_target(self, pipeline_state: base.State) -> jax.Array:
        """Get the distance to the target."""
        t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]
        dist = jp.linalg.norm(t_location - self.target_pos)
        return dist
    
    def _ee_dist_to_t(self, pipeline_state: base.State) -> jax.Array:
        """Get the distance from the end effector to the target."""
        t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]
        panda1_pusher_pos = pipeline_state.site_xpos[self.panda1_pusher_geom_idx]
        panda2_pusher_pos = pipeline_state.site_xpos[self.panda2_pusher_geom_idx]
        dist1 = jp.linalg.norm(t_location - panda1_pusher_pos)
        dist2 = jp.linalg.norm(t_location - panda1_pusher_pos)
        return dist1, dist2
    
    
    def _get_t_floor_contact(self, pipeline_state: base.State) -> jax.Array:
        """Get the contact between the T shape and the floor."""
        
        contact_forces = []
        for i in self.contact_id_tmain:
            force = contact_force(self.sys, pipeline_state, i, False)
            contact_forces.append(force)
        
        for i in self.contact_id_tcross:
            force = contact_force(self.sys, pipeline_state, i, False)
            contact_forces.append(force)
        
        contact_forces = jp.array(contact_forces)

        return (jp.sum(contact_forces) != 0).astype(bool)   
    
    # TODO: implement termination condition 
    
    # Could add this in for heterogenous rewards
    # def _get_t_contact(self, contact_id, pipeline_state: base.State) -> jax.Array:
        
