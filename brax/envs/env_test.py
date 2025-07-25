# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for brax envs."""

import os
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
)
import functools

from absl.testing import absltest
from absl.testing import parameterized
import brax
from brax import envs
from brax import test_utils
import jax
from jax import numpy as jp


_EXPECTED_SPS = {"mjx": {"scratchitch": 1000}}
BATCH_SIZES = [64,128,256]


class EnvTest(parameterized.TestCase):
    params = [
        (backend, batch_size, env, _EXPECTED_SPS[backend][env])
        for backend in _EXPECTED_SPS
        for env in _EXPECTED_SPS[backend]
        for batch_size in BATCH_SIZES
    ]

    @parameterized.parameters(params)
    def testSpeed(self, backend, batch_size, env_name, expected_sps):
        episode_length = 100 if expected_sps < 10_000 else 1000

        env = envs.create(
            env_name,
            backend=backend,
            episode_length=episode_length,
            auto_reset=True,
        )
        zero_action = jp.zeros(env.action_size)
        step_fn = functools.partial(env.step, action=zero_action)

        mean_sps = test_utils.benchmark(
            f"{backend}_{env_name}",
            env.reset,
            step_fn,
            batch_size=batch_size,
            length=episode_length,
        )
        self.assertGreater(mean_sps, expected_sps * 0.99)


if __name__ == "__main__":
    absltest.main()
