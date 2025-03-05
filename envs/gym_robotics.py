import gymnasium as gym
import numpy as np
import gymnasium_robotics
import os

os.environ["MUJOCO_GL"] = "egl"


class Robotics(gym.Env):
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name="FetchReach-v2",
        action_repeat=1,
        size=(100, 100),
        render_mode="rgb_array",
        seed=None,
    ):
        super().__init__()

        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()

        self._repeat = action_repeat
        self._size = size
        self._random = np.random.RandomState(seed)
        self._render_mode = render_mode

        gym.register_envs(gymnasium_robotics)

        with self.LOCK:
            try:
                self._env = gym.make(name, render_mode=self._render_mode)
            except gym.error.Error as e:
                raise ValueError(f"Error creating environment {name}: {e}")

        if not hasattr(self._env, "step") or not hasattr(self._env, "reset"):
            raise TypeError(f"Invalid environment: {name}")

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self._done = True
        self._step = 0
        self.reward_range = [-np.inf, np.inf]

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        for _ in range(self._repeat):
            obs, step_reward, done, truncated, info = self._env.step(action)
            reward += (
                step_reward or 0.0
            )  # 1) Why `or 0`. 2) Should it not be an average?
            if done:
                break
        obs = dict(obs)
        obs = {
            key: [val] if len(np.shape(val)) == 0 else val for key, val in obs.items()
        }

        obs["is_terminal"] = done
        obs["is_first"] = self._step == 0
        info["discount"] = np.array(
            1.0 if not done else 0.0, np.float32
        )  # Why 0 when done
        obs["image"] = self.render()
        self._step += 1
        return obs, reward, done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed, options=options)

        self._done = False
        self._step = 0

        obs = dict(obs)
        obs = {
            key: [val] if len(np.shape(val)) == 0 else val for key, val in obs.items()
        }
        obs["image"] = self.render()
        obs["is_terminal"] = False  # Reset state is never terminal
        obs["is_first"] = True  # First step of episode

        info["discount"] = np.array(1.0, np.float32)  # Discount factor at start

        return obs, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
