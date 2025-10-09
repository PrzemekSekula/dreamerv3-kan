# dmc.py (Gymnasium-compatible and Wrapper-safe)
import gymnasium as gym
import numpy as np


class DeepMindControl(gym.Env):
    """
    dm_control wrapper that:
      - Subclasses gymnasium.Env (so Gymnasium wrappers accept it),
      - Exposes .action_space and .observation_space as attributes,
      - Keeps legacy 4-tuple step: (obs, reward, done, info),
      - Keeps legacy reset() -> obs (no (obs, info)) expected by your wrappers.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0):
        # Parse domain/task
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"

        # Create dm_control env
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(
                domain,
                task,
                task_kwargs={"random": seed},
            )
        else:
            # Allow passing a callable that constructs a dm_control env.
            assert task is None
            self._env = domain()

        # Config
        self._action_repeat = int(action_repeat)
        self._size = tuple(size)
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = int(camera)

        self.reward_range = [-np.inf, np.inf]

        # ---- Build spaces as attributes (Gymnasium wrappers expect these) ----
        # Observation space from dm_control spec + rendered image + flags.
        obs_spec = self._env.observation_spec()
        obs_spaces = {}
        for key, value in obs_spec.items():
            shape = (1,) if len(value.shape) == 0 else tuple(value.shape)
            obs_spaces[key] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
            )

        obs_spaces["image"] = gym.spaces.Box(
            low=0, high=255, shape=self._size + (3,), dtype=np.uint8
        )
        # Flags you include in obs dict; simple binary boxes keep your pipeline unchanged.
        obs_spaces["is_terminal"] = gym.spaces.Box(0, 1, shape=(), dtype=bool)
        obs_spaces["is_first"] = gym.spaces.Box(0, 1, shape=(), dtype=bool)

        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Action space from dm_control
        act_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=np.asarray(act_spec.minimum, dtype=np.float32),
            high=np.asarray(act_spec.maximum, dtype=np.float32),
            dtype=np.float32,
        )

    # -------------------- Core API --------------------

    def step(self, action):
        """
        Returns (obs: dict, reward: float, done: bool, info: dict).
        Keeps legacy 4-tuple to match your wrappers.
        """
        action = np.asarray(action, dtype=np.float32)
        assert np.isfinite(action).all(), action

        reward = 0.0
        time_step = None
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += float(time_step.reward or 0.0)
            if time_step.last():
                break

        obs = self._build_obs(time_step)
        done = bool(time_step.last())
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Legacy reset to match your wrappers: returns obs (no (obs, info)).
        Accepts **kwargs so Gymnasium-style wrappers can pass seed/options.
        """
        time_step = self._env.reset()
        return self._build_obs(time_step)

    # -------------------- Helpers --------------------

    def _build_obs(self, time_step):
        # Scalar dm_control observations -> length-1 arrays for shape stability.
        obs = dict(time_step.observation)
        obs = {
            key: (np.asarray([val]) if len(val.shape) == 0 else np.asarray(val))
            for key, val in obs.items()
        }
        obs["image"] = self.render()
        # DMC has no absorbing terminal; treat discount == 0 as terminal (except first).
        obs["is_terminal"] = False if time_step.first() else (time_step.discount == 0)
        obs["is_first"] = bool(time_step.first())
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
