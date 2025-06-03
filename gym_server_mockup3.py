import numpy as np
import gymnasium as gym
from gymnasium import spaces
from laser_mockup import MockupACFPulse

class MockupMaskEnv(gym.Env):
    """
    RL environment for pulse-shaping: agent sets a 20-value mask to maximize pulse quality.
    Observations are images rendered from the synthetic ACF. Reward is based on the
    improvement in (negative) new_fitness, so maximizing reward is minimizing fitness.

    The mockup converts the 20value input into a discrete action space:
    120 actions corresponding to increments of -100, -10, -1, 1, 10, 100 for every of the 20 mask values.
    """
    metadata = {"render_modes": ["rgb_array"]}
    _INCREMENTS = np.array([-100, -10, -1, 1, 10, 100], dtype=np.int32)
    ACTION_DIM = 20

    def __init__(
        self,
        true_state: np.ndarray | None = None,
        *,
        max_episode_steps: int = 400,
        laser_seed: int = 0,
    ):
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)
        self._step = 0

        # Action is: (index * len(INCREMENTS) + increment_id)
        self.action_space = spaces.Discrete(self.ACTION_DIM * len(self._INCREMENTS))
        self._action_vec_space = spaces.Box(
            low=MockupACFPulse.GENE_MIN,
            high=MockupACFPulse.GENE_MAX,
            shape=(self.ACTION_DIM,),
            dtype=np.int32
        )

        # Handle true_state and device
        self._seed = laser_seed
        self._device_true_state = (
            np.copy(true_state) if true_state is not None else None
        )
        self._reinit_device()

        # Mask state for agent, starts at 500 everywhere
        self.current_state = np.zeros(self.ACTION_DIM, dtype=np.int32) + 500

        # Observation space = shape of the rendered image from mockup
        img_shape = self._device.img_shape
        self.observation_space = spaces.Box(0, 255, shape=img_shape, dtype=np.uint8)

        # Previous fitness for reward shaping
        self.prev_fitness = None

    def _reinit_device(self):
        self._device = MockupACFPulse(
            true_state=self._device_true_state, seed=self._seed
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._reinit_device()
        self.current_state.fill(0 if self._device_true_state is None else 500)

        delay, acf = self._device.get_pulse(self.current_state)
        delay = np.asarray(delay)
        acf = np.asarray(acf)
        fitness = self._device.new_fitness(delay, acf)
        obs = self._device.render(delay, acf)
        self.prev_fitness = fitness

        return obs

    def step(self, action: int):
        # Discrete action: map to index and increment
        idx = action // len(self._INCREMENTS)
        delta = self._INCREMENTS[action % len(self._INCREMENTS)]
        self.current_state[idx] = np.clip(
            self.current_state[idx] + delta,
            MockupACFPulse.GENE_MIN,
            MockupACFPulse.GENE_MAX,
            dtype=np.int32,
        )

        # Get new observation and fitness
        delay, acf = self._device.get_pulse(self.current_state)
        delay = np.asarray(delay)
        acf = np.asarray(acf)
        fitness = self._device.new_fitness(delay, acf)
        obs = self._device.render(delay, acf)

        # Reward is improvement in (-fitness): maximize reward = minimize fitness
        reward = float(self.prev_fitness - fitness)
        self.prev_fitness = fitness

        self._step += 1
        terminated = False  # Could be fitness threshold, but none specified
        truncated = self._step >= self.max_episode_steps
        terminated = terminated or truncated
        #image, reward, done, info,_
        info = {},  # No additional info for now
        return obs, reward, terminated, truncated, {}
        #return obs, reward, terminated, info

    def render(self):
        # Return latest frame (call get_pulse with current mask)
        delay, acf = self._device.get_pulse(self.current_state)
        return self._device.render(np.asarray(delay), np.asarray(acf))

    def close(self):
        pass

if __name__ == "__main__":
    np.random.seed(42)
    env = MockupMaskEnv(max_episode_steps=10)
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}, dtype={obs.dtype}")

    for t in range(10):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(a)
        print(f"t={t:2d} | a={a:3d} | reward={reward:+.3f} | trunc={truncated}")
        if terminated or truncated:
            break

    import matplotlib.pyplot as plt
    plt.imshow(obs)
    plt.title("Last observation")
    plt.axis("off")
    plt.show()
    env.close()
