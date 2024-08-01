import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack
import torch
import numpy as np
from torchvision import transforms as T

class SeedWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            print(f"Warning: seed {seed} is ignored in JoypadSpace environment")
        return self.env.reset()

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        # Get amount of frames to skip
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        # Do the same action for each n frames and return to total reward 
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return (next_state, total_reward, done ,trunc ,info)
    
class GrayScaleObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Get width and height of observation space
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def permute(self, observation):
        # Change from numpy's [Height, Width, Channel] to PyTorch's [C, H, W] tensor form
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation
    
    def observation(self, observation):
        observation = self.permute(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation
    
class ResizeObs(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        # Handles shape being given as a list, tuple, or singular value
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        # Get the new shape by appending the channel count to the given dimensions
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Resizes observation_space, and normalizes pixel values to a range of [0, 1]
        transform = T.Compose([T.Resize(self.shape, antialias=True), T.Normalize(0, 255)])
        # We squeeze the first dimension, as sometimes the dimension is expanded
        observation = transform(observation).squeeze(0)
        return observation