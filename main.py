import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers import SeedWrapper, FrameSkip, GrayScaleObs, ResizeObs, FrameStack
import torch
from pathlib import Path
import datetime
from agent import Mario
from logger import MetricLogger
import sys


# Change render_mode to "human" to see visual gameplay
env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="rgb_array", apply_api_compatibility=True)
# Limit the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# Apply the wrappers to the env
env = SeedWrapper(env)
env = FrameSkip(env, skip=4)
env = GrayScaleObs(env)
env = ResizeObs(env, shape=84)
# Use frame stack to squish frames into one observation point
env = FrameStack(env, num_stack=4)
# Reset the environment to start from the beginning
env.reset()

mps = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using MPS: {mps}")
print()

# Create a directory for this training loops records
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# Create the directory including the parent directory checkpoints if needed
save_dir.mkdir(parents=True)
# Four black and white  84 x 84 frames stacked together as input
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)

if len(sys.argv) == 2:
    mario.net.load_state_dict(torch.load(save_dir / sys.argv[1]))

episodes = 40000
for e in range(episodes):
    # Start from the beginning each episode
    state, info = env.reset()
    while True:
        # Get action from state
        action = mario.act(state)
        # Perform said action
        next_state, reward, done, trunc, info = env.step(action)
        # Add experience to memory
        mario.cache(state, next_state, action, reward, done)
        # Learn
        q, loss = mario.learn()
        # Log
        logger.log_step(reward, loss, q)
        # Update state
        state = next_state
        # Check if Mario lost or reached the flag
        if done or info["flag_get"]:
            break
        
    logger.log_episode()

    if (e % 20 == 0):
        print(f"Episode {e}")

    # Only record every 2000 episodes or at the last episode
    if (e % 2000 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.step)
        mario.save()