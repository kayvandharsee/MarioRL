import torch
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import numpy as np
from tensordict import TensorDict
from neural import MarioNet

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        # Using a mac so cuda is not an option
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.step = 0
        self.gamma = 0.9

        # Save MarioNet every 5 x 10^5 experiences
        self.save_every = 5e5
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        # Use the Adam optimizer for adjusting the parameters
        self.optim = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_func = torch.nn.SmoothL1Loss()

        # Do not begin updating the weights until 1 x 10^4 experiences have been collected
        self.burnin = 1e4
        # Update Q_online every 3 experiences, and sync Q_target every 1 x 10^4 experiences
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        '''
        Use Epsilon-Greedy to choose an action with the given state

        Inputs: state(LazyFrame)
        Outputs: action(int)
        '''
        # Epsilon Chance
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)

        # 1 - Epsilon Chance
        else:
            # Convert the state to a numpy array. We check if its a tuple, as sometimes the environment might return
            # a tuple of states
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            # Add a dimension of 1 for batch size, so the nn accepts the tensor
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_vals = self.net(state, model="online")
            # Gets the highest action value index for the given state
            action = torch.argmax(action_vals, axis=1).item()
        
        # Decay the exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        # Increment step counter
        self.step += 1
        return action
    
    def cache(self, state, next_state, action, reward, done):
        '''
        Stores the experience to self.memory (replay buffer)
        
        Inputs: state(LazyFrame), next_state(LazyFrame), action(int), reward(float), done(bool)
        '''
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        # Store state and next_state as numpy arrays
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        # Convert parameters to tensors, put singular values in [] to store as a tensor
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # Store tensors in self.memory in a TensorDict (batch_size=[] implies that individual samples are being added
        # and not batches)
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, 
                                    "reward": reward, "done": done}, batch_size=[]))
        
    def recall(self):
        '''
        Retrieve a batch of experiences from memory
        '''
        keys = ("state", "next_state", "action", "reward", "done")
        # Move a batch from cpu to device(gpu if available)
        batch = self.memory.sample(self.batch_size).to(self.device)
        # Load tensor values from the batch
        state, next_state, action, reward, done = (batch.get(key) for key in keys)
        # We squeeze to remove any unnecessary single dimensions which might occur when getting the batch
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def td_estimate(self, state, action):
        '''
        Using the indexes given by action, return the best Q-values for each state by forward passing through 
        the online network

        Our batch size is expected to be 1 (one state), so we should just be getting a single value tensor
        which is our Temporal Difference Estimate for the given state
        '''
        return self.net(state, model="online")[np.arange(0, self.batch_size), action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        '''
        Using the reward, next_state, and done boolean, we determine the Temporal Difference Target with
        both the online and target models
        '''
        next_state_Q = self.net(next_state, model="online")
        # Get the action with highest Q-value for each state in the next_state batch with the online nn 
        # (we only have one state per batch)
        best_action = torch.argmax(next_state_Q, axis=1)
        # Get the Q-value of best_action for each state in the next_state batch with the target nn 
        # (we only have one state per batch)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        # Return the TD target
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_online(self, td_estimate, td_target):
        '''
        Calculate the loss, zero the gradient, backpropogate, update parameters, and return the loss
        '''
        loss = self.loss_func(td_estimate, td_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()
    
    def sync_target(self):
        '''
        Update the target parameters to match those of the online nn
        '''
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        '''
        Save the DDQN model every self.save_every experiences
        '''
        # Create the path of the file to save to
        save_path = (self.save_dir / f"mario_net_{int(self.step // self.save_every)}.chkpt")
        # Save model to the created path
        torch.save({"model": self.net.state_dict(), "exploration_rate": self.exploration_rate}, save_path)
        print(f"MarioNet saved to {save_path} at step {self.step}")

    def learn(self):
        '''
        Put together all the class methods to create a learning loop
        '''
        if self.step % self.sync_every == 0:
            self.sync_target()
        if self.step % self.save_every == 0:
            self.save()
        # Do not learn if we havent passed the burn-in period, or 
        # if we haven't passed 3 experienced from the last learning loop
        if (self.step < self.burnin) or (self.step % self.learn_every != 0):
            return None, None
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state=state, action=action)
        td_tgt = self.td_target(reward=reward, next_state=next_state, done=done)
        loss = self.update_online(td_estimate=td_est, td_target=td_tgt)
        return (td_est.mean().item(), loss)