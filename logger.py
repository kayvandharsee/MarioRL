import time, datetime
import numpy as np
import matplotlib.pyplot as plt

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as file:
            # Create right-aligned column headers with 8 - 20 characters wide space depending on the header
            file.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}"
                f"{'MeanReward':>15}{'MeanLength':>15}{'MeanLoss':>15}"
                f"{'MeanQValue':>15}{'Time':>20}\n"
            )
        file.close()

        # Create filepaths for plots to be saved to
        self.ep_rewards_plt = save_dir / "reward_plot.jpg"
        self.ep_lengths_plt = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plt = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plt = save_dir / "q_plot.jpg"

        # Stores data for each episode
        self.ep_rewards = []
        self.ep_length = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Stores moving averages across a certain amount of episodes
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Initialize variables for the current episode
        self.init_episode()

        self.record_time = time.time()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
    
    # We are only considering the Q-value for actions where there is a non-zero loss value.
    # As parameters do not change with zero loss, finding the Q-value for these actions would be redundant.
    # It also allows for us to only keep track of the amount of times a change in loss has occured, and use
    # this number to also know the amount of times a Q-value was computed

    def log_step(self, reward, loss, q):
        '''
        Record values for each step / action made
        '''
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        # Update current variables if there is any loss i.e. change occuring in parameters
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_loss_length += 1
            self.curr_ep_q += q

    def log_episode(self):
        '''
        Record values once an episode is over
        '''
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_length.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            # Avg Q-value is not actually 0, it is just we did not have any loss and thus no Q-value calculations
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        # Reset current values for the next episode
        self.init_episode()

    def record(self, episode, epsilon, step):
        '''
        Record various moving averages over the last 100 episodes
        '''
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_length[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_delta = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_length} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q-Value {mean_ep_q} - "
            f"Time Delta {time_delta} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open (self.save_log, "a") as file:
            file.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_delta:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            # Clear the plot figure
            plt.clf()
            # Create a line graph for the metric
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            # Add the label specified in the line above
            plt.legend()
            # Save the figure to the filepath
            plt.savefig(getattr(self, f"{metric}_plt"))