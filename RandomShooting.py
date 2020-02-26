
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
### Random shooting planner ###
# Just samples a bunch of trajectories at random and picks the best one.

class RandomShootingPlanner(nn.Module):
    def __init__(self, env,
    plan_horizon,
    N_samples,
    action_noise_sigma,
    discount_factor=1,
    save_states = False,
    save_actions=False,
    device='cpu'):
        super().__init__()
        self.env = deepcopy(env)
        self.action_size = env.action_space.shape[0]
        self.plan_horizon = plan_horizon
        self.N_samples = N_samples
        self.action_noise_sigma = action_noise_sigma
        self.discount_factor = discount_factor
        self.save_states = save_states
        self.save_actions = save_actions
        self.device=device
        if self.discount_factor <1:
            self.discount_factor_matrix = self._initialize_discount_factor_matrix()
        else:
            self.discount_factor_matrix = np.ones([self.plan_horizon,1])

        if self.save_states:
            self.states = []
        if self.save_actions:
            self.actions = []

    def _initialize_discount_factor_matrix(self):
        discounts = np.zeros([self.plan_horizon,1])
        for t in range(self.plan_horizon):
            discounts[t,:] = self.discount_factor ** self.plan_horizon
        #discounts=torch.from_numpy(discounts).repeat(1, self.N_samples, 1).to(self.device)
        return discounts

    def forward(self, state):
        self.state_size = state.shape[0]
        state = torch.from_numpy(state)
        state = state.unsqueeze(dim=0)
        state = state.repeat(self.N_samples,1)

        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        actions = action_mean + action_std_dev * torch.randn(
            self.plan_horizon,
            self.N_samples,
            self.action_size,
            device=self.device,
        )

        #full rewards
        rewards = self.perform_rollout(state, actions)
        #print("rollout complete: ", rewards.size())
        #sum across timesteps
        rewards = rewards.sum(dim=1)
        #print("REWARDS: ", rewards)
        #print("MAX:", torch.max(rewards))
        #find best trajectory
        best_idx =torch.argmax(rewards).item()
        best_action_trajectory = actions[:,best_idx,:]
        best_action = best_action_trajectory[0,:]
        return best_action.numpy()


    def perform_rollout(self,current_state, actions):
        current_state = current_state.cpu()
        actions = actions.cpu()
        returns = torch.zeros([self.N_samples,self.plan_horizon])
        for k in range(self.N_samples):
            s = self.env.reset()

            self.env._env.state = self.env._env.state_from_obs(current_state[0,:].numpy())
            for t in range(self.plan_horizon):
                action = actions[t,k,:].numpy()
                s, reward, _ = self.env.step(action)
                if self.save_states:
                    self.states.append(s)
                if self.save_actions:
                    self.actions.append(action)
                returns[k,t] = reward * self.discount_factor_matrix[t,:][0]
        return returns
