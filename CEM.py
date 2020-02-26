
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

class CEMPlanner(nn.Module):
    def __init__(
        self,
        env,
        plan_horizon,
        n_candidates,
        top_candidates,
        optimisation_iters,
        action_noise_sigma,
        discount_factor=1,
        save_states=False,
        save_actions=False,
        device="cpu",
    ):
        super().__init__()
        self.env = deepcopy(env)
        self.action_size = self.env.action_space.shape[0]

        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.n_candidates = n_candidates
        self.top_candidates = top_candidates
        self.action_noise_sigma = action_noise_sigma
        self.discount_factor = discount_factor
        self.device = device
        self.save_states = save_states
        self.save_actions = save_actions
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

    def perform_rollout(self,current_state, actions):
        current_state = current_state.cpu()
        actions = actions.cpu()
        returns = torch.zeros([self.n_candidates,self.plan_horizon])
        for k in range(self.n_candidates):
            s = self.env.reset()

            self.env._env.set_state(self.env._env.state_from_obs(current_state[0,:].numpy()))
            for t in range(self.plan_horizon):
                action = actions[t,k,:].numpy()
                #print(action)
                #print(type(action))
                s, reward, _ = self.env.step(action)
                if self.save_states:
                    self.states.append(s)
                if self.save_actions:
                    self.actions.append(action)
                returns[k,t] = reward * self.discount_factor_matrix[t,:][0]
        return returns

    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.state_size = state.size(0)
        state = state.unsqueeze(dim=0)
        state = state.repeat(self.n_candidates, 1)

        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(self.device)
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(self.device) * self.action_noise_sigma

        for _ in range(self.optimisation_iters):
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )

            returns = self.perform_rollout(state, actions)
            returns = torch.sum(returns,dim=1)
            #print("RETURNS: ", returns.size())
            _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)
            #print("TOPK:", topk.size())
            #print("ACTIONS: ", actions.size())
            #print(topk.view(-1)[:])
            topk = topk.numpy()
            #print(topk.shape)

            best_actions = actions[:,topk,:]
            beest_actions = best_actions.reshape(self.top_candidates, self.plan_horizon, self.action_size)
            #print("BEST ACTIONS", best_actions.size())
            action_mean, action_std_dev = (
                best_actions.mean(dim=1, keepdim=True),
                best_actions.std(dim=1, unbiased=False, keepdim=True),
            )


        return action_mean[0].squeeze(dim=0).numpy()
# so this is super frustrating...
