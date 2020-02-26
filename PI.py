import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import scipy.signal as signal


### Path Integral Planner ###
class PIPlanner(nn.Module):
    def __init__(
        self,
        env,
        N_samples,
        plan_horizon,
        lambda_,
        noise_mu,
        noise_sigma,
        save_states = False,
        save_actions = False,
        device="cpu"
    ):
        super().__init__()
        self.env = deepcopy(env)
        self.action_size = self.env.action_space.shape[0]
        self.N_samples = N_samples
        self.plan_horizon = plan_horizon
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma=noise_sigma
        self.save_states = save_states
        self.save_actions = save_actions
        self.device = device
        self.action_trajectory= torch.zeros([self.plan_horizon, self.action_size]).to(self.device)

        self.times_called = 0
        if self.save_states:
            self.states =[]
        if self.save_actions:
            self.actions = []

    def real_env_rollout(self, current_state, noise):
        noise = noise.cpu()
        costs = torch.zeros([self.N_samples,self.action_size])
        for k in range(self.N_samples):
            s = self.env.reset()
            self.env._env.state = self.env._env.state_from_obs(current_state)
            for t in range(self.plan_horizon):
                action = self.action_trajectory[t].cpu() + noise[k,t,:]
                action = action.numpy()
                s, reward, _ = self.env.step(action)
                if self.save_states:
                    self.states.append(s)
                if self.save_actions:
                    self.actions.append(action)
                costs[k,:] -= reward
        return None, costs.to(self.device)


    def SG_filter(self, action_trajectory):
        WINDOW_SIZE = 5
        POLY_ORDER = 3
        return torch.tensor(signal.savgol_filter(action_trajectory, WINDOW_SIZE,POLY_ORDER,axis=0))

    def forward(self, current_state):
        noise = torch.randn([self.N_samples, self.plan_horizon,self.action_size]) * self.noise_sigma
        noise = noise.to(self.device)
        print("Noise: ", noise.size())
        """ costs: [Ensemble_size, N_samples] """
        states, costs = self.real_env_rollout(current_state,noise)
        #print("COSTS: ",costs.size())

        #costs = costs /torch.mean(torch.sum(torch.abs(costs),dim=1))
        #print("COSTS: ",costs) #normalize first here might be the easiest way of getting this sorted. then I can adjust params around
        #print("costs: ", costs[0,1:10,:])
        """ beta is for numerical stability. Aim is that all costs before negative exponentiation are small and around 1 """
        beta = torch.min(costs)
        costs = torch.exp(-(1/self.lambda_ * (costs - beta)))
        print("COSTS: ",costs)
        eta = torch.mean(torch.sum(costs,dim=1)) + 1e-10
        """ weights: [Ensemble_size, N_samples] """
        weights = ((1/eta) * costs) / self.plan_horizon
        print("weights: ", weights[1:10,:])
        print("SUM :", torch.sum(weights))
        #print("weights shape: ", weights.size())
        """ Multiply weights by noise and sum across time dimension """
        #print("noise: ", noise[0,1:10,0,:])
        #print("multiplied: ", weights[0,1:10,:] * noise[0,1:10,0,:])

        add = torch.stack([torch.sum(torch.sum(weights * noise[:,t,:],dim=1),dim=0) for t in range(self.plan_horizon)])
        add =add.unsqueeze(1)
        #print("ADD: ", add[0].size())
        #print("ADD: ",add.size())
        #self.times_called+=1
        #if self.times_called >= 100:
        #  print("costs: ", costs[0,1:10,:])
        #  print("weights: ", weights[0,1:10,:])
        #  print("add: ", add)
        #  self.times_called = 0
        #print("ACTION TRAJ: ",self.action_trajectory.size())
        self.action_trajectory += add#self.SG_filter(add.cpu()).to(self.device)
        action = self.action_trajectory[0] #* 5
        """ Move forward action trajectory by 1 in preparation for next time-step """
        self.action_trajectory = torch.roll(self.action_trajectory,-1)
        self.action_trajectory[self.plan_horizon-1] = 0
        #print('action: ',action.item())
        return action.numpy()
