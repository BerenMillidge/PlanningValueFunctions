
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import time
from RandomShooting import RandomShootingPlanner
from PI import PIPlanner
from CEM import CEMPlanner
from copy import deepcopy
import sys
import argparse
import subprocess
from datetime import datetime

from baselines.envs import TorchEnv, const

def boolcheck(x):
    return str(x).lower() in ["true", "1", "yes"]

def main(args):
    env = TorchEnv(args.env_name,args.max_episode_len)
    s = env.reset()
    if args.planner_type == "RandomShooting":
        planner = RandomShootingPlanner(env,
         plan_horizon=args.plan_horizon,
         N_samples=args.N_samples,
         action_noise_sigma=args.action_noise_sigma,
         discount_factor = args.discount_factor,
         save_states = args.save_states,
         save_actions = args.save_actions)
    elif args.planner_type == "PI":
        planner = PIPlanner(env,
        plan_horizon=args.plan_horizon,
        N_samples=args.N_samples,
        lambda_=args.PI_lambda_,
        noise_mu=0,
        noise_sigma=args.action_noise_sigma,
        save_states = args.save_states,
        save_actions = args.save_actions,
        )
    elif args.planner_type == "CEM":
        planner = CEMPlanner(env,
        plan_horizon=args.plan_horizon,
        n_candidates = args.N_samples,
        top_candidates = args.CEM_top_candidates,
        optimisation_iters=args.CEM_iterations,
        action_noise_sigma = args.action_noise_sigma,
        save_states = args.save_states,
        save_actions = args.save_actions,
        discount_factor = args.discount_factor)

    results = np.zeros([args.N_episodes, args.max_episode_len])
    for i_ep in range(args.N_episodes):
        s = env.reset()
        for j in range(args.max_episode_len):
            a = planner(s)
            #print("action: ",a)
            s,r,done= env.step(a)
            print("reward: ",r)
            results[i_ep, j] = r
            #env.render()
            if done:
                s = env.reset()

        np.save(args.logdir, results)
        if args.save_states:
            np.save(args.logdir + args.states_logdir, np.array(planner.states))
        if args.save_actions:
            np.save(args.logdir + args.actions_logdir, np.array(planner.actions))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner_type", type=str)
    parser.add_argument("--plan_horizon", type=int)
    parser.add_argument("--N_samples",type=int)
    parser.add_argument("--action_noise_sigma",type=float, default=1.0)
    parser.add_argument("--CEM_iterations", type=int, default=5)
    parser.add_argument("--CEM_top_candidates", type=int, default=50)
    parser.add_argument("--PI_lambda_",type=float, default=1)
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--max_episode_len", type=int, default=2)
    parser.add_argument("--N_episodes", type=int, default=1)
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--save_states", type=boolcheck, default=False)
    parser.add_argument("--states_logdir", type=str, default="_states.npy")
    parser.add_argument("--save_actions", type=boolcheck, default=False)
    parser.add_argument("--actions_logdir", type=str, default="_actions.npy")

    args = parser.parse_args()
    main(args)
