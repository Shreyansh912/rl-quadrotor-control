import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for stable policy gradient training."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Continuous-action Actor-Critic Network for 6-DOF Quadrotor Control.
    Actor: 12-dim state -> Mean action (4-dim motor RPMs) + learned log-std.
    Critic: 12-dim state -> Scalar State-Value V(s).
    """
    def __init__(self, obs_dim=12, action_dim=4):
        super().__init__()

        # Critic (Value Function Evaluator)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # Actor (Policy Mean Generator)
        # bias_const=-0.34 initializes output near physical equilibrium hover throttle (~0.33)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01, bias_const=-0.34)
        )

        # Independent learnable parameter for action standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(x)

        return action, log_prob, entropy, value