from training_algorithms.BaseTrainingInterface import BaseTrainingInterface
from utilities.ReplayMemory import ReplayMemory
from math import exp
import torch
import torch.nn as nn
import torch.optim as optim
import random

class DeepQLearning(BaseTrainingInterface):

    def __init__(
        self, game_env, policy_net, target_net, batch_size=128, 
        gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200, 
        target_update=10, criterion=nn.SmoothL1Loss(), optimizer="RMSprop"
    ) -> None:
        super().__init__()
        
        def get_optimizer(optimizer):
            if (optimizer == "RMSprop"):
                return optim.RMSprop(policy_net.get_model().parameters())

            #TODO: implement for other kinds of optimizers
            return 

        self.env = game_env
        self.policy = policy_net
        self.target = target_net
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.criterion = criterion
        self.optimizer = get_optimizer(optimizer)

    def train(self):  

        steps_done = 0
        memory = ReplayMemory(10000)

        def select_action(state):

            sample = random.random()
            threshold = self.eps_end + (self.eps_start - self.eps_end) * exp(-1 * (steps_done/self.eps_decay))

            if sample > threshold:
                with torch.no_grad():
                    return self.policy.forward(state).max(1)[1].view(1, 1)  

            return torch.tensor([[random.randrange(self.env.get_action_space())]], dtype=torch.long)
        
        # TODO: remember to add steps_done += 1 everytime select_action() is called

        

        return 

    def optimize_model(self):
        return 

