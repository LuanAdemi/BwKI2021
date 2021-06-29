import random
import torch

class RandomAgent:
    def selectAction(self, state, actionMask):
        return torch.tensor(random.choice(actionMask))
