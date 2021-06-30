import random
import torch

class RandomAgent:
    def selectAction(self, state, actionMask):
        actions = [i for i in range(54) if actionMask[i] == 1]
        return torch.tensor(random.choice(actions))
