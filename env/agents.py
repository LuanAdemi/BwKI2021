import random

class RandomAgent:
    def selectAction(self, state, actionMask):
        return random.choice(actionMask)