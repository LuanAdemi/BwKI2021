import random

class RandomAgent:
    def selectAction(self, actionMask):
        return random.choice(actionMask)