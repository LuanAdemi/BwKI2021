from maumau import MauMauEnv
import random

env = MauMauEnv(5, 4)

obs, reward, done = env.reset()
turns = 0
while not done:
    action = random.choice(env.currentPlayer.getActionMask(env.playStack, binary=False))
    obs, reward, done = env.step(action)
    turns += 1
print(f"winner: {env.currentPlayer}, iters: {turns}")
