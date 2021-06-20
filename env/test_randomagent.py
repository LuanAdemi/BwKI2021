from maumau import MauMauEnv

env = MauMauEnv(5, 4)

obs, reward, done = env.reset()
print(obs, env.currentPlayer.getActionMask(env.playStack, binary=False))
