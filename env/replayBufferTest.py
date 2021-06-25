from td3 import ReplayBuffer

from maumau import MauMauEnv

buff = ReplayBuffer((10, 6, 9), (1, 6, 9), 2)

env = MauMauEnv(2, 5)
obs, reward, done = env.reset()
action = env.cardToTensor("SJ")
buff.add(env.currentPlayerID, obs, action, obs, reward, done)
print(buff.sample(1))
