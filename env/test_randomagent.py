from maumau import MauMauEnv
import random
import multiprocessing as mp

# runs a random game
def runRandomGame():
    # create a new environment
    env = MauMauEnv(5, 4)
    # reset the env
    obs, reward, done = env.reset()
    turns = 0
    
    # main loop
    while not done:
        action = random.choice(
            env.currentPlayer.getActionMask(env.playStack, binary=False)
        )
        obs, reward, done = env.step(action)
        turns += 1
    
    print(f"winner: {env.currentPlayer}, iters: {turns}")

# store all the process objects
processes = []

# launch 8 games
for _ in range(8):
    p = mp.Process(target=runRandomGame)
    processes.append(p)
    p.start()

# join all games
for p in processes:
    p.join()

