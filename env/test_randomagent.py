from maumau import MauMauEnv
import random
import multiprocessing as mp

# takes about 7 sec for JIT compilation, then goes brrrrr 
# (10 + 7 = 17 sec for 4096 with 8 processes)

num_processes = 8 # eight cores on my laptop

# runs a random game
def runRandomGame(n_games):
    for _ in range(n_games):
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
    
        #print(f"winner: {env.currentPlayer}, iters: {turns}")

# store all the process objects
processes = []

# launch the processes
for _ in range(num_processes):
    p = mp.Process(target=runRandomGame, args=(512,)) # 8*512 = 4096 games
    processes.append(p)
    p.start()

# join all games
for p in processes:
    p.join()

