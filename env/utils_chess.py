import random as rn
import os

from math import floor
from numba import jitclass
import numba as nm
from numba.types import List, string
import chess as ch

import numpy as np

from collections import namedtuple

# a decorator function for logging events
def logged(f):
    def wrapper(*args):
        # log with the logger attached to the class
        logger = args[0].logger
        if logger is not None:
            print(f.__name__)
            logger.log(type=f.__name__, data=args[0:])
        return f(*args)
    return wrapper

# a player class
class Player:
    """
    A player class containing the player color and methods to play
    """
    # returns the number of cards in the player hand
    @property
    def numCards(self):
        return len(self.hand)
    
    def __init__(self, color, logger=None):
        self.color = color
        self.eval = -1
        # the logger for this player
        self.logger = logger

    def __repr__(self):
        return f"Player(color={self.color}, eval={self.eval})"
    
    def act(self, action):
        if action == "draw":
            self.getCards(1, pullStack)
        elif action == "pass":
            """pass"""
        else:
            assert action in CARDS, "The provided action is invalid"
            self.playCard(action, playStack)

    # returns a bool array containing all legal moves
    # WARNING: If you wish to understand the mysteries and secrets of this method,
    # you will be forced to draw or something like that idk
    def getActionMask(self, legal_moves):
        bMask = np.zeros((8, 8, 73), dtype=int)
        for move in legal_moves:
            dim1 = ord(move[0])-96 #a->1 and so on
            dim2 = int(move[1])
            file = ord(move[2])-96
            rank = int(move[3])
            
            if dim1 == file: #moves vertically
                steps = dim2-rank
                if steps<0: #moves up (from white perspective)
                    dim3 = abs(steps)
                elif steps>0: #moves down
                    dim3 = steps+7
                else:
                    print("Piece does not move wtf???")
                    pass
            
            elif dim2 == rank: #moves horicontally
                steps = dim1-file
                if steps<0: #moves to right (f. w. p.)
                    dim3 = steps+(2*7)
                elif steps>0: #moves to left
                    dim3 = steps+(3*7)
                else:
                    print("Piece does not move wtf???")
                    pass
            
            elif dim1<file and dim2<rank: #right up
                stepsUp = rank-dim2
                stepsRight = file-dim1
                if stepsUp == stepsRight: #diagonal movement
                    dim3 = stepsUp+(4*7)
                elif stepsUp==2 and stepsRight==1: #first knight movement
                    dim3 = 57 #places 1-56 are reserved for straight or diagonal movement (7 steps possible * 8 directions) 
                    #(dim3 - 1 comes at the end for first index to be 0)
                elif stepsUp==1 and stepsRight==2: #second k.m.
                    dim3 = 58
                else:
                    print("Illegal movement: ", stepsUp, "steps up and ", stepsRight, " steps right.")
                    pass
            
            elif dim1<file and dim2>rank: #right down
                stepsDown = dim2-rank
                stepsRight = file-dim1
                if stepsDown == stepsRight: #diagonal movement
                    dim3 = stepsDown+(5*7)
                elif stepsDown==1 and stepsRight==2: #third km
                    dim3 = 59
                elif stepsDown==2 and stepsRight==1: #fourth km
                    dim3 = 60
                else:
                    print("Illegal movement: ", stepsDown, "steps down and ", stepsRight, " steps right.")
                    pass
            
            elif dim1>file and dim2>rank: #left down
                stepsDown = dim2-rank
                stepsLeft = dim1-file
                if stepsDown == stepsLeft: #diagonal movement
                    dim3 = stepsDown+(6*7)
                elif stepsDown==2 and stepsLeft==1: #fifth km
                    dim3 = 61
                elif stepsDown==1 and stepsLeft==2: #sixth km
                    dim3 = 62
                else:
                    print("Illegal movement: ", stepsDown, "steps down and ", stepsLeft, " steps left.")
                    pass
            
            elif dim1>file and dim2<rank: #left up
                stepsUp = rank-dim2
                stepsLeft = dim1-file
                if stepsUp == stepsLeft: #diagonal movement
                    dim3 = stepsUp+(7*7) #fills up to 56 max
                elif stepsUp==1 and stepsLeft==2: #7th km
                    dim3 = 63
                elif stepsUp==2 and stepsLeft==1: #8th km
                    dim3 = 64
                else:
                    print("Illegal movement: ", stepsUp, "steps up and ", stepsLeft, " steps left.")
                    pass
            
            #almost done, just promotions left
            if len(move)==5: #if promotion
                typeOfProm = move[4]
                placeOfProm = dim1-file #can be -1 for right, 0 for straight, 1 for left
                if typeOfProm == "n":
                    dim3 = 66+placeOfProm #65, 66 and 67
                elif typeOfProm == "b":
                    dim3 = 69+placeOfProm
                elif typeOfProm == "r":
                    dim3 = 72+placeOfProm
                elif typeOfProm == "q":
                    pass
                else:
                    print("Invalid promotion type: ", typeOfProm)
                    pass
            
            #turn variables to indeces
            dim1-=1
            dim2-=1
            dim3-=1
            bMask[dim1, dim2, dim3] = 1
        return bMask
                
                
# loggs object methods decorated with the @logged decorator
class Logger:
    """
    A class that logs games to a file or wandb

    TODO: Add a layer above the games, such as there is a instance 
          only creating a wandb run for each training run with multiple epochs of games.

         -|Logger|-------> File & WandB
       /      |      \
    Game    Game    Game
      |       |       |
   Players Players Players
      |       |       |
    Moves   Moves   Moves
    
    """
    def __init__(self, env, level=1, wandb=False, wandb_project="board-game-agent"):
        self.env = env
        self.level = level
        self.wandb = wandb
        self.wandb_project = wandb_project
        
        # logging containers
        self.games = {}
        self.game = namedtuple("Game", ["env_args", "moves", "winner"])
        self.move = namedtuple("Move", ["player_id", "hand", "played_card"])

        # if wandb logging is enabled, initialize wandb
        if self.wandb:
            wandb.login()
            self.run = wandb.init(project=self.wandb_project, tags=[self.env.__name__])
    
    # returns the game count of the logger
    @property
    def gameCount(self):
        return len(list(self.games.keys()))
    
    # returns a list of the current logged games and their ids
    @property
    def gameKeys(self):
        return list(self.games.keys())

    # logs specific types of method calls
    def log(self, game_id, type, data):
        assert game_id in list(self.games.keys()), f"The game with the id {game_id} does not exist."
        
        # type definition
        if type == "playCard":
            self.games[game_id].moves.append(self.move(data[0].id, data[0].hand, data[1]))
    
    # creates a new game
    def addGame(self, game_id, env_args):
        assert game_id not in list(self.games.keys()), f"The game with the id {game_id} does already exist."
        # init a new game
        self.games[game_id] = self.game(env_args, [], "")

    # uploads the model checkpoints to wandb
    def uploadCheckpoints(self, path):
        if self.wandb:
            wandb.save(os.path.join(path, "checkpoint*"))

    # generates a playback file from the stored moves
    def generatePlayback(self, game_id, path):
        return NotImplementedError

    # closes the wandb run
    def close(self):
        if self.wandb:
            wandb.run.finish()


from numba import typed, types


specs = [
    ('id', string),
    ('deck', List(string)),
    ('stack', types.ListType(string)),
    ('colors', types.ListType(string)),
    ('numbers', types.ListType(string)),
    ('idx', nm.int32),
    ('idxs', List(nm.int32)),
    ('item', string),
    ('n', nm.int32)
]

# Just In Time compilation with numba for the stack datastructure
# enables runtimes in nanosecounds >>I AM SPEED<<
@jitclass(specs)
class Stack:
    """
    A class representing a stack of cards
    """

    #ids = 'pull', 'play',
    def __init__(self, id):
        self.id = id   
        self.stack = typed.List.empty_list(string)
        
    def __getitem__(self, idx):
        return self.stack[idx]
    
    def __len__(self):
        return self.stack.shape[0]

    def __repr__(self):
        if len(self.stack) > 2:
            return f"Stack(id={self.id}, cards=[{self.first}, ..., {self.last}], len={len(self.stack)})"
        elif len(self.stack) == 2:
            return f"Stack(id={self.id}, cards=[{self.first}, {self.last}], len={len(self.stack)})"
        else:
            return f"Stack(id={self.id}, cards=[{self.first}], len={len(self.stack)})"

    # returns the first element of the stack
    @property
    def first(self):
        return self.stack[0]

    # returns the last element of the stack
    @property
    def last(self):
        return self.stack[-1]
    
    # returns true if the stack is empty
    @property
    def empty(self):
        if len(self.stack) > 0:
            return False
        else:
            return True

    # creates a deck using the given colors and numbers
    def createDeck(self, colors, numbers):
        for color in colors:
            for number in numbers:
                self.append(color+number)

        # shuffle
        self.shuffle()

    # appends an item to the current stack
    def append(self, item):
        self.stack.append(item)

    # removes an item from the current stack
    def remove(self, item):
        self.stack.remove(item)
    
    def clear(self):
        self.stack.clear()
    # shuffles the stack
    def shuffle(self):
        amnt_to_shuffle = len(self.stack)
        while amnt_to_shuffle > 1:
            i = int(floor(rn.random() * amnt_to_shuffle))
            amnt_to_shuffle -= 1 
            self.stack[i], self.stack[amnt_to_shuffle] = self.stack[amnt_to_shuffle], self.stack[i]

    # draws n cards from the stack
    def draw(self, n):
        cards = self.stack[:n]
        for c in cards:
            self.remove(c)
        return cards
