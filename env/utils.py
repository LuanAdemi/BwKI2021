import random as rn
import os

from math import floor
from numba.experimental import jitclass
import numba as nm
from numba.types import List, string

import numpy as np

from collections import namedtuple

# a standard 52 cards deck
CARDS = ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "DJ", "DQ", "DK", "DA",
         "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "SJ", "SQ", "SK", "SA",
         "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "HJ", "HQ", "HK", "HA",
         "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "CJ", "CQ", "CK", "CA"]

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

# converts a card string to an ascii art representation
def cardToASCII(card):
    symbol = {"S": "♠", "H" :"♥", "C": "♣", "D": "♦"}
    template = ["" for _ in range(7)]
    template[0] = "┌─────────┐"
    template[1] = f"│ {(symbol[card[0]] + card[1:]):3}     │"
    template[2] = "│         │"
    template[3] = f"│    {card[1:]:2}   │"
    template[4] = "│         │"
    template[5] = f"│      {(symbol[card[0]] + card[1:]):3}│"
    template[6] = "└─────────┘"
    return template

# a player class
class Player:
    """
    A player class containing the player hand 
    and several methods to manipulate it.
    """
    # returns the number of cards in the player hand
    @property
    def numCards(self):
        return len(self.hand)
    
    def __init__(self, id, logger=None):
        self.hand = []
        self.id = id
        
        # the logger for this player
        self.logger = logger

    def __repr__(self):
        return f"Player(id={self.id}, hand={self.hand})"
    
    def act(self, action, playStack, pullStack):
        if action == "draw":
            self.getCards(1, pullStack)
        elif action == "pass":
            """pass"""
        else:
            assert action in CARDS, "The provided action is invalid"
            self.playCard(action, playStack)

    # gets n cards from the stack
    @logged
    def getCards(self, n,  pullStack):
        cards = pullStack.draw(n)
        for c in cards:
            self.hand.append(c)

    # plays a card
    @logged
    def playCard(self, card, playStack):
        if card in self.hand:
            # rule set
            if any(ele in playStack.last for ele in list(card)) or "J" in card:
                playStack.append(card)
                if "J" not in card:
                    self.hand.remove(card)
                else:
                    js = [c for c in self.hand if "J" in c]
                    self.hand.remove(js[0])
                return True
                
        print("[Warning] Card not playable.")
        return False

    # returns a bool array containing all legal moves from the 54 moves in total
    def getActionMask(self, pullStack, playStack, binary=True):
        # all cards + drawing
        mask = []
        b_mask = [0 for _ in range(len(CARDS)+2)]
        for card in self.hand:
            if any(ele in playStack.last for ele in list(card)) or "J" in card:
                b_mask[CARDS.index(card)] = 1
                mask.append(card)

        # drawing is ALWAYS an option ( ͡° ͜ʖ ͡°), except if there is nothing to draw from
        if not pullStack.empty:
            b_mask[-2] = 1
            mask.append("draw")

        # pass
        b_mask[-1] = 1
        mask.append("pass")
        
        if binary:
            return b_mask
        else:
            return mask

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
