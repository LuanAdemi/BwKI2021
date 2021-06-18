import random as rn
import os

from collections import namedtuple
import inspect
from functools import wraps


def logged(f):
    def wrapper(*args):
        logger = args[0].logger
        if logger is not None:
            print(f.__name__)
            logger.log(type=f.__name__, data=args[0:])
        return f(*args)
    return wrapper

class Player:
    """
    A player class containing the player hand 
    and several methods to manipulate it.
    """
    @property
    def numCards(self):
        return len(self.hand)
    
    def __init__(self, id, logger=None):
        self.hand = []
        self.id = id

        self.logger = logger

    def __repr__(self):
        return f"Player(id={self.id}, hand={self.hand})"
        
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
            if any(ele in playStack[-1] for ele in list(card)):
                playStack.append(card)
                self.hand.remove(card)
                return True
                
        print("[Warning] Card not playable.")
        return False


class Logger:
    """A class that logs games to a file or wandb"""
    def __init__(self, env, level=1, wandb=True, wandb_project="board-game-agent"):
        self.env = env
        self.level = level
        self.wandb = wandb
        self.wandb_project = wandb_project
        
        # logging containers
        self.moves = []
        self.move = namedtuple("Move", ["player_id", "hand", "played_card"])

        # if wandb logging is enabled, initialize wandb
        if self.wandb:
            wandb.login()
            self.run = wandb.init(project=self.wandb_project)
    
    def log(self, type, data):
        if type == "playCard":
            self.moves.append(self.move(data[0].id, data[0].hand, data[1]))

    def register_calls(f):
        @wraps(f)
        def f_call(*args, **kw):
            self.moves.append(self.move(args[0].id, args[1]))
            return f(*args, **kw)
        return f_call

    def uploadCheckpoints(self, path):
        wandb.save(os.path.join(path, "checkpoint*"))

    def generatePlayback(self):
        return NotImplementedError

    def close(self):
        wandb.run.finish()        

class Stack:
    """
    A class representing a stack of cards
    """

    #ids = 'pull', 'play',
    def __init__(self, id):
        self.id = id   
        self.stack = []
        self.colors = ["D", "H", "C", "S"]
        self.numbers = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        
    def __getitem__(self, idx):
        return self.stack[idx]
    
    def __len__(self):
        return len(self.stack)

    def __repr__(self):
        return f"Stack(id={self.id}, cards=[{self.first}, ..., {self.last}], len={len(self.stack)})"
    
    # returns the first element of the stack
    @property
    def first(self):
        return self.stack[0]

    # returns the last element of the stack
    @property
    def last(self):
        return self.stack[-1]

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
    
    # shuffles the stack
    def shuffle(self):
        rn.shuffle(self.stack)

    # draws n cards from the stack
    def draw(self, n):
        cards = self.stack[:n]
        for c in cards:
            self.stack.remove(c)
        return cards

    # deals n cards to n player from the current deck
    def deal(self, players, num_cards):
        for _ in range(num_cards):
            for player in players:
                player.getCards(1, self)
