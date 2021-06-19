import random as rn
import os

from collections import namedtuple
import inspect
from functools import wraps

# a standart 52 cards deck
CARDS = ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "DJ", "DQ", "DK", "DA",
         "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "SJ", "SQ", "SK", "SA",
         "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "HJ", "HQ", "HK", "HA",
         "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "CJ", "CQ", "CK", "CA"]

# a decorator function for logging events
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
            if any(ele in playStack.last for ele in list(card)):
                playStack.append(card)
                self.hand.remove(card)
                return True
                
        print("[Warning] Card not playable.")
        return False

    # returns a bool array containing all legal moves from the 53 moves in total
    def getActionMask(self, playStack):
        # all cards + drawing
        mask = [0 for _ in range(len(CARDS)+1)]
        for card in self.hand:
            if any(ele in playStack.last for ele in list(card)):
                mask[CARDS.index(card)] = 1

        # drawing is always an option
        mask[-1] = 1
        return mask


class Logger:
    """A class that logs games to a file or wandb"""
    def __init__(self, env, level=1, wandb=False, wandb_project="board-game-agent"):
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

    # logs specific types of method calls
    def log(self, type, data):
        if type == "playCard":
            self.moves.append(self.move(data[0].id, data[0].hand, data[1]))

    # uploads the model checkpoints to wandb
    def uploadCheckpoints(self, path):
        if self.wandb:
            wandb.save(os.path.join(path, "checkpoint*"))

    # generates a playback file from the stored moves
    def generatePlayback(self, path):
        return NotImplementedError

    # closes the wandb run
    def close(self):
        if self.wandb:
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
