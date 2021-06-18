import random as rn

class Player:
    """
    A player class containing the player hand 
    and several methods to manipulate it.
    """
    @property
    def numCards(self):
        return len(self.hand)
    
    def __init__(self, id):
        self.hand = []
        self.id = id

    def __repr__(self):
        return f"Player(id={self.id}, hand={self.hand})"
        
    # gets n cards from the stack
    def getCards(self, n,  pullStack):
        cards = pullStack.draw(n)
        for c in cards:
            self.hand.append(c)

    # plays a card
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

        # if wandb logging is enabled, initialize wandb
        if self.wandb:
            wandb.login()
            self.run = wandb.init(project=self.wandb_project)


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
