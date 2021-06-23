from utils import Player, Stack, Logger
from numba import typed, types
import numpy as np
import torch


CARDS = ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "DJ", "DQ", "DK", "DA",
         "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "SJ", "SQ", "SK", "SA",
         "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "HJ", "HQ", "HK", "HA",
         "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "CJ", "CQ", "CK", "CA"]

class MauMauEnv:
    def __init__(self, num_cards, num_players, nhistory=8):
        
        # containers
        self.players = []
        self.currentPlayerID = 0
        self.pullStack = Stack('pull')
        self.playStack = Stack('play')
        self.num_cards = num_cards
        
        self.pile = 0 # a pile for card drawing
        self.nhistory = nhistory # the history of actions passed to the observation
        self.history = []

        # define our deck
        self.colors = typed.List(["D", "H", "C", "S"])
        self.numbers = typed.List(["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"])
        self.pullStack.createDeck(self.colors, self.numbers)

        # create the players
        for i in range(num_players):
            player = Player(i)
            player.getCards(num_cards, self.pullStack)
            self.players.append(player)
        
        # set up the stacks
        firstCard = self.pullStack.first
        self.pullStack.remove(firstCard)
        self.playStack.append(firstCard)

    # returns the current player
    @property
    def currentPlayer(self):
        return self.players[self.currentPlayerID]
    
    # returns the current players hand
    def getCurrentHand(self, tensor=True):
        hand = self.currentPlayer.hand
        if tensor:
            handMask = np.zeros(54)
            for i, card in enumerate(CARDS):
                if card in hand:
                    handMask[i] = 1

            return torch.tensor(handMask).reshape(6, 9)
        else:
            return self.hand
   
    # returns a tensor representation of a card
    def cardToTensor(self, card):
        cardTensor = np.zeros(54)
        cardTensor[CARDS.index(card)] = 1
        return torch.tensor(cardTensor).reshape(6, 9)
    
    # switches to the next player
    def nextPlayer(self):
        if self.currentPlayerID < len(self.players)-1:
            self.currentPlayerID += 1
        else:
            self.currentPlayerID = 0
    
    # performs a step in the environment
    # action is either a card string or the string "draw"
    # reward is like AlphaZero: Winner=1, Losers=-1
    def step(self, action):
        done = False
        reward = [0 for _ in range(len(self.players))]

        # check if there are pending cards in the pile
        if self.pile > 0 and "7" not in action:
            # if the current player has not played a seven, give the player these cards
            self.currentPlayer.getCards(self.pile, self.pullStack)
            self.pile = 0 # reset the pile
        
        # check whether the pullStack is empty
        if self.pullStack.empty:
            # get the top card of the play stack
            topCard = self.playStack.last

            # move all cards except the top one from the play stack to the pull stack
            self.pullStack.stack = self.playStack.stack[:-1]
            self.playStack.clear()
            self.playStack.append(topCard)

            # shuffle the pull stack
            self.pullStack.shuffle()
        
        # perform a action with the current player -> plays a card or draws
        self.currentPlayer.act(action, self.playStack, self.pullStack)

       
        # if the hand is empty after the action, the game ends
        if len(self.currentPlayer.hand) <= 0:
            done = True
            reward = [-1 if idx is not self.currentPlayerID else 1 for idx in range(len(self.players))]
        else:
            # switch to the next player
            self.nextPlayer()
        
            # special cards
            if "8" in action:
                # skip next player
                self.nextPlayer()
        
            elif "7" in action:
                self.pile += 2

        # the next observation
        obs = (self.getCurrentHand(), self.cardToTensor(self.playStack.last))
        
        return obs, reward, done

    # resets the environment
    def reset(self):
        # reset the player ID
        self.currentPlayerID = 0
        self.playStack.clear()
        self.pullStack.clear()

        self.pullStack.createDeck(self.colors, self.numbers)

        # create the players
        for p in self.players:
            p.hand.clear()
            p.getCards(self.num_cards, self.pullStack)
        
        # set up the stacks
        firstCard = self.pullStack.first
        self.pullStack.remove(firstCard)
        self.playStack.append(firstCard)

        done = False
        reward = [0 for _ in range(len(self.players))]
        obs = (self.getCurrentHand(), self.cardToTensor(self.playStack.last))
        return obs, reward, done
