from utils import Player, Stack, Logger
from numba import typed, types
import numba as nb
import numpy as np
import torch
from collections import deque

ACTIONS = {"D2":0, "D3":1, "D4":2, "D5":3, "D6":4, "D7":5, "D8":6, "D9":7, "D10":8, "DJ":9, "DQ":10, "DK":11, "DA":12,
           "S2":13, "S3":14, "S4":15, "S5":16, "S6":17, "S7":18, "S8":19, "S9":20, "S10":21, "SJ":22, "SQ":23, "SK":24, "SA":25,
           "H2":26, "H3":27, "H4":28, "H5":29, "H6":30, "H7":31, "H8":32, "H9":33, "H10":34, "HJ":35, "HQ":36, "HK":37, "HA":38,
           "C2":39, "C3":40, "C4":41, "C5":42, "C6":43, "C7":44, "C8":45, "C9":46, "C10":47, "CJ":48, "CQ":49, "CK":50, "CA":51,
           "draw":52, "pass":53}

ACTION_STR = {v : k for v,k in zip(ACTIONS.values(), ACTIONS.keys())}

class MauMauEnv:
    # env constants
    state_dim = (10, 6, 9)
    action_dim = (1, 6, 9)
    
    def __init__(self, num_players, num_cards, nhistory=8):
        # containers
        self.players = []
        self.currentPlayerID = 0
        self.pullStack = Stack('pull')
        self.playStack = Stack('play')
        self.num_cards = num_cards
        
        self.pile = 0 # a pile for card drawing
        self.nhistory = nhistory # the length of the history of actions passed to the observation
        self.history = deque(maxlen=nhistory) # the deque for the history
        
        # fill the history with blanks
        for _ in range(nhistory):
            self.history.append("N")

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
    def handToTensor(self):
        hand = self.currentPlayer.hand
        handMask = np.zeros(54)
        for card in hand:
            handMask[ACTIONS[card]] = 1

        return np.array(handMask).reshape(1, 6, 9)
    
    # OneHot-encodes a card string and turns it into a tensor
    def cardToTensor(self, card):
        oneHot = np.zeros(54)
        if card in ACTIONS.keys():
            oneHot[ACTIONS[card]] = 1
        return np.array(oneHot).reshape(1, 6, 9)

    # switches to the next player
    def nextPlayer(self):
        if self.currentPlayerID < len(self.players)-1:
            self.currentPlayerID += 1
        else:
            self.currentPlayerID = 0

    # convert the history into a tensor
    def historyToTensor(self):
        h = []
        for c in self.history:
            h.append(self.cardToTensor(c))

        return np.stack(h).reshape(8, 6, 9)
    
    # performs a step in the environment
    # action is either a number corresponding to the action
    # reward is like AlphaZero: Winner=1, Losers=-1
    def step(self, action):
        done = False
        reward = [0 for _ in range(len(self.players))]
        
        action = ACTION_STR[action]

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
            
            # add the performed action to the history
            self.history.append(action)
        
            # special cards
            if "8" in action:
                # skip next player
                self.nextPlayer()
        
            elif "7" in action:
                self.pile += 2

        # the next observation
        # currentPlayer's hand, current top card of the playstack and history
        # (1, 6, 9)                (1, 6, 9)                                (8, 6, 9)
        obs = np.vstack((self.handToTensor(), self.cardToTensor(self.playStack.last), self.historyToTensor()))
        
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
        obs = np.vstack((self.handToTensor(), self.cardToTensor(self.playStack.last), self.historyToTensor()))
        return obs, reward, done
