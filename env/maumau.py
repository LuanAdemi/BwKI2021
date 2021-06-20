from utils import Player, Stack, Logger
from numba import typed, types

# TODO: Implement special cards like the seven

class MauMauEnv:
    def __init__(self, num_cards, num_players):
        
        # containers
        self.players = []
        self.currentPlayerID = 0
        self.pullStack = Stack('pull')
        self.playStack = Stack('play')
        self.num_cards = num_cards
        
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

    # switches to the next player
    def nextPlayer(self):
        self.currentPlayerID += 1 if self.currentPlayerID < len(self.players)-1 else -len(self.players)-1

    # performs a step in the environment
    # action is either a card string or the string "draw"
    # TODO: end of game & REWARD <-- the hard part :/
    def step(self, action):
        self.currentPlayer.act(action)
        self.nextPlayer()

        reward = 0
        done = False
        obs = (self.currentPlayer.hand, self.playStack.last)
        
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


        reward = 0
        done = False
        obs = (self.currentPlayer.hand, self.playStack.last)
        return obs, reward, done


