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
        
        self.pile = 0 # a pile for card drawing

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
        if self.currentPlayerID < len(self.players)-1:
            self.currentPlayerID += 1
        else:
            self.currentPlayerID = 0

    # performs a step in the environment
    # action is either a card string or the string "draw"
    # TODO: REWARD <-- the hard part :/
    def step(self, action):
        done = False
        reward = 0

        # check if there are pending cards in the pile
        if self.pile > 0 and "7" not in action:
            self.currentPlayer.getCards(self.pile, self.pullStack)
        
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


