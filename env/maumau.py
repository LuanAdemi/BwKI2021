from utils import Player, Stack, Logger

class MauMauEnvironment:
    def __init__(self, num_cards, num_players):
        
        self.players = []
        self.pullStack = Stack('pull')
        self.playStack = Stack('play')

        # define our deck
        self.colors = ["D", "H", "C", "S"]
        self.numbers = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        self.pullStack.createDeck(self.colors, self.numbers)

        # create the players
        for i in range(num_players):
            player = Player(i)
            self.players.append(player)
            
        # set up the playStack
        self.pullStack.deal(self.players, num_cards)
        firstCard = self.pullStack.first

        self.pullStack.remove(firstCard)
        self.playStack.append(firstCard)

    def step(self, action):
        return NotImplementedError

    def reset(self):
        return NotImplementedError


