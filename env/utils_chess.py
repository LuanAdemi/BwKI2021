import random as rn
import os

from math import floor
from numba import jitclass
import numba as nm
from numba.types import List, string
import chess as ch

import numpy as np

from collections import namedtuple

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

    # returns a bool array containing all moves in the list
    # WARNING: If you wish to understand the mysteries and secrets of this method,
    # you will be forced to draw or something like that idk
    def getActionMask(self, moves):
        bMask = np.zeros((8, 8, 76), dtype=int)
        for move in moves:
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
                    dim3 = abs(steps)+(2*7)
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
                    dim3 = 75+placeOfProm
                else:
                    print("Invalid promotion type: ", typeOfProm)
                    pass
            
            #turn variables to indeces
            dim1-=1
            dim2-=1
            dim3-=1
            bMask[dim1, dim2, dim3] = 1
        return bMask
    
    def getAction(self, mask):
        dimensions = np.array(np.where(mask == 1)).T
        moves = []
        for move in dimensions: #[0, 0, 0] for a1a2
            dim1 = move[0] #going to be endFile
            dim2 = move[1] #going to be endRank
            dim3 = move[2]
            dim4 = 0 #potential promotion piece
            moveStr = ""
            #startFile = chr(dim1+97)
            #startRank = str(dim2+1)
            
            if(dim3<0 and dim3>73): #i wrote this and then read the doc of match-case :(
                print("WTF IS THIS SHIT??? HONESTLY WHICH IDIOT DID... oh (dim3=", dim3, "), which should NOT happen")
                pass
            elif(dim3>=0 and dim3<7): #moves up
                dim2+=dim3+1
            elif(dim3>=7 and dim3<14): #moves down
                dim2-=dim3-6
            elif(dim3>=14 and dim3<21): #moves right
                dim1+=dim3-13
            elif(dim3>=21 and dim3<28): #moves left
                dim1-=dim3-20
            elif(dim3>=28 and dim3<35): #moves up right
                dim1+=dim3-27
                dim2+=dim3-27
            elif(dim3>=35 and dim3<42): #moves down right
                dim1-=dim3-34
                dim2+=dim3-34
            elif(dim3>=42 and dim3<49): #moves down left
                dim1-=dim3-41
                dim2-=dim3-41
            elif(dim3>=49 and dim3<56): #moves up left
                dim1+=dim3-48
                dim2-=dim3-48
            #queen movement done
            elif(dim3==56): #8 knight moves
                dim2+=2
                dim1+=1
            elif(dim3==57):
                dim2+=1
                dim1+=2
            elif(dim3==58):
                dim2-=1
                dim1+=2
            elif(dim3==59):
                dim2-=2
                dim1+=1
            elif(dim3==60):
                dim2-=2
                dim1-=1
            elif(dim3==61):
                dim2-=1
                dim1-=2
            elif(dim3==62):
                dim2+=1
                dim1-=2
            elif(dim3==63):
                dim2+=2
                dim1-=1
            #knight movement done
            elif(dim3==64): #promotions
                dim1-=1
                dim2+=1
                dim4="n"
            elif(dim3==65):
                dim2+=1
                dim4="n"
            elif(dim3==66):
                dim1+=1
                dim2+=1
                dim4="n"
            elif(dim3==67):
                dim1-=1
                dim2+=1
                dim4="b"
            elif(dim3==68):
                dim2+=1
                dim4="b"
            elif(dim3==69):
                dim1+=1
                dim2+=1
                dim4="b"
            elif(dim3==70):
                dim1-=1
                dim2+=1
                dim4="r"
            elif(dim3==71):
                dim2+=1
                dim4="r"
            elif(dim3==72):
                dim1+=1
                dim2+=1
                dim4="r"
            elif(dim3==73):
                dim1-=1
                dim2+=1
                dim4="q"
            elif(dim3==74):
                dim2+=1
                dim4="q"
            elif(dim3==75):
                dim1+=1
                dim2+=1
                dim4="q"
                pass
            #promos done
            #finalisation
            startFile = chr(move[0]+97)
            startRank = str(move[1]+1)
            endFile = chr(dim1+97)
            endRank = str(dim2+1)
            moveStr+=startFile
            moveStr+=startRank
            moveStr+=endFile
            moveStr+=endRank
            
            if(dim4==0):
                pass
            else:
                moveStr+=dim4
            moves.append(moveStr)
        return moves