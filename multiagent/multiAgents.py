# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        score=float(0)
        foodPos=currentGameState.getFood().asList()
        for i in range(len(newGhostStates)):
            manhdis=manhattanDistance(newPos,newGhostStates[i].getPosition())
            if newPos==newGhostStates[i].getPosition():
                score=-float("inf")
            if manhdis<2:
                score-=2
            if newPos in foodPos:
                score += 1
            if  manhdis <= newScaredTimes[i] :
                score += manhdis
        distance = []
        for food in foodPos:
            dis= manhattanDistance(newPos,food)
            distance.append(dis) 
        score-=min(distance)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maxv = -float('inf')
        legal = gameState.getLegalActions(0)
        successor = [gameState.generateSuccessor(0, action) for action in legal]
        goalIndex = 0
        for x in range(len(successor)):
            actionValue = self.value(successor[x], 1, 0)
            if actionValue > maxv:
                maxv = actionValue
                goalIndex = x
        return legal[goalIndex]
    
    def value(self, gameState, index, currentdepth):
        if index == gameState.getNumAgents():
            index=0
            currentdepth+=1
        if currentdepth == self.depth  or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.maxvalue(gameState, index, currentdepth)
        return self.minvalue(gameState, index, currentdepth)
    
    def maxvalue(self, gameState, index, currentdepth):
        x = -float('inf')
        legalactions=gameState.getLegalActions(index)
        for legalact in legalactions:
            node=gameState.generateSuccessor(index, legalact)
            if x<self.value(node, index+1, currentdepth):
                x=self.value(node, index+1, currentdepth)
        return x
    
    def minvalue(self, gameState, index, currentdepth):
        x = float('inf')
        legalactions=gameState.getLegalActions(index)
        for legalact in legalactions:
            node=gameState.generateSuccessor(index, legalact)
            if x>self.value(node, index+1, currentdepth):
                x=self.value(node, index+1, currentdepth)
        return x
    '''util.raiseNotDefined()'''

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha=-float('inf')
        beta=float('inf')
        maxv = -float('inf')
        legal = gameState.getLegalActions(0)
        successor = [gameState.generateSuccessor(0, action) for action in legal]
        goalIndex = 0
        for x in range(len(successor)):
            actionValue = self.value(successor[x], 1, 0,alpha, beta)
            if actionValue > maxv:
                maxv = actionValue
                goalIndex = x
                alpha = actionValue
        return legal[goalIndex]
    
    def value(self, gameState, index, currentdepth,alpha, beta):
        if index == gameState.getNumAgents():
            index=0
            currentdepth+=1
        if currentdepth == self.depth  or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.maxvalue(gameState, index, currentdepth,alpha, beta)
        return self.minvalue(gameState, index, currentdepth,alpha, beta)
    
    def maxvalue(self, gameState, index, currentdepth,alpha, beta):
        x = -float('inf')
        legalactions=gameState.getLegalActions(index)
        for legalact in legalactions:
            node=gameState.generateSuccessor(index, legalact)
            if x<self.value(node, index+1, currentdepth,alpha, beta):
                x=self.value(node, index+1, currentdepth,alpha, beta)
            if x > beta:
                return x
            alpha = max(alpha, x)
        return x
    
    def minvalue(self, gameState, index, currentdepth,alpha, beta):
        x = float('inf')
        legalactions=gameState.getLegalActions(index)
        for legalact in legalactions:
            node=gameState.generateSuccessor(index, legalact)
            if x>self.value(node, index+1, currentdepth,alpha, beta):
                x=self.value(node, index+1, currentdepth,alpha, beta)
            if x < alpha:
                return x
            beta = min(beta, x)
        return x
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
    
        maxv = -float('inf')
        legal = gameState.getLegalActions(0)
        successor = [gameState.generateSuccessor(0, action) for action in legal]
        goalIndex = 0
        for x in range(len(successor)):
            actionValue = self.value(successor[x], 1, 0)
            if actionValue > maxv:
                maxv = actionValue
                goalIndex = x
        return legal[goalIndex]
    
    def value(self, gameState, index, currentdepth):
        if index == gameState.getNumAgents():
            index=0
            currentdepth+=1
        if currentdepth == self.depth  or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.maxvalue(gameState, index, currentdepth)
        return self.expvalue(gameState, index, currentdepth)
    
    def maxvalue(self, gameState, index, currentdepth):
        x = -float('inf')
        legalactions=gameState.getLegalActions(index)
        for legalact in legalactions:
            node=gameState.generateSuccessor(index, legalact)
            if x<self.value(node, index+1, currentdepth):
                x=self.value(node, index+1, currentdepth)
        return x
    
    def expvalue(self, gameState, index, currentdepth):
        x = 0
        legalactions=gameState.getLegalActions(index)
        for legalact in legalactions:
            node=gameState.generateSuccessor(index, legalact)
            x+=self.value(node, index+1, currentdepth)
        return x/len(legalactions)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

    """
    "*** YOUR CODE HERE ***"

        # Choose one of the best actions
        # Useful information you can extract from a GameState (pacman.py)
    score=float(0)
    foodPos=currentGameState.getFood().asList()
    newPos = currentGameState.getPacmanPosition()
    x, y = newPos
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    "*** YOUR CODE HERE ***"
    distanceghost=[manhattanDistance(newPos,newGhost.getPosition()) for newGhost in newGhostStates]
    a= -1/(max(distanceghost)+1)
    b=min(newScaredTimes)
    distancefood=[manhattanDistance(newPos,food) for food in foodPos ]
    c=1/(min(distancefood)+1)if distancefood else 0
    d= currentGameState.getScore()
    e=-len(foodPos)
    score=a+b+c+d+e
    return score


# Abbreviation
better = betterEvaluationFunction
