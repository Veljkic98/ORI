# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MojAgent', second = 'MojAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)







class ReflexCaptureAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    foodLeft = len(self.getFood(gameState).asList())
    actions = gameState.getLegalActions(self.index)

    #ako je ostalo manje od dve hrane vrati se nazad
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction



    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]





    #ako cu ja pojesti hranu onda stavi flag da sam bas ja pojeo a ne kolega pacman
    choice = random.choice(bestActions)
    succState = self.getSuccessor(gameState, choice)
    succAgent = succState.getAgentState(self.index)
    succFood = len(self.getFood(succState).asList())
    currFood = len(self.getFood(gameState).asList())
    if succFood != currFood:
      print('pojscu hranu')
      self.jaSamPojeo = True
    else:
      self.jaSamPojeo = False

    return choice

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights


class MojAgent(ReflexCaptureAgent):
  def getFeatures(self, gameState, action):

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    foodList = self.getFood(successor).asList()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    if myState.isPacman:
      features['onDefense'] = 0 #ako je pacman, znaci u napadu je, to se ne kaznjava
    else:
      features['onDefense'] = 1 #ako nije pacman vec ghost, znaci u odbrani je, to se kaznjava malo

    features['successorScore'] = -len(foodList) #dobija poene kada pojede bilo sta

    # ako nema napadaca idi da jedes hranu
    if len(invaders) <= 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    #ako ima napadaca trci nazad  u odbranu
    else:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)


    # ako sam u napadu onda bezi od protivnika
    if myState.isPacman:
      defenders = [ a for a in enemies if a.isPacman is False and a.getPosition() != None]
      defendersDists= [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      total = 0
      for dist in defendersDists:
        total += dist
      features['defendersDistance'] = 1/total

    #stajanje u mestu se kaznjava
    if action == Directions.STOP:
      features['stop'] = 1

    #vracanje u nazad se isto kaznjava
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    #da ne budu preblizu oba agenta
    features['distanceBetweenAgents'] = self.getMazeDistance(
      successor.getAgentPosition(1),
      successor.getAgentPosition(3)
    )

    #ako sam pojeo hranu vrati se nazad
    #mora try blok jer baca exception u getMazeDistance
    # kada probam da se pomerim u jednu stranu, a protivnicki pacman me je pojeo
    try:
      predecessor = self.getPreviousObservation()
      if predecessor is not None:
        pojedenaHrana = len(self.getFood(predecessor).asList()) != len(self.getFood(gameState).asList())
        if pojedenaHrana is True and self.jaSamPojeo is True:
          xHome = CaptureAgent.getFood(self,gameState).width // 3 + 1
          yHome = gameState.getAgentState(self.index).getPosition()[1]
          homeCoords = (xHome, yHome)
          features['distanceToHome'] = self.getMazeDistance(myPos, homeCoords)
    except:
      pass



    return features

  def getWeights(self, gameState, action):
    return {
      'successorScore': 100, #ovo je pozitivno, ostalo su penali
      'onDefense': -10, #dajem prednost napadu
      'distanceToFood': -1,
      'invaderDistance': -10,
      'stop': -10,
      'reverse': -2,
      'distanceToHome': -300,
      'distanceBetweenAgents': 1,
      'defendersDistance': -100,
    }