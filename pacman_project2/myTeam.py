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




class MojAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self._foodEaten = 0

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

    choice = random.choice(bestActions)
    # ako cu ja pojesti hranu onda stavi flag da sam bas ja pojeo a ne kolega pacman
    self.willFoodBeEaten(choice)
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


  def getWeights(self, gameState, action):
    return {
      'score': 100, #ovo je pozitivno, ostalo su penali
      'onDefense': -10, #dajem prednost napadu
      'distanceToFood': -1,
      'invaderDistance': -10,
      'stop': -10,
      'reverse': -2,
      'distanceToHome': -10,
      'distanceBetweenAgents': 1,
      'defendersDistance': -100,
    }

  def getFeatures(self, gameState, action):
    #inicijalizuj promenljive koje ce mi trebati
    self._features = util.Counter()
    self._succ = self.getSuccessor(gameState, action)
    self._succState = self._succ.getAgentState(self.index)
    self._succPos = self._succState.getPosition()
    self._foodList = self.getFood(self._succ).asList()
    self._enemies = [self._succ.getAgentState(i) for i in self.getOpponents(self._succ)]
    self._invaders = [a for a in self._enemies if a.isPacman and a.getPosition() != None]
    self._gameState = gameState
    self._action = action

    #pozovi sve funkcije za setovanje feature-a
    self._initRules()
    #vrati feature-a evaluate funkciji
    return self._features

  def _initRules(self):
    self.goAttack()
    self.setScore()
    self.playDefenseOrOffense()
    self.runAwayFromOpponent()
    self.stop()
    self.reverse()
    self.distanceBetweenAgents()
    self.returnFood()
    self.willTheyEatMe()

  ### pravila kojih moj pacman mora da se pridrzava ###
  #---------------------------------------------------#

  def goAttack(self):
    '''
    funkcija setujes vrednost agentu.
    ako je agent pacman to znaci u napadu je, to se ne kaznjava
    ako nije pacman vec ghost, znaci u odbrani je, to se kaznjava malo
    '''
    if self._succState.isPacman:
      self._features['onDefense'] = 0
    else:
      self._features['onDefense'] = 1

  def setScore(self):
    '''
    funkcija govori agentu da treba da pojede hranu i time se povecava score
    '''
    self._features['score'] = -len(self._foodList)

  def playDefenseOrOffense(self):
    '''
    funkcija proverava da li ima napadaca na mojoj polovini mape.
    ako nema napadaca agent ide da jede najblizu hranu
    ako ima napadaca agent trci nazad  u odbranu
    '''
    if len(self._invaders) <= 0:
      minDistance = min([self.getMazeDistance(self._succPos, food) for food in self._foodList])
      self._features['distanceToFood'] = minDistance

    else:
      dists = [self.getMazeDistance(self._succPos, a.getPosition()) for a in self._invaders]
      self._features['invaderDistance'] = min(dists)

  def runAwayFromOpponent(self):
    '''
    racuna gde je pacman trenutno u odnosu na protivnika, duha.
    ako je agent u napadu onda bezi od protivnika.
    Ovde postoji i druga provera da agent ne stoji u mestu ako je protivnik odmah pored agenta, jer
    ce ga u tom slucaju svaki normalan protivnik pojesti
    '''
    if self._succState.isPacman:
      defenders = [a for a in self._enemies if a.isPacman is False and a.getPosition() != None]
      defendersDists = [self.getMazeDistance(self._succPos, a.getPosition()) for a in defenders]
      total = 0
      for dist in defendersDists:
        #druga provera
        if dist == 1:
          self._features['defendersDistance'] = 20
          return
        else:
          total += dist
          self._features['defendersDistance'] = 1 / total

  def willTheyEatMe(self):
    '''
    ovo sluzi da agent ne izvrsi samoubistvo.
    ako trenutna akcija koju razmatramo vodi do toga da ce mene pojesti protivnik, to je JAKO lose.
    u koliko me protivnik pojede, zavrsicu na pocetnoj poziciji. To ne zelim da se desi.
    Kako bih proverio da li cu zavrsiti na pocetnoj poziciji,
    moram prvo da proverim da li sam crveni ili plavi tim
    '''

    #prva provera da ne odem bas u usta protivniku
    myTeam = []
    currPos = self._gameState.getAgentState(self.index).getPosition()
    isMyTeamRed = self.red
    if isMyTeamRed is True:
      myTeam = self._gameState.redTeam
    else:
      myTeam = self._gameState.blueTeam

    for i in myTeam:
      initPos = self._gameState.getInitialAgentPosition(i)
      if self._succPos == initPos and self.getMazeDistance(currPos, self._succPos) > 1:
        self._features['defendersDistance'] = 20


  def stop(self):
    '''
    stajanje u mestu se kaznjava
    '''
    if self._action == Directions.STOP:
      self._features['stop'] = 1

  def reverse(self):
    '''
    vracanje u nazad se isto kaznjava
    '''
    rev = Directions.REVERSE[self._gameState.getAgentState(self.index).configuration.direction]
    if self._action == rev:
      self._features['reverse'] = 1

  def distanceBetweenAgents(self):
    '''
    tera agente da se priblizavaju jedan drugome previse
    '''
    self._features['distanceBetweenAgents'] = self.getMazeDistance(
      self._succ.getAgentPosition(1),
      self._succ.getAgentPosition(3)
    )

  def returnFood(self):
    '''
    ako sam JA pojeo hranu vrati se nazad
    mora try blok jer baca exception u getMazeDistance, kada probam da
    se pomerim u jednu stranu, a protivnicki pacman me je pojeo.
    Sto sam dalje od kuce to je evalute funkcija losija
    '''
    try:
      predecessor = self.getPreviousObservation()
      if predecessor is not None:
        pojedenaHrana = len(self.getFood(predecessor).asList()) != len(self.getFood(self._gameState).asList())
        if pojedenaHrana is True and self.jaSamPojeo is True:
          xHome = CaptureAgent.getFood(self, self._gameState).width // 3 + 1
          yHome = self._gameState.getAgentState(self.index).getPosition()[1]
          homeCoords = (xHome, yHome)
          distanceToHome = self._foodEaten *  self._foodEaten *  self.getMazeDistance(self._succPos, homeCoords)
          self._features['distanceToHome'] = distanceToHome
    except:
      pass


  def willFoodBeEaten(self, choice):
    '''
    ova funkcija proverava da li ce pacman nakon izabrane akcije pojesti hranu.
    ovo je neophodno zbog returnFood pravila kako bi pacman tezio tome da kad skupi dovoljne hrane da se vrati,
    sto vise hrane pacman pojede, to vise tezi tome da se vrati kuci
    '''
    succState = self.getSuccessor(self._gameState, choice)
    succAgent = succState.getAgentState(self.index)
    succFood = len(self.getFood(succState).asList())
    currFood = len(self.getFood(self._gameState).asList())
    if succFood != currFood:
      # print('pojscu hranu')
      self.jaSamPojeo = True
      self._foodEaten += 1
    else:
      self.jaSamPojeo = False