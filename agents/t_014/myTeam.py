from template import Agent
import random
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
import math
from copy import deepcopy
from Azul.azul_utils import *
import time

# Constants for Azul game
NUM_PLAYERS = 2
MAX_DIFF = 30  # Max score difference to normalize the heuristic value
THINK_TIME = 0.95
FLOOR_DEDUCTION = [-1, -1, -2, -2, -2, -3, -3]  # Penalty for overflows in the floor line
WARM_UP = 0

class myAgent():
    def __init__(self, _id, think_time=THINK_TIME):
        self.id = _id  # Player ID
        self.game_rule = GameRule(NUM_PLAYERS)  # Initialize game rule for Azul
        self.think_time = think_time
        self.first_turn = True
        self.action_counter = 0

    # Expands the given node by adding a child with an unexplored move
    def Expand(self, node, current_player):
        move = node.untried_moves.pop()  # Get an untried move
        next_state = deepcopy(node.state)  # Deepcopy the current state
        next_state = self.game_rule.generateSuccessor(next_state, move, current_player)
        actions = self.game_rule.getLegalActions(next_state, current_player)  # Get next legal actions
        # Create a new child node and add it to the current node's children
        child_node = self.MCTSNode(next_state, actions, self, move=move, parent=node)
        node.children.append(child_node)
        return child_node
    
    # Checks if the game has ended based on row completion
    def GameEnds(self, state):
        return any(agent.GetCompletedRows() > 0 for agent in state.agents)

    class MCTSNode():
        def __init__(self, azul_state, actions, agent, move=None, parent=None):
            self.state = azul_state  # The state associated with this node
            self.untried_moves = actions  # Available moves not yet explored
            self.move = move  # The move that led to this node
            self.parent = parent  # Parent node in the tree
            self.children = []  # List of child nodes
            self.visits = 0  # Number of visits to this node
            self.win = 0  # Win score for this node
            self.agent = agent  # Reference to the agent using this node
            
        # Returns True if all possible moves from this node have been explored
        def is_fully_expanded(self):
            return len(self.untried_moves) == 0

        # Updates the node's win score and visits based on the reward
        def update(self, reward):
            self.visits += 1
            self.win += reward
            # print(f"Node Update -> Move: {self.move}, Visits: {self.visits}, Wins: {self.win}, Reward: {reward}")

        # Returns the best child node based on UCB value
        def best_child(self, exploration_param=1.41):
            best_score = -float('inf')
            best_child = None

            """# Determine adaptive alpha based on round count
            if self.agent.action_counter < 10:
                alpha = 1
            elif self.agent.action_counter < 15:
                alpha = .8
            elif self.agent.action_counter < 20:
                alpha = .4
            else:
                alpha = .2"""
            
            alpha = 1

            for child in self.children:
                exploitation = child.win / child.visits if child.visits > 0 else 0
                exploration = exploration_param * math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
                heuristic = alpha * self.agent.evaluate_action(self.state, child.move)
                score = exploitation + exploration + heuristic
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child
            
        # Returns the move associated with this node
        def getMove(self):
            return self.move

    # Selects the best action to take from the root state using MCTS
    def SelectAction(self, actions, root_state):
        start_time = time.time()
        root_node = myAgent.MCTSNode(root_state, actions, self)  # Initialize the root node
        current_player = self.id  # Start with this agent as the current player
        if self.first_turn:
            self.think_time+=WARM_UP
            self.first_turn = False
        else:
            self.think_time = THINK_TIME

        current_score = root_state.agents[current_player].score
        while time.time() - start_time < self.think_time:
            node = root_node
            state = deepcopy(root_state)  # Start with the current root state
            sim_player = current_player  # Ensure simulation starts with the correct player
            
            # Selection: Traverse to the most promising child
            while node.is_fully_expanded() and node.children:
                node = node.best_child(current_score)

            # Expansion: Expand the node if possible
            if node.untried_moves:
                node = self.Expand(node, sim_player)
            
            # Simulation: Simulate the rest of the game randomly
            simulation_state = deepcopy(state)
            while simulation_state.TilesRemaining():
                legal_moves = self.game_rule.getLegalActions(simulation_state, sim_player)
                if legal_moves:
                    move = random.choice(legal_moves)
                    # Apply the move
                    self.game_rule.generateSuccessor(simulation_state, move, sim_player)

                # Switch player for the next simulation move
                sim_player = 1 - sim_player

            # Backpropagation: Score the result of the simulation
            scaled_reward = self.evaluate_state(simulation_state, self.id)
            
            # Backpropagation: Update nodes with the result
            while node is not None:
                node.update(scaled_reward)
                node = node.parent

        # Return the best move from the root node
        best_child = root_node.best_child(exploration_param=0)
        if best_child is None:
            return random.choice(actions)  # Fallback to a random action if no best child found

        self.action_counter+=1
        return best_child.getMove()
    
    # Evaluates the final state for MCTS by calculating bonuses and penalties
    def evaluate_state(self, state: AzulState, player_id: int) -> float:
        player_score, opponent_score = 0, 0

        for agent in state.agents:
            total_score = (
                agent.ScoreRound()[0] +
                agent.GetCompletedRows() * (agent.ROW_BONUS * 1.5) +
                agent.GetCompletedColumns() * (agent.COL_BONUS * 1.5) +
                agent.GetCompletedSets() * (agent.SET_BONUS * 2)
            )

            if agent.id == player_id:
                player_score = total_score
                #player_pattern = pattern
            else:
                opponent_score = total_score
                #opponent_pattern = pattern
        
        score_diff = player_score - opponent_score

        # Normalize the score between -1.0 and 1.0
        return max(0, min(score_diff / MAX_DIFF, 1))

    # Evaluates a potential action using domain knowledge and bonuses
    def evaluate_action(self, state: AzulState, action) -> float:
        # Simulate the action to see its effects
        curr_state = deepcopy(state)
        next_state = self.game_rule.generateSuccessor(curr_state, action, self.id)
        next_state.agents[self.id].ScoreRound()  # Score the round for this agent
        agent_score = next_state.agents[self.id].ScoreRound()[0]  # Get score

        agent = next_state.agents[self.id]
        completed_rows = agent.GetCompletedRows()
        completed_cols = agent.GetCompletedColumns()
        completed_sets = agent.GetCompletedSets()

        # Calculate bonuses for completed rows, columns, and sets
        row_bonus = completed_rows * agent.ROW_BONUS
        col_bonus = completed_cols * agent.COL_BONUS
        set_bonus = completed_sets * agent.SET_BONUS
        complete_bonus = row_bonus + col_bonus + set_bonus  # Total bonus for completed rows, columns, sets

        # Calculate the total heuristic score combining agent score, bonuses, penalties, and domain knowledge
        heuristic_score = agent_score + complete_bonus
        return heuristic_score