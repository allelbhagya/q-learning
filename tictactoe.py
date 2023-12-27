import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'

    def is_winner(self, player):
        for i in range(3):
            if all(self.board[i * 3 + j] == player for j in range(3)) or \
               all(self.board[j * 3 + i] == player for j in range(3)):
                return True
        if all(self.board[i] == player for i in [0, 4, 8]) or \
           all(self.board[i] == player for i in [2, 4, 6]):
            return True
        return False

    def is_full(self):
        return ' ' not in self.board

    def is_game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_full()

    def legal_moves(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, move):
        self.board[move] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def display(self):
        for i in range(3):
            print(' '.join(self.board[i * 3:i * 3 + 3]))
        print()

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=1.0):
        self.q_values = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def update_q_value(self, state, action, value):
        current_q = self.get_q_value(state, action)
        self.q_values[(state, action)] = current_q + self.alpha * (value - current_q)

    def choose_action(self, state, legal_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(legal_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in legal_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(legal_actions, q_values) if q == max_q]
            return random.choice(best_actions)

def train_q_learning_agent(agent, environment, num_episodes=10000):
    for episode in range(num_episodes):
        environment.reset()
        state = tuple(environment.board)
        while not environment.is_game_over():
            legal_moves = environment.legal_moves()
            action = agent.choose_action(state, legal_moves)
            environment.make_move(action)
            new_state = tuple(environment.board)

            reward = 0
            if environment.is_winner('X'):
                reward = 1
            elif environment.is_winner('O'):
                reward = -1

            best_future_q = max(agent.get_q_value(new_state, a) for a in legal_moves)
            agent.update_q_value(state, action, reward + agent.gamma * best_future_q)

            state = new_state

if __name__ == "__main__":
    game = TicTacToe()
    agent = QLearningAgent()

    print("Training the Q-learning agent...")
    train_q_learning_agent(agent, game)
    game.reset()
    while not game.is_game_over():
        game.display()

        if game.current_player == 'X':
            player_move = int(input("Enter your move (0-8): "))
            game.make_move(player_move)
        else:
            agent_move = agent.choose_action(tuple(game.board), game.legal_moves())
            game.make_move(agent_move)

    game.display()

    if game.is_winner('X'):
        print("You win!")
    elif game.is_winner('O'):
        print("Q-learning agent wins!")
    else:
        print("It's a draw!")
