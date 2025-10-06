import random
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from players import RLAgent


class SimpleNN(nn.Module):
    def __init__(self, in_features, out_features, hidden):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, out_features),
        ])

    def forward(self, x):
        x = x.to('cuda')
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)

        return x


class SimpleNNAgent(RLAgent):
    def __init__(self, name: str = "RL Agent", exploration_rate: float = 0.02):
        super().__init__(name, exploration_rate)
        self.model = SimpleNN(214, 40, 256).to('cuda')

        # extra info to remember
        self.episode_input = []
        self.episode_action_id = []
        self.episode_input_to_remember = None
        self.episode_action_id_to_remember = None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01)

    @staticmethod
    def __board_state_to_input(state: Dict[str, Any]) -> np.ndarray:
        ''' turns state dictionary into an array of usable inputs '''

        board = state['board']
        current_piece = state['current_piece']
        next_piece = state['next_piece']

        piece_type = {'I' : 0, 'O' : 1, 'L' : 2, 'J' : 3, 'Z' : 4, 'S' : 5, 'T' : 6}
        current_piece_v = [0.0 for _ in range(7)]
        current_piece_v[piece_type[current_piece]] = 1.0
        next_piece_v = [0.0 for _ in range(7)]
        next_piece_v[piece_type[next_piece]] = 1.0

        input_vals = np.append(np.array(board.flatten(), dtype = 'bool'), [current_piece_v, next_piece_v])
        # print(input_vals)
        return input_vals

    def remember_extra(self, input_params, action_id):
        self.episode_input_to_remember = input_params
        self.episode_action_id_to_remember = action_id

    def remember(self, state, action, reward):
        super().remember(state, action, reward)
        self.episode_input.append(self.episode_input_to_remember)
        self.episode_action_id.append(self.episode_action_id_to_remember)

    def get_best_action(self, state: Dict[str, Any]) -> Tuple[int, int]:
        input_params = self.__board_state_to_input(state)

        predictions = self.model.forward(torch.FloatTensor(input_params))
        # if random.randint(0, 50) == 25:
        #     print("predictions:")
        #     print(predictions)

        valid_actions = state['valid_actions']
        max_pos = 0
        max_pred = predictions[0]
        for i, prediction in enumerate(predictions):
            col, rotation = i // 4, i % 4
            if (col, rotation) not in valid_actions:
                continue
            if prediction > max_pred:
                max_pred = prediction
                max_pos = i
        # print(max_pos)

        action = max_pos // 4, max_pos % 4
        self.remember_extra(input_params, max_pos)

        return action

    def train(self, decay_factor = 0.8):
        # rewards are gained through multiple actions, propagate the reward back in time
        for i in range(len(self.episode_rewards) - 2, -1, -1):
            self.episode_rewards[i] += decay_factor * self.episode_rewards[i + 1]

        # print(len(self.episode_rewards))
        # print(self.episode_rewards)

        self.optimizer.zero_grad()

        for episode in range(len(self.episode_states)):
            input_params = self.episode_input[episode]
            output = self.model.forward(torch.FloatTensor(input_params))
            action_id = self.episode_action_id[episode]
            ((output[action_id] - self.episode_rewards[episode]) ** 2).backward()

        # print(self.model.fc1.weight)

        self.optimizer.step()

        import random
        if random.randint(1, 50) == 25:
            print(self.episode_rewards)
        #     print(self.episode_action_id)
        #     print(self.episode_action_id)
        #     print(self.episode_rewards)
        #     print(self.model.fc1.weight)

        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_states.clear()
        self.episode_input.clear()
        self.episode_action_id.clear()


    def calculate_reward(self, prev_state: Dict, new_state: Dict, game_over: bool) -> float:
        """
        Calculate reward for state transition.

        Args:
            prev_state: Previous game state
            new_state: New game state
            game_over: Whether game ended

        Returns:
            Reward value
        """
        reward = 0.0
        if game_over:
            # point penalty for losing
            reward -= 200
            pass

        # reward for not dying
        reward += 5

        # small penalty for increasing the column size
        prev_max_height = self._get_max_height(prev_state['board'])
        current_max_height = self._get_max_height(new_state['board'])
        # print(current_max_height - prev_max_height)
        reward -= 2 * (current_max_height - prev_max_height)

        # Reward for increasing score
        reward += 2.0 * (new_state['score'] - prev_state['score'])

        return reward

    def choose_action(self, state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Choose action with exploration-exploitation.

        Args:
            state: Current game state

        Returns:
            (column, rotation) action
        """
        valid_actions = state['valid_actions']
        if not valid_actions:
            return (0, 0)

        # Exploration
        if self.training_mode and random.random() < self.exploration_rate:
            random_action = random.choice(valid_actions)

            column, rotation = random_action
            input_params = self.__board_state_to_input(state)
            self.remember_extra(input_params, column * 4 + rotation)
            return random_action

        # Exploitation
        return self.get_best_action(state)