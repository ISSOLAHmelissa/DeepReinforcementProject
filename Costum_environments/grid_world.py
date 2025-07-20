import numpy as np
from collections import defaultdict
from collections import defaultdict

class GridWorld:
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4  # 0: gauche, 1: droite, 2: haut, 3: bas
        self.terminal_states = [4, 24]  # état 4 (-3), état 24 (+1)
        self.rewards_dict = {4: -3.0, 24: 1.0}
        self.R = sorted(list(set(self.rewards_dict.values()) | {0.0}))  # [-3.0, 0.0, 1.0]

        self.model = defaultdict(lambda: defaultdict(list))
        self.valid_actions = defaultdict(list)
        self.forbidden_actions = defaultdict(list)

        self._build_model()

    def _coords(self, s):
        return divmod(s, self.cols)

    def _to_state(self, row, col):
        return row * self.cols + col

    def _build_model(self):
        for s in range(self.n_states):
            row, col = self._coords(s)

            for a in range(self.n_actions):
                if s in self.terminal_states:
                    # Terminal state: transition vers soi-même avec reward 0
                    self.model[s][a].append((1.0, s, 0.0, True))
                    continue

                new_row, new_col = row, col
                if a == 0 and col > 0:         # gauche
                    new_col -= 1
                elif a == 1 and col < self.cols - 1:  # droite
                    new_col += 1
                elif a == 2 and row > 0:       # haut
                    new_row -= 1
                elif a == 3 and row < self.rows - 1:  # bas
                    new_row += 1
                else:
                    # Action interdite (mur)
                    self.forbidden_actions[s].append(a)
                    continue

                s_prime = self._to_state(new_row, new_col)
                reward = self.rewards_dict.get(s_prime, 0.0)
                done = s_prime in self.terminal_states

                self.model[s][a].append((1.0, s_prime, reward, done))
                self.valid_actions[s].append(a)

    def get_all_states(self):
        return list(range(self.n_states))

    def get_all_actions(self):
        return list(range(self.n_actions))

    def get_rewards(self):
        return self.R

    def get_model(self):
        return self.model

    def get_valid_actions(self):
        return self.valid_actions

    def get_terminal_states(self):
        return self.terminal_states

    def is_terminal(self, s):
        return s in self.terminal_states


class MonteCarloEnv:
    def num_states(self) -> int:
        raise NotImplementedError()

    def num_actions(self) -> int:
        raise NotImplementedError()

    def step(self, a: int):
        """
        Applique l'action `a` et renvoie (nouvel état, récompense, terminé)
        """
        raise NotImplementedError()

    def score(self) -> float:
        """
        Score cumulé (pour calculer les rewards par différence)
        """
        raise NotImplementedError()

    def is_game_over(self) -> bool:
        """
        Retourne True si l'épisode est terminé
        """
        raise NotImplementedError()

    def reset(self):
        """
        Réinitialise l'environnement et retourne l'état initial
        """
        raise NotImplementedError()

    def get_valid_actions(self, state: int) -> list[int]:
        """
        Liste des actions possibles depuis l'état `state`
        """
        raise NotImplementedError()

    def state_id(self) -> int:
        """
        Identifiant de l'état actuel
        """
        raise NotImplementedError()

    def available_actions(self) -> list[int]:
        """
        Actions valides depuis l'état actuel
        """
        raise NotImplementedError()

import random
class GridWorld_MC(MonteCarloEnv):
    def __init__(self):
        self.s = 0
        self.inner_score = 0.0

    def num_states(self) -> int:
        return 25

    def num_actions(self) -> int:
        return 4  # gauche, droite, haut, bas

    def state(self) -> int:
        return self.s

    def state_id(self) -> int:
        return self.s  # <--- C’est cette méthode qu’il manquait

    def available_actions(self) -> list[int]:
        return self.get_valid_actions(self.s)

    def step(self, a: int):
        if self.is_game_over():
            raise Exception("Épisode terminé")

        row, col = divmod(self.s, 5)

        if a == 0 and col > 0:
            col -= 1
        elif a == 1 and col < 4:
            col += 1
        elif a == 2 and row > 0:
            row -= 1
        elif a == 3 and row < 4:
            row += 1

        new_state = row * 5 + col

        if new_state == 4:
            reward = -3.0
        elif new_state == 24:
            reward = 1.0
        else:
            reward = 0.0

        self.s = new_state
        self.inner_score += reward

        done = self.is_game_over()
        return new_state, reward, done

    def score(self) -> float:
        return self.inner_score

    def is_game_over(self) -> bool:
        return self.s == 4 or self.s == 24

    def reset(self):
        self.s = 0
        self.inner_score = 0.0
        return self.s

    def get_valid_actions(self, state: int) -> list[int]:
        row, col = divmod(state, 5)
        valid = []
        if col > 0:
            valid.append(0)
        if col < 4:
            valid.append(1)
        if row > 0:
            valid.append(2)
        if row < 4:
            valid.append(3)
        return valid
    def from_random_state(self):

        self.s = random.randint(0, self.num_states() - 1)
        self.inner_score = 0.0
        return self.s
    
    def display(self):
        grid = ['.'] * self.num_states()
        grid[self.s] = 'A'
        for i in range(0, self.num_states(), 5):
            print("".join(grid[i:i+5]))

