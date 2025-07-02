class MonteCarloEnv:

  def num_states(self) -> int:
    raise NotImplementedError()

  def num_actions(self) -> int:
    raise NotImplementedError()

  def step(self, a: int):
    raise NotImplementedError()

  def score(self) -> float:
    raise NotImplementedError()

  def is_game_over(self) -> bool:
    raise NotImplementedError()

  def reset(self):
    raise NotImplementedError()

class GridWorld_MC(MonteCarloEnv):
    def __init__(self):
        self.s = 0  # position initiale (état)
        self.inner_score = 0.0

    def num_states(self) -> int:
        return 25

    def num_actions(self) -> int:
        return 4  # 0: gauche, 1: droite, 2: haut, 3: bas

    def state(self) -> int:
        return self.s

    def step(self, a: int):
        if self.is_game_over():
            raise Exception("Épisode terminé")

        row, col = divmod(self.s, 5)

        if a == 0 and col > 0:  # gauche
            col -= 1
        elif a == 1 and col < 4:  # droite
            col += 1
        elif a == 2 and row > 0:  # haut
            row -= 1
        elif a == 3 and row < 4:  # bas
            row += 1

        new_state = row * 5 + col

        # Calcul de la récompense immédiate
        if new_state == 4:
            reward = -3.0
        elif new_state == 24:
            reward = 1.0
        else:
            reward = 0.0

        self.s = new_state
        self.inner_score += reward  # si tu souhaites garder score cumulé

        done = self.is_game_over()
        return new_state, reward, done

    def score(self) -> float:
        return self.inner_score

    def is_game_over(self) -> bool:
        return self.s == 4 or self.s == 24

    def reset(self):
        self.s = 0
        self.inner_score = 0.0
        return self.s  # Explicitly return the initial state

    def get_all_states(self):
        """Return all possible states in the grid world"""
        return list(range(self.num_states()))  # States are 0 to 24

    def get_all_actions(self):
        """Return all possible actions in the grid world"""
        return list(range(self.num_actions()))  # Actions are 0 to 3