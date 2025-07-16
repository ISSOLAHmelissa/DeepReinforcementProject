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
