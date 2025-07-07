import random

class RockPaperScissorsEnv:
    def __init__(self):
        self.choices = [0, 1, 2]  # 0: Pierre, 1: Feuille, 2: Ciseaux
        self.result = ((0, -1, 1),  # Matrice des résultats
                      (1, 0, -1),
                      (-1, 1, 0))
        self.reset()

    def reset(self):
        self.state = ('first_round', None, None)  # (phase, player_choice, opponent_choice)
        self.reward = 0
        self.done = False
        self.round = 1
        return self.state

    def get_all_states(self):
        # Retourne tous les états possibles pour DP
        states = [('first_round', None, None)]
        for player_choice in self.choices:
            for opponent_choice in self.choices:
                states.append(('second_round', player_choice, opponent_choice))
                states.append(('terminal', player_choice, opponent_choice))
        return states

    def get_all_actions(self, state=None):
        if state is None:
            state = self.state
        if state[0] in ['first_round', 'second_round']:
            return [0, 1, 2]  # Pierre, Feuille, Ciseaux
        return []

    def step(self, action):
        if self.state[0] == 'first_round':
            player_choice = action
            opponent_choice = random.choice(self.choices)
            self.state = ('second_round', player_choice, opponent_choice)
            outcome = self.result[player_choice][opponent_choice]
            if outcome == 1:
                self.reward += 1
            elif outcome == -1:
                self.reward -= 1
            return self.state, self.reward, False

        elif self.state[0] == 'second_round':
            player_choice = action
            opponent_choice = self.state[1]  # L'adversaire choisit le premier choix du joueur
            self.state = ('terminal', player_choice, opponent_choice)
            outcome = self.result[player_choice][opponent_choice]
            if outcome == 1:
                self.reward += 1
            elif outcome == -1:
                self.reward -= 1
            self.done = True
            return self.state, self.reward, True

        else:
            raise ValueError("Game is already over.")

    def is_terminal(self):
        return self.done

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.state
        if state[0] in ['first_round', 'second_round']:
            return [0, 1, 2]  # Pierre, Feuille, Ciseaux
        return []

# Exemple d'utilisation
if __name__ == "__main__":
    env = RockPaperScissorsEnv()
    state = env.reset()
    print("--------------PREMIER ROUND--------------")
    player_choice = int(input("Pierre (0), Feuille (1) ou Ciseaux (2) ? "))
    state, reward, done = env.step(player_choice)
    print(f"Choix de l'adversaire : {state[2]}")
    print(f"Choix du joueur : {state[1]}")
    if env.result[state[1]][state[2]] == 1:
        print(f"Le joueur a gagné la manche ! Reward : {reward}")
    elif env.result[state[1]][state[2]] == -1:
        print(f"Le joueur a perdu la manche ! Reward : {reward}")
    else:
        print(f"Egalité !")

    if not done:
        print("--------------DEUXIEME ROUND--------------")
        player_choice = int(input("Pierre (0), Feuille (1) ou Ciseaux (2) ? "))
        state, reward, done = env.step(player_choice)
        print(f"Choix de l'adversaire : {state[2]}")
        print(f"Choix du joueur : {state[1]}")
        if env.result[state[1]][state[2]] == 1:
            print(f"Le joueur a gagné la manche ! Reward : {reward}")
        elif env.result[state[1]][state[2]] == -1:
            print(f"Le joueur a perdu la manche ! Reward : {reward}")
        else:
            print(f"Egalité !")

    print("--------------PARTIE TERMINEE--------------")
    print(f"Reward final : {reward}")