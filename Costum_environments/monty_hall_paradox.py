import random

class MontyHallLevel01Env:
    def __init__(self):
        self.doors = [0, 1, 2]
        self.reset()

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.state = ('start', None, None)
        self.done = False
        return self.state

    def get_all_states(self):
        # Utile pour DP
        states = []
        for first in self.doors:
            for revealed in self.doors:
                if revealed != first:
                    states.append(('second_choice', first, revealed))
        states.append(('start', None, None))
        return states

    def get_all_actions(self, state=None):
        if self.state[0] == 'start':
            return [0, 1, 2]  # choisir une porte
        elif self.state[0] == 'second_choice':
            return ['keep', 'switch']
        return []

    def step(self, action):
        if self.state[0] == 'start':
            first_choice = action
            remaining = [d for d in self.doors if d != first_choice and d != self.winning_door]
            revealed = random.choice(remaining) if remaining else random.choice([d for d in self.doors if d != first_choice])
            self.state = ('second_choice', first_choice, revealed)
            return self.state, 0.0, False

        elif self.state[0] == 'second_choice':
            first_choice, revealed = self.state[1], self.state[2]
            final_choice = first_choice if action == 'keep' else next(
                d for d in self.doors if d != first_choice and d != revealed
            )
            reward = 1.0 if final_choice == self.winning_door else 0.0
            self.state = ('terminal', final_choice, self.winning_door)
            self.done = True
            return self.state, reward, True

        else:
            raise ValueError("Game is already over.")

    def is_terminal(self):
        return self.done

    def get_transition_model(self):
        # Pour DP : retourner un modèle {s: {a: [(p, s’, r, done), ...]}}
        # Optionnel — à coder si DP nécessaire
        pass
    
    def get_valid_actions(self, state=None):
        if state is None:
           state = self.state
        if state[0] == 'start':
           return [0, 1, 2]
        elif state[0] == 'second_choice':
           return ['keep', 'switch']
        return []

