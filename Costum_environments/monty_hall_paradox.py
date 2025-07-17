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
    
import random

from itertools import combinations, permutations

import random
from itertools import combinations, permutations

class MontyHallLevel02Env:
    def __init__(self):
        self.doors = [0, 1, 2, 3, 4]
        self.max_steps = 3
        self.reset()

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.choices = []
        self.revealed_doors = []
        self.done = False
        self.state = ('start', tuple(self.choices), tuple(self.revealed_doors))  # ✅ tuple
        return self.state   

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.state

        if state[0] == 'start':
            return [d for d in self.doors if d not in self.revealed_doors]
        elif state[0] == 'final_choice':
            return ['keep', 'switch']
        return []

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over.")

        if self.state[0] == 'start':
            self.choices.append(action)

            # Révéler une porte ≠ choix du tour courant, ≠ gagnante, ≠ déjà révélée
            candidates = [
                d for d in self.doors
                if d != action and d != self.winning_door and d not in self.revealed_doors
            ]

            if candidates:
                revealed = random.choice(candidates)
                self.revealed_doors.append(revealed)
                print(f"🧑‍🔧 Monty a révélé la porte {revealed} (perdante).")

            if len(self.choices) < self.max_steps:
                self.state = ('start', tuple(self.choices), tuple(self.revealed_doors))  # ✅ YES
                return self.state, 0.0, False
            else:
                still_closed = [d for d in self.doors if d not in self.revealed_doors]
                last_choice = self.choices[-1]

                if last_choice not in still_closed:
                    raise ValueError(f"La dernière porte choisie ({last_choice}) a été révélée — bug !")

                remaining_closed = [d for d in still_closed if d != last_choice]
                if len(remaining_closed) != 1:
                    raise AssertionError(f"Erreur logique : {remaining_closed} portes restantes. "
                                         f"Choix: {self.choices}, Révélées: {self.revealed_doors}, Fermées: {still_closed}")

                self.remaining_closed = remaining_closed[0]
                self.state = ('final_choice', last_choice, self.remaining_closed)
                return self.state, 0.0, False

        elif self.state[0] == 'final_choice':
            last_choice = self.state[1]
            if action == 'keep':
                final_choice = last_choice
            elif action == 'switch':
                final_choice = self.remaining_closed
            else:
                raise ValueError(f"Action invalide : {action}")

            reward = 1.0 if final_choice == self.winning_door else 0.0
            self.state = ('terminal', final_choice)
            self.done = True
            return self.state, reward, True

    def is_terminal(self):
        return self.done
    
    def get_all_states(self):
        states = set()
        doors = self.doors

        def build_states(choices, revealed, depth):
            if depth > 3:
                return

            # Ajouter l’état actuel — toujours avec des tuples
            states.add(('start', tuple(choices), tuple(revealed)))

            if depth < 3:
                valid_choices = [d for d in doors if d not in revealed]

                for choice in valid_choices:
                    possible_reveal = [
                        d for d in doors
                        if d != choice and d not in revealed
                    ]
                    for r in possible_reveal:
                        # Ajout proprement en tuple
                        new_choices = tuple(list(choices) + [choice])
                        new_revealed = tuple(list(revealed) + [r])
                        build_states(new_choices, new_revealed, depth + 1)

        # États 'start'
        build_states(tuple(), tuple(), 0)

        # États 'final_choice'
        for last_choice in doors:
            for remaining in doors:
                if remaining != last_choice:
                    states.add(('final_choice', last_choice, remaining))

        # États 'terminal'
        for d in doors:
            states.add(('terminal', d))

        return list(states)
    
    def state_id(self):
        return self.state


    def available_actions(self):
        return self.get_valid_actions(self.state)
    
    def is_game_over(self):
        return self.done
    
    def score(self):
        if self.state[0] == 'terminal':
            return 1.0 if self.state[1] == self.winning_door else 0.0
        return 0.0


