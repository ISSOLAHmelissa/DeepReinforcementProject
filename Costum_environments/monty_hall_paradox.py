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
    

import itertools
from collections import defaultdict

class MontyHallLevel02Env:
    def __init__(self):
        self.doors = [0, 1, 2, 3, 4]
        self.reset()

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.choices = []
        self.revealed = []
        self.available_doors = self.doors.copy()  # Nouvelle liste pour les portes disponibles
        self.state = ('start', tuple(self.choices), tuple(self.revealed))
        self.done = False
        return self.state

    def is_terminal(self):
        return self.done

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.state
        phase, choices, revealed = state

        if phase not in ['start', 'progress']:
            return []

        choices = list(choices)
        revealed = list(revealed)

        if phase == 'start':
            return self.available_doors.copy()  # Utiliser seulement les portes disponibles
        elif phase == 'progress':
            return [d for d in self.available_doors if d not in choices and d not in revealed]

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over.")

        if len(self.choices) < 4:
            # Validate action
            if action not in self.available_doors or action in self.choices or action in self.revealed:
                raise ValueError(f"Invalid action {action}. Available: {self.available_doors}, Chosen: {self.choices}, Revealed: {self.revealed}")

            self.choices.append(action)
            
            # Determine doors that can be revealed
            possible_reveals = [
                d for d in self.available_doors 
                if d not in self.choices 
                and d != self.winning_door
            ]
            
            # If no "safe" doors to reveal (not winning door), allow revealing any available door except chosen ones
            if not possible_reveals:
                possible_reveals = [
                    d for d in self.available_doors 
                    if d not in self.choices
                ]
                
            # This should never be empty at this point
            if not possible_reveals:
                self.done = True
                return self.state, -10.0, True  # Penalize invalid state
                
            revealed_door = random.choice(possible_reveals)
            self.revealed.append(revealed_door)
            
            # Update available doors
            if revealed_door in self.available_doors:
                self.available_doors.remove(revealed_door)
            
            self.state = ('progress', tuple(self.choices), tuple(self.revealed))
            return self.state, 0.0, False
        else:
            # Final decision logic
            last_choice = self.choices[-1]
            remaining_doors = [
                d for d in self.available_doors 
                if d not in self.choices[:-1] 
                and d not in self.revealed
            ]

            if len(remaining_doors) != 2:
                self.done = True
                return self.state, -10.0, True  # Penalize invalid final state

            if action == 'keep':
                final_choice = last_choice
            else:
                final_choice = next(d for d in remaining_doors if d != last_choice)

            reward = 1.0 if final_choice == self.winning_door else 0.0
            self.state = ('terminal', final_choice, self.winning_door)
            self.done = True
            return self.state, reward, True

    def get_all_states(self):
        states = []
        doors = self.doors

        # Start state
        states.append(('start', (), ()))

        # Progress states (1-4 choices)
        for num_choices in range(1, 5):
            for choices in itertools.permutations(doors, num_choices):
                # Les portes disponibles sont celles qui n'ont pas été révélées
                available_doors = [d for d in doors if d not in self.revealed]
                remaining_doors = [d for d in available_doors if d not in choices]
                
                # Can reveal up to num_choices doors (1 reveal per choice)
                max_reveals = num_choices
                for num_reveals in range(1, max_reveals + 1):
                    for reveals in itertools.permutations(remaining_doors, num_reveals):
                        # Check no duplicates in reveals and no overlap with choices
                        if len(set(reveals)) == len(reveals) and not set(reveals) & set(choices):
                            states.append(('progress', choices, reveals))

        # Terminal states
        for chosen in doors:
            for win in doors:
                states.append(('terminal', (chosen,), (win,)))

        return states

    def get_all_actions(self, state=None):
        return self.get_valid_actions(state)