import random

import random

class MontyHallLevel01Env_DP:
    def __init__(self):
        self.doors = [0, 1, 2]

    def get_all_states(self):
        states = [('start', None, None)]
        for first in self.doors:
            for revealed in self.doors:
                if revealed != first:
                    states.append(('second_choice', first, revealed))
        states.append(('terminal_win',))
        states.append(('terminal_lose',))
        return states

    def get_all_actions(self, state):
        if state[0] == 'start':
            return [0, 1, 2]
        elif state[0] == 'second_choice':
            return ['keep', 'switch']
        return []

    def is_terminal(self, state):
        return state in [('terminal_win',), ('terminal_lose',)]

    def get_valid_actions(self, state):
        return self.get_all_actions(state)

    def get_transition_model(self):
        model = dict()
        states = self.get_all_states()

        for state in states:
            if self.is_terminal(state):
                continue
            model[state] = dict()
            for action in self.get_valid_actions(state):
                transitions = []
                if state[0] == 'start':
                    # Agent choisit une porte, nature choisit une porte gagnante et révèle une autre
                    for winning_door in self.doors:
                        for revealed in self.doors:
                            if revealed != action and revealed != winning_door:
                                next_state = ('second_choice', action, revealed)
                                prob = 1.0 / 3  # Probabilité que la porte gagnante soit choisie
                                transitions.append((prob, next_state, 0.0))
                    model[state][action] = transitions

                elif state[0] == 'second_choice':
                    first_choice, revealed = state[1], state[2]
                    for winning_door in self.doors:
                        if first_choice == winning_door:
                            final_choice = first_choice if action == 'keep' else \
                                [d for d in self.doors if d != first_choice and d != revealed][0]
                            reward = 1.0 if final_choice == winning_door else 0.0
                            next_state = ('terminal_win',) if reward == 1.0 else ('terminal_lose',)
                            transitions = [(1.0 / 3, next_state, reward)]
                        else:
                            final_choice = first_choice if action == 'keep' else \
                                [d for d in self.doors if d != first_choice and d != revealed][0]
                            reward = 1.0 if final_choice == winning_door else 0.0
                            next_state = ('terminal_win',) if reward == 1.0 else ('terminal_lose',)
                            transitions = [(1.0 / 3, next_state, reward)]
                        model[state][action] = model[state].get(action, []) + transitions

        return model


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
        pass

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.state
        if state[0] == 'start':
            return [0, 1, 2]
        elif state[0] == 'second_choice':
            return ['keep', 'switch']
        return []

    def is_terminal(self):
        return self.done

    def score(self):
        if self.state[0] == 'terminal':
            return 1.0 if self.state[1] == self.state[2] else 0.0
        return 0.0

    def is_game_over(self):
        return self.is_terminal()

    def available_actions(self, state=None):
        return self.get_valid_actions(state)

    def state_id(self, state=None):
        if state is None:
            state = self.state
        return str(state)

    def num_actions(self, state=None):
        if state is None:
            state = self.state
        return len(self.get_valid_actions(state))

    def display(self):
        print(f"[État actuel] : {self.state}")

import random
from collections import defaultdict
from itertools import permutations

class MontyHallLevel02Env_DP:
    def __init__(self):
        self.doors = [0, 1, 2, 3, 4]
        self.max_steps = 3
        self.reset()

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.choices = []
        self.revealed_doors = []
        self.remaining_closed = None
        self.done = False
        self.state = ('start', tuple(self.choices), tuple(self.revealed_doors))
        return self.state

    def get_all_states(self):
        states = set()
        def build(choices, revealed, depth):
            states.add(('start', tuple(choices), tuple(revealed)))
            if depth < self.max_steps:
                for c in self.doors:
                    if c in revealed:
                        continue
                    for r in self.doors:
                        if r != c and r not in revealed:
                            build(choices + [c], revealed + [r], depth + 1)

        build([], [], 0)
        for c in self.doors:
            for r in self.doors:
                if r != c:
                    states.add(('final_choice', c, r))
        for d in self.doors:
            states.add(('terminal', d))
        return list(states)

    def get_all_actions(self, state):
        if state[0] == 'start':
            choices, revealed = state[1], state[2]
            return [d for d in self.doors if d not in revealed]
        elif state[0] == 'final_choice':
            return ['keep', 'switch']
        else:
            return []

    def get_valid_actions(self, state):
        return self.get_all_actions(state)

    def get_transition_model(self):
        model = defaultdict(lambda: defaultdict(list))
        for winning in self.doors:
            for state in self.get_all_states():
                if state[0] == 'start':
                    choices, revealed = list(state[1]), list(state[2])
                    if len(choices) < self.max_steps:
                        for action in self.get_valid_actions(state):
                            new_choices = choices + [action]
                            candidates = [d for d in self.doors if d != action and d != winning and d not in revealed]
                            if not candidates:
                                candidates = [d for d in self.doors if d != action and d not in revealed]
                            for r in candidates:
                                new_revealed = revealed + [r]
                                if len(new_choices) < self.max_steps:
                                    next_state = ('start', tuple(new_choices), tuple(new_revealed))
                                    model[state][action].append((1/len(candidates), next_state, 0.0))
                                else:
                                    still_closed = [d for d in self.doors if d not in new_revealed]
                                    last_choice = new_choices[-1]
                                    remaining_closed = [d for d in still_closed if d != last_choice]
                                    if len(remaining_closed) == 1:
                                        rc = remaining_closed[0]
                                        next_state = ('final_choice', last_choice, rc)
                                        model[state][action].append((1/len(candidates), next_state, 0.0))
                elif state[0] == 'final_choice':
                    last, remaining = state[1], state[2]
                    for action in ['keep', 'switch']:
                        final = last if action == 'keep' else remaining
                        reward = 1.0 if final == winning else 0.0
                        next_state = ('terminal', final)
                        model[state][action].append((1.0 / len(self.doors), next_state, reward))
        return model


import random
from itertools import combinations, permutations

import random

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
        self.state = ('start', tuple(self.choices), tuple(self.revealed_doors))
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

            candidates = [
                d for d in self.doors
                if d != action and d != self.winning_door and d not in self.revealed_doors
            ]

            if candidates:
                revealed = random.choice(candidates)
                self.revealed_doors.append(revealed)

            if len(self.choices) < self.max_steps:
                self.state = ('start', tuple(self.choices), tuple(self.revealed_doors))
                return self.state, 0.0, False
            else:
                still_closed = [d for d in self.doors if d not in self.revealed_doors]
                last_choice = self.choices[-1]

                if last_choice not in still_closed:
                    raise ValueError(f"La dernière porte choisie ({last_choice}) a été révélée — bug !")

                remaining_closed = [d for d in still_closed if d != last_choice]
                if len(remaining_closed) != 1:
                    raise AssertionError(f"Erreur logique : {remaining_closed} portes restantes.")

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

            states.add(('start', tuple(choices), tuple(revealed)))

            if depth < 3:
                valid_choices = [d for d in doors if d not in revealed]
                for choice in valid_choices:
                    possible_reveal = [
                        d for d in doors
                        if d != choice and d not in revealed
                    ]
                    for r in possible_reveal:
                        new_choices = tuple(list(choices) + [choice])
                        new_revealed = tuple(list(revealed) + [r])
                        build_states(new_choices, new_revealed, depth + 1)

        build_states(tuple(), tuple(), 0)

        for last_choice in doors:
            for remaining in doors:
                if remaining != last_choice:
                    states.add(('final_choice', last_choice, remaining))

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
    def display(self):
        print(f"[État actuel] : {self.state}")