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

class MontyHallLevel02Env:
    """
    Monty Hall Paradox Level 2:
    - 5 doors
    - Agent makes 4 successive selections
    - Then 3 losing doors are revealed among the unchosen ones
    - Agent can keep or switch between the last 2 closed doors
    - Reward = 1 if final choice is the winning door
    """
    def __init__(self):
        self.doors = [0, 1, 2, 3, 4]
        self.max_steps = 4
        self.reset()

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.choices = []
        self.step_count = 0
        self.done = False
        self.state = ('start', tuple(self.choices))
        return self.state

    def get_all_states(self):
        # All possible sequences of choices (for DP)
        states = []
        for steps in range(self.max_steps + 1):
            for c in self._generate_choices_sequences(steps):
                states.append(('start', tuple(c)))
        # After revealing 3 losing doors
        for c in self._generate_choices_sequences(self.max_steps):
            states.append(('final_choice', c[-1], None))
        states.append(('terminal', None))
        return states

    def _generate_choices_sequences(self, length):
        if length == 0:
            return [[]]
        shorter = self._generate_choices_sequences(length - 1)
        return [s + [d] for s in shorter for d in self.doors]

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.state
        if state[0] == 'start':
            return [0, 1, 2, 3, 4]
        elif state[0] == 'final_choice':
            return ['keep', 'switch']
        return []

    def step(self, action):
        if self.done:
            raise ValueError("Game is already over.")

        if self.state[0] == 'start':
            # Record selection
            self.choices.append(action)
            self.step_count += 1

            if self.step_count < self.max_steps:
                self.state = ('start', tuple(self.choices))
                return self.state, 0.0, False
            else:
                # After 4 selections, reveal 3 losing doors
                last_choice = self.choices[-1]
                remaining = [d for d in self.doors if d != last_choice]
                revealable = [d for d in remaining if d != self.winning_door]
                self.revealed = random.sample(revealable, 3)
                self.remaining_closed = [d for d in self.doors if d not in self.revealed and d != last_choice]
                assert len(self.remaining_closed) == 1, "Should be exactly 1 door remaining besides last choice"
                self.state = ('final_choice', last_choice, None)
                return self.state, 0.0, False

        elif self.state[0] == 'final_choice':
            last_choice = self.state[1]
            other_closed = self.remaining_closed[0]
            if action == 'keep':
                final_choice = last_choice
            else:
                final_choice = other_closed
            reward = 1.0 if final_choice == self.winning_door else 0.0
            self.state = ('terminal', final_choice)
            self.done = True
            return self.state, reward, True

        else:
            raise ValueError("Invalid state transition.")

    def is_terminal(self):
        return self.done