class LineWorld:
    ACTIONS = ['left', 'right']
    
    def __init__(self, size=5, start_pos=2):
        assert size >= 3, "LineWorld doit avoir au moins 3 états"
        self.size = size
        self.start_pos = start_pos
        self.reset()

    def reset(self):
        self.position = self.start_pos
        self.done = False
        return self.position

    def available_actions(self):
        return self.ACTIONS

    def step(self, action):
        assert action in self.ACTIONS, "Action invalide"

        if self.done:
            raise Exception("Épisode terminé, appelez reset()")

        if action == 'left':
            self.position -= 1
        elif action == 'right':
            self.position += 1

        # Vérifie les états terminaux
        if self.position == 0:
            self.done = True
            return self.position, -1, True
        elif self.position == self.size - 1:
            self.done = True
            return self.position, +1, True
        else:
            return self.position, 0, False

    def render(self):
        line = ['.'] * self.size
        line[self.position] = 'A'
        print("".join(line))

    def is_done(self):
        return self.done

class LineWorldEnv:
    ACTIONS = ['left', 'right']
    ACTION_TO_INDEX = {a: i for i, a in enumerate(ACTIONS)}
    INDEX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

    def __init__(self, size=5, start_pos=2):
        assert size >= 3, "LineWorld doit avoir au moins 3 états"
        self.size = size
        self.start_pos = start_pos
        self.reset()

    def reset(self):
        self.position = self.start_pos
        self.done = False
        return self.position

    def step(self, action_index):
        action = self.INDEX_TO_ACTION[action_index]
        if self.done:
            raise Exception("Épisode terminé, appelez reset()")

        if action == 'left':
            self.position -= 1
        elif action == 'right':
            self.position += 1

        reward = 0.0
        if self.position == 0:
            self.done = True
            reward = -1.0
        elif self.position == self.size - 1:
            self.done = True
            reward = +1.0

        return self.state_id(), reward, self.done

    def is_game_over(self):
        return self.done

    def available_actions(self):
        return list(range(len(self.ACTIONS)))  # [0, 1]

    def state_id(self):
        return self.position

    def score(self):
        if self.position == self.size - 1:
            return 1.0
        elif self.position == 0:
            return -1.0
        return 0.0

    def display(self):
        line = ['.'] * self.size
        line[self.position] = 'A'
        print("".join(line))

    def is_forbidden(self, a):
        return False  # Toutes les actions sont permises dans LineWorld
    def num_actions(self):
        return len(self.available_actions())


