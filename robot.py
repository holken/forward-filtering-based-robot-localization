import random


class Robot:

    def __init__(self, position, direction, size):
        self.position = position
        self.size = size
        self.directions = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        self.direction = self.directions[direction]
        self.direction_state = direction

    def inside_boundary(self, dir):
        if 0 <= self.position[0] + dir[0] < self.size and 0 <= self.position[1] + dir[1] < self.size:
            return True
        else:
            return False

    def new_direction(self):
        available_dir = []
        for dir in self.directions.values():
            if self.inside_boundary(dir) and dir != self.direction:
                available_dir.append(dir)
        return available_dir[random.randint(0, len(available_dir) - 1)]

    def move_robot(self):
        facing_wall = not self.inside_boundary(self.direction)
        if facing_wall:
            direction = self.new_direction()
        else:
            chose_new_direction = random.randint(1, 10) <= 7
            if chose_new_direction:
                direction = self.new_direction()
            else:
                direction = self.direction
        self.position = (self.position[0] + direction[0], self.position[1] + direction[1])
        return self.position, self.direction

    def get_position(self):
        return self.position

    def get_direction(self):
        return self.direction

    def get_direction_state(self):
        return self.direction_state
