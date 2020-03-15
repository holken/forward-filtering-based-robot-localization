"""
This exercise is concerned with filtering in an environment with no landmarks. Consider a vacuum robot in an empty room, represented by an n×m rectangular grid. The robot’s
location is hidden; the only evidence available to the observer is a noisy location sensor that
gives an approximation to the robot’s location. If the robot is at location (x, y) then with
probability .1 the sensor gives the correct location, with probability .05 each it reports one
of the 8 locations immediately surrounding (x, y), with probability .025 each it reports one
of the 16 locations that surround those 8, and with the remaining probability of .1 it reports
“no reading.” The robot’s policy is to pick a direction and follow it with probability .8 on
each step; the robot switches to a randomly selected new heading with probability .2 (or with
probability 1 if it encounters a wall). Implement this as an HMM and do filtering to track the
robot. How accurately can we track the robot’s path?
"""
import random
import math
import numpy as np
import time
import robot as robot

import sys

class State:
    def __init__(self, location, direction):
        self.pos = location
        self.dir = direction


def print_first_layer(board):
    if robot_pos[0] > 0:
        for i in range(max(0, robot_pos[1]-1), min(SIZE, robot_pos[1] + 2)):
            board[robot_pos[0] - 1][i] = 1
    if robot_pos[0] < SIZE - 1:
        for i in range(max(0, robot_pos[1]-1), min(SIZE, robot_pos[1] + 2)):
            board[robot_pos[0] + 1][i] = 1
    if robot_pos[1] < SIZE - 1:
        board[robot_pos[0]][robot_pos[1] + 1] = 1
    if robot_pos[1] > 0:
        board[robot_pos[0]][robot_pos[1] - 1] = 1
    return board


def print_second_layer(board):
    if robot_pos[0] > 1:
        for i in range(max(0, robot_pos[1]-2), min(SIZE, robot_pos[1] + 3)):
            board[robot_pos[0] - 2][i] = 2
    if robot_pos[0] < SIZE - 2:
        for i in range(max(0, robot_pos[1]-2), min(SIZE, robot_pos[1] + 3)):
            board[robot_pos[0] + 2][i] = 2
    if robot_pos[1] < SIZE - 2:
        for i in range(max(0, robot_pos[0] - 1), min(SIZE, robot_pos[0] + 2)):
            board[i][robot_pos[1] + 2] = 2
    if robot_pos[1] > 1:
        for i in range(max(0, robot_pos[0] - 1), min(SIZE, robot_pos[0] + 2)):
            board[i][robot_pos[1] - 2] = 2
    return board


def print_board(robot_pos, estimated_position):
    board = [["#" for _ in range(SIZE)] for _ in range(SIZE)]
    board[robot_pos[0]][robot_pos[1]] = 0

    board = print_first_layer(board)
    board = print_second_layer(board)

    for i in range(SIZE):
        for j in range(SIZE):
            if (i, j) == estimated_position:
                print('X', end=' ')
            else:
                print(board[i][j], end=' ')
        print('')
    print('')

def get_distance(robot_pos, current_pos):
    return math.sqrt(abs(robot_pos[0] - current_pos[0]) ** 2 + abs(robot_pos[1] - current_pos[1]) ** 2)

def get_sensor_readings(robot_pos):
    del first_layer[:]
    del second_layer[:]
    sensor = [[0 for _ in range(nbr_of_states)] for _ in range(nbr_of_states)]
    for i in range(SIZE):
        for j in range(SIZE):
            distance = get_distance(robot_pos, (i, j))
            probability = 0
            if distance <= 0.001:
                probability = 0.1
            elif distance < 2:
                probability = 0.05
                first_layer.append((i, j))
            elif distance < 3:
                probability = 0.025
                second_layer.append((i, j))

            sensor_state = (i * SIZE + j) * 4
            for x in range(sensor_state, sensor_state + 4):
                sensor[x][x] = probability / 4  # Diagonal matrix
    return np.array(sensor)


def check_facing_wall(pos1, dir):
    y1, x1 = pos1 // SIZE, pos1 % SIZE
    if 0 == y1 and dir == 0:
        return True
    elif x1 == 7 and dir == 1:
        return True
    elif y1 == 7 and dir == 2:
        return True
    elif x1 == 0 and dir == 3:
        return True
    else:
        return False


def check_surrounding_walls(pos):
    if pos == 0 or pos == SIZE - 1 or pos == SIZE * 7 or pos == (SIZE * SIZE) - 1:
        return 2
    if 0 <= pos <= 7:
        return 1
    elif SIZE * 7 <= pos < SIZE * SIZE:
        return 1
    elif pos % SIZE == 1:
        return 1
    elif pos % SIZE == 0:
        return 1
    else:
        return 0


def in_range(pos1, pos2):
    x1, y1 = pos1 // SIZE, pos1 % SIZE
    x2, y2 = pos2 // SIZE, pos2 % SIZE
    return abs(x1 - x2) + abs(y1 - y2) <= 1


def normalize(m):
    alpha = np.amax(m)
    return m * (1/alpha)


def get_estimate():
    best_position = (0, 0)
    highest_probability = 0
    for i in range(SIZE):
        for j in range(SIZE):
            probability = np.sum(f[(i * SIZE + j) * 4:(i * SIZE + j) * 4 + 4])
            if probability > highest_probability:
                highest_probability = probability
                best_position = (i, j)
    return best_position


def can_move(pos1, pos2, dir):
    y1, x1 = pos1 // SIZE, pos1 % SIZE
    y2, x2 = pos2 // SIZE, pos2 % SIZE

    if y2 > y1 and dir == 2:  # SOUTH
        return True
    elif y2 < y1 and dir == 0:  # NORTH
        return True
    elif x2 < x1 and dir == 3:  # WEST
        return True
    elif x2 > x1 and dir == 1:  # EAST
        return True
    else:
        return False


SIZE = 8  # Board size
# 0 = north, 1 = east, 2 = south, 3 = west
if __name__ == "__main__":
    DELAY = 0.0001 if len(sys.argv) <= 1 else float(sys.argv[1])  # delay to help with displaying the movement
    ITERATIONS = 100 if len(sys.argv) <= 2 else int(sys.argv[2])  # Iterations on the board
    nbr_of_iterations = 0
    nbr_of_states = 4 * (SIZE ** 2)
    right_guesses = 0
    right_random_guesses = 0
    accumulated_distance_error = 0
    accumulated_random_distance_error = 0
    first_layer = []
    second_layer = []

    # Initialize Transition
    T = [[0 for _ in range(nbr_of_states)] for _ in range(nbr_of_states)]
    for i in range(len(T)):
        first_pos, first_dir = i // 4, i % 4
        facing_wall = check_facing_wall(first_pos, first_dir)
        walls = check_surrounding_walls(first_pos)
        for j in range(len(T[0])):
            second_pos, second_dir = j // 4, j % 4
            if second_pos == first_pos:
                continue
            if facing_wall and first_dir == second_dir:
                continue

            if in_range(first_pos, second_pos) and can_move(first_pos, second_pos, second_dir):

                if first_dir != second_dir:
                    if facing_wall:
                        T[i][j] = 1.0 / (4 - walls)
                    else:
                        T[i][j] = 0.3 / (4 - walls - 1)
                else:
                    T[i][j] = 0.7
            else:
                T[i][j] = 0.0
    T = np.array(T)
    T = np.transpose(T)

    # Initiate f
    f = [1.0 / nbr_of_states] * nbr_of_states
    f = np.array(f)

    robot_pos = (random.randint(0, 7), random.randint(0, 7))
    direction = random.randint(0, 3)
    robot = robot.Robot(robot_pos, direction, SIZE)

    while nbr_of_iterations < ITERATIONS:

        # Get sensor reading
        sensor_reading = get_sensor_readings(robot_pos)

        # Get new forward vector
        sensor_T = np.dot(sensor_reading,  T)
        f = normalize(np.dot(sensor_T, f))

        # Get estimated position of robot
        estimated_position = get_estimate()

        # Check if position match true location
        accumulated_distance_error += get_distance(robot_pos, estimated_position)
        if estimated_position[0] == robot_pos[0] and estimated_position[1] == robot_pos[1]:
            right_guesses += 1

        # Random guess
        random_guess = (random.randint(0, SIZE), random.randint(0, SIZE))
        accumulated_random_distance_error += get_distance(robot_pos, random_guess)
        if random_guess[0] == robot_pos[0] and random_guess[1] == robot_pos[1]:
            right_random_guesses += 1

        # Update our visual board
        print_board(robot_pos, estimated_position)

        # Move the robot
        robot_pos, direction = robot.move_robot()
        time.sleep(DELAY)

        nbr_of_iterations += 1

    res = right_guesses / nbr_of_iterations
    res_guess = right_random_guesses / nbr_of_iterations
    print(f"you guessed right: {res * 100}%")
    print(f"The average eucledian distance error was: {accumulated_distance_error/ITERATIONS}")

    print(f"Random guess has accuracy: {res_guess * 100}%")
    print(f"The average eucledian distance error was: {accumulated_random_distance_error/ITERATIONS}")
    print(f"Forward filtering is {res/max(res_guess, 0.01)} times better")