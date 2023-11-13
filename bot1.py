# Importing the necessary libraries
from collections import deque

import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('error')
import heapq
import math

# Creating the dataframe
df = pd.DataFrame(columns=['Bot', 'NumAliens', 'CrewSaved', 'TimeAlive', 'IsAlive'])

# Declaring the global variables
D = 10 # Dimension of the ship
k = 3 # Alien detection range
alpha = 0.075  # crew beep
ship = np.ones(())  # Layout of the ship
beliefNetworkAlien = np.zeros((D, D), dtype=np.longdouble)  # Belief of alien in each cell
beliefNetworkCrew = np.zeros((D, D), dtype=np.longdouble)  # Belief of crew in each cell
tempNetwork = np.zeros((D, D), dtype=np.longdouble)  # Temp
distances = [[[[]]]]
alien_cells = []  # Position of the aliens in the cells
bot_cell = (0, 0)  # Current position of the bot cell
crew_cell = (0, 0)  # Current position of the crew cell
noOfOpenCell = 0
noOfAlien = 1
h1 = 1
h2 = 1
isBeepAlien = 0
isBeepCrew = 0
isAlive = 1
isCrewSaved = 0
distances = [[[[]]]]


# Function to display all the basic details
def display():
    print(ship)
    print(noOfOpenCell)
    print("Bot at: ", bot_cell)
    print("Crew at: ", crew_cell)
    print("Alien cells: ", alien_cells)


# Checks if the passed cell co-ordinates lie in the range of the ship dimensions
def in_range(x, y):
    return (0 <= x < D) and (0 <= y < D)


# Generates the layout of the ship
def generate_ship_layout():
    global ship
    global noOfOpenCell
    # Initialize the grid with all cells as blocked (1 -> blocked, 0 -> open)
    ship = np.ones((D, D))

    # Actions to reach neighbors of cell as the adjacent cells in the up/down/right/left direction
    actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    # Choose a random cell in the interior to open
    x = random.randint(0, D - 1)
    y = random.randint(0, D - 1)
    ship[x][y] = 0
    noOfOpenCell += 1

    # Iteratively open cells with exactly one open neighbor
    while True:
        validBlockedCells = []
        for i in range(D):
            for j in range(D):
                if ship[i][j] == 1 and sum(
                        1 for dx, dy in actions if in_range(i + dx, j + dy) and ship[i + dx][j + dy] == 0) == 1:
                    validBlockedCells.append((i, j))

        if len(validBlockedCells) == 0:
            break

        cell = random.choice(validBlockedCells)
        ship[cell[0]][cell[1]] = 0
        noOfOpenCell += 1

    # Identify and open approximately half the dead-end cells
    dead_ends = [(i, j) for i in range(D) for j in range(D) if ship[i][j] == 0 and sum(
        1 for dx, dy in actions if in_range(i + dx, j + dy) and ship[i + dx][j + dy] == 0) == 1]
    random.shuffle(dead_ends)
    for i in range(len(dead_ends) // 2):
        x, y = dead_ends[i]
        neighbors = [(x + dx, y + dy) for dx, dy in actions if in_range(x + dx, y + dy) and ship[x + dx][y + dy] == 1]
        if neighbors:
            nx, ny = random.choice(neighbors)
            ship[nx][ny] = 0
            noOfOpenCell += 1


# Function to find a random empty cell on the grid
def find_empty_cell(key=0):
    global ship
    global bot_cell
    # print(bot_cell)
    # print(max(0, bot_cell[0] - k), min(D - 1, bot_cell[0] + k))
    while True:
        x = random.randint(0, len(ship) - 1)
        y = random.randint(0, len(ship[0]) - 1)
        if key == 0:
            if ship[x][y] in [0, 9]:
                return (x, y)

        elif key == 1:
            # print(x,y)
            if ship[x][y] in [0, 9]:
                if (x < bot_cell[0] - k or x > bot_cell[0] + k) and (y < bot_cell[1] - k or y > bot_cell[1] + k):
                    return (x, y)


# Function to set up the bot
def setup_bot():
    global ship
    global bot_cell
    # Find a random empty cell for the bot
    bot_cell = find_empty_cell()
    ship[bot_cell] = 3


# Function to set up the aliens
def setup_aliens(aliens):
    global ship
    global alien_cells
    # Find a random empty cell for each alien
    num_aliens = aliens
    alien_cells = []
    for x in range(num_aliens):
        temp = find_empty_cell(1)
        alien_cells.append(temp)
        ship[temp] = 9


# Function to set up the crew member
def setup_crew():
    global ship
    global crew_cell
    # print(bot_cell)
    # Find a random empty cell for the crew member
    crew_cell = find_empty_cell()
    # ship[crew_cell] = 6


# Function to generate ship, bot, crew and aliens in a single call
def generate_ship_bot_aliens(no_of_aliens):
    # Generate your ship layout
    generate_ship_layout()

    # Set up bot, aliens, and crew
    setup_bot()
    setup_crew()
    setup_aliens(no_of_aliens)

    # display()


# Function to update the position of the bot after it's movement
def update_bot_position(old_bot_cell, new_bot_cell):
    global ship
    ship[old_bot_cell] = 0
    ship[new_bot_cell] = 3


# Function to update the position of the alien after it's movement
def update_alien_position(old_alien_cell, new_alien_cell):
    global ship
    ship[old_alien_cell] = 0
    ship[new_alien_cell] = 9


# Function to get valid neighbor cells
def get_valid_neighbors(x, y):
    actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    neighbors = [(x + dx, y + dy) for dx, dy in actions if in_range(x + dx, y + dy) and ship[x + dx][y + dy] != 1]
    # Randomize the order of neighbors
    random.shuffle(neighbors)
    return neighbors


# Function to generate alien movements
def generate_alien_movements():
    global alien_cells
    movements = []

    # Shuffle the order of aliens
    random.shuffle(alien_cells)
    new_alien_cells = []
    for alien_cell in alien_cells:
        x, y = alien_cell
        valid_neighbors = get_valid_neighbors(x, y)

        # Choose a random valid neighbor cell or stay in place
        if valid_neighbors:
            new_x, new_y = random.choice(valid_neighbors)
            movements.append((x, y, new_x, new_y))
            update_alien_position(alien_cell, (new_x, new_y))
            new_alien_cells.append((new_x, new_y))
        else:
            # Stay in place
            movements.append((x, y, x, y))
            new_alien_cells.append((x, y))

    alien_cells.clear()
    alien_cells = new_alien_cells

    return movements


def detect_within_2k(cell, key=1):
    a, b = cell
    if bot_cell[0] - k <= a <= bot_cell[0] + k and bot_cell[1] - k <= b <= bot_cell[1] + k:
        return key
    return int(not key)


def detect_alien():
    global ship
    for alien_cell in alien_cells:
        if detect_within_2k(alien_cell) == 1:
            return 1

    return 0

def detect_crew():
    x, y = bot_cell
    i, j = crew_cell
    prob = np.exp(-alpha * (dist(x, y, i, j)-1))
    if random.random() < prob:
        return 1
    else:
        return 0


def dist(x, y, i, j):
    global distances
    return distances[x][y][i][j]

def init_belief_network_Alien():
    global ship
    global beliefNetworkAlien
    for i in range(D):
        for j in range(D):
            if ship[i][j] not in [1, 3]:
                beliefNetworkAlien[i][j] = 1 / (noOfOpenCell - 1)


def init_belief_network_crew():
    global ship
    global beliefNetworkCrew
    for i in range(D):
        for j in range(D):
            if ship[i][j] not in [1, 3]:
                beliefNetworkCrew[i][j] = 1 / (noOfOpenCell - 1)


def calc_sum_prob_alien():
    global beliefNetworkAlien
    score = np.longdouble(0)

    for i in range(D):
        for j in range(D):
            score = score + (beliefNetworkAlien[i][j] * detect_within_2k((i, j), isBeepAlien))
    if score==0:
        score = 10**(-308)
    return score


def update_belief_network_alien():
    global beliefNetworkAlien
    denom = np.longdouble(calc_sum_prob_alien())
    #print(denom)
    for i in range(D):
        for j in range(D):
            beliefNetworkAlien[i][j] = ((beliefNetworkAlien[i][j] * detect_within_2k((i, j), isBeepAlien)) / denom)
            #print()


def calc_sum_prob_crew():
    global beliefNetworkCrew

    score = 0.0
    for i in range(D):
        for j in range(D):
            prob = np.exp(-alpha * (dist(bot_cell[0], bot_cell[1], i, j) - 1))
            if isBeepCrew == 1:
                score = score + (beliefNetworkCrew[i][j] * prob)
            elif isBeepCrew == 0:
                score = score + (beliefNetworkCrew[i][j] * (1 - prob))
            else:
                print("How!!!")
            #score = score + (beliefNetworkCrew[i][j] * prob)
    return score


def update_belief_network_crew():
    global beliefNetworkCrew
    global isBeepCrew

    denom = calc_sum_prob_crew()
    for i in range(D):
        for j in range(D):
            prob = np.exp(-alpha * (dist(bot_cell[0], bot_cell[1], i, j) - 1))
            if isBeepCrew == 1:
                beliefNetworkCrew[i][j] = ((beliefNetworkCrew[i][j] * prob) / denom)
            elif isBeepCrew == 0:
                beliefNetworkCrew[i][j] = ((beliefNetworkCrew[i][j] * (1 - prob)) / denom)
            else:
                print("bhai kesa!!!")


def bot_movement():
    global bot_cell
    global isAlive
    global isCrewSaved
    x, y = bot_cell
    score = -1
    move = (0, 0)
    # print(score)
    valid_neighbors = get_valid_neighbors(x, y)
    # print(valid_neighbors)
    for valid_neighbor in valid_neighbors:
        temp = h1 * beliefNetworkCrew[valid_neighbor[0]][valid_neighbor[1]] + h2 * (1 - beliefNetworkAlien[valid_neighbor[0]][valid_neighbor[1]])
        if temp > score:
            move = (valid_neighbor[0], valid_neighbor[1])
            score = temp
        elif temp == score:
            if beliefNetworkCrew[valid_neighbor[0]][valid_neighbor[1]] > beliefNetworkCrew[move[0]][move[1]]:
                move = (valid_neighbor[0], valid_neighbor[1])

    #print(bot_cell)
    #print(move)
    update_bot_position(bot_cell, move)
    bot_cell = move
    # print(bot_cell)

    if bot_cell in alien_cells:
        isAlive = 0
        #print("dead")
    if bot_cell == crew_cell:
        isCrewSaved = 1
        #print("saved")
    update_after_bot_movement()



def update_after_alien_movement():
    global beliefNetworkAlien
    global tempNetwork
    tempNetwork = np.zeros((D, D))

    for i in range(D):
        for j in range(D):
            valid_neighbors = get_valid_neighbors(i, j)
            n = len(valid_neighbors)
            for valid_neighbor in valid_neighbors:
                x, y = valid_neighbor
                tempNetwork[x][y] += beliefNetworkAlien[i][j] / n

    beliefNetworkAlien = tempNetwork


def update_after_bot_movement():
    global beliefNetworkAlien
    global beliefNetworkCrew

    denom = 1 - beliefNetworkAlien[bot_cell]
    beliefNetworkAlien[bot_cell] = 0
    #print(denom)
    for i in range(D):
        for j in range(D):
            beliefNetworkAlien[i][j] = ((beliefNetworkAlien[i][j] * detect_within_2k((i, j), isBeepAlien)) / denom)

    denom = 1 - beliefNetworkCrew[bot_cell]
    beliefNetworkCrew[bot_cell] = 0
    # print(denom)
    for i in range(D):
        for j in range(D):
            prob = np.exp(-alpha * (dist(bot_cell[0], bot_cell[1], i, j) - 1))
            if isBeepCrew == 1:
                beliefNetworkCrew[i][j] = ((beliefNetworkCrew[i][j] * prob) / denom)
            elif isBeepCrew == 0:
                beliefNetworkCrew[i][j] = ((beliefNetworkCrew[i][j] * (1 - prob)) / denom)


def bfs(grid, start):
    """Compute the shortest distances from the start cell to all other cells using BFS."""
    rows, cols = len(grid), len(grid[0])
    distances = [[-1] * cols for _ in range(rows)]
    distances[start[0]][start[1]] = 0
    queue = deque([start])

    while queue:
        i, j = queue.popleft()
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < rows and 0 <= new_j < cols and distances[new_i][new_j] == -1:
                distances[new_i][new_j] = distances[i][j] + 1
                queue.append((new_i, new_j))

    return distances


def compute_all_pairs_shortest_distances(grid):
    """Compute the shortest distances between all pairs of cells."""
    rows, cols = len(grid), len(grid[0])
    all_pairs_distances = [[[] for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            all_pairs_distances[i][j] = bfs(grid, (i, j))

    return all_pairs_distances


def bot1():
    global ship
    global beliefNetworkAlien
    global beliefNetworkCrew
    global tempNetwork
    global bot_cell
    global alien_cells
    global crew_cell
    global isBeepAlien
    global isBeepCrew
    global distances
    global isAlive
    global isCrewSaved
    global noOfOpenCell

    t = 0
    isAlive = 1
    isCrewSaved = 0
    noOfOpenCell = 0

    generate_ship_bot_aliens(noOfAlien)
    #display()

    init_belief_network_crew()
    init_belief_network_Alien()
    ship2 = ship.tolist()
    distances = compute_all_pairs_shortest_distances(ship2)

    #print(beliefNetworkCrew)
    #print(beliefNetworkAlien)
    isBeepAlien = detect_alien()
    isBeepCrew = detect_crew()
    #print(isBeepAlien)
    #print(isBeepCrew)
    update_belief_network_alien()
    update_belief_network_crew()

    while True:
        #print("time", t)
        #print("*************************************************************************************")
        #print("time", t)
        # print(ship)
        # print(beliefNetworkCrew)
        # print(beliefNetworkAlien)

        t += 1
        bot_movement()
        #print("before Bot, alien, crew", bot_cell, alien_cells, crew_cell)
        if isAlive != 1 or isCrewSaved != 0:
            return (t, isAlive, isCrewSaved)
        isBeepAlien = detect_alien()
        isBeepCrew = detect_crew()
        update_belief_network_alien()
        update_belief_network_crew()

        generate_alien_movements()
        #print("after Bot, alien, crew", bot_cell, alien_cells, crew_cell)
        if bot_cell in alien_cells:
            isAlive = 0
        if isAlive != 1 or isCrewSaved != 0:
            return (t, isAlive, isCrewSaved)
        isBeepAlien = detect_alien()
        update_after_alien_movement()
        update_belief_network_alien()

    return "maa chudi"




# print(distances)

# print([bot_position[0]])

saved = 0
for i in range(30):
    #print(i)
    t, a, s = bot1()
    saved += s
    #print(t, a, s)
    #print("#################################################################################################")

print("saved ", saved)