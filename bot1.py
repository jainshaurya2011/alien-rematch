# Importing the necessary libraries
from collections import deque

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('error')
import heapq

# Creating the dataframe
df = pd.DataFrame(columns=['Bot', 'NumAliens', 'CrewSaved', 'TimeAlive', 'IsAlive'])

# Declaring the global variables
D = 30  # Dimension of the ship
k = 3  # Alien detection range
alpha = 0.055  # crew beep
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
    while True:
        x = random.randint(0, len(ship) - 1)
        y = random.randint(0, len(ship[0]) - 1)
        if key == 0:
            if ship[x][y] in [0, 9]:
                return (x, y)

        elif key == 1:
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
def setup_aliens():
    global ship
    global alien_cells

    alien_cells = []
    for x in range(noOfAlien):
        temp = find_empty_cell(1)
        alien_cells.append(temp)
        ship[temp] = 9


# Function to set up the crew member
def setup_crew():
    global ship
    global crew_cell
    # Find a random empty cell for the crew member
    crew_cell = find_empty_cell()


# Function to generate ship, bot, crew and aliens in a single call
def generate_ship_bot_aliens():
    # Generate your ship layout
    generate_ship_layout()

    # Set up bot, aliens, and crew
    setup_bot()
    setup_crew()
    setup_aliens()


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
    prob = np.exp(-alpha * (dist(x, y, i, j) - 1))
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
    return score


def update_belief_network_alien():
    global beliefNetworkAlien
    denominator = np.longdouble(calc_sum_prob_alien())
    for i in range(D):
        for j in range(D):
            beliefNetworkAlien[i][j] = (
                        (beliefNetworkAlien[i][j] * detect_within_2k((i, j), isBeepAlien)) / denominator)


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
    valid_neighbors = get_valid_neighbors(x, y)
    for valid_neighbor in valid_neighbors:
        temp = h1 * beliefNetworkCrew[valid_neighbor[0]][valid_neighbor[1]] + h2 * (
                1 - beliefNetworkAlien[valid_neighbor[0]][valid_neighbor[1]])
        if temp > score:
            move = (valid_neighbor[0], valid_neighbor[1])
            score = temp
        elif temp == score:
            if beliefNetworkCrew[valid_neighbor[0]][valid_neighbor[1]] > beliefNetworkCrew[move[0]][move[1]]:
                move = (valid_neighbor[0], valid_neighbor[1])

    update_bot_position(bot_cell, move)
    bot_cell = move

    if bot_cell in alien_cells:
        isAlive = 0
    if bot_cell == crew_cell:
        isCrewSaved = 1
    update_after_bot_movement()


def update_after_alien_movement():
    global beliefNetworkAlien
    global tempNetwork
    tempNetwork = np.zeros((D, D))

    for i in range(D):
        for j in range(D):
            valid_neighbors = get_valid_neighbors(i, j)
            n = len(valid_neighbors) + 1
            tempNetwork[i][j] += beliefNetworkAlien[i][j] / n
            for valid_neighbor in valid_neighbors:
                x, y = valid_neighbor
                tempNetwork[x][y] += beliefNetworkAlien[i][j] / n

    beliefNetworkAlien = tempNetwork


def update_after_bot_movement():
    global beliefNetworkAlien
    global beliefNetworkCrew

    if isCrewSaved == 1:
        return 0

    denominator = 1 - beliefNetworkAlien[bot_cell]
    beliefNetworkAlien[bot_cell] = 0
    for i in range(D):
        for j in range(D):
            beliefNetworkAlien[i][j] = ((beliefNetworkAlien[i][j]) / denominator)

    denominator = 1 - beliefNetworkCrew[bot_cell]
    beliefNetworkCrew[bot_cell] = 0
    for i in range(D):
        for j in range(D):
            beliefNetworkCrew[i][j] = ((beliefNetworkCrew[i][j]) / denominator)


def a_star(start_cell):
    global ship

    tempMatrix = np.zeros((D, D))

    def is_valid(x, y):
        return 0 <= x < len(ship) and 0 <= y < len(ship[0]) and ship[x][y] in [0, 9]

    came_from = {}
    g_score = {(i, j): float('inf') for i in range(D) for j in range(D)}
    g_score[start_cell] = 0
    f_score = {(i, j): float('inf') for i in range(D) for j in range(D)}
    f_score[start_cell] = 0

    open_set = [(0, 0, start_cell)]

    while open_set:
        _, _, current = heapq.heappop(open_set)

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if is_valid(*neighbor):
                tentative_g_score = g_score[current] + 1
                tentative_f_score = tentative_g_score + 0
                if tentative_f_score < f_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_f_score
                    heapq.heappush(open_set, (f_score[neighbor], 0, neighbor))
                    tempMatrix[neighbor] = f_score[neighbor]
    return tempMatrix


def compute_all_pairs_shortest_distances(grid):
    """Compute the shortest distances between all pairs of cells."""
    rows, cols = len(grid), len(grid[0])
    all_pairs_distances = [[[] for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            all_pairs_distances[i][j] = a_star((i, j)).tolist()

    return all_pairs_distances


def heatmap(data, titt):
    ax = sns.heatmap(data, fmt="d", linewidths=1, linecolor='white')
    ax.plot([bot_cell[1] + 0.5], [bot_cell[0] + 0.5], marker='o', markersize=10, markeredgewidth=2, markeredgecolor='b',
            markerfacecolor='b')
    for alien_cell in alien_cells:
        ax.plot([alien_cell[1] + 0.5], [alien_cell[0] + 0.5], marker='o', markersize=10, markeredgewidth=2,
                markeredgecolor='r', markerfacecolor='r')
    ax.plot([crew_cell[1] + 0.5], [crew_cell[0] + 0.5], marker='o', markersize=10, markeredgewidth=2,
            markeredgecolor='g', markerfacecolor='g')
    plt.title(titt)
    plt.show()


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
    ship = np.ones(())  # Layout of the ship
    beliefNetworkAlien = np.zeros((D, D), dtype=np.longdouble)  # Belief of alien in each cell
    beliefNetworkCrew = np.zeros((D, D), dtype=np.longdouble)  # Belief of crew in each cell
    tempNetwork = np.zeros((D, D), dtype=np.longdouble)  # Temp

    generate_ship_bot_aliens()
    init_belief_network_crew()
    init_belief_network_Alien()
    ship2 = ship.tolist()
    distances = compute_all_pairs_shortest_distances(ship2)

    isBeepAlien = detect_alien()
    isBeepCrew = detect_crew()
    update_belief_network_alien()
    update_belief_network_crew()

    while True:
        t += 1
        bot_movement()
        if isAlive != 1 or isCrewSaved != 0:
            return (t, isAlive, isCrewSaved)
        isBeepAlien = detect_alien()
        isBeepCrew = detect_crew()
        update_belief_network_alien()
        update_belief_network_crew()

        generate_alien_movements()
        if bot_cell in alien_cells:
            isAlive = 0
            return (t, isAlive, isCrewSaved)
        isBeepAlien = detect_alien()
        update_after_alien_movement()
        update_belief_network_alien()

    return "maa chudi"


asdf = [2, 4, 6, 8, 10, 12]
saved = 0
timeAlive = 0
for q in asdf:
    k = q
    saved = 0
    timeAlive = 0
    print(k)
    for i in range(50):
        t, a, s = bot1()
        saved += s
        timeAlive += t

    print("saved: ", saved)
    print("avg time: ", timeAlive / 50)
    print("***************************************************************************************")
