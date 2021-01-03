import os
import numpy as np
import operator
import matplotlib.pyplot as plt
from itertools import product


class actions:
    north = 0
    east = 1
    south = 2
    west = 3

    @classmethod
    def all(cls):
        return {cls.north, cls.east, cls.south, cls.west}


def convert(list):
    """ Convert [i, j] to (i, j)

    Args:
        list

    Returns:
        tuple
    """
    return (*list,)


def find_best_move_simple(current_pos, goal, M):
    """ Do simple shortest path heuristic using  the start cell and the transition matrix. The function looks at the directions which should be moved
    from the current position to get to the goal position, and chooses the direction with the highest transition probability. 
    Afterwards, calculate the expected travel time for the optimal path

    Args:
        start ([position]): starting position
        goal ([position]): goal position
        M : transition matrix

    Returns:
        travel time
    """
    trvl_time = 0

    while current_pos != goal:
        # potential moves
        moves_x = goal[0] - current_pos[0]
        moves_y = goal[1] - current_pos[1]

        # This will be the vector which contains the values of the possible moves we can do. Order: N, E, S, W
        possible_moves = np.zeros(len(actions.all()))

        # Determining if we move North, South or no vertical movement
        if moves_x > 0:
            possible_moves[actions.north] = M[
                current_pos[0], current_pos[1], [actions.north]
            ]
            possible_moves[actions.south] = 0
        elif moves_x < 0:
            possible_moves[actions.north] = 0
            possible_moves[actions.south] = M[
                current_pos[0], current_pos[1], [actions.south]
            ]
        else:
            possible_moves[actions.north] = 0
            possible_moves[actions.south] = 0

        # Determining if we move East, West or no horizontal movement
        if moves_y > 0:
            possible_moves[actions.east] = M[
                current_pos[0], current_pos[1], [actions.east]
            ]
            possible_moves[actions.west] = 0
        elif moves_y < 0:
            possible_moves[actions.east] = 0
            possible_moves[actions.west] = M[
                current_pos[0], current_pos[1], [actions.west]
            ]
        else:
            possible_moves[actions.east] = 0
            possible_moves[actions.west] = 0

        # From this vector, we can choose the optimal move by selecting the direction with the highest probability of no congestion - the max value.
        if np.argmax(possible_moves) == actions.north:
            current_pos[0] += 1
        elif np.argmax(possible_moves) == actions.east:
            current_pos[1] += 1
        elif np.argmax(possible_moves) == actions.south:
            current_pos[0] -= 1
        else:
            current_pos[1] -= 1

        # updating the travel time
        trvl_time += prob_to_exp_trvl_time(max(possible_moves))
    return trvl_time


def dijkstra(start, goal, M):
    """ Do Dijkstra's shortest path algorithm given the start cell and the transition matrix. 
    Afterwards, calculate the expected travel time for the optimal path

    Args:
        start ([position]): starting position
        goal ([position]): goal position
        M : transition matrix

    Returns:
        travel time
    """
    visited = {}
    visited[start] = 0
    unvisited = {}

    while len(visited) < 2500:

        # update the distances of the neigbours from the visited points
        for node in visited:
            potential_neigbours = neighbours(node, M)
            for potential_neigbour in potential_neigbours:
                # check if there is a shorter way to a neigbour already in the unvisited set
                if convert(potential_neigbour) in unvisited:
                    # print("unvisited in loop", unvisited)
                    # print("new route? ", potential_neigbour, M[node[0], node[1], get_direction(node, potential_neigbour)], prob_to_exp_trvl_time(M[node[0], node[1], get_direction(node, potential_neigbour)]) + visited[node], unvisited[convert(potential_neigbour)])
                    unvisited[convert(potential_neigbour)] = min(
                        prob_to_exp_trvl_time(
                            M[node[0], node[1], get_direction(node, potential_neigbour)]
                        )
                        + visited[node],
                        unvisited[convert(potential_neigbour)],
                    )
                    # print("added: ", potential_neigbour, unvisited[convert(potential_neigbour)])
                    break

                # if neigbour not yet in the visited set, add to unvisited
                if convert(potential_neigbour) not in visited:
                    unvisited[convert(potential_neigbour)] = visited[
                        node
                    ] + prob_to_exp_trvl_time(
                        M[node[0], node[1], get_direction(node, potential_neigbour)]
                    )
                    # print("added: ", potential_neigbour, visited[node] + prob_to_exp_trvl_time(M[node[0], node[1], get_direction(node, potential_neigbour)]))

        # print("unvisited", unvisited)

        # from the unvisited set, choose the node with the shortest distance
        visited[min(unvisited.items(), key=operator.itemgetter(1))[0]] = min(
            unvisited.items(), key=operator.itemgetter(1)
        )[1]
        del unvisited[min(unvisited.items(), key=operator.itemgetter(1))[0]]

    return visited[goal]


def value_iteration(start, goal, M, plot=False):
    """ Do value iteration given the start cell and the transition matrix. Afterwards, calculate the expected travel time for the optimal path

    Args:
        start ([position]): starting position
        goal ([position]): goal position
        M : transition matrix

    Returns:
        travel time
    """
    size_V = len(M)
    V = np.zeros((size_V, size_V))
    V_tmp = np.zeros((size_V, size_V))

    # how many steps for value iteration - this might be enough when starting in (0, 0) and going to (1, 10). When the whole matrix is used, we might
    # have to increase the steps
    for _ in range(100):
        for i in range(len(V)):
            for j in range(len(V)):
                V_sum = []
                neigbours_list = neighbours([i, j], V)

                for neigbour in neigbours_list:
                    V_sum.append(
                        r[i, j]
                        + gamma
                        * M[i, j, get_direction([i, j], neigbour)]
                        * V_tmp[neigbour[0], neigbour[1]]
                    )
                V_tmp[i][j] = max(V_sum)
        V = V_tmp.copy()

    if plot:
        plt.imshow(V)
        plt.title("Heatmap of V matrix")
        os.makedirs("img", exist_ok=True)
        plt.savefig("img/V_heatmap")
        plt.show()

    # Value iteration done, now choose the optimal path through the matrix
    travel_time = 0
    current_pos = start
    for i in range(100):
        moves = []

        # find the neigbour with the highest V value and go to this place
        neigbours_list = neighbours([current_pos[0], current_pos[1]], V)
        for neigbour in neigbours_list:
            moves.append(V[neigbour[0], neigbour[1]])

        # update the travel time
        travel_time += prob_to_exp_trvl_time(
            M[
                current_pos[0],
                current_pos[1],
                get_direction(
                    [current_pos[0], current_pos[1]], neigbours_list[np.argmax(moves)]
                ),
            ]
        )
        current_pos = neigbours_list[np.argmax(moves)]

        # goal has been reached
        if convert(current_pos) == goal:
            break

    return travel_time


def prob_to_exp_trvl_time(prob):
    """ Convert probability to expected travel time. Always +1 for the step that needs to be done

    Args:
        prob ([float]): probility of transition

    Returns:
        travel time [int]: expected travel time
    """
    return ((1 - prob) * 10) + 1


def get_direction(current, neigbour):
    """ This function gives the direction between the current cell and the neigbour. Direction is indicated as integer 0-3. 0 = North, 1 = East, etc

    Args:
        current ([type]): current cell
        neigbour ([type]): neigbour cell

    Returns:
        direction [int]:
    """
    if current[1] - neigbour[1] == 1:
        return actions.north
    # southern neigbour
    elif current[0] - neigbour[0] == -1:
        return actions.east
    # eastern neigbour
    elif current[1] - neigbour[1] == -1:
        return actions.south
    # western neigbour
    return actions.west


def neighbours(cell, V):
    """ This function provides all the possible neigbours from the current cell, respecting the boundries of the matrix

    Args:
        cell ([type]): current position
        V ([type]): Matrix for dimensions

    Returns:
        neigbours: list with all possible neigbours
    """
    neigbours = []
    if cell[0] != 0:
        neigbours.append([cell[0] - 1, cell[1]])
    if cell[0] != len(V) - 1:
        neigbours.append([cell[0] + 1, cell[1]])
    if cell[1] != 0:
        neigbours.append([cell[0], cell[1] - 1])
    if cell[1] != len(V) - 1:
        neigbours.append([cell[0], cell[1] + 1])
    return neigbours


if __name__ == "__main__":

    goal = [1, 10]
    current_pos = [0, 0]

    # Transition matrix with 3 dimensions. x,y is grid, z is transitions. dim0 = north, dim1 = east, dim2 = south, dim3 = west
    M = np.zeros((50, 50, 4))
    r = np.zeros((50, 50))
    gamma = 0.99

    r[1][10] = 1

    # Generate transition matrix with congestion probabilities
    for i in range(len(M)):
        for j in range(len(M)):
            for k in range(M.shape[2]):
                M[i][j][k] = np.random.choice(np.arange(0.1, 1.1, 0.1))
    np.save("M.npy", M)

    # Simple heuristic - take the shortest path to (1,10)
    current_pos = [0, 0]
    exp_trvl_time_heuristic = find_best_move_simple(current_pos, goal, M)
    print(
        "Expected travel time for simple heuristic is: \t%d" % (exp_trvl_time_heuristic)
    )

    # Set of equations, aka Dijkstra
    current_pos = (0, 0)
    exp_trvl_time_dijkstra = dijkstra(current_pos, convert(goal), M)
    print("Expected travel time in Dijkstra is: \t\t%d" % (exp_trvl_time_dijkstra))

    # Dynamic Programming, aka value iteration
    current_pos = (0, 0)
    exp_trvl_time_value_iteration = value_iteration(current_pos, convert(goal), M, True)
    print(
        "Expected travel time in Value Iteration is: \t%d"
        % (exp_trvl_time_value_iteration)
    )

