import numpy as np
from assignment4 import (
    neighbours,
    get_direction,
    convert,
    prob_to_exp_trvl_time,
    actions,
)
from q_learning import get_valid_actions, make_step

q_table = np.load("q-table-0.0.npy")
M = np.load("M.npy")


start = (0, 0)
goal = (1, 10)


def translate(action):
    if action == actions.north:
        return "N"
    if action == actions.south:
        return "S"
    if action == actions.west:
        return "W"
    if action == actions.east:
        return "E"


travel_time = 0
current_pos = start
for i in range(1000):
    moves = get_valid_actions(current_pos, q_table.shape[0], q_table.shape[1])
    q_vals = []
    for i in (0, 1, 2, 3):
        if i not in moves:
            q_vals.append(-10000)
        else:
            q_vals.append(q_table[current_pos[0], current_pos[1], i])
    print(current_pos)
    q_vals = np.array(q_vals)
    action = np.argmax(q_vals)
    next_state, _, _ = make_step(current_pos, action, False)

    print(f"{current_pos}, {q_vals}, A={translate(action)} -> {next_state}")
    # update the travel time
    travel_time += prob_to_exp_trvl_time(M[current_pos[0], current_pos[1], action])
    current_pos = next_state

    # goal has been reached
    if current_pos == goal:
        break

print(travel_time)
