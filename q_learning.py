import os
from pickle import TRUE
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numba as nb

goal = (1, 10)
current_pos = (0, 0)

n_states_x = 50
n_states_y = 50
n_actions = 4
goal_reward = 1
max_steps_per_epidode = 1000
move_reward = 0
# move_reward = -(goal_reward / max_steps_per_epidode)
n_episodes = 10000

lr = 0.1
discount_rate = 0.99

max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 1.0 / n_episodes
USE_DIST = True

reward_fun = np.ones((n_states_x, n_states_y), dtype=np.float64) * move_reward

if USE_DIST:
    xg, yg = goal
    for x in range(50):
        for y in range(50):
            dist = np.sqrt((x - xg) * (x - xg) + (y - yg) * (y - yg))
            reward_fun[x, y] = (
                -0.00816 * dist + -0.091
            )  # scale negative rewards from ~(-0.1:-0.5)
reward_fun[goal] = goal_reward

# Generate transition matrix with congestion probabilities
if os.path.exists("M.npy"):
    M = np.load("M.npy")
else:
    M = np.zeros((n_states_x, n_states_y, n_actions))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                M[i][j][k] = np.random.choice(np.arange(0.1, 1.1, 0.1))
    np.save("M.npy", M)

# Transition matrix with 3 dimensions. x,y is grid, z is transitions. dim0 = north, dim1 = east, dim2 = south, dim3 = west
@nb.njit(cache=True)
def make_step_congest(x: int, y: int, action: int):
    """ 
    Make a step. 
    
    * Action has to be valid (cannot go outside of the borad and has to be number from {0,1,2,3}). 
    * Takes congestions into account.
    """
    moves = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

    congestion_probability = M[x, y, action]
    if np.random.uniform(0, 1) >= congestion_probability:
        # Stay in current state due to congestion
        return x, y
    move_x, move_y = moves[action]
    nx = move_x + x
    ny = move_y + y

    return nx, ny


@nb.njit(cache=True)
def make_step(x: int, y: int, action: int):
    """ 
    Make a step from (x,y) position. Action has to be valid i.e. cannot go outside of the borad and has to be number from {0,1,2,3}. 
    """
    moves = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

    move_x, move_y = moves[action]
    nx = move_x + x
    ny = move_y + y

    return nx, ny


@nb.njit(cache=True)
def choose_action(state_q, exploration_rate):
    """ Choose action using exploration/exploitation. """
    valid = np.where(state_q > np.NINF)[0]

    if np.sum(state_q[valid]) < 1e-5:
        exploit = False
    else:
        exploration_rate_threshold = np.random.uniform(0, 1)
        exploit = exploration_rate_threshold > exploration_rate
    if exploit:
        action = np.argmax(state_q)
    else:
        action = np.random.choice(valid)
    return action


@nb.njit(cache=True)
def get_explotation_rate(min_eps, max_eps, decay, episode):
    """ Decay exploration rate logarithmically """
    return min_eps + (max_eps - min_eps) * np.exp(-decay * episode)


@nb.njit(cache=True)
def do_episode(ini_x, ini_y, exploration_rate, q_table):
    """ Perform episode of Q-Learning """
    x, y = ini_x, ini_y
    # print(f"Exploration rate: {exploration_rate}")
    rewards_cur_episode = 0  # we start with no reweards

    steps = 0
    for _ in range(max_steps_per_epidode):
        state_q = q_table[x, y, :]
        action = choose_action(state_q, exploration_rate)

        nx, ny = make_step_congest(x, y, action)
        new_state = np.array([nx, ny])
        reward = reward_fun[x, y] if nx == x and ny == y else reward_fun[nx, ny]

        q_table[x, y, action] = state_q[action] * (1 - lr) + lr * (
            reward + discount_rate * np.max(q_table[nx, ny, :])
        )

        rewards_cur_episode += reward
        steps += 1

        done = np.array_equal(new_state, goal)
        if done:
            break

        x, y = nx, ny

    return steps, rewards_cur_episode


@nb.njit(cache=True)
def count_expected_time(q_table, xs, ys, xg, yg, debug=False):
    travel_time = 0
    current_pos = (xs, ys)
    g = (xg, yg)
    steps = 0
    d = {0: "N", 1: "E", 2: "S", 3: "W"}
    for _ in range(1000):
        x, y = current_pos
        state_q = q_table[x, y, :]
        action = np.argmax(state_q)
        next_state = make_step(x, y, action)
        if debug:
            print(current_pos, "Q =", state_q, "A =", d[action], "->", next_state)
        # update the travel time
        prob = M[current_pos[0], current_pos[1], action]
        travel_time += ((1 - prob) * 10) + 1
        current_pos = next_state
        steps += 1
        # goal has been reached
        if current_pos == g:
            break

    return travel_time, steps


@nb.njit(cache=True)
def q_learning():
    q_table = np.zeros((n_states_x, n_states_y, n_actions))
    q_table[n_states_x - 1, :, 1] = np.NINF
    q_table[0, :, 3] = np.NINF
    q_table[:, n_states_y - 1, 2] = np.NINF
    q_table[:, 0, 0] = np.NINF

    exploration_rate = max_exploration_rate
    rewards_all_episodes = np.zeros(n_episodes, dtype=np.float64)
    travel_times = np.zeros(n_episodes, dtype=np.int64)

    avg_R = 0
    episode = 0
    # while avg_R < 100:
    for episode in range(n_episodes):
        x, y = np.random.randint(0, n_states_x - 1, size=2)
        # x, y = (0, 0)
        steps, rewards_cur_episode = do_episode(x, y, exploration_rate, q_table)
        travel_times[episode] = steps
        rewards_all_episodes[episode] = rewards_cur_episode

        exploration_rate = get_explotation_rate(
            min_exploration_rate, max_exploration_rate, exploration_decay_rate, episode
        )
        if episode <= 100:
            avg_R = 0  # don't take average into account yet
            avg_S = 0
        else:
            avg_R = np.sum(rewards_all_episodes[episode - 100 : episode]) / 100
            avg_S = np.sum(travel_times[episode - 100 : episode]) / 100
        exp, stp = count_expected_time(q_table, 0, 0, 1, 10)

        print(
            "Episode:",
            episode,
            "| eps =",
            exploration_rate,
            "| R =",
            round(rewards_cur_episode, 2),
            "| exp =",
            exp,
            "| exp_S =",
            stp,
            "| S =",
            steps,
            "| avg_R =",
            round(avg_R, 4),
            "| avg_S =",
            round(avg_S, 4),
        )

    return rewards_all_episodes, travel_times, q_table


def rolling_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def save_plots(rewards_all_episodes, travel_times, q_table):
    suf = "_dist" if USE_DIST == True else ""

    n, e, s, w = [q_table[:, :, i] for i in (0, 1, 2, 3)]
    for action, name in zip((n, e, s, w), "NESW"):
        plt.figure()
        sns.heatmap(action, square=True)
        plt.title(f"Q-table {name}")
        plt.xlabel("y")
        plt.ylabel("x")
        os.makedirs("img", exist_ok=True)
        plt.savefig(f"img/q_table-{name}" + suf)

    q_table_cum = np.sum(q_table, axis=2)
    plt.figure()
    sns.heatmap(q_table_cum, square=True)
    plt.xlabel("y")
    plt.ylabel("x")

    plt.title(f"Q-table cum")
    os.makedirs("img", exist_ok=True)
    plt.savefig(f"img/q_table-cum" + suf)

    spl = n_episodes / 10
    reward_per_thousand_episodes = np.split(
        np.array(rewards_all_episodes), n_episodes / spl
    )
    times_per_thousand_episodes = np.split(np.array(travel_times), n_episodes / spl)
    cnt = spl
    print("Iters :       avg reward     | avg steps to goal")
    for r, tr in zip(reward_per_thousand_episodes, times_per_thousand_episodes):
        print(cnt, ": ", str(sum(r / spl)), "|", str(sum(tr / spl)))
        cnt += spl

    plt.figure()
    plt.plot(rolling_average(travel_times, 50))
    plt.title("Travel time rolling average from 50 episodes")
    plt.xlabel("Episode")
    plt.ylabel("50-avg travel time")

    os.makedirs("img", exist_ok=True)
    plt.savefig("img/travel_time" + suf)

    plt.figure()
    plt.plot(rolling_average(rewards_all_episodes, 50))
    plt.title("Reward rolling average from 50 episodes")
    plt.xlabel("Episode")
    plt.ylabel("50-avg reward")

    os.makedirs("img", exist_ok=True)
    plt.savefig("img/reward" + suf)
    n, e, s, w = [M[:, :, i] for i in (0, 1, 2, 3)]
    for action, name in zip((n, e, s, w), "NESW"):
        plt.figure()
        sns.heatmap(action, square=True)
        plt.xlabel("y")
        plt.ylabel("x")
        plt.title(f"Congestion prob {name}")
        os.makedirs("img", exist_ok=True)
        plt.savefig(f"img/M-{name}" + suf)

    eps = [
        get_explotation_rate(
            min_exploration_rate, max_exploration_rate, exploration_decay_rate, i
        )
        for i in range(n_episodes)
    ]
    plt.figure()
    plt.title("Exploration rate")
    plt.plot(eps)
    plt.xlabel("Episode")
    plt.savefig("img/eps" + suf)

    plt.figure()
    sns.heatmap(reward_fun, square=True)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title("Reward")
    plt.savefig("img/reward_fun" + suf)


if __name__ == "__main__":
    # ts = time.time()
    rewards_all_episodes, travel_times, q_table = q_learning()
    # print(f"Q-learning finished in: {time.time() - ts}")
    exp_trvl_time_q_lrn, stp = count_expected_time(
        q_table, 0, 0, goal[0], goal[1], debug=True
    )
    print(
        "Expected travel time for Q-learning is: \t%d in %d steps"
        % (exp_trvl_time_q_lrn, stp)
    )

    np.save("q-table.npy", q_table)

    save_plots(rewards_all_episodes, travel_times, q_table)

