# %%
import numpy as np
import random
from tqdm import tqdm
from assignment4 import actions, neighbours
import matplotlib.pyplot as plt
import os
import seaborn as sns

goal = (1, 10)
current_pos = (0, 0)

n_states_x = 50
n_states_y = 50
n_actions = 4
goal_reward = 1
congestion_reward = 0
max_steps_per_epidode = 1000

move_reward = -(goal_reward / max_steps_per_epidode)
move_reward = 0

# Transition matrix with 3 dimensions. x,y is grid, z is transitions. dim0 = north, dim1 = east, dim2 = south, dim3 = west
M = np.zeros((n_states_x, n_states_y, n_actions))
r = np.ones((n_states_x, n_states_y)) * move_reward
gamma = 0.99

r[goal] = goal_reward

# Generate transition matrix with congestion probabilities

if os.path.exists("M.npy"):
    print("Loading M.npy")
    M = np.load("M.npy")
else:
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                M[i][j][k] = np.random.choice(np.arange(0.1, 1.1, 0.1))
    np.save("M.npy", M)
# %%
q_table = np.zeros((n_states_x, n_states_y, n_actions))
n_episodes = 10000

lr = 0.1
discount_rate = 0.99

max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 1.0 / n_episodes
exploration_rate = max_exploration_rate
# %%
rewards_all_episodes = list()
# %%


def make_step(state, action, use_congestions=False):
    """ 
    Make a step. 
    
    * Action has to be valid (cannot go outside of the borad and has to be number from {0,1,2,3}). 
    * Takes congestions into account.
    """
    x, y = state
    nx, ny = x, y
    if action == actions.north:
        ny -= 1
    elif action == actions.south:
        ny += 1
    elif action == actions.west:
        nx -= 1
    elif action == actions.east:
        nx += 1
    else:
        AssertionError(f"action=={action} is not valid!")

    # Check congestion
    congestion_probability = M[x, y, action]
    if use_congestions and np.random.uniform(0, 1) >= congestion_probability:
        # Stay in current state due to congestion
        reward = congestion_reward
        new_state = (x, y)
        # print(f"({x}, {y}) . C R={reward}")
    else:
        # Progress to the new state
        new_state = (nx, ny)
        reward = r[nx, ny]
        # print(f"({x}, {y}) -> ({nx}, {ny}) R={reward}")

    done = True if new_state == goal else False

    return new_state, reward, done


def get_valid_actions(state, max_x, max_y):
    possible_actions = set(actions.all())
    x, y = state
    if x >= max_x - 1:
        possible_actions.remove(actions.east)
    if x <= 0:
        possible_actions.remove(actions.west)
    if y >= max_y - 1:
        possible_actions.remove(actions.south)
    if y <= 0:
        possible_actions.remove(actions.north)
    return possible_actions


if __name__ == "__main__":
    # %%
    travel_times = []
    explores = []
    import time

    ts = time.time()
    all_actions = actions.all()
    for episode in range(n_episodes):
        # state = (0, 0)
        state = (
            np.random.randint(0, n_states_x - 1),
            np.random.randint(0, n_states_y - 1),
        )
        # print(f"Exploration rate: {exploration_rate}")
        done = False
        rewards_cur_episode = 0  # we start with no reweards

        steps = 0
        n_explore = 0
        for step in range(max_steps_per_epidode):
            # epsilon - greedy decision
            exploration_rate_threshold = random.uniform(0, 1)
            exploit = exploration_rate_threshold > exploration_rate
            possible_actions = get_valid_actions(state, n_states_x, n_states_y)

            state_q = q_table[state[0], state[1], :]
            if np.sum(state_q) < 1e-5:
                exploit = False
                # print("Empty q, going explore")

            if exploit:
                q_vals = np.zeros(len(all_actions))
                for i in actions.all():
                    if i not in possible_actions:
                        q_vals[i] = -10000
                    else:
                        q_vals[i] = state_q[i]
                action = np.argmax(q_vals)
            else:
                action = random.choice(list(possible_actions))
                n_explore += 1
            # print(f"{state} | A={action} | Explore={exploit}")

            new_state, reward, done = make_step(state, action)
            q_value = state_q[action] * (1 - lr) + lr * (
                reward + discount_rate * np.max(q_table[new_state[0], new_state[1], :])
            )

            q_table[state[0], state[1], action] = q_value

            rewards_cur_episode += reward
            # print(f"E={episode} S={step} | {state} -(r{reward},q{q_value})-> {new_state} ")
            # stopping condition
            steps = step

            if done:
                # print(f"DONE! State: {new_state}, old-state: {state}, reward: {reward}")
                break

            state = new_state
        # print("explore rate:", n_explore, "/", max_steps_per_epidode)
        explores.append(n_explore)
        travel_times.append(steps)
        exploration_rate = min_exploration_rate + (
            max_exploration_rate - min_exploration_rate
        ) * np.exp(-exploration_decay_rate * episode)

        # print(f"Episode: {episode}, reward: {rewards_cur_episode}")
        rewards_all_episodes.append(rewards_cur_episode)
        print(
            f"{episode} | eps={exploration_rate} lr={lr} | R={rewards_cur_episode:.2f} | S={steps} | avg_R={np.mean(rewards_all_episodes):.4f} | avg_S={np.mean(travel_times):.4f} ",
        )

    print(f"Q-learning time: {time.time() - ts}")
    np.save("q-table-0.0.npy", q_table)

    n, e, s, w = [q_table[:, :, i] for i in (0, 1, 2, 3)]
    for action, name in zip((n, e, s, w), "NESW"):
        print(action.shape)
        plt.figure()
        sns.heatmap(action.T, square=True)
        plt.title(f"Q-table {name}")
        os.makedirs("img", exist_ok=True)
        plt.savefig(f"img/q_table-{name}")

    q_table_cum = np.sum(q_table, axis=2)
    plt.figure()
    sns.heatmap(q_table_cum.T, square=True)
    plt.title(f"Q-table cum")
    os.makedirs("img", exist_ok=True)
    plt.savefig(f"img/q_table-cum")

    reward_per_thousand_episodes = np.split(
        np.array(rewards_all_episodes), n_episodes / 1000
    )
    times_per_thousand_episodes = np.split(np.array(travel_times), n_episodes / 1000)
    cnt = 1000
    print("\nAverage reward per 1000 episodes ***********\n")
    print("Iters: avg reward | avg steps to goal")
    for r, tr in zip(reward_per_thousand_episodes, times_per_thousand_episodes):
        print(cnt, ": ", str(sum(r / 1000)), "|", str(sum(tr / 1000)))
        cnt += 1000

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

    plt.figure()
    plt.plot(moving_average(travel_times, 100))
    plt.title("Travel time rolling average from 100 episodes")
    os.makedirs("img", exist_ok=True)
    plt.savefig("img/travel_time")

    plt.figure()
    plt.plot(moving_average(rewards_all_episodes, 100))
    plt.title("Reward rolling average from 100 episodes")
    os.makedirs("img", exist_ok=True)
    plt.savefig("img/reward")

    print(np.array(explores))
    n, e, s, w = [M[:, :, i] for i in (0, 1, 2, 3)]
    for action, name in zip((n, e, s, w), "NESW"):
        print(action.shape)
        plt.figure()
        sns.heatmap(action.T, square=True)
        plt.title(f"Congestion prob {name}")
        os.makedirs("img", exist_ok=True)
        plt.savefig(f"img/M-{name}")

