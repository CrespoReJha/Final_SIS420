import numpy as np
import random
import matplotlib.pyplot as plt  # librería que proporciona funciones para graficar datos

class PuzzleEnvironment:
    def __init__(self, initial_state, max_moves=100):
        self.initial_state = initial_state
        self.state = np.copy(initial_state)
        self.rows, self.cols = self.state.shape
        self.goal_state = np.array([[1, 2, 3, 4, 5],
                                    [6, 7, 8, 9, 10], 
                                    [11, 12, 13, 14, 15],
                                    [16, 17, 18, 19, 0]])  # 0 represents the empty space
        self.empty_pos = self.find_empty_pos()
       
        self.max_moves = max_moves
        self.current_moves = 0
        
    def find_empty_pos(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.state[i, j] == 0:
                    return (i, j)
        
    def is_solved(self):
        return np.array_equal(self.state, self.goal_state)
    
    def count_correct_positions(self):
        correct_positions = 0
        for i in range(self.rows):
            for j in range(self.cols):
                if self.state[i, j] == self.goal_state[i, j]:
                    correct_positions += 1
        return correct_positions
    
    def move(self, action):
        row, col = self.empty_pos
        moved = False
        if action == 'up' and row > 0:
            self.state[row, col], self.state[row-1, col] = self.state[row-1, col], self.state[row, col]
            self.empty_pos = (row-1, col)
            moved = True
        elif action == 'down' and row < self.rows - 1:
            self.state[row, col], self.state[row+1, col] = self.state[row+1, col], self.state[row, col]
            self.empty_pos = (row+1, col)
            moved = True
        elif action == 'left' and col > 0:
            self.state[row, col], self.state[row, col-1] = self.state[row, col-1], self.state[row, col]
            self.empty_pos = (row, col-1)
            moved = True
        elif action == 'right' and col < self.cols - 1:
            self.state[row, col], self.state[row, col+1] = self.state[row, col+1], self.state[row, col]
            self.empty_pos = (row, col+1)
            moved = True
        
        if moved:
            self.current_moves += 1
        
        #reward = self.count_correct_positions() - 1
        reward = 200 if self.is_solved() else -1
        done = self.is_solved() or self.current_moves >= self.max_moves
        return self.state, reward, done
    
    def reset(self):
        self.state = np.copy(self.initial_state)
        np.random.shuffle(self.state)
        self.empty_pos = self.find_empty_pos()
        self.current_moves = 0
        return self.state

def train(env, episodes, epsilon_start, learning_rate=0.1, discount_factor=0.9, epsilon_decay_rate=0.0011):
    q_table = {}
    actions = ['up', 'down', 'left', 'right']

    def get_q_value(state, action):
        return q_table.get((tuple(state.flatten()), action), 0)

    def set_q_value(state, action, value):
        q_table[(tuple(state.flatten()), action)] = value

    epsilon = epsilon_start
    rewards_per_episode = np.zeros(episodes)
    rng = np.random.default_rng()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            if rng.random() < epsilon:
                action = random.choice(actions)#exploracion accion aleatoria
            else:
                q_values = [get_q_value(state, action) for action in actions]
                action = actions[np.argmax(q_values)]#explotacion, toma la mejor accion segun la tabla

            new_state, reward, done = env.move(action)#realiza la accion
            total_reward += reward
            #print(f"recompensa: {reward}, done: {done}")

            if not done:
                future_q_values = [get_q_value(new_state, future_action) for future_action in actions]
                target = reward + discount_factor * max(future_q_values)
            else:
                target = reward

            q_value = get_q_value(state, action)
            new_q_value = q_value + learning_rate * (target - q_value) #actualiza la tabla
            set_q_value(state, action, new_q_value)

            state = new_state

        rewards_per_episode[episode] = total_reward
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if (episode + 1) % 100 == 0:
            print(f"Episodio {episode + 1} - Recompensa total: {total_reward}")

    rewards_mean_per_100_episodes = np.mean(rewards_per_episode.reshape(-1, 100), axis=1)
    return rewards_mean_per_100_episodes

# Estado inicial del rompecabezas (un ejemplo)
initial_state = np.array([[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10],  # Empty space represented by 0
                          [11, 12, 13, 14, 15],
                          [16, 17, 18, 0, 19]])

env = PuzzleEnvironment(initial_state, max_moves=200)

if __name__ == "__main__":
    episodes = 100
    epsilons = [0.9, 0.5, 0.1]  # Valores iniciales de epsilon
    all_rewards = []

    for epsilon in epsilons:
        rewards = train(env, episodes, epsilon)
        all_rewards.append(rewards)

    # Gráfica de la recompensa media cada 100 episodios para cada valor de epsilon
    plt.figure(figsize=(8, 5))
    for i, epsilon in enumerate(epsilons):
        plt.plot(all_rewards[i], label=f'$\epsilon$ inicial = {epsilon}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Episodios (x100)')
    plt.ylabel('Recompensa Media')
    plt.title('Recompensa Media cada 100 Episodios para diferentes valores de epsilon')
    plt.show()
