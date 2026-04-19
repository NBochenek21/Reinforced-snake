import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ukryj INFO i WARNING z TF

class DQNAgent:
    def __init__(self, state_size=11, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        # hiperparametry
        self.gamma = 0.9           # dyskontowanie przyszłych nagród
        self.epsilon = 1.0         # szansa na losową akcję (eksploracja)
        self.epsilon_min = 0.01    # minimum epsilonu
        self.epsilon_decay = 0.995 # tempo spadku epsilonu

        # pamięć doświadczeń (replay buffer)
        self.memory = deque(maxlen=100_000)

        # sieć neuronowa
        self.model = self._build_model()

    def _build_model(self):
        '''
        Prosta sieć: 11 wejść → 256 neuronów → 256 neuronów → 3 wyjścia.
        Wejście: stan gry (11 cech).
        Wyjście: Q-value dla każdej z 3 akcji.
        '''
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size),  # bez aktywacji - Q może być ujemne
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       loss='mse')
        return model
    
    def act(self, state):
        '''
        Wybiera akcję na podstawie stanu.
        Z szansą epsilon - losowa akcja (eksploracja).
        W przeciwnym razie - akcja z najwyższym Q-value (eksploatacja).
        '''
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = np.array([state])          # sieć oczekuje batcha: (1, 11)
        q_values = self.model(state, training=False).numpy()  # zwraca np. [[2.1, -0.5, 1.3]]
        return int(np.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done):
        '''Zapisuje jedno doświadczenie w replay bufferze.'''
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        '''
        Uczy sieć na losowej paczce doświadczeń z pamięci.

        Idea: Q-value akcji powinno być równe
        reward + gamma * max(Q przyszłego stanu)

        Czyli: "wartość tego co zrobiłem = nagroda którą dostałem
        + zdyskontowana wartość najlepszej przyszłej opcji"
        '''
        if len(self.memory) < batch_size:
            return   # za mało doświadczeń, jeszcze nie uczymy

        # losowa paczka doświadczeń
        batch = random.sample(self.memory, batch_size)

        # rozpakowujemy paczkę na osobne tablice
        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch])
        rewards     = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch])

        # Q-values dla obecnych stanów (co sieć TERAZ myśli)
        q_now = self.model(states, training=False).numpy()
        
        # Q-values dla przyszłych stanów (żeby wiedzieć ile warta jest przyszłość)
        q_next = self.model(next_states, training=False).numpy()

        # korekta: dla każdego doświadczenia poprawiamy Q wybranej akcji
        for i in range(batch_size):
            if dones[i]:
                target = rewards[i]       # gra się skończyła, nie ma przyszłości
            else:
                target = rewards[i] + self.gamma * np.max(q_next[i])

            q_now[i][actions[i]] = target  # nadpisujemy Q tej konkretnej akcji

        # trenujemy sieć: "dla tych stanów, Q powinno wyglądać TAK"
        self.model.fit(states, q_now, epochs=1, verbose=0)

        # zmniejszamy epsilon (mniej eksploracji z czasem)
        

if __name__ == "__main__":
    agent = DQNAgent()
    print("model zbudowany!")
    agent.model.summary()

    # symulujemy jedno doświadczenie
    fake_state = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    action = agent.act(fake_state)
    print(f"\nepsilon: {agent.epsilon:.2f}, wybrana akcja: {action}")