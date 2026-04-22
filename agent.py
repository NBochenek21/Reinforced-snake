import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ukrycie info o GPU

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras



class DQNAgent:
    def __init__(self, state_size=16, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        # hiperparametry
        self.gamma = 0.9           #gamma - obniżanie nastęnpych nagród
        self.epsilon = 1.0         #szansa na losową akcję (eksploracja)
        self.epsilon_min = 0.01    #minimum epsilonu
        self.epsilon_decay = 0.995 #tempo spadku epsilonu

        #pamięć doświadczeń (replay buffer)
        self.memory = deque(maxlen=100_000) # pamięta 100k ostatnich doświadczeń
        #automatycznie wyrzuca najstarsze gdy się zapełni 

        #100k bo jeden epizod to ok 50-200 kroków, 100k to 500-2000 epizodów.

        # sieć neuronowa
        self.model = self._build_model()
        self.target_model = self._build_model()          
        self.target_model.set_weights(self.model.get_weights())  

    def _build_model(self):
        '''
        Prosta sieć: 16 wejść - 128 - 128 - 3 wyjścia.
        Wejście: stan gry 16 cech.
        Wyjście: Q-value dla każdej z 3 akcji.
        '''
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_size),  # bez aktywacji - Q może być ujemne
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model
    
    def act(self, state):
        '''
        Tu wykorzystyjemy zamysł epsilon-greedy do wyboru akcji.
        Wybiera akcję na podstawie stanu.
        Z szansą epsilon - losowa akcja (eksploracja).
        W przeciwnym razie - akcja z najwyższym Q-value (eksploatacja).
        '''
        if random.random() < self.epsilon: #losowa akcja z szansą epsilon
            return random.randrange(self.action_size)

        state = np.array([state]) # sieć oczekuje batcha: (1, 16)
        q_values = self.model(state, training=False).numpy() # np. [[2.1, -0.5, 1.3]]
        return int(np.argmax(q_values[0])) # zwraca indeks najlepszej akcji

    def remember(self, state, action, reward, next_state, done):
        #Zapisuje jedno doświadczenie w replay buffer
        self.memory.append((state, action, reward, next_state, done))
        #bylem w X zrobilem Y dostalem R , jestem w X` i stan gry.

    def update_target_model(self):
        #Kopiuje wagi z głównej sieci do target sieci.
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size=32):
        '''
        Losujemy batch doświadczen z pamieci - bez powtórzen
        - losowosc pomaga 'nie zapomnieć' starych doświadzen
        - losowanie łamie korelcje

        Q-value akcji powinno być równe
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

        # co siec teraz mysli o kazdym stanie
        q_now = self.model(states, training=False).numpy() #c(32,3)
        
        # co target siec mysli o przyszlych stanach - pytamy zamrozona kopie modelu
        q_next = self.target_model(next_states, training=False).numpy()

        # petla korekty
        for i in range(batch_size):
            if dones[i]:
                target = rewards[i] #śmierć - nie ma przyszłości
            else:
                target = rewards[i] + self.gamma * np.max(q_next[i]) # rónwanie Bellmana

            q_now[i][actions[i]] = target  # nadpisujemy Q tylko tej akcji ktora agent wybral

        # trenujemy sieć: "dla tych stanów, Q powinno wyglądać TAK"
        self.model.fit(states, q_now, epochs=1, verbose=0)


# ten blok jest do testowania agenta bez trenowania
if __name__ == "__main__":
    agent = DQNAgent()
    print("model zbudowany!")
    agent.model.summary()

    # symulujemy jedno doświadczenie
    fake_state = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    action = agent.act(fake_state)
    print(f"\nepsilon: {agent.epsilon:.2f}, wybrana akcja: {action}")

