import json
import os
from model.game import SnakeGame
from model.agent import DQNAgent
import random
import numpy as np
import sys
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
''' 
Tutaj łączymy grę z agenetem w pętle treningową.
'''
EPISODES = 2000 #ilość rund gry
BATCH_SIZE = 64 #ile doświadczeń z pamieci losujemy na jedno uczenie

#zapewniamy powtarzalność wyników - faza testów
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

env = SnakeGame(size=10)
agent = DQNAgent()

best_score = 0
step_count = 0 # globalny licznik kroków (nie resetuje sie miedzy ep)

history = {
    "scores": [],
    "epsilons": [],
    "episode_lengths": [],
}

try:# po to zeby ctrl C nie gubił danych
    for ep in range(EPISODES):
        state = env.reset() # nowa gra
        total_reward = 0 #suma nagród
        step_count_ep = 0 # kroki w epizodzie

        while not env.done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            step_count_ep += 1

            if step_count % 4 == 0 and len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE) 
                #złoty środek uczenia się , nie fit co krok bo to bardzo wymagające
                # wtedy by mogło byc wywołane np. 200 razy na epizod - cięzary

        if agent.epsilon > agent.epsilon_min: #epsilon decay - raz na epizod
            agent.epsilon *= agent.epsilon_decay

        if ep % 50 == 0: #target model update - co 100 epizodow
            agent.update_target_model()

        #zapisywanie historii modelu
        history["scores"].append(env.score)
        history["epsilons"].append(agent.epsilon)
        history["episode_lengths"].append(step_count_ep)

        # zapisywanie najlepszego modelu
        if env.score > best_score:
            best_score = env.score
            agent.model.save('best_model.keras')

        if ep % 50 == 0:#infro do terminala co 50 epizodów
            print(f"epizod {ep:4d} best: {best_score:2d}  "
                  f"eps: {agent.epsilon:.3f}")

except KeyboardInterrupt:
    print(f"\n\nPrzerwano na epizodzie {ep}!")

agent.model.save('last_model.keras')

with open('training_history.json', 'w') as f:
    json.dump(history, f)

print(f"Najlepszy wynik: {best_score}")
print("Historia zapisana w training_history.json")