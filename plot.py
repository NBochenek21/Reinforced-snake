import json
import matplotlib.pyplot as plt

with open('training_history.json', 'r') as f:
    history = json.load(f) # wczytujemy plik JSON do pythonowego słownika

scores = history["scores"]
epsilons = history["epsilons"]
episode_lengths = history["episode_lengths"]
episodes = range(len(scores)) #długość listy score to liczba eps

# średnia krocząca (wygładza szum - bo tu jest bardzo nieregularnie)
def moving_avg(data, window=50):
    avg = []
    for i in range(len(data)):
        start = max(0, i - window)
        avg.append(sum(data[start:i+1]) / (i - start + 1))
    return avg

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# wykres 1: score
axes[0].plot(episodes, scores, alpha=0.3, color='dodgerblue', label='Score')
axes[0].plot(episodes, moving_avg(scores), color='dodgerblue', linewidth=2, label='Avg (50 ep)')
axes[0].set_ylabel('Score')
axes[0].legend()
axes[0].set_title('Trening Snake AI - DQN')

# wykres 2: ilosc kroków w epizodzie
lengths = history["episode_lengths"]
axes[1].plot(episodes, lengths, alpha=0.3, color='limegreen', label='Steps')
axes[1].plot(episodes, moving_avg(lengths), color='limegreen', linewidth=2, label='Avg (50 ep)')
axes[1].set_ylabel('Episode Length')
axes[1].legend()

# wykres 3: epsilon
axes[2].plot(episodes, epsilons, color='tomato', linewidth=2)
axes[2].set_ylabel('Epsilon')
axes[2].set_xlabel('Epizod')

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150)
plt.show()
print("Wykres zapisany jako training_plot.png")