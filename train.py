from game import SnakeGame
from agent import DQNAgent

EPISODES = 1000
BATCH_SIZE = 32

env = SnakeGame(size=10)
agent = DQNAgent()

best_score = 0
step_count = 0

for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0

    while not env.done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step_count += 1

        # ucz co 4 kroki — kompromis między szybkością a jakością
        if step_count % 4 == 0 and len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

    # epsilon spada raz na epizod, nie na krok
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if env.score > best_score:
        best_score = env.score
        agent.model.save('best_model.keras')

    if ep % 50 == 0:
        print(f"ep {ep:4d}  score: {env.score:2d}  best: {best_score:2d}  "
              f"eps: {agent.epsilon:.3f}  reward: {total_reward:+.0f}")

print(f"\nTrening zakończony! Najlepszy wynik: {best_score}")