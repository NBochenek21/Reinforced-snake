import pygame
import numpy as np
from game import SnakeGame
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─── konfiguracja ───
GRID_SIZE = 10         # rozmiar planszy (musi być taki sam jak w treningu)
CELL_SIZE = 50         # rozmiar jednego pola w pikselach
FPS = 8                # klatki na sekundę (im mniej, tym wolniej gra agent)

# kolory (RGB)
BG_COLOR    = (30, 30, 40)       # tło
GRID_COLOR  = (50, 50, 60)       # linie siatki
SNAKE_COLOR = (0, 200, 80)       # ciało węża
HEAD_COLOR  = (0, 255, 120)      # głowa
FOOD_COLOR  = (220, 50, 50)      # jedzenie
TEXT_COLOR  = (220, 220, 220)    # tekst

WINDOW_SIZE = GRID_SIZE * CELL_SIZE

def draw(screen, game, font, model_name=""):
    screen.fill(BG_COLOR)

    # siatka
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

    # jedzenie
    fx, fy = game.food
    food_rect = pygame.Rect(fx * CELL_SIZE + 4, fy * CELL_SIZE + 4,
                            CELL_SIZE - 8, CELL_SIZE - 8)
    pygame.draw.rect(screen, FOOD_COLOR, food_rect, border_radius=10)

    # ciało węża
    for (x, y) in game.snake[1:]:
        body_rect = pygame.Rect(x * CELL_SIZE + 2, y * CELL_SIZE + 2,
                                CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(screen, SNAKE_COLOR, body_rect, border_radius=6)

    # głowa
    hx, hy = game.snake[0]
    head_rect = pygame.Rect(hx * CELL_SIZE + 2, hy * CELL_SIZE + 2,
                            CELL_SIZE - 4, CELL_SIZE - 4)
    pygame.draw.rect(screen, HEAD_COLOR, head_rect, border_radius=6)

    # tekst: score i długość
    info = font.render(f"Score: {game.score}   Len: {len(game.snake)}", True, TEXT_COLOR)
    screen.blit(info, (10, WINDOW_SIZE + 5))

    pygame.display.flip()



def main():
    # ładujemy model
    model = keras.models.load_model('best_model.keras')
    print("Model załadowany!")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 40))
    pygame.display.set_caption("Snake AI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    game = SnakeGame(size=GRID_SIZE)
    state = game.reset()
    running = True

    while running:
        # obsługa zdarzeń
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # strzałki góra/dół zmieniają szybkość
                if event.key == pygame.K_UP:
                    clock.tick(0)   # bez limitu — pełna prędkość
                if event.key == pygame.K_r:
                    game = SnakeGame(size=GRID_SIZE)
                    state = game.reset()

        if not game.done:
            # agent wybiera akcję (epsilon=0, zawsze najlepsza)
            state_input = np.array([state])
            q_values = model(state_input, training=False).numpy()
            action = int(np.argmax(q_values[0]))

            state, reward, done = game.step(action)
        else:
            # auto-restart po krótkiej pauzie
            pygame.time.wait(1000)
            game = SnakeGame(size=GRID_SIZE)
            state = game.reset()

        draw(screen, game, font)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()