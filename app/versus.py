import pygame
import numpy as np
import random
import math
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.game import SnakeGame, absolute_to_relative, KEYS
from tensorflow import keras

# konfiguracja 
GRID_SIZE = 10 #rozmiar planszy
CELL_SIZE = 40 #rozmiar kwadratu w pxl
FPS = 5  #klatki na sekundę (im wyższe, tym szybciej gra)
GAP = 20 #przerwa miedzy planszami w pxl

# kolory w RGB
BG_COLOR    = (30, 30, 40) #tlo
GRID_COLOR  = (50, 50, 60) # siatka
TEXT_COLOR  = (220, 220, 220) # txt
DIM_COLOR   = (140, 140, 140) # przyciemniony txt 
MENU_BG     = (20, 20, 30) # tlo menu
HIGHLIGHT   = (255, 220, 80) # podsiwetlanie
DEAD_COLOR  = (200, 30, 30) # kolor smierci weza

# kolory graczy
P1_COLOR    = (0, 150, 255)  #wsad - niebieski
P1_HEAD     = (80, 200, 255)
P2_COLOR    = (255, 160, 0)  #strzalki - pomarancz
P2_HEAD     = (255, 200, 60)
AI_COLOR    = (0, 200, 80)   #siec neuronowa -zielony
AI_HEAD     = (0, 255, 120)

BOARD_SIZE = GRID_SIZE * CELL_SIZE #rozmiar w pxl
WINDOW_W = BOARD_SIZE * 2 + GAP # szerokosc okna
WINDOW_H = BOARD_SIZE + 80 # wysokosc okna 


#owoce rysowane z kształtów pygame

def draw_apple(screen, cx, cy, size):
    '''jabłko'''
    r = size // 2 - 2
    pygame.draw.circle(screen, (220, 40, 40), (cx, cy + 2), r)
    #połysk
    pygame.draw.circle(screen, (255, 100, 100), (cx - r // 3, cy - r // 3), r // 3)
    #ogonek
    pygame.draw.line(screen, (100, 60, 30), (cx, cy - r), (cx + 2, cy - r - 5), 2)
    #listek
    leaf_points = [(cx + 2, cy - r - 3), (cx + 8, cy - r - 7), (cx + 5, cy - r)]
    pygame.draw.polygon(screen, (60, 180, 60), leaf_points)


def draw_cherry(screen, cx, cy, size):
    '''wisienki'''
    r = size // 4
    #gałązki
    pygame.draw.line(screen, (100, 60, 30), (cx, cy - r - 4), (cx - r, cy + 2), 2)
    pygame.draw.line(screen, (100, 60, 30), (cx, cy - r - 4), (cx + r, cy + 2), 2)
    #wisienki
    pygame.draw.circle(screen, (180, 20, 40), (cx - r, cy + 4), r)
    pygame.draw.circle(screen, (200, 30, 50), (cx + r, cy + 4), r)
    #połysk
    pygame.draw.circle(screen, (255, 100, 120), (cx - r - 2, cy + 1), r // 3)
    pygame.draw.circle(screen, (255, 100, 120), (cx + r - 2, cy + 1), r // 3)


def draw_banana(screen, cx, cy, size):
    '''banan'''
    banana_color = (255, 220, 50)
    dark_yellow = (200, 170, 30)
    r = size // 2 - 3

    #łuk z punktów
    points = []
    for angle in range(30, 180, 10):
        rad = math.radians(angle)
        x = cx + int(r * math.cos(rad))
        y = cy - int(r * 0.6 * math.sin(rad)) + 3
        points.append((x, y))

    if len(points) > 2:
        pygame.draw.lines(screen, banana_color, False, points, 5)
        pygame.draw.lines(screen, dark_yellow, False, points, 2)

    #końcówki
    pygame.draw.circle(screen, (120, 90, 20), points[0], 3)
    pygame.draw.circle(screen, (120, 90, 20), points[-1], 3)


FRUIT_DRAWERS = [draw_apple, draw_cherry, draw_banana]


#narzędzia graficzne

def lerp_color(c1, c2, t): #interpolacja liniowa kolorów
    '''Mieszanie dwóch kolorów'''
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
    # gradient na ciele weża od głowy do ogona

def draw_board(screen, game, offset_x, offset_y, body_color, head_color, #offset to przesuniecie planszy tej drugiej o 420pxl
               dead=False, flash=False, fruit_type=0):
    # siatka
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(offset_x + x * CELL_SIZE, #x
                               offset_y + y * CELL_SIZE, #y
                               CELL_SIZE, CELL_SIZE) #szerokość i wysokość
            pygame.draw.rect(screen, GRID_COLOR, rect, 1) #rysuje prostokat - 1 to grubosc obramowania

    # jedzenie 
    if not dead: # nie rysujemy gdy waz jest martwy 
        fx, fy = game.food #zmieniamy z wspolrzednych na pxl
        fruit_cx = offset_x + fx * CELL_SIZE + CELL_SIZE // 2
        fruit_cy = offset_y + fy * CELL_SIZE + CELL_SIZE // 2
        drawer = FRUIT_DRAWERS[fruit_type % len(FRUIT_DRAWERS)] # zabezpieczenie zakresu przez modulo
        drawer(screen, fruit_cx, fruit_cy, CELL_SIZE) #wywolujemy funckje z listy do narysowania owocka jedndego z 3

    # kolory dedek
    if dead and flash:
        head_color = DEAD_COLOR
        body_color = DEAD_COLOR

    # ciało z gradientem - okrągłe segmenty
    tail_color = lerp_color(body_color, BG_COLOR, 0.7)
    snake_len = len(game.snake)

    for i, (x, y) in enumerate(game.snake[1:], start=1): #pomijamey glowe
        t = i / max(snake_len - 1, 1) #normalizacja 0-1
        seg_color = lerp_color(body_color, tail_color, t) #kolor danego segmentu
        cx = offset_x + x * CELL_SIZE + CELL_SIZE // 2 
        cy = offset_y + y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 3 #lekki margines dla estetyki
        pygame.draw.circle(screen, seg_color, (cx, cy), radius) #rysujemy segment ciala jako kółko

    # głowa 
    hx, hy = game.snake[0]
    head_cx = offset_x + hx * CELL_SIZE + CELL_SIZE // 2
    head_cy = offset_y + hy * CELL_SIZE + CELL_SIZE // 2
    head_r = CELL_SIZE // 2 - 1 #margines
    pygame.draw.circle(screen, head_color, (head_cx, head_cy), head_r) # głowka

    # kierunek węża (potrzebny do oczu i języka)
    dx, dy = game.direction

    # oczy
    eye_offset = head_r // 3 # jak daleko oczy od środka
    eye_r = head_r // 4 # rozmiar oka
    pupil_r = eye_r // 2 # rozmiar źrenicy

    #oczy prostopadłe do kierunku ruchu
    perp_x, perp_y = -dy, dx # wektor prostopadły - tak jak w tym obrocie w game
    eye1_x = head_cx + perp_x * eye_offset + dx * eye_offset 
    eye1_y = head_cy + perp_y * eye_offset + dy * eye_offset
    eye2_x = head_cx - perp_x * eye_offset + dx * eye_offset
    eye2_y = head_cy - perp_y * eye_offset + dy * eye_offset

    pygame.draw.circle(screen, (255, 255, 255), (eye1_x, eye1_y), eye_r) # 
    pygame.draw.circle(screen, (255, 255, 255), (eye2_x, eye2_y), eye_r)
    pygame.draw.circle(screen, (0, 0, 0), (eye1_x + dx * 2, eye1_y + dy * 2), pupil_r)
    pygame.draw.circle(screen, (0, 0, 0), (eye2_x + dx * 2, eye2_y + dy * 2), pupil_r)

    # język 
    tongue_start_x = head_cx + dx * head_r
    tongue_start_y = head_cy + dy * head_r
    tongue_len = CELL_SIZE // 3
    tongue_end_x = tongue_start_x + dx * tongue_len
    tongue_end_y = tongue_start_y + dy * tongue_len

    pygame.draw.line(screen, (220, 30, 30),
                     (tongue_start_x, tongue_start_y),
                     (tongue_end_x, tongue_end_y), 2)
    # rozciecie języka
    fork = tongue_len // 3
    pygame.draw.line(screen, (220, 30, 30),
                     (tongue_end_x, tongue_end_y),
                     (tongue_end_x + perp_x * fork + dx * fork,
                      tongue_end_y + perp_y * fork + dy * fork), 2)
    pygame.draw.line(screen, (220, 30, 30),
                     (tongue_end_x, tongue_end_y),
                     (tongue_end_x - perp_x * fork + dx * fork,
                      tongue_end_y - perp_y * fork + dy * fork), 2)


# tryby ekranow

def draw_menu(screen, font, big_font, selected):
    screen.fill(MENU_BG)

    title = big_font.render("REINFORCED SNAKE", True, HIGHLIGHT) #antyaliasing i highlight
    screen.blit(title, title.get_rect(centerx=WINDOW_W // 2, y=40))# wklejamy obrazek w dane miejsce

    options = [
        "Ty vs AI",
        "1 vs 1 (WSAD vs Strzalki)",
        "Ogladaj grę AI",
        "Wyjdz",
    ]

    #4 opcje menu, podświetlamy aktualnie wybraną (żółty)
    for i, text in enumerate(options):
        color = HIGHLIGHT if i == selected else TEXT_COLOR
        label = font.render(text, True, color)
        y = 150 + i * 45
        screen.blit(label, label.get_rect(centerx=WINDOW_W // 2, y=y))
        if i == selected:
            arrow = font.render(">", True, HIGHLIGHT)
            screen.blit(arrow, (WINDOW_W // 2 - 160, y))

    hint = font.render("W/S = wybierz    ENTER = zatwierdz", True, DIM_COLOR)
    screen.blit(hint, hint.get_rect(centerx=WINDOW_W // 2, y=WINDOW_H - 40))

    pygame.display.flip()

#overlay z wynikiem i opcjami po zakończeniu gry
def draw_game_over(screen, font, big_font, result, p1_score, p2_score, p2_name):
    overlay = pygame.Surface((WINDOW_W, WINDOW_H))
    overlay.fill((0, 0, 0))
    overlay.set_alpha(180)
    screen.blit(overlay, (0, 0))

    if result == "win":
        msg = big_font.render("WYGRALES!", True, P1_HEAD)
    elif result == "lose":
        msg = big_font.render("PRZEGRALES!", True, DEAD_COLOR)
    else:
        msg = big_font.render("REMIS!", True, HIGHLIGHT)

    screen.blit(msg, msg.get_rect(centerx=WINDOW_W // 2, y=WINDOW_H // 2 - 60))

    scores = font.render(f"P1: {p1_score}    {p2_name}: {p2_score}", True, TEXT_COLOR)
    screen.blit(scores, scores.get_rect(centerx=WINDOW_W // 2, y=WINDOW_H // 2))

    hint = font.render("ENTER = menu    R = jeszcze raz", True, DIM_COLOR)
    screen.blit(hint, hint.get_rect(centerx=WINDOW_W // 2, y=WINDOW_H // 2 + 50))

    pygame.display.flip()


def death_animation(screen, font, games, positions, colors, dead_index):
    '''Animacja smierci'''
    for flash_i in range(6): # 6 sekwencji - 3 blyski
        screen.fill(BG_COLOR)
        flash_on = flash_i % 2 == 0

        for i, (game, (ox, oy, label, bc, hc, ft)) in enumerate(zip(games, positions)):
            lbl = font.render(label, True, hc)
            screen.blit(lbl, (ox + 10, 5))

            is_dead = (i == dead_index)
            draw_board(screen, game, ox, oy, bc, hc,
                       dead=is_dead, flash=(is_dead and flash_on), fruit_type=ft)

        pygame.display.flip()
        pygame.time.wait(150)


#główna pętla gry

def run_game(screen, font, big_font, model, mode):
    '''mode: "versus", "pvp", "watch"'''
    clock = pygame.time.Clock() #zegar do kntroli FPS
    left_x = 0 #lewa plansza
    right_x = BOARD_SIZE + GAP #prawa plansza
    board_y = 30 # 30pxl od góry

    # inicjalizacja gier
    game1 = SnakeGame(size=GRID_SIZE)
    game2 = SnakeGame(size=GRID_SIZE)
    game1.reset() # czlowiek - stan niepotrzebny
    state2 = game2.reset() #zapis stanu dla sieci 

    #losowe typy owoców
    fruit1 = random.randrange(len(FRUIT_DRAWERS))
    fruit2 = random.randrange(len(FRUIT_DRAWERS))

    #sterowanie
    p1_action = None
    p2_action = None

    #gracz 2 (strzałki)
    P2_KEYS = {
        pygame.K_UP:    (0, -1),
        pygame.K_DOWN:  (0,  1),
        pygame.K_LEFT:  (-1, 0),
        pygame.K_RIGHT: (1,  0),
    }

    # etykiety i kolory
    if mode == "versus":
        p2_name = "AI"
        p2_bc, p2_hc = AI_COLOR, AI_HEAD
    elif mode == "pvp":
        p2_name = "P2"
        p2_bc, p2_hc = P2_COLOR, P2_HEAD
    else:
        p2_name = "AI"
        p2_bc, p2_hc = AI_COLOR, AI_HEAD

    p1_label = "P1 (WSAD)" if mode != "watch" else ""
    p2_label = f"{p2_name} (Strzalki)" if mode == "pvp" else p2_name

    game_over = False
    result = None

    while True:
        # eventy
        for event in pygame.event.get(): #pobieramy event od ostatniej klatki
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"

                if game_over:
                    if event.key == pygame.K_RETURN:
                        return "menu"
                    if event.key == pygame.K_r:
                        return mode
                    continue

                #P1: WSAD
                if mode in ("versus", "pvp"):
                    wsad_map = {
                        pygame.K_w: 'w', pygame.K_a: 'a',
                        pygame.K_s: 's', pygame.K_d: 'd',
                    }
                    if event.key in wsad_map:
                        p1_action = absolute_to_relative(
                            game1.direction, KEYS[wsad_map[event.key]]
                        )

                # P2: strzałki (tylko pvp)
                if mode == "pvp" and event.key in P2_KEYS:
                    p2_action = absolute_to_relative(
                        game2.direction, P2_KEYS[event.key]
                    )

        if game_over:
            clock.tick(30)
            continue

        #ruch P1
        if mode in ("versus", "pvp") and not game1.done:
            action = p1_action if p1_action is not None else 0
            _, h_reward, h_done = game1.step(action)
            p1_action = None

            if h_reward == 10:
                fruit1 = random.randrange(len(FRUIT_DRAWERS))

            if h_done:
                positions = [
                    (left_x, board_y, p1_label, P1_COLOR, P1_HEAD, fruit1),
                    (right_x, board_y, p2_label, p2_bc, p2_hc, fruit2),
                ]
                death_animation(screen, font, [game1, game2], positions, None, 0)

                if game1.score > game2.score:
                    result = "win"
                elif game1.score < game2.score:
                    result = "lose"
                else:
                    result = "draw"
                game_over = True

        # P2 / AI
        if not game2.done:
            if mode in ("versus", "watch"):
                # AI
                state_input = np.array([state2])
                #wrzucamy stan do sieci i dostajemy q_values akcji
                q_values = model(state_input, training=False).numpy()
                ai_action = int(np.argmax(q_values[0])) #wybieramy akcje z najwyższ

                state2, a_reward, a_done = game2.step(ai_action)
            else:
                #P2 (pvp)
                action = p2_action if p2_action is not None else 0
                state2, a_reward, a_done = game2.step(action)
                p2_action = None

            if a_reward == 10:
                fruit2 = random.randrange(len(FRUIT_DRAWERS))

            if a_done:
                if mode == "watch":
                    game2 = SnakeGame(size=GRID_SIZE)
                    state2 = game2.reset()
                    fruit2 = random.randrange(len(FRUIT_DRAWERS))
                elif mode == "versus":
                    # AI umarło — cichy restart
                    game2 = SnakeGame(size=GRID_SIZE)
                    state2 = game2.reset()
                    fruit2 = random.randrange(len(FRUIT_DRAWERS))
                else:
                    #pvp — P2 umarł
                    positions = [
                        (left_x, board_y, p1_label, P1_COLOR, P1_HEAD, fruit1),
                        (right_x, board_y, p2_label, p2_bc, p2_hc, fruit2),
                    ]
                    death_animation(screen, font, [game1, game2], positions, None, 1)

                    if game1.score > game2.score:
                        result = "win"
                    elif game1.score < game2.score:
                        result = "lose"
                    else:
                        result = "win"  # P2 umarł przy remisie to P1 wygrywa
                    game_over = True

        # rysowanie
        screen.fill(BG_COLOR)

        if mode != "watch":
            lbl1 = font.render(p1_label, True, P1_HEAD)
            screen.blit(lbl1, (left_x + 10, 5))
            draw_board(screen, game1, left_x, board_y, P1_COLOR, P1_HEAD, fruit_type=fruit1)

            h_info = font.render(f"Score: {game1.score}", True, P1_HEAD)
            screen.blit(h_info, (left_x + 10, board_y + BOARD_SIZE + 5))

        # AI / P2
        if mode == "watch":
            ai_x = WINDOW_W // 2 - BOARD_SIZE // 2
        else:
            ai_x = right_x

        lbl2 = font.render(p2_label, True, p2_hc)
        screen.blit(lbl2, (ai_x + 10, 5))
        draw_board(screen, game2, ai_x, board_y, p2_bc, p2_hc, fruit_type=fruit2)

        a_info = font.render(f"Score: {game2.score}", True, p2_hc)
        screen.blit(a_info, (ai_x + 10, board_y + BOARD_SIZE + 5))

        if game_over:
            draw_game_over(screen, font, big_font, result,
                           game1.score, game2.score, p2_name)
        else:
            pygame.display.flip()

        clock.tick(FPS)


# main

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(ROOT, './results/base1.keras')
    model = keras.models.load_model(model_path)
    print("Model zaladowany!")

    pygame.init() 
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H)) #towrzymy okno o danych wymiarach
    pygame.display.set_caption("REINFORCED SNAKE")
    font = pygame.font.SysFont("consolas", 18)
    big_font = pygame.font.SysFont("consolas", 36)

    state = "menu"
    selected = 0
    num_options = 4

    while state != "quit":
        if state == "menu":
            draw_menu(screen, font, big_font, selected)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    state = "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_w, pygame.K_UP):
                        selected = (selected - 1) % num_options
                    if event.key in (pygame.K_s, pygame.K_DOWN):
                        selected = (selected + 1) % num_options
                    if event.key == pygame.K_RETURN:
                        if selected == 0:
                            state = "versus"
                        elif selected == 1:
                            state = "pvp"
                        elif selected == 2:
                            state = "watch"
                        else:
                            state = "quit"

        elif state in ("versus", "pvp", "watch"):
            state = run_game(screen, font, big_font, model, state)

    pygame.quit()


if __name__ == "__main__":
    main()