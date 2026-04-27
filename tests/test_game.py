import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from model.game import SnakeGame, absolute_to_relative


# ─── fixtures ───

@pytest.fixture
def game():
    '''Świeża gra 10x10 do każdego testu.'''
    return SnakeGame(size=10)

@pytest.fixture
def game_with_long_snake():
    '''Gra z wężem długości 4 w kształcie poziomym.'''
    g = SnakeGame(size=10)
    g.snake = [(5, 5), (4, 5), (3, 5), (2, 5)]
    g.direction = (1, 0)
    g.food = (8, 8)
    return g


# ─── reset ───

class TestReset:
    def test_snake_position(self, game):
        '''Wąż startuje na środku planszy.'''
        assert game.snake == [(5, 5)]

    def test_direction(self, game):
        '''Wąż startuje patrząc w prawo.'''
        assert game.direction == (1, 0)

    def test_returns_state(self, game):
        '''Reset zwraca stan o 16 elementach.'''
        state = game.reset()
        assert len(state) == 16

    def test_initial_values(self, game):
        '''Score=0, done=False, hunger=0.'''
        assert game.score == 0
        assert game.done == False
        assert game.steps_since_food == 0

    def test_food_valid(self, game):
        '''Jedzenie na planszy i nie na wężu.'''
        fx, fy = game.food
        assert 0 <= fx < 10
        assert 0 <= fy < 10
        assert game.food not in game.snake


# ─── turn ───

class TestTurn:
    def test_straight(self, game):
        '''Akcja 0 nie zmienia kierunku.'''
        game._turn(0)
        assert game.direction == (1, 0)

    @pytest.mark.parametrize("start,action,expected", [
        ((1, 0),  1, (0, 1)),    # prawo → skręt w prawo → dół
        ((0, 1),  1, (-1, 0)),   # dół → skręt w prawo → lewo
        ((-1, 0), 1, (0, -1)),   # lewo → skręt w prawo → góra
        ((0, -1), 1, (1, 0)),    # góra → skręt w prawo → prawo
        ((1, 0),  2, (0, -1)),   # prawo → skręt w lewo → góra
        ((0, -1), 2, (-1, 0)),   # góra → skręt w lewo → lewo
        ((-1, 0), 2, (0, 1)),    # lewo → skręt w lewo → dół
        ((0, 1),  2, (1, 0)),    # dół → skręt w lewo → prawo
    ])
    def test_all_rotations(self, game, start, action, expected):
        '''Sprawdza obrót dla każdego kierunku i akcji.'''
        game.direction = start
        game._turn(action)
        assert game.direction == expected

    def test_full_circle_right(self, game):
        '''4 skręty w prawo = pełny obrót.'''
        original = game.direction
        for _ in range(4):
            game._turn(1)
        assert game.direction == original

    def test_right_then_left_cancels(self, game):
        '''Prawo + lewo = powrót.'''
        original = game.direction
        game._turn(1)
        game._turn(2)
        assert game.direction == original


# ─── collision ───

class TestCollision:
    @pytest.mark.parametrize("pos", [(-1, 5), (10, 5), (5, -1), (5, 10)])
    def test_wall(self, game, pos):
        '''Wyjście poza planszę = kolizja.'''
        assert game._is_collision(pos) == True

    @pytest.mark.parametrize("pos", [(0, 0), (9, 0), (0, 9), (9, 9), (5, 5)])
    def test_valid_positions(self, game, pos):
        '''Pozycje na planszy bez węża = brak kolizji.'''
        game.snake = [(7, 7)]  # wąż daleko
        game.food = (0, 1)
        assert game._is_collision(pos) == False

    def test_body(self, game_with_long_snake):
        '''Kolizja z ciałem.'''
        assert game_with_long_snake._is_collision((3, 5)) == True

    def test_tail_vanishes(self, game_with_long_snake):
        '''Ogon zaraz zniknie → brak kolizji.'''
        tail = game_with_long_snake.snake[-1]
        assert game_with_long_snake._is_collision(tail) == False

    def test_tail_stays_when_eating(self, game_with_long_snake):
        '''Ogon zostaje gdy jemy → kolizja.'''
        tail = game_with_long_snake.snake[-1]
        game_with_long_snake.food = tail
        assert game_with_long_snake._is_collision(tail) == True


# ─── step ───

class TestStep:
    def test_move(self, game):
        '''Wąż przesuwa się o jedno pole.'''
        game.food = (9, 9)
        game.step(0)
        assert game.snake[0] == (6, 5)

    def test_returns_tuple(self, game):
        '''Zwraca (state, reward, done).'''
        state, reward, done = game.step(0)
        assert len(state) == 16
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)

    def test_normal_reward(self, game):
        '''Zwykły krok → reward 0.'''
        game.food = (9, 9)
        _, reward, _ = game.step(0)
        assert reward == 0

    def test_eat_food(self, game):
        '''Zjedzenie: score+1, reward+10, wąż rośnie.'''
        game.food = (6, 5)
        old_len = len(game.snake)
        _, reward, done = game.step(0)
        assert reward == 10
        assert game.score == 1
        assert len(game.snake) == old_len + 1
        assert done == False

    def test_new_food_after_eating(self, game):
        '''Po zjedzeniu nowe jedzenie w innym miejscu.'''
        game.food = (6, 5)
        game.step(0)
        assert game.food != (6, 5)
        assert game.food not in game.snake

    def test_no_growth_without_food(self, game):
        '''Bez jedzenia wąż nie rośnie.'''
        game.food = (9, 9)
        old_len = len(game.snake)
        game.step(0)
        assert len(game.snake) == old_len

    def test_wall_death(self, game):
        '''Ściana → done, reward -10.'''
        game.snake = [(9, 5)]
        game.direction = (1, 0)
        game.food = (0, 0)
        _, reward, done = game.step(0)
        assert done == True
        assert reward == -10

    def test_body_death(self):
        '''Wjechanie w siebie → done, reward -10.'''
        game = SnakeGame(size=10)
        game.snake = [(5, 5), (4, 5), (4, 6), (5, 6)]
        game.direction = (0, 1)
        game.food = (0, 0)
        _, reward, done = game.step(1)
        assert done == True
        assert reward == -10

    def test_starvation(self, game):
        '''Za dużo kroków bez jedzenia → śmierć.'''
        game.food = (0, 0)
        game.steps_since_food = 100 * len(game.snake)
        _, reward, done = game.step(0)
        assert done == True
        assert reward == -10

    def test_hunger_resets_on_eat(self, game):
        '''Zjedzenie zeruje licznik głodu.'''
        game.steps_since_food = 50
        game.food = (6, 5)
        game.step(0)
        assert game.steps_since_food == 0
    
    
    def test_win_full_board(self):
        '''Wąż zajmuje całą planszę → wygrana.'''
        game = SnakeGame(size=3)  # mała plansza 3x3 = 9 pól
        # wąż zajmuje 8 pól, jedzenie na 9-tym
        game.snake = [
            (0,0), (1,0), (2,0),
            (2,1), (1,1), (0,1),
            (0,2), (1,2),
        ]
        game.direction = (1, 0)  # w prawo
        game.food = (2, 2)       # ostatnie wolne pole
        _, reward, done = game.step(0)
        assert done == True
        assert reward == 10      # wygrana, nie kara
        assert game.score == 1


# ─── get_state ───

class TestGetState:
    def test_length(self, game):
        '''Stan ma 16 elementów.'''
        assert len(game.get_state()) == 16

    def test_danger_at_wall(self, game):
        '''Przy ścianie → danger ahead.'''
        game.snake = [(9, 5)]
        game.direction = (1, 0)
        game.food = (0, 0)
        state = game.get_state()
        assert state[0] == True

    def test_no_danger_center(self, game):
        '''Na środku → brak danger.'''
        game.snake = [(5, 5)]
        game.direction = (1, 0)
        game.food = (0, 0)
        state = game.get_state()
        assert state[0:3] == [False, False, False]

    @pytest.mark.parametrize("direction", [(1, 0), (-1, 0), (0, 1), (0, -1)])
    def test_direction_one_hot(self, game, direction):
        '''Kierunek → dokładnie jedna jedynka w [3:7].'''
        game.direction = direction
        state = game.get_state()
        assert sum(state[3:7]) == 1

    def test_food_direction(self, game):
        '''Jedzenie na prawo i niżej.'''
        game.snake = [(3, 3)]
        game.food = (7, 8)
        state = game.get_state()
        assert state[7:11] == [False, True, False, True]

    def test_food_distance_values(self, game):
        '''Konkretne znormalizowane odległości.'''
        game.snake = [(5, 5)]
        game.food = (8, 3)
        state = game.get_state()
        assert state[11] == pytest.approx(0.3)
        assert state[12] == pytest.approx(-0.2)

    def test_wall_distance_range(self, game):
        '''Odległości do ściany w [0, 1].'''
        state = game.get_state()
        for i in [13, 14, 15]:
            assert 0 <= state[i] <= 1

    def test_wall_distance_at_edge(self, game):
        '''Przy ścianie → distance = 0.'''
        game.snake = [(9, 5)]
        game.direction = (1, 0)
        game.food = (0, 0)
        state = game.get_state()
        assert state[13] == 0


# ─── absolute_to_relative ───

class TestAbsoluteToRelative:
    @pytest.mark.parametrize("current,desired,expected", [
        ((1, 0),  (1, 0),  0),   # ten sam → prosto
        ((1, 0),  (0, 1),  1),   # prawo→dół = skręt w prawo
        ((1, 0),  (0, -1), 2),   # prawo→góra = skręt w lewo
        ((1, 0),  (-1, 0), 0),   # przeciwny → ignoruj
        ((0, 1),  (1, 0),  2),   # dół→prawo = skręt w lewo
        ((0, -1), (1, 0),  1),   # góra→prawo = skręt w prawo
    ])
    def test_conversion(self, current, desired, expected):
        assert absolute_to_relative(current, desired) == expected