import random

class SnakeGame:
    def __init__(self, size = 10):
        self.size = size
        self.reset()

    def reset(self):
        '''
        -wąż zaczyna na srodku z długością 1
        - prawo  = ( 1,  0)       x +1, y bez zmian
            lewo   = (-1,  0)     x -1, y bez zmian
            dół    = ( 0,  1)     x bez zmian, y +1  - y rosnie w dol
            góra   = ( 0, -1)     x bez zmian, y -1
            
        
        '''
        cx, cy = self.size // 2, self.size // 2
        self.snake = [(cx, cy)] # lista a nie set bo głowa to [0] dlatego ma znaczenie kolejnosc
        self.direction = (1, 0) 
        self.food = None # pozycja jedzenia (krotka)
        self.done = False # czy gra się skończyła
        self._place_food()

    def _place_food(self):
        #losowe dopóki nie trafi na puste pole
        while True:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1)) #losujemy od 0 do 9
            if pos not in self.snake: # sprawdzamy czy jedzenie nie ląduje w weżu
                self.food = pos
                return
            

    def step(self, action):
        #1. aktualizujemy kierunek na podstawie akcji

        self._turn(action)

        #2. liczmy nowa pozycje glowy
        head = self.snake[0]
        new_head = (head[0] + self.direction[0],
                    head[1] + self.direction[1])
        
        #3. sprawdzamy kolizje
        if self._is_collision(new_head):
            self.done = True #jak jest no to end
            return
        

        #4. nowa głowa na początek losty
        self.snake.insert(0, new_head) 

        #5. sprawdzamy czy zjedlismy jedzenie
        if new_head == self.food:
            self._place_food() #jesli tak to nowe jedzenie
        else:
            self.snake.pop() #jesli nie to usuwamy ogon

    def _turn(self, action):
            '''
            Obrót wektora kierunku (dx, dy) o 90°.
            Konwencja: x rośnie w prawo, y rośnie w dół (jak w pygame).

            Skręt w prawo: (dx, dy) to (-dy, dx)
            Skręt w lewo:  (dx, dy) to (dy, -dx)

            Sprawdzenie skrętu w prawo:
                prawo (1, 0)  -> ( 0,  1) = dół
                dół   (0, 1)  -> (-1,  0) = lewo
                lewo (-1, 0)  -> ( 0, -1) = góra
                góra  (0,-1)  -> ( 1,  0) = prawo

            '''
            dx, dy = self.direction
            if action == 0:                # prosto
                return
            elif action == 1:              # skręć w prawo
                self.direction = (-dy, dx)
            elif action == 2:              # skręć w lewo
                self.direction = (dy, -dx)

    

    def _is_collision(self, pos):
        x, y = pos
        # ściana
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        # własne ciało — z wyjątkiem ogona, który zaraz zniknie
        # (ale tylko jeśli wąż NIE rośnie w tej klatce)
        will_grow = (pos == self.food)
        body = self.snake if will_grow else self.snake[:-1]
        return pos in body

    def __str__ (self):
        '''
        Rysowanie planszy.

        Każdy wiersz odpowiada jednej wartości y (pionowa oś).
        Wewnątrz wiersza wybieramy kolumnę po x (pozioma oś).

        Dlatego piszemy grid[y][x], NIE grid[x][y].
        Pierwszy indeks = wiersz = y
        Drugi indeks   = kolumna = x

        '''

        # zbuduj pustą planszę: lista list znaków
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        #jedzonko
        fx, fy = self.food
        grid[fy][fx] = '%'

        #wąż -cialo
        for (x,y) in self.snake[1:]: # ogon
            grid[y][x] = 'o'

        #wąż - głowa
        hx, hy = self.snake[0]
        grid[hy][hx] = '@'

        #zaminiamy na tekst - każdwy wiersz będzie jedną linią tekstu
        rows = [' '.join(row) for row in grid]

        #ramka
        border = '+' + '-' * (self.size * 2 - 1) + '+' #kazdy znak na planszy zajmuje 2 pozycje (znak + spacja) więc szerokość to size * 2 - 1 (bo ostatni znak nie ma spacji)
        body = '\n'.join('|' + row + '|' for row in rows) # generator który dodaje ramkę z lewej i prawej strony do każdego wiersza
        return f"{border}\n{body}\n{border} \nlen: {len(self.snake)}  status: {self.done}"



if __name__ == "__main__":

    # mapowanie WSAD na wektory kierunku absolutnego
    KEYS = {
        'w': (0, -1),   # góra
        's': (0,  1),   # dół
        'a': (-1, 0),   # lewo
        'd': (1,  0),   # prawo
    }

    def absolute_to_relative(current_dir, desired_dir):
        '''
        Zamienia "chcę iść w tym absolutnym kierunku" na akcję względną
        (0=prosto, 1=skręć w prawo, 2=skręć w lewo) względem obecnego
        kierunku węża. Jeśli gracz wciska kierunek przeciwny do obecnego
        (czyli "wróć w siebie") - ignorujemy i jedziemy prosto.
        - dla sieci nie będzie to miało znaczenia dalej są 3 klasy akcji.
        '''
        if desired_dir == current_dir:
            return 0  # nic nie zmieiamy - prosto
        dx, dy = current_dir
        right = (-dy, dx)   # wektor po skręcie w prawo
        left  = (dy, -dx)   # wektor po skręcie w lewo
        if desired_dir == right:
            return 1   #gracz chce iść tam gdzie zaprowadziłby skręt w prawo
        if desired_dir == left:
            return 2   #analogicznie dla lewej
        return 0  # przeciwny kierunek - ignorujemy

    game = SnakeGame(size=10)
    print(game)
    print('\nSterowanie: WSAD   wyjście: q')

    while not game.done:
        cmd = input('> ').strip().lower()
        if cmd == 'q':
            break
        if cmd not in KEYS:
            print('Nieznana komenda!')
            continue

        action = absolute_to_relative(game.direction, KEYS[cmd])
        game.step(action)
        print(game)

    print('Koniec gry!')