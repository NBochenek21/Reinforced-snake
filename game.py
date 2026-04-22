import random

'''
To jest środowisko (env) niezależne od reszty aplikacji,
dzięki temu można podpiąc do tej samej gry człowieka z klawiatury,
AI lub inny algorytm sterujący wężem.

'''

class SnakeGame:
    #konstruktor planszy - odrazu reset - zawsze gotowy do użycia od
    #momentu stworzenia obiektu
    def __init__(self, size = 10):
        self.size = size
        self.reset()

    def reset(self):
        '''
        Ustawia gre w stan początkowy - wywołwane przy tworzeniu gry

        -wąż zaczyna na srodku z długością 1
        - prawo  = ( 1,  0)       x +1, y bez zmian
            lewo   = (-1,  0)     x -1, y bez zmian
            dół    = ( 0,  1)     x bez zmian, y +1  - y rosnie w dol
            góra   = ( 0, -1)     x bez zmian, y -1
            
        
        '''
        cx, cy = self.size // 2, self.size // 2 # // - dzielenie całkowite, zawsze zwraca int
        self.snake = [(cx, cy)] # lista a nie set bo głowa to [0] dlatego ma znaczenie kolejnosc
        self.direction = (1, 0) 
        self.food = None # pozycja jedzenia (krotka)
        self._place_food()
        self.score = 0
        self.steps_since_food = 0
        self.done = False # czy gra się skończyła
        
        return self.get_state()


    def _place_food(self): #metoda prywatna - nie wywołujemy jej z zewnatrz klasy) 
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
        new_head = (head[0] + self.direction[0], #x + dx
                    head[1] + self.direction[1]) #y + dy
        
        self.steps_since_food += 1
        
        #3. sprawdzamy kolizje
        if self._is_collision(new_head) or self.steps_since_food > 100 * len(self.snake):
            self.done = True #jak jest no to end
            reward = -10
            return self.get_state(), reward, self.done
        

        #4. nowa głowa na początek losty
        self.snake.insert(0, new_head) 

        #5. sprawdzamy czy zjedlismy jedzenie
        if new_head == self.food:
            self._place_food() #jesli tak to nowe jedzenie
            self.score += 1
            self.steps_since_food = 0
            reward = 10
        else:
            self.snake.pop() #jesli nie to usuwamy ogon
            reward = 0

        return self.get_state(), reward, self.done

    def _turn(self, action):
            '''
            Obrót wektora kierunku (dx, dy) o 90°.
            Konwencja: x rośnie w prawo, y rośnie w dół (tak jest w  pygame)

            Skręt w prawo: (dx, dy) to (-dy, dx)
            Skręt w lewo:  (dx, dy) to (dy, -dx)

                        x 
                0   1   2   3
            0   .   .   .   .
          y 1   .   .   .   .
            2   .   .   .   .
            3   .   .   .   .

            skręt w prawo sposobem geometrycznym (najłatwiej pomyslec o strzalce na kartce papieru):
                prawo (1, 0)  - przesuń x o + 1, y bez zmian
                lewo (-1, 0)  - przesuń x o - 1, y bez zmian
                dół   (0, 1)  - przesuń y o + 1, x bez zmian
                góra  (0,-1)  - przesuń y o - 1, x bez zmian

                PRZED   i  PO
                (1, 0)  na  (0, 1)      prawo na dół
                (0, 1)  na  (-1, 0)     dół na lewo
                (-1, 0) na  (0, -1)     lewo na góra
                (0, -1) na  (1, 0)      góra na prawo

                z tego wynika wzor:
                nowy_dx = -dy
                nowy_dy = dx        

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
        # sprawdza dwa rodzaje smierci  - sciana i wlasne cialo
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        # tu sprawdzenie czy glowa nie jest w cielie
        # jesli waz rosnie to ogon zostaje - cały, jesli nie rośnie to bez ostatniego segmentu
        will_grow = (pos == self.food)
        body = self.snake if will_grow else self.snake[:-1] 
        return pos in body
    
    def get_state(self):
        '''
        Zwraca 11 cech opisujących stan gry z perspektywy węża.
        Każda cecha to 0 lub 1 (False/True zamienione na float).

        [0-2]  niebezpieczeństwo: prosto, w prawo, w lewo - cechy wzgledne - siec nie musi uczyc sie rozumiec co to jest prawo czy lewo, tylko czy w tym kierunku jest niebezpiecznie
        [3-6]  kierunek węża: lewo, prawo, góra, dół (one-hot)
        [7-10] jedzenie względem głowy: lewo, prawo, góra, dół
        [11-12] znormalizowana odległość do jedzenia (dx, dy)
        [13-15] znormalizowana odległość do ściany: prosto, w prawo, w lewo
        '''
        head = self.snake[0]
        dx, dy = self.direction

        # wektory kierunkowe, te same wzory co w _turn
        dir_right = (-dy, dx)
        dir_left  = (dy, -dx)

        #pola o jeden krok w każdą stronę
        ahead = (head[0] + dx,           head[1] + dy)
        right = (head[0] + dir_right[0], head[1] + dir_right[1])
        left  = (head[0] + dir_left[0],  head[1] + dir_left[1])

        #odległość do jedzenia - odrazu znormalizowana do [-1;1] - ujemne zachowuja informacje o kierunku
        food_dx = (self.food[0] - head[0]) / self.size
        food_dy = (self.food[1] - head[1]) / self.size

        # odległość do ściany w trzech kierunkach (względnych)
        def wall_distance(dir_x, dir_y):
            '''Ile kroków od głowy do ściany w danym kierunku.'''
            steps = 0
            x, y = head
            while True:
                x += dir_x
                y += dir_y
                if x < 0 or x >= self.size or y < 0 or y >= self.size:
                    break
                steps += 1
            return steps / self.size   #normalizacja [0;1]

        wall_ahead = wall_distance(dx, dy)
        wall_right = wall_distance(dir_right[0], dir_right[1])
        wall_left  = wall_distance(dir_left[0], dir_left[1])


        state = [
            #grupa 1: niebezpieczeństwo
            self._is_collision(ahead),
            self._is_collision(right),
            self._is_collision(left),

            #grupa 2: kierunek
            dx == -1,   # lewo
            dx == 1,    # prawo
            dy == -1,   # góra
            dy == 1,    # dół

            #grupa 3: gdzie jedzenie
            self.food[0] < head[0],   # jedzenie na lewo
            self.food[0] > head[0],   # jedzenie na prawo
            self.food[1] < head[1],   # jedzenie wyżej (mniejsze y = wyżej)
            self.food[1] > head[1],   # jedzenie niżej

            #grupa 4: odleglosc od jedznia
            food_dx, # odl w poziomie
            food_dy, # odl w pionie

            #grupa 5: odleglosc od sciany
            wall_ahead, # odl do sciany prosto
            wall_right, # odl do sciany w prawo
            wall_left # odl do sciany w lewo
        ]

        return state

    def __str__ (self):
        '''
        Rysowanie planszy.

        Każdy wiersz odpowiada jednej wartości y (pionowa oś).
        Wewnątrz wiersza wybieramy kolumnę po x (pozioma oś).

        Dlatego piszemy grid[y][x], NIE grid[x][y].
        Pierwszy indeks = wiersz = y
        Drugi indeks   = kolumna = x

        '''

        # pusta plansza: lista list znaków
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
        return f"{border}\n{body}\n{border}\nscore: {self.score}  len: {len(self.snake)}  done: {self.done}"

KEYS = {
    'w': (0, -1),
    's': (0,  1),
    'a': (-1, 0),
    'd': (1,  0),
}

def absolute_to_relative(current_dir, desired_dir):
    '''
    Tłumaczy kierunek absolutny (WSAD) na akcję względną (0/1/2).
    Kierunek przeciwny do obecnego jest ignorowany (zwraca 0 = prosto).
    '''
    if desired_dir == current_dir:
        return 0
    dx, dy = current_dir
    right = (-dy, dx)
    left  = (dy, -dx)
    if desired_dir == right:
        return 1
    if desired_dir == left:
        return 2
    return 0 # jesli jest taki jak aktualny to nic



#można odpalić osobno do testów
#nie ma to konfliktu z agentem i trenigniem
if __name__ == "__main__":
    game = SnakeGame(size=10)
    print(game)
    print('\nSterowanie: WSAD   wyjście: q')

    while not game.done:
        cmd = input('> ').strip().lower()
        if cmd == 'q': #wjście
            break
        if cmd not in KEYS: # jesli nie wsad
            print('Nieznana komenda!')
            continue
        action = absolute_to_relative(game.direction, KEYS[cmd]) #zmiana dla czlwieka
        game.step(action) #zaaplikowanie akcji
        print(game) #wyświetlenie planszy po ruchu

    print('Koniec gry!')
