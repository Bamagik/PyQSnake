from utils import STEP

class Player:
    x = [0]
    y = [0]
    direction = 0
    length = 3

    update_count_max = 2
    update_count = 0

    def __init__(self, length, screen_size):
        self.length = length
        for i in range(0, screen_size):
            self.x.append(0)
            self.y.append(0)

        for i in range(length, -1, -1):
            self.x[i] = (length - i) * STEP

    def update(self):
        self.update_count += 1
        if self.update_count > self.update_count_max:
            for i in range(self.length-1, 0, -1):
                # print("self.x[", str(i), "] = self.x[", str(i-1), "]")
                self.x[i] = self.x[i-1]
                self.y[i] = self.y[i-1]

            if self.direction == 0:
                self.x[0] += STEP
            if self.direction == 1:
                self.x[0] -= STEP
            if self.direction == 2:
                self.y[0] -= STEP
            if self.direction == 3:
                self.y[0] += STEP

            self.update_count = 0

    def move_right(self):
        self.direction = 0
    
    def move_left(self):
        self.direction = 1

    def move_up(self):
        self.direction = 2

    def move_down(self):
        self.direction = 3

    def draw(self, surface, image):
        for i in range(0, self.length):
            surface.blit(image, (self.x[i], self.y[i]))