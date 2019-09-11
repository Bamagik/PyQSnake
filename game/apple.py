from utils import STEP

class Apple:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x * STEP
        self.y = y * STEP

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))