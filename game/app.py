
import pygame
from pygame.locals import *
from player import Player
from apple import Apple
from utils import STEP
from random import randint
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

STARTING_LENGTH = 3

class Game:
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

    def is_collision(self, x1, y1, x2, y2):
        return x1 == x2 and y1 == y2
    
    def is_wall_collision(self, x, y):
        return x < 0 or x > self.window_width or y < 0 or y > self.window_height

class App:
    window_width = 600
    window_height = 600
    player = 0
    apple = 0
    running = True

    def __init__(self):
        self.running = True
        self._display_surf = None
        self._image_surf = None
        self.game = Game(self.window_width, self.window_height)
        self.player = Player(STARTING_LENGTH, self.window_height * self.window_width)
        self.apple = Apple(randint(0, self.window_width//STEP - 1), randint(0,self.window_height//STEP - 1))

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.window_width, self.window_height), pygame.HWSURFACE)
        pygame.display.set_caption('Pygame Snake AI Game')
        self.running = True
        self._image_surf = pygame.Surface((10,10))
        self._image_surf.fill((255,255,255))
        self._apple_surf = pygame.Surface((10,10))
        self._apple_surf.fill((0,255,0))

        return get_observations()

    def get_observations(self):
        observs = np.zeros((self.window_width, self.window_height))
        observs[self.apple.x, self.apple.y] = 1
        
        for i in range(1,self.player.length):
            observs[self.player.x[i], self.player.y[i]] = -1

        return observs

    def on_event(self, event):
        if event.type == QUIT:
            self.running = False

    def on_loop(self):
        self.player.update()

        # does snake eat apple?
        for i in range(0,self.player.length):
            if self.game.is_collision(self.apple.x,self.apple.y,self.player.x[i], self.player.y[i]):
                self.apple.x = randint(0, self.window_width//STEP - 1) * STEP
                self.apple.y = randint(0, self.window_height//STEP - 1) * STEP
                print("Apple Location, (", self.apple.x, ", ", self.apple.y, ")" )
                self.player.length = self.player.length + 1
 
 
        # does snake collide with itself?
        lost = False
        for i in range(1,self.player.length):
            lost = self.game.is_collision(self.player.x[0],self.player.y[0],self.player.x[i], self.player.y[i])
            if lost: break

        if self.game.is_wall_collision(self.player.x[0], self.player.y[0]) or lost:
            print("You lose! Collision: ")
            print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
            print("x[" + str(i) + "] (" + str(self.player.x[i]) + "," + str(self.player.y[i]) + ")")
            self.running = False

    def on_render(self):
        self._display_surf.fill((20,20,15))
        self.player.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_key(self, keys):
        if keys[K_RIGHT]:
            self.player.move_right()

        if keys[K_LEFT]:
            self.player.move_left()

        if keys[K_UP]:
            self.player.move_up()
        
        if keys[K_DOWN]:
            self.player.move_down()

        if keys[K_ESCAPE]:
            self.running = False

    def on_execute(self):
        if self.on_init() == False:
            self.running = False
        
        while self.running:
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            self.on_key(keys)

            self.on_loop()
            self.on_render()

            time.sleep (20.0 / 1000.0)
        self.on_cleanup()


if __name__ == "__main__":
    app = App()
    # app.on_execute()

    tf.reset_default_graph()
    # set up... might not be correct yet
    inputs1 = tf.placeholder(shape=[1,60*60], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([60,60], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argsmax(Qout, 1)
    # loss function set up
    nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
    loss = td.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.trian.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()
    y = .99
    e = 0.1
    num_episodes = 2000
    # list of rewards and steps per episode
    jList = []
    rList = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # init instead of execute
            app.on_init()
            s = 3 # starting location
            rAll = 0
            d = False
            j = 0
            # Q network
            while j < 200:
                j += 1
                # choose action greedily
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1:np.identity(60*60)[s:s+1]})
                if np.random.random(1) < e:
                    a[0] = [K_DOWN, K_UP, K_LEFT, K_RIGHT][randint(0,4)]
                # get new state and reward from environment
                s1 = app.get_observations()
                r = app.player.length - STARTING_LENGTH
                d = not app.running # if running, continue, if not running, failed
                # Obtain Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1:np.identity(600*600)[s1:s1+1]})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y*maxQ1
                # Train our network using Target and predicted
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1:np.identity(60*60)[s: s+1], nextQ: targetQ})
                rAll += r
                s = s1
                if d == True:
                    # reduce chance of random action as we train
                    e = 1. / ((i / 50) + 10)
                    break
            jList.append(j)
            rList.append(rAll)
        print("Percent of successful episodes: " + str(sum(rList)/num_episodes0) + "%")
