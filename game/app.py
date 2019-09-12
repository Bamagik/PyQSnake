# Learning expiriment using a tutorial snake game and a tutorial q-learning function
# and jamming them together into an experiment with snake and q-learning
# 
# this is purely for self teaching purposes and is my initial deep dive into what is deep learning
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
ACTIONS = [K_DOWN, K_UP, K_LEFT, K_RIGHT]
MAX_DISTANCE = 120 # relates to screen height/width and STEP

class Game:
    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

    def is_collision(self, x1, y1, x2, y2):
        return x1 == x2 and y1 == y2
    
    def is_wall_collision(self, x, y):
        return x < 0 or x > self.window_width or y < 0 or y > self.window_height

    def get_distance(self, x1, y1, x2, y2):
        return (np.abs(x1 - x2) + np.abs(y1 - y2))//STEP

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

        return self.get_observations()

    def get_observations(self):
        observs = np.zeros((self.window_width//STEP, self.window_height//STEP))
        observs[self.apple.x//STEP - 1, self.apple.y//STEP - 1] = 1
        x = self.player.x[0]//STEP - 1
        y = self.player.y[0]//STEP - 1
        if not (x > self.window_width // STEP or x < 0 or y > self.window_height // STEP or y < 0):
            observs[x,y] = -2
        for i in range(1,self.player.length):
            observs[self.player.x[i]//STEP - 1, self.player.y[i]//STEP - 1] = -1

        return observs.reshape((1,-1)) # reshape to vector

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
                # print("Apple Location, (", self.apple.x, ", ", self.apple.y, ")" )
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

    def on_key(self, key, loop=False):
        if key == K_RIGHT:
            self.player.move_right()
        if key == K_LEFT:
            self.player.move_left()
        if key == K_UP:
            self.player.move_up()
        if key == K_DOWN:
            self.player.move_down()

        if loop:
            self.on_loop()
            self.on_render()

    def on_execute(self):
        self.on_init()
        while self.running:
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[K_RIGHT]:
                self.on_key(K_RIGHT)

            if keys[K_LEFT]:
                self.on_key(K_LEFT)

            if keys[K_UP]:
                self.on_key(K_UP)
            
            if keys[K_DOWN]:
                self.on_key(K_DOWN)

            if keys[K_ESCAPE]:
                self.running = False

            self.on_loop()
            self.on_render()

            time.sleep (20.0 / 1000.0)
        self.on_cleanup()

    def get_reward(self):
        r = self.game.get_distance(self.player.x[0], self.player.y[0], self.apple.x, self.apple.y)
        r += 240 * (self.player.length - STARTING_LENGTH)
        r -= (not self.running) * 200
        # r = self.player.length - STARTING_LENGTH
        return r

if __name__ == "__main__":
    # app = App()
    # app.on_execute()

    tf.reset_default_graph()
    # set up... might not be correct yet
    inputs1 = tf.placeholder(shape=[1,60*60], dtype=tf.float32)
    V = tf.Variable(tf.random_uniform([60*60, 60], 0, 0.01))
    middle = tf.matmul(inputs1, V)
    W = tf.Variable(tf.random_uniform([60,4], 0, 0.01))
    Qout = tf.matmul(middle, W)
    predict = tf.argmax(Qout, 1)
    # loss function set up
    nextQ = tf.placeholder(shape=[1,4], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()
    y = .99
    e = 0.1
    num_episodes = 100
    # list of rewards and steps per episode
    jList = []
    rList = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # init instead of execute
            app = App()
            s = app.on_init()
            rAll = 0
            d = False
            j = 0
            # Q network
            while j < 300:
                time.sleep (10.0 / 1000.0)
                j += 1
                # choose action greedily
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1:s})
                if np.random.random(1) < e:
                    a[0] = randint(0,3)
                # do action
                app.on_key(ACTIONS[a[0]], True)
                # get new state and reward from environment
                d = not app.running # if running, continue, if not running, failed
                s1 = app.get_observations()
                r = app.get_reward()
                if r >= rAll + 240:
                    j = 0 # reset the counter that ends if staying still too long
                # Obtain Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1:s1})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y*maxQ1
                # Train our network using Target and predicted
                sess.run([updateModel], feed_dict={inputs1:s, nextQ: targetQ})
                rAll += r
                s = s1
                if d == True:
                    # reduce chance of random action as we train
                    e = 1. / ((i / 50) + 10)
                    break
            jList.append(j)
            rList.append(rAll)
        print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
