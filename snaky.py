import random, pygame, sys
from pygame.locals import *

FPS = 150
WINDOWWIDTH = 200
WINDOWHEIGHT = 200
CELLSIZE = 40
assert WINDOWWIDTH % CELLSIZE == 0, "Window width must be a multiple of cell size."
assert WINDOWHEIGHT % CELLSIZE == 0, "Window height must be a multiple of cell size."
CELLWIDTH = int(WINDOWWIDTH / CELLSIZE)
CELLHEIGHT = int(WINDOWHEIGHT / CELLSIZE)

#             R    G    B
WHITE     = (255, 255, 255)
BLACK     = (  0,   0,   0)
RED       = (255,   0,   0)
GREEN     = (  0, 255,   0)
BRIGREEN  = (150, 255, 150)
DARKGREEN = (  0, 155,   0)
DARKGRAY  = ( 40,  40,  40)
BGCOLOR = BLACK

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

HEAD = 0 # syntactic sugar: index of the worm's head
pygame.init()
FPSCLOCK = pygame.time.Clock()
DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
pygame.display.set_caption('Snake')
episode = 0
class gameState:
    def __init__(self):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, episode
        episode = episode + 1
        self.startx = random.randint(3, 4)
        self.starty = random.randint(3, 4)
        self.wormCoords = [{'x': self.startx,     'y': self.starty},
                    {'x': self.startx - 1, 'y': self.starty},
                    {'x': self.startx - 2, 'y': self.starty}]
        self.direction = RIGHT
        self.totalscore = 0

        # Start the apple in a random place.
        self.apple = self.getRandomLocation(self.wormCoords)

    def frameStep(self, action):
        image_data, reward, done = self.runGame(action)

        return image_data, reward, done

    def runGame(self, action):
        global episode
        pygame.event.pump()
        
        self.pre_direction = self.direction
        #action[0] up
        #action[1] down
        #action[2] left
        #action[3] right
        if (action[0] == 1) and self.direction != DOWN:
            self.direction = UP
        elif (action[1] == 1) and self.direction != UP:
            self.direction = DOWN
        elif (action[2] == 1) and self.direction != RIGHT:
            self.direction = LEFT
        elif (action[3] == 1) and self.direction != LEFT:
            self.direction = RIGHT
        
        # check if the worm has hit itself or the edge
        reward = -0.1
        done = False
        if self.wormCoords[HEAD]['x'] == -1 or self.wormCoords[HEAD]['x'] == CELLWIDTH or self.wormCoords[HEAD]['y'] == -1 or self.wormCoords[HEAD]['y'] == CELLHEIGHT:
            done = True
            #self.__init__() # game over
            reward = -1
            image_data = pygame.surfarray.array3d(pygame.display.get_surface())
            return image_data, reward, done
        for self.wormBody in self.wormCoords[1:]:
            if self.wormBody['x'] == self.wormCoords[HEAD]['x'] and self.wormBody['y'] == self.wormCoords[HEAD]['y']:
                done = True
                #self.__init__() # game over
                reward = -1
                image_data = pygame.surfarray.array3d(pygame.display.get_surface())
                return image_data, reward, done

        # check if worm has eaten an apple
        if self.wormCoords[HEAD]['x'] == self.apple['x'] and self.wormCoords[HEAD]['y'] == self.apple['y']:
            # don't remove worm's tail segment
            self.apple = self.getRandomLocation(self.wormCoords) # set a new apple somewhere
            reward = 2
            self.totalscore = self.totalscore + 1
        else:
            del self.wormCoords[-1] # remove worm's tail segment

        # move the worm by adding a segment in the direction it is moving
        if not self.examine_direction(self.direction, self.pre_direction):
            self.direction = self.pre_direction
        if self.direction == UP:
            self.newHead = {'x': self.wormCoords[HEAD]['x'], 'y': self.wormCoords[HEAD]['y'] - 1}
        elif self.direction == DOWN:
            self.newHead = {'x': self.wormCoords[HEAD]['x'], 'y': self.wormCoords[HEAD]['y'] + 1}
        elif self.direction == LEFT:
            self.newHead = {'x': self.wormCoords[HEAD]['x'] - 1, 'y': self.wormCoords[HEAD]['y']}
        elif self.direction == RIGHT:
            self.newHead = {'x': self.wormCoords[HEAD]['x'] + 1, 'y': self.wormCoords[HEAD]['y']}
        self.wormCoords.insert(0, self.newHead)
        DISPLAYSURF.fill(BGCOLOR)
        #self.drawGrid()
        self.drawWorm(self.wormCoords)
        self.drawApple(self.apple)
        #self.drawScore(len(self.wormCoords) - 3)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, done

    def examine_direction(self, temp , direction):
        if direction == UP:
            if temp == DOWN:
                return False
        elif direction == RIGHT:
            if temp == LEFT:
                return False
        elif direction == LEFT:
            if temp == RIGHT:
                return False
        elif direction == DOWN:
            if temp == UP:
                return False
        return True
    
    def retScore(self):
        global episode
        tmp1 = self.totalscore
        tmp2 = episode
        self.__init__()
        return tmp1, tmp2 

    def drawPressKeyMsg(self):
        pressKeySurf = BASICFONT.render('Press a key to play.', True, DARKGRAY)
        pressKeyRect = pressKeySurf.get_rect()
        pressKeyRect.topleft = (WINDOWWIDTH - 200, WINDOWHEIGHT - 30)
        DISPLAYSURF.blit(pressKeySurf, pressKeyRect)


    def checkForKeyPress(self):
        if len(pygame.event.get(QUIT)) > 0:
            terminate()

        keyUpEvents = pygame.event.get(KEYUP)
        if len(keyUpEvents) == 0:
            return None
        if keyUpEvents[0].key == K_ESCAPE:
            terminate()
        return keyUpEvents[0].key

    def terminate(self):
        pygame.quit()
        sys.exit()

    def getRandomLocation(self, worm):
        temp = {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}
        while self.test_not_ok(temp, worm):
            temp = {'x': random.randint(0, CELLWIDTH - 1), 'y': random.randint(0, CELLHEIGHT - 1)}
        return temp

    def test_not_ok(self, temp, worm):
        for body in worm:
            if temp['x'] == body['x'] and temp['y'] == body['y']:
                return True
        return False

    def showGameOverScreen(self):
        pygame.event.get() # clear event queue
        return

    def drawScore(self, score):
        scoreSurf = BASICFONT.render('Score: %s' % (score), True, WHITE)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (WINDOWWIDTH - 120, 10)
        DISPLAYSURF.blit(scoreSurf, scoreRect)


    def drawWorm(self, wormCoords):
        a = 0
        for coord in wormCoords:
            x = coord['x'] * CELLSIZE
            y = coord['y'] * CELLSIZE
            
            wormSegmentRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
            if a == 0:
                pygame.draw.rect(DISPLAYSURF, BRIGREEN, wormSegmentRect)
            else:
                pygame.draw.rect(DISPLAYSURF, DARKGREEN, wormSegmentRect)
            a = a + 1
            wormInnerSegmentRect = pygame.Rect(x + 4, y + 4, CELLSIZE - 8, CELLSIZE - 8)
            pygame.draw.rect(DISPLAYSURF, GREEN, wormInnerSegmentRect)


    def drawApple(self, coord):
        x = coord['x'] * CELLSIZE
        y = coord['y'] * CELLSIZE
        appleRect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
        pygame.draw.rect(DISPLAYSURF, RED, appleRect)


    def drawGrid(self):
        for x in range(0, WINDOWWIDTH, CELLSIZE): # draw vertical lines
            pygame.draw.line(DISPLAYSURF, DARKGRAY, (x, 0), (x, WINDOWHEIGHT))
        for y in range(0, WINDOWHEIGHT, CELLSIZE): # draw horizontal lines
            pygame.draw.line(DISPLAYSURF, DARKGRAY, (0, y), (WINDOWWIDTH, y))
