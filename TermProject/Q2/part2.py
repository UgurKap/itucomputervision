import time
import dlib
import sys
import pyautogui as pg
import numpy as np
import cv2 as cv
from copy import deepcopy

R, L, U, D = range(4) # Enum

first_move = True

predictor_name = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_name)

worried = cv.imread("ssr.png")
img = worried
img = img[600:, 1200:, :]
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
rectangles = detector(gray)
points = predictor(gray, rectangles[0])
w_points_x = [points.part(k).x for k in range(68)]
w_points_y = [points.part(k).y for k in range(68)]

def move(move_let, sleep_time=0.54):
    keys = ["D", "A", "W", "S"]
    pg.keyDown(keys[move_let])
    time.sleep(sleep_time)
    pg.keyUp(keys[move_let])
    time.sleep(sleep_time)


def is_worried(img):
    img = img[600:, 1200:, :]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rectangles = detector(gray)
    points = predictor(gray, rectangles[0])
    points_x = [points.part(k).x for k in range(68)]
    points_y = [points.part(k).y for k in range(68)]
    a = np.sum(np.abs(np.array(w_points_x) - np.array(points_x)))
    b = np.sum(np.abs(np.array(w_points_y) - np.array(points_y)))
    if (a+b) > 3500:
        return False
    return True

def fix_pos():
    time.sleep(1)
    ss = pg.screenshot()
    ss = np.array(ss)
    img = ss[90:350, 500:850]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray = cv.GaussianBlur(gray,(5,5),0)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    row_pos, col_pos = np.where(dst != 0)
    r1, r2, c1, c2 = min(row_pos), max(row_pos), min(col_pos), max(col_pos)
    
    if np.sum(np.abs(np.array([r1, r2, c1, c2]) - np.array([25, 222, 82, 284]))) > 20:
        if c1 > 82:
            move(R, (c1 - 82)/100)
        else:
            move(L, (82 - c1)/100)
        if r1 > 25:
            move(D, (r1-25)/100)
        else:
            move(U, (25-r1)/100)
    
    return r1, r2, c1, c2


class State():
    def __init__(self, cur_pos, world):
        self.cur_pos = cur_pos
        self.world = deepcopy(world)
        if self.world[cur_pos] == 1:
            print("Win!")
            pg.press("esc")
            sys.exit(0)
        self.world[cur_pos] = -100
        self.move_coords = []
    
    def next_states(self):
        self.right_pos = (self.cur_pos[0], self.cur_pos[1] + 1)
        self.left_pos = (self.cur_pos[0], self.cur_pos[1] - 1)
        self.up_pos = (self.cur_pos[0] - 1, self.cur_pos[1])
        self.down_pos = (self.cur_pos[0] + 1, self.cur_pos[1])
       
        
        candidates = [self.right_pos, self.left_pos, self.up_pos, self.down_pos]
        self.move_coords = candidates
        dirs = []
        for dr, pos in enumerate(candidates):
            if self.world[pos[0], pos[1]] > -1:
                dirs.append(dr)
                
        return dirs
    
    def probe(self):
        global first_move
        fix_pos()
        self.moves = self.next_states()
        ss = np.array(pg.screenshot())
        
        time.sleep(2)
        if R in self.moves:
            move(R)
            if first_move:
            	first_move = False
            	time.sleep(2)
            ss = pg.screenshot()
            move(L)
            if is_worried(np.array(ss)):
                self.moves.remove(R)
                self.world[self.right_pos[0], self.right_pos[1]] = -100
            
        if L in self.moves:
            move(L)
            ss = pg.screenshot()
            move(R)
            if is_worried(np.array(ss)):
                self.moves.remove(L)
                self.world[self.left_pos[0], self.left_pos[1]] = -100
            
        if U in self.moves:
            move(U)
            ss = pg.screenshot()
            move(D)
            if is_worried(np.array(ss)):
                self.moves.remove(U)
                self.world[self.up_pos[0], self.up_pos[1]] = -100
              
        if D in self.moves:
            move(D)
            ss = pg.screenshot()
            move(U)
            if is_worried(np.array(ss)):
                self.moves.remove(D)
                self.world[self.down_pos[0], self.down_pos[1]] = -100
                
    def play(self):
        if len(self.moves) == 0:
            return -1
        
        for m in self.moves:
            st = State(self.move_coords[m], self.world)
            move(m, 2.0)
            st.probe()
            p = st.play()

            if p == -1:
                if m == R or m == U:
                    move(m+1, 2.0)
                else:
                    move(m-1, 2.0)

        return -1


maze = np.zeros((7, 12))
maze[1:4, -2] = 1
maze[:, -1] = -100
maze[4:,4:] = -100
maze[0, :] = -100
maze[:, 0] = -100
maze[-1, :] = -100
a = State((5, 2), maze)
print("Starting to move in 3 seconds")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("Go!")
a.probe()
a.play()