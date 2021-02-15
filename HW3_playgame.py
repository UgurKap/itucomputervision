import pyautogui as pg
import numpy as np
import cv2
import time

time.sleep(3)
found = 0
while found < 19:
    time.sleep(0.5)
    ss = np.array(pg.screenshot())
    ss = cv2.cvtColor(ss, cv2.COLOR_BGR2GRAY)
    
    region = ss[580:764, 510:850]
    cannyEdges = cv2.Canny(region,100,200)
    contours, _ = cv2.findContours(cannyEdges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) >= 2:
        found += 1
        # We have a shape to find out
        region = ss[580:764, 510:860]
        region[region == 206] = 255
        
        # Detect corners
        dst = cv2.cornerMinEigenVal(region, 2, 1)

        # Normalize
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
        
        e = dst_norm_scaled[dst_norm_scaled > 1].shape[0]

        if e > 1110:
            # hexagon
            #print(f"Hexagon {e}")
            pg.keyDown("f")
            pg.keyUp("f")
        elif e > 400:
            # star
            #print(f"Star {e}")
            pg.keyDown("d")
            pg.keyUp("d")
        elif e > 100:
            # triangle
            #print(f"Triangle {e}")
            pg.keyDown("a")
            pg.keyUp("a")
        else:
            # square
            #print(f"Square {e}")
            pg.keyDown("s")
            pg.keyUp("s")