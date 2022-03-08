import sys
import numpy as np
import cv2

oldx = oldy = -1
img = np.ones((750, 1200, 3), dtype=np.uint8) * 255

def on_mouse(event, x, y, flags, param):
    global oldx, oldy

    if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
        oldx, oldy = x, y

    elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (0, 0, 0), 3, cv2.LINE_AA)
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (255, 255, 255), 30, cv2.LINE_AA)
        cv2.imshow('painter', img)
        oldx, oldy = x, y


def main():
    cv2.namedWindow('painter')
    cv2.setMouseCallback('painter', on_mouse, img)

    cv2.imshow('painter', img)
    cv2.waitKey()

    cv2.destroyAllWindows()