import sys
import numpy as np
import cv2, test_ui_1

oldx = oldy = -1
clocX, clocY = 0, 0
iimg = np.ones((750, 1200, 3), dtype=np.uint8) * 255
virtual_event = 'default'
pen_toggle = True

def on_mouse(event, x, y, flags, param):
    global oldx, oldy, virtual_event

    if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:

        oldx, oldy = x, y

    elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
        pass
    elif event == cv2.EVENT_MOUSEMOVE:
        print(x, y)
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(iimg, (oldx, oldy), (x, y), (0, 0, 0), 3, cv2.LINE_AA)
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.line(iimg, (oldx, oldy), (x, y), (255, 255, 255), 30, cv2.LINE_AA)
        cv2.imshow('painter', iimg)
        oldx, oldy = x, y


def on_hand(event, x, y, flags, param):
    global oldx, oldy

    clocX, clocY = x, y
    if virtual_event == 'leftclick':
        oldx, oldy = clocX, clocY
    elif virtual_event == 'drag':
        if pen_toggle: cv2.line(iimg, (oldx, oldy), (int(clocX), int(clocY)), (0, 0, 0), 3, cv2.LINE_AA)
        else: cv2.line(iimg, (oldx, oldy), (x, y), (255, 255, 255), 30, cv2.LINE_AA)
        oldx, oldy = clocX, clocY



def main(param):
    global virtual_event, pen_toggle
    virtual_event = param
    cv2.namedWindow('painter')
    if virtual_event == 'rightclick': pen_toggle = not pen_toggle
    if virtual_event in ['drag', 'leftclick']:
        cv2.setMouseCallback('painter', on_hand, iimg)
    else:
        cv2.setMouseCallback('painter', on_mouse, iimg)
    cv2.imshow('painter', iimg)
    if cv2.waitKey(1) == ord('w'):
        cv2.destroyWindow(iimg)