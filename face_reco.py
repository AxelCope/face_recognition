import cv2 as cv
from time import perf_counter


t1_start = perf_counter()
frame = 0

webcam = cv.VideoCapture(0)
NB_IMAGES = 100

print(webcam.isOpened())

if webcam.isOpened():
    while True:
        bImgready, imageFrame = webcam.read()
        if bImgready:
            cv.imshow('My webcam', imageFrame)
        else:
            print('Image indisponible')
        keystroke = cv.waitKey(20)
        if(keystroke == 27):
                break
    webcam.release()
    cv.destroyAllWindow()



