import cv2 as cv
import os

dirCascadeFiles = r'haarcascades_cuda'
classCascadeFacial = cv.CascadeClassifier(os.path.join(dirCascadeFiles, "haarcascade_frontalface_default.xml"))
classCascadeEyes = cv.CascadeClassifier(os.path.join(dirCascadeFiles, "haarcascade_eye.xml"))
classCascadeSmile = cv.CascadeClassifier(os.path.join(dirCascadeFiles, "haarcascade_profileface.xml"))


def facialDetectionAndMark(_image, _classCascade):
    imgreturn = _image.copy()
    gray = cv.cvtColor(imgreturn, cv.COLOR_BGR2GRAY)
    faces = _classCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv.rectangle(imgreturn, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return imgreturn

def videoDetection(_haarclass):
    webcam = cv.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            bImgReady, imageframe = webcam.read()  # get frame per frame from the webcam
            if bImgReady:
                face = facialDetectionAndMark(imageframe, _haarclass)
                cv.imshow('My webcam', face)  # show the frame
            else:
                print('No image available')
            keystroke = cv.waitKey(20)  # Wait for Key press
            if keystroke == 27:
                break  # if key pressed is ESC then escape the loop
    except Exception as e:
        print(f"Error: {e}")
    finally:
        webcam.release()
        cv.destroyAllWindows()

videoDetection(classCascadeSmile)
