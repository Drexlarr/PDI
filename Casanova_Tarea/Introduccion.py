import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from pygame import mixer
import matplotlib.pylab as plt

lena_gris = cv2.imread('./Lena.png')
# Cambia a la imagen de BGR a GRAY
lena_gris = cv2.cvtColor(lena_gris, cv2.COLOR_BGR2GRAY)
lena_colores = cv2.imread('./Lena_colores.png')


# cv2.imshow("Lenita",lena_gris)
# cv2.imshow('Lenita colorida', lena_colores)
# cv2.waitKey()

mixer.init()
mixer.music.load('./Eminem - The Way I Am [HD Best Quality].mp3')

baseImage = []
cont = True
playing = False

def thresholdImage(frame):
    _, binaryImage = cv2.threshold(frame, 125, 255, cv2.THRESH_BINARY)
    return binaryImage


def contDiferencies(baseFrame, currentFrame, dim):
    # binaryImage = thresholdImage(currentFrame)
    row, col = baseFrame.shape
    r_point = row//2
    c_point = col//2

    w = plt.gradydiffweight(baseFrame,c_point,r_point,'Pesos',25)
    cv2.imshow(w)



def screem_2(frame):
    rows, cols, channels = frame.shape
    div = cols//2

    myFrame = frame[:, :div]
    grayFrame = cv2.cvtColor(frame[:, div:], cv2.COLOR_RGB2GRAY)
    global baseImage
    global cont

    if cont:
        baseImage = np.flip(grayFrame, 1)
        cont = False
    contDiferencies(baseImage, grayFrame, rows*cols)

    comb = np.zeros(shape=(rows, cols, channels), dtype=np.uint8)

    comb[:rows, :div] = myFrame
    comb[:rows, div:] = grayFrame[:, :, None]

    return comb


def screem_3(frame):
    div = 213
    myFrame = frame[:, :div]
    # Azul
    yoAzulito = cv2.cvtColor(frame[:, div:div*2], cv2.COLOR_RGB2BGR)
    # Blanco y Negro
    yoByN = cv2.cvtColor(frame[:, div*2:], cv2.COLOR_RGB2GRAY)

    rows, cols, channels = frame.shape

    comb = np.zeros(shape=(rows, cols, channels), dtype=np.uint8)

    comb[:rows, :div] = myFrame
    comb[:rows, div:div*2] = yoAzulito
    comb[:rows, div*2:] = yoByN[:, :, None]

    return comb


video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()

    comb = screem_2(frame)
    comb = np.flip(comb, 1)
    cv2.imshow('Tarea: Video-Music', comb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

video_capture = cv2.VideoCapture(0)
