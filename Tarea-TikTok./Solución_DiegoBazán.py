import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
import matplotlib.pylab as plt
from imutils.video import VideoStream
from pygame import mixer

mixer.init()

mixer.music.load('tom.mp3')
mixer.music.play(-1)
video_capture = cv2.VideoCapture(0)
cambios = 0
while True:
  _,frame = video_capture.read()
  yonas1 = cv2.cvtColor(frame[:,:600], cv2.COLOR_RGB2BGR)
  yonas2 = frame[:,600:]
  yonassuper = cv2.hconcat((yonas1,yonas2))
  cv2.imshow('Procesamiento Digital de Imágenes', yonassuper)
  print(yonas2[300])
  for i in range(len(yonas2[300])):
      for j in range(len(yonas2[300][i])):
            if yonas2[300][i][j] < 100 and cambios <= 2000:
              cambios += 1;
            elif cambios >= 0 : 
                cambios -=1
  if cambios > 1000: mixer.music.pause()
  if cambios < 1000: mixer.music.unpause()
  print(cambios)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()
