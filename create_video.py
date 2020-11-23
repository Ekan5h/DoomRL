import cv2
import numpy as np
import glob

frameSize = (640,480)

out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, frameSize)

for filename in range(len(glob.glob('gif/*'))):
    print(filename)
    img = cv2.imread("gif/"+str(filename)+".png")
    out.write(img)

out.release()
