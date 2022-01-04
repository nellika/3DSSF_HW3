import numpy as np
import cv2 as cv

k = np.array([[640, 0, 640],
    [0, 480, 480],
    [0,   0,   1]], dtype=np.float32)

for i in range(1,10):
    # if i == 10: continue
    sed = np.loadtxt('../data/dat_disp/frame_0000'+str(i)+'.dat', unpack = True, delimiter=',',dtype=np.float64)*1000.0

    to_file = ""

    for h in range(0,sed.shape[1]):
        for w in range(0,sed.shape[0]):
            disp = sed[w, h]
            if disp < 1000:
                p_c = np.linalg.inv(k) @ (disp * np.array([h, w, 1]))

                x = p_c[0]
                y = p_c[1]
                z = p_c[2]

                row = str(x) + " " + str(y) + " " + str(z) + "\n"
                to_file += row
    
    f= open("poses/pose_"+str(i)+".xyz","w+")
    f.write(to_file)
    f.close()