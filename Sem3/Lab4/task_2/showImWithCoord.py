import os, cv2 
import numpy as np
import matplotlib.pyplot as plt

palm_im_path = os.path.join(os.getcwd(),'Sem3','Lab4','task_2','palm.png')

print(palm_im_path)
img = cv2.imread(palm_im_path ) 
height,width,channels = img.shape
print(img.ndim)
transparent_img = np.zeros((height, width, channels), dtype=np.uint8)
img_padded = np.pad(img,((0,0),(0,width),(0,0)), mode='constant',constant_values=255) 


plt.imshow(img_padded, cmap='gray')
plt.show()