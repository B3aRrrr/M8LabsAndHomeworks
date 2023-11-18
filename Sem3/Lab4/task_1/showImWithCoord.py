import os, cv2 
import numpy as np
import matplotlib.pyplot as plt

railway_im_path = os.path.join(os.getcwd(),'Sem3','Lab4','task_1','railway.jpeg')

# print(railway_im_path)
img = cv2.imread(railway_im_path, 0)
img_copy = img.copy()
plt.imshow(img, cmap='gray')
plt.show()