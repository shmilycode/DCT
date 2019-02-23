import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./test.jpg",0)

img1 = img.astype('float')

img_dct=cv2.dct(img1)
img_dct_log=np.log(abs(img_dct))

img_recor=cv2.idct(img_dct)

recor_tmp=img_dct[0:10,0:10]
recor_tmp2=np.zeros(img.shape)
recor_tmp2[0:10,0:10]=recor_tmp

img_recor1=cv2.idct(recor_tmp2)

plt.subplot(221)
plt.imshow(img,'gray')

#plt.subplot(222)
#plt.imshow(img_recor,'gray')

plt.subplot(223)
plt.imshow(img_recor1,'gray')

#plt.subplot(224)
#plt.imshow(img_dct_log)
#
plt.show()
