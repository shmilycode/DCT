import cv2
import numpy as np
import matplotlib.pyplot as plt

block_width=8
dct_arg=1

img = cv2.imread("./test.jpg",0)
img_float = img.astype('float')

x_block_count=img.shape[0]/block_width
y_block_count=img.shape[1]/block_width
print("block count: %sx%s"%(x_block_count,y_block_count))

idct_img_recor=np.zeros(img.shape)
for y_num in range(0, y_block_count):
    for x_num in range(0, x_block_count):
        x_offset=x_num*block_width
        y_offset=y_num*block_width
        block_img=img_float[x_offset:x_offset+block_width,y_offset:y_offset+block_width]

        #dct
        block_img_dct=cv2.dct(block_img)
        recor_tmp1=block_img_dct[0:dct_arg,0:dct_arg]
        recor_tmp2=np.zeros(block_img.shape)
        recor_tmp2[0:dct_arg,0:dct_arg]=recor_tmp1

        #idct
        block_img_idct=cv2.idct(recor_tmp2)
        idct_img_recor[x_offset:x_offset+block_width,y_offset:y_offset+block_width]=block_img_idct

plt.subplot(221)
plt.imshow(img,'gray')

plt.subplot(222)
plt.imshow(idct_img_recor,'gray')
plt.show()
