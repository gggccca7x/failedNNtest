import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def convertImage(image):
    image = image[:,:,:3]
    image = 1- np.mean(image, axis=2)
    paddingAmount = max(28-image.shape[0],28-image.shape[1])
    image = np.pad(image, (0,paddingAmount))
    return image[:28, :28]

# what i found: for this image resolution and font type, 10x23y seems to display letters quite well 
# but I haven't tried longer letters such as 'w'

img = mpimg.imread('invoice2.png')
# print(img.shape)
# shape (2200, 1700, 3) - (Y, X, RGB) - note: x and y reversed from normal
img_cropped_J = img[515:538, 307:317, :] # letter J (23y , 10x)

img_cropped1 = img[515:538, 318:328, :]
img_cropped2 = img[515:538, 328:338, :]
img_cropped3 = img[515:538, 338:348, :]
img_cropped4 = img[515:538, 355:370, :]
img_cropped5 = img[515:538, 1240:1247, :]
img_cropped6 = img[875:900, 75:89, :]

image = convertImage(img_cropped4)
# image = img_cropped6
print(image.shape)

exit()

# print(image)
plt.figure()
plt.imshow(image)

# images = [img_cropped_J, img_cropped1, img_cropped2, img_cropped3]

# for i in images:
#     plt.figure()
#     plt.imshow(i)

plt.show()