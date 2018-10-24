from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import pylab
def convolve2d(image, kernel):

    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()
    return output

img = io.imread('test1.jpeg')    # Load the image
img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)
# Adjust the contrast of the image by applying Histogram Equalization
#image_equalized = exposure.equalize_adapthist(img/np.max(np.abs(img)), clip_limit=0.03)
#plt.imshow(image_equalized, cmap=plt.cm.gray)
#plt.axis('off')
#plt.show()
# Convolve the sharpen kernel and the image
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
image_edge = convolve2d(img,kernel)
print ('\n First 5 columns and rows of the image_sharpen matrix: \n')
print( image_sharpen[:5,:5]*255)
# Plot the filtered image
plt.imshow(image_edge, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
# Adjust the contrast of the filtered image by applying Histogram Equalization
image_edge_equalized = exposure.equalize_adapthist(image_edge/np.max(np.abs(image_edge)), clip_limit=0.03)
plt.imshow(image_edge_equalized, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
