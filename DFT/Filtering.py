import cv2
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt
# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv


class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order


    #Modified to add dummy parameter for order
    def get_ideal_low_pass_filter(self, shape, cutoff, dummy):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""

        cutoff = int(cutoff)
        x = shape[0]
        y = shape[1]
        #center pixel
        crow = int(x/2)
        ccol = int(y/2)

        y1, x1 = np.ogrid[-crow:x - crow, -ccol:y - ccol]
        mask = x1 * x1 + y1 * y1 <= cutoff * cutoff

        lowpassfilter = np.zeros((x, y, 2),np.uint8)
        lowpassfilter[mask] = 1

        #print(lowpassfilter[crow-5:crow+5,ccol-5:ccol+5])

        return lowpassfilter


    def get_ideal_high_pass_filter(self, shape, cutoff, dummy):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""
        #Hint: May be one can use the low pass filter function to get a high pass mask

        lowpassfilter = self.get_ideal_low_pass_filter(shape, cutoff, dummy)
        highpassfilter  = 1 - lowpassfilter

        return highpassfilter


    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""

        cutoff = int(cutoff)
        x, y = shape

        # center pixel
        crow = int(x / 2)
        ccol = int(y / 2)

        butterworth_filter = np.zeros((x, y, 2))

        for i in range(0, x):
            for j in range(0, y):

                dist = np.sqrt(np.square(i-crow)+np.square(j-ccol))
                butterworth_filter[i, j, :] = 1/(1 + np.power((dist/cutoff),2*int(order)))

        #print(butterworth_filter[crow - 5:crow + 5, ccol - 5:ccol + 5])

        return butterworth_filter


    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""
        #Hint: May be one can use the low pass filter function to get a high pass mask

        cutoff = int(cutoff)
        x, y = shape

        # center pixel
        crow = int(x / 2)
        ccol = int(y / 2)

        butterworth_high_pass_filter = np.zeros((x, y, 2))

        for i in range(0, x):
            for j in range(0, y):
                dist = np.sqrt(np.square(i - crow) + np.square(j - ccol))
                if dist == 0:
                    butterworth_high_pass_filter[i, j, :] = 0

                else:
                    butterworth_high_pass_filter[i, j, :] = 1 / (1 + np.power((cutoff / dist), 2 * int(order)))


        #butterworth_high_pass_filter = 1 - self.get_butterworth_low_pass_filter(shape, cutoff, order)
        #return butterworth_high_pass_filter

        return butterworth_high_pass_filter


    def get_gaussian_low_pass_filter(self, shape, cutoff, dummy):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""

        cutoff = int(cutoff)
        x, y = shape
        # center pixel
        crow = int(x / 2)
        ccol = int(y / 2)

        gaussian_filter = np.zeros((x, y, 2))

        for i in range(0, x):
            for j in range(0, y):
                dist = np.square(i - crow) + np.square(j - ccol)
                gaussian_filter[i, j, :] = math.pow(math.e,(-1*dist)/(2*cutoff*cutoff))

        return gaussian_filter


    def get_gaussian_high_pass_filter(self, shape, cutoff, dummy):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""
        #Hint: May be one can use the low pass filter function to get a high pass mask

        gaussian_high_pass_filter = 1 - self.get_gaussian_low_pass_filter(shape, cutoff, dummy)
        return gaussian_high_pass_filter


    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """


        return image


    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """

        #print(self.image.dtype)
        img_float32 = np.float32(self.image)

        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        #save DFT
        magnitude = 10 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        gray = np.array(magnitude, dtype=np.uint8)

        #mask
        mask = self.filter(self.image.shape, self.cutoff, self.order)

        # apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        i_dft = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        img_back = i_dft.astype(np.uint8)

        #img_back = cv2.magnitude(i_dft[:, :, 0], i_dft[:, :, 1])

        # Full scale contrast stretch
        x, y = img_back.shape
        img_back2 = np.array(img_back)
        fc_stretch = np.ones((x, y))
        #fc_stretch1 = np.ones((x, y),np.uint8)
        k = 255 #k-1
        A = np.min(img_back2)
        B = np.max(img_back2)
        diff = B - A
        print("a",A)
        print("b", B)
        print("diff",diff)
        #print(img_back2)
        #print("*******")
        #print(img_back2.dtype)
        print(fc_stretch.dtype)

        for i in range(0, x):
            for j in range(0, y):
                fc_stretch[i, j] = (np.round(((k / diff) * (img_back2[i, j] - 1)) + 0.5))
                #fc_stretch1[i, j] = int(np.round(((k / diff) * (img_back2[i, j] - 1)) + 0.5))

        print(fc_stretch)
        print("*****")
        #print(fc_stretch1)

        plt.subplot(131), plt.imshow(self.image, cmap='gray')
        plt.title('Input'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img_back, cmap='gray')
        plt.title('IFT '), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(fc_stretch, cmap='gray')
        plt.title('Filtered image'), plt.xticks([]), plt.yticks([])
        plt.show()

        cv2.imshow('image',(fc_stretch))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        return [gray, img_back, fc_stretch]
