import cv2
import numpy as np
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

        lowpassfilter = np.zeros((x, y),np.uint8)
        lowpassfilter[mask] = 1

        return lowpassfilter


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        
        return 0

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""

        
        return 0

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        
        return 0

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""

        
        return 0

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        
        return 0

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

        #FFT
        ft = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(ft)

        #img_float64 = np.float64(self.image)
        #ft = cv2.dft(img_float64, flags=cv2.DFT_COMPLEX_OUTPUT)
        #fshift = np.fft.fftshift(ft)
        #magnitude = 20*np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        magnitude = 10*np.log(np.abs(fshift))
        gray = np.array(magnitude,dtype = np.uint8)

        #Get filter
        mask = self.filter(gray.shape, self.cutoff, self.order)

        #Apply mask
        filtered_dft = gray*mask
        #print(filtered_dft)

        #inverse shift
        i_fshift = np.fft.ifftshift(filtered_dft)
        i_ft = np.fft.ifft2(i_fshift)
        img_back = np.round(np.abs(i_ft))

        #img_back = cv2.idft(i_fshift)
        #img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        print(img_back)

        #Full scale contrast strech
        x, y = img_back.shape
        k = (x * y) - 1
        A = np.min(img_back)
        B = np.max(img_back)
        diff = B-A
        print("b",B)

        fc_stretch = np.zeros((x,y),np.uint8)

        for i in range(0, x):
            for j in range(0, y):
                fc_stretch[i, j] = np.round((k/diff)* (img_back[i, j] - 1) + 0.5)

        print(fc_stretch)
        cv2.imshow('image', fc_stretch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        #return [gray, filtered_dft, self.image]
