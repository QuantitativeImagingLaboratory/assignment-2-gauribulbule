# Report

Question 1 -> DFT

Four given transformations (FFT, inverse FT, magnitude & cosine) are implemented using mathematical formulae. No in-built
library functions are used. Forward fourier transformed matrix values are complex conjugates of each other. I have used
this idea to compute half of the matrix values from their complex conjugate part. Hence reducing the iterations by half.

Question 2 - > Filtering:

To compute fourier transform and inverse fourier transform of an image, I have used opencv (cv2) functions. Full scale
contrast stretch is applied on final filtered grayscale image. Negative of image is taken only for high pass filters.

I also observed that with opencv, when a high pass filter is applied, final output image looks different than an image
when a numpy functions are used. For opencv edges of an image looked better than when numpy functions are applied. So I
have used opencv.

Method:

1) Converted image to float32
2) Applied fourier transform (dft()) to image and then shift
3) Calculate the mask (filter) and apply it to an image in step 2
4) Calculated inverse shift and inverse fourier transform
5) Took magnitude of the image in step 4 and converted it to uint8 to perform full scale contrast stretch
6) Took negative of contrast stretched image if high pass filter is applied (final filtered image)
7) Returned final filtered image, magnitude of image in step2 (dft) and magnitude of image in step 3 (filtered dft)

I) Low pass filters:

They are used for smoothing, to remove noise. It allows to pass only frequencies lower than a cutoff (radius).

As the cutoff radius is increased, blurring of an image is reduced.
In Ideal low pass filters, cutoff of 30 and below gives a very blurred image. Filtered image with cutoff 50 and above is
much better.

Butterworth filter results are better than ideal filters. I observed a ringing effect in an ideal and butterworth
filtered images. Ringing effect increased with increasing order of filter. Filtered image with cutoff 30 and order 2 has
less ringing effect than that of image with cutoff 30 and order 10.

There were no ringing effects in gaussian filtered image. Filtered images with Gaussian are smoother
than that of butterworth for a same cutoff. For a given image, Lenna0.jpg, filtered image is much clear than that of
original image. Noise in the form of dots is removed.

As a part of post-processing, I did not need to take a negative of image after a contrast stretch.

II) High pass Filters:

They are used for image sharpening. They allow to pass only frequencies greater than a cutoff (radius)

To apply a full contrast stretch on filtered image, inverse fourier image (float32) is converted to a grayscale (uint8)
image. Hence grayscale image looks different than that of inverse fourier transformed image. Also, a negative of an image
is taken to improve the visibility of a processed image.

Butterworth filter results are better than that of ideal high pass filter for a particular cutoff. Ex. for cutoff 50 and
order 2, Butterworth gave better result than ideal high pass filter.

For both the filters as ringing effect reduced with increasing cutoff value and constant order 2 for Butterworth.
But as the cutoff increased more than 50, filtered image quality became poorer; edges were not so clear as it allowed less
higher frequencies to pass. Same was the case for Gaussian. But overall, I felt that for a particular cutoff,
Gaussian performed better than Butterworth. Edges were prominent in image filtered using Gaussian.