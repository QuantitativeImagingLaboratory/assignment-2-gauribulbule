# Report

Question 1 ->


Question 2 - > Filtering:

To compute fourier transform and inverse fourier transform of an image, I have used opencv (cv2) functions. Full stretch
contrast stretch is applied on final Filtered image.

Low pass filters:

They are used for smoothing, to remove noise. It allows to pass only frequencies lower than a cutoff (radius).

As the cutoff radius is increased, blurring of an image is reduced.
In Ideal low pass filters, cutoff of 30 and below gives a very blurred image. Filtered image with cutoff 50 and above is
much better.

Butterworth filter results are better than ideal filters. I observed a ringing effect in an ideal and butterworth
filtered images. Ringing effect increased with increasing order of filter. Filtered image with cutoff 30 and order 2 has
less ringing effect than that of image with cutoff 30 and order 10.

There were no ringing effects in gaussian filtered image. Filtered images with Gaussian are smoother
than that of butterworth for a same cutoff. For a given image, Lenna0.jpg, filtered image is much clear than that of
original image. Noise if the form of dots is removed.

High pass Filters:

They are used for image sharpening. They allow to pass only frequencies greater than a cutoff (radius)

To apply a full contrast stretch on filtered image, inverse fourier image (float32) is convered to grayscale (uint8)
image. Hence grayscale image looks different than that of inverse fourier transformed image. Also, a negative of an image
is taken to improve the visibility of a processed image.

Butterworth filter results are better rhan that of ideal high pass filter for a particular cutoff. Ex. for cutoff 50 and
order 2, Butterworth gave better result than ideal high pass filter.

For both the filters as ringing effect reduced with increasing cutoff value and constant order 2 for Butterworth.
But as the cutoff increased more than 50, filtered image quality became poorer; edges were not so clear as it allowed less
higher frequencies to pass. Same was the case for Gaussian. But overall, I felt that for a particular cutoff,
Gaussian performed better than Butterworth. Edges were prominent in image filtered using Gaussian.