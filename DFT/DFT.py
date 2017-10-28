import cv2
import numpy as np

# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries


class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        x = matrix.shape[0]
        y = matrix.shape[1]
        N = x

        #Fourier Transform matrix:
        ft = np.zeros([x,y], complex)

        for u in range(0, x):
            for v in range(0, y):
                sum_ft = 0
                for i in range(0, x):
                    for j in range(0, y):

                        sum_ft = sum_ft + matrix[i, j] * (np.cos(((2*np.pi)/N)*(u*i + v*j)) - 1j*np.sin(((2*np.pi)/N)*(u*i + v*j)))

                ft[u, v] = sum_ft

        return ft

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        x = matrix.shape[0]
        y = matrix.shape[1]
        N = x

        # Inverse Fourier Transform matrix:
        ift = np.zeros([x, y], complex)

        for i in range(0, x):
            for j in range(0, y):
                sum_ift = 0
                for u in range(0, x):
                    for v in range(0, y):
                        sum_ift = sum_ift + matrix[u, v] * (np.cos(((2 * np.pi) / N) * (u * i + v * j)) + 1j * np.sin(((2 * np.pi) / N) * (u * i + v * j)))

                ift[i, j] = sum_ift

        return ift/(x*x)



    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""


        x = matrix.shape[0]
        y = matrix.shape[1]
        N = x

        #Fourier Transform matrix:
        dct = np.zeros([x,y])

        for u in range(0, x):
            for v in range(0, y):
                sum_ft = 0
                for i in range(0, x):
                    for j in range(0, y):

                        sum_ft = sum_ft + matrix[i, j] * (np.cos(((2*np.pi)/N)*(u*i + v*j)))

                dct[u, v] = sum_ft

        return dct


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        x = matrix.shape[0]
        y = matrix.shape[1]

        # Magnitude matrix:
        dft = np.zeros([x, y], float)

        for i in range(0, x):
            for j in range(0, y):
                dft[i, j] = np.sqrt(np.square(np.real(matrix[i, j])) + np.square(np.imag(matrix[i, j])))

        # x = np.fft.ifft2(matrix)
        # print(x)


        return dft