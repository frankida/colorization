import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy import misc

def getimagesRGB(bw_image_name, marked_image_name):
    print "Get images", bw_image_name, marked_image_name
    bw_image = misc.imread(bw_image_name) #loaded as RGB
    # plt.imshow(bw_image)
    # plt.show()
    marked_image = misc.imread(marked_image_name)
    return bw_image.astype('float')/255, marked_image.astype('float')/255

def getMarkedPix(bw, marked):
    print "Get the marked pixels"
    if bw.shape[2]>3:
        bw=bw[:,:,0:3]
    colorPix=  (np.sum(abs(bw - marked), 2) > 0.01)
    # print type(colorPix[0,0])
    return colorPix

def create_YUV_with_marks(bw_image, marked_image):
    print "Create YUV image with marks"
    # print bw_image

    y, place, place = rgb_to_yiq(bw_image[:, :, 0], bw_image[:, :, 1], bw_image[:, :, 2])
    place, u, v = rgb_to_yiq(marked_image[:, :, 0], marked_image[:, :, 1], marked_image[:, :, 2])

    # print bw_image

    bw_yuv = np.zeros(bw_image.shape, dtype='float')
    bw_yuv[:, :, 0] = y
    bw_yuv = bw_yuv
    # print bw_yuv
    marked_yuv = np.zeros(marked_image.shape,dtype='float')
    marked_yuv[:, :, 0] = y
    marked_yuv[:, :, 1] = u
    marked_yuv[:, :, 2] = v
    marked_yuv = marked_yuv

    YUV_image = np.zeros(marked_image.shape,dtype='float')
    YUV_image[:,:,0]= bw_yuv[:, :, 0]
    YUV_image[:, :, 1] = marked_yuv[:, :, 1]
    YUV_image[:, :, 2] = marked_yuv[:, :, 2]

    return YUV_image

# YIQ: used by composite video signals (linear combinations of RGB)
# Y: perceived grey level (0.0 == black, 1.0 == white)
# I, Q: color components

def rgb_to_yiq(r, g, b): # from color.sys https://github.com/python/cpython/blob/2.7/Lib/colorsys.py
    y = 0.299*r + 0.587*g + 0.114*b
    i = 0.596*r - 0.274*g - 0.322*b
    q = 0.211*r - 0.523*g + 0.312*b
    return (y, i, q)

def yiq_to_rgb(y, i, q): # from color.sys https://github.com/python/cpython/blob/2.7/Lib/colorsys.py
    #modified for arrays
    r = y + 0.948262*i + 0.624013*q
    g = y - 0.276066*i - 0.639810*q
    b = y - 1.105450*i + 1.729860*q
    r[r < 0.0]=0.0
    g[g < 0.0]= 0.0
    b[b < 0.0]= 0.0
    r[r > 1.0]= 1.0
    g[g > 1.0]= 1.0
    b[b > 1.0]= 1.0
    return (r, g, b)

def Convertwholeimg_YIQ2RGB(YUV, show=False):
    R, G, B = yiq_to_rgb(YUV[:, :, 0], YUV[:, :, 1], YUV[:, :, 2])
    RGB_image = np.zeros(YUV.shape).astype('float')
    RGB_image[:, :, 0] = R
    RGB_image[:, :, 1] = G
    RGB_image[:, :, 2] = B
    # print RGB_image
    if show:
        plt.imshow(RGB_image)
        plt.show()
    return RGB_image

def get_sigma(window_values):
    variance= np.mean((window_values-np.mean(window_values))**2)
    sigma = variance *.4  # sqrt causes the image to blur
    return sigma

def colorization_setup(YUV_image, windowpx):
    print "Setting up colorization"
    n = YUV_image.shape[0]
    m = YUV_image.shape[1]
    imgSize = n * m
    windowpx_by_image = imgSize * windowpx
    index_matrix = np.arange(imgSize).reshape(n, m, order='F')
    col_index = np.zeros(windowpx_by_image)
    row_index = np.zeros(windowpx_by_image)
    values = np.zeros(windowpx_by_image)
    return n,m, imgSize, index_matrix, row_index, col_index, values

def optimize(YUV_image, marked_pixels, A):
    print "Sparse matrix - optimize GO"
    newImage = np.zeros(YUV_image.shape)
    newImage[:, :, 0] = YUV_image[:, :, 0]  # intensities are still the same; UV changes
    marked_index = np.array(np.nonzero(marked_pixels.flatten('F')))

    b1 = np.zeros((A.shape[0]))
    I_img = YUV_image[:, :, 1].reshape(imgSize, order='F')
    b1[marked_index] = I_img[marked_index] # load only the values that are marked
    i_lucky_charms = linalg.spsolve(A, b1)
    newImage[:, :, 1] = i_lucky_charms.reshape(n, m, order='F')
    b2 = np.zeros((A.shape[0]))
    Q_img = YUV_image[:, :, 2].reshape(imgSize, order='F')
    b2[marked_index] = Q_img[marked_index]
    q_lucky_charms = linalg.spsolve(A, b2)
    newImage[:, :, 2] = q_lucky_charms.reshape(n, m, order='F')
    return newImage

def get_index_values(row_index, col_index, values, YUV_image, index_matrix,  window):
    pixel_in_image_idx = 0
    counter = 0

    for j in range(m):
        for i in range(n):

            if (not marked_pixels[i, j]):
                neighbor_idx = 0
                neighbor_values = np.zeros(window)

                for ii in range(max(0, i - 1), min(i + 2, n)):
                    for jj in range(max(0, j - 1), min(j + 2, m)):
                        if (ii != i) | (jj != j):
                            # print "Not center"
                            row_index[counter] = pixel_in_image_idx
                            col_index[counter] = index_matrix[ii, jj]

                            neighbor_values[neighbor_idx] = YUV_image[ii, jj, 0]
                            counter = counter + 1
                            neighbor_idx = neighbor_idx + 1

                center_pixel = YUV_image[i, j, 0]  # intensity at i,j
                neighbor_values[neighbor_idx] = center_pixel
                # print neighbor_idx
                neighbor_weights = get_neighbor_weights(center_pixel,neighbor_idx, neighbor_values)

                values[counter - neighbor_idx:counter] = -neighbor_weights[0:neighbor_idx]  # loading values

            row_index[counter] = pixel_in_image_idx
            col_index[counter] = index_matrix[i, j]
            values[counter] = 1
            # print counter
            counter = counter + 1
            pixel_in_image_idx = pixel_in_image_idx + 1

    return values, col_index, row_index
    # return values[0:counter], col_index[0:counter], row_index[0:counter]


def get_neighbor_weights(center_pixel,neighbor_idx,  neighbor_values):

    Sigma = get_sigma(neighbor_values)
    if (Sigma < 0.000001): Sigma = 0.000001
    neighbor_values[0:neighbor_idx] = np.exp(
        -(neighbor_values[0:neighbor_idx] - center_pixel) ** 2 / Sigma)  # weighting function
    neighbor_values[0:neighbor_idx] = neighbor_values[0:neighbor_idx] / np.sum(
        neighbor_values[0:neighbor_idx])  # make it sum to one
    return neighbor_values


if __name__ == '__main__':
    print "I can * do this!"

#    bw_image, marked_image= getimagesRGB('example.bmp', 'example_marked.bmp')
    #bw_image, marked_image = getimagesRGB('yellow_bw.bmp', 'yellow_m.bmp')
    bw_image, marked_image = getimagesRGB('cats_bw.bmp', 'cats_res.bmp')

    marked_pixels = getMarkedPix(bw_image, marked_image)
    YUV_image = create_YUV_with_marks(bw_image, marked_image)

    window=9 #nine pixels total

    n, m, imgSize, index_matrix, row_index, col_index, values= \
        colorization_setup(YUV_image, windowpx=window)

    values, col_index, row_index = get_index_values(row_index, col_index, values, YUV_image,
                                                    index_matrix, window)
# csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    A=csr_matrix((values, (row_index, col_index)), shape=(imgSize, imgSize))
    newImage= optimize(YUV_image, marked_pixels, A)
    newImageRGB= Convertwholeimg_YIQ2RGB(newImage, show=False)
    plt.imsave("colorized_image.bmp", newImageRGB)
    print "The End"
