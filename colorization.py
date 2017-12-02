import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy import sparse
from scipy import misc
# import colorsys

def getimagesRGB(bw_image_name, marked_image_name):
    print "Get images", bw_image_name, marked_image_name
    bw_image = misc.imread(bw_image_name) #loaded as RGB
    # plt.imshow(bw_image)
    # plt.show()
    marked_image = misc.imread(marked_image_name)
    return bw_image.astype('float')/255, marked_image.astype('float')/255

def getMarkedPix(bw, marked):

    colorPix=  (np.sum(abs(bw - marked), 2) > 0.01)
    # print type(colorPix[0,0])
    return colorPix

def create_YUV_with_marks(bw_image, marked_image):
    print "Create YUV image with marks"
    # print bw_image

    y, _, _ = rgb_to_yiq(bw_image[:, :, 0], bw_image[:, :, 1], bw_image[:, :, 2])
    _, u, v = rgb_to_yiq(marked_image[:, :, 0], marked_image[:, :, 1], marked_image[:, :, 2])

    print bw_image

    bw_yuv = np.zeros(bw_image.shape, dtype='float')
    bw_yuv[:, :, 0] = y
    bw_yuv = bw_yuv
    print bw_yuv
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

def yiq_to_rgb(y, i, q): # modified to allow arrays to work... getting a weird color
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

def Convertwholeimg_YIQ2RGB(YUV):
    R, G, B = yiq_to_rgb(YUV[:, :, 0], YUV[:, :, 1], YUV[:, :, 2])
    print R.shape
    print YUV.shape
    RGB_image = np.zeros(YUV.shape).astype('float')
    RGB_image[:, :, 0] = R
    RGB_image[:, :, 1] = G
    RGB_image[:, :, 2] = B
    # print RGB_image
    # plt.imshow(RGB_image)
    # plt.show()
    return RGB_image

def dude_code():
    bw_image, marked_image = getimagesRGB('example.bmp', 'example_marked.bmp')
    isColored = getMarkedPix(bw_image, marked_image)
    YUV = create_YUV_with_marks(bw_image, marked_image)


    n = YUV.shape[0]  # n = image height
    m = YUV.shape[1]  # m = image width
    image_size = n * m
    indices_matrix = np.arange(image_size).reshape(n, m, order='F').copy()  # indices_matrix as indsM
    wd = 1  # The radius of window around the pixel to assess
    nr_of_px_in_wd = (2 * wd + 1) ** 2  # The number of pixels in the window around one pixel
    max_nr = image_size * nr_of_px_in_wd  # Maximal size of pixels to assess for the hole image
    # (for now include the full window also for the border pixels)
    row_inds = np.zeros(max_nr, dtype=np.int64)
    col_inds = np.zeros(max_nr, dtype=np.int64)
    vals = np.zeros(max_nr)
    # ----------------------------- Interation ----------------------------------- #
    length = 0  # length as len
    pixel_nr = 0  # pixel_nr as consts_len
    # Nr of the current pixel == row index in sparse matrix
    # iterate over pixels in the image
    for j in range(m):
        for i in range(n):

            # If current pixel is not colored
            if (not isColored[i, j]):
                window_index = 0  # window_index as tlen
                window_vals = np.zeros(nr_of_px_in_wd)  # window_vals as gvals

                # Then iterate over pixels in the window around [i,j]
                for ii in range(max(0, i - wd), min(i + wd + 1, n)):
                    for jj in range(max(0, j - wd), min(j + wd + 1, m)):

                        # Only if current pixel is not [i,j]
                        if (ii != i or jj != j):
                            row_inds[length] = pixel_nr
                            col_inds[length] = indices_matrix[ii, jj]
                            window_vals[window_index] = YUV[ii, jj, 0]
                            length += 1
                            window_index += 1

                center = YUV[i, j, 0].copy()  # t_val as center
                window_vals[window_index] = center
                # calculate variance of the intensities in a window around pixel [i,j]
                variance = np.mean(
                    (window_vals[0:window_index + 1] - np.mean(
                        window_vals[0:window_index + 1])) ** 2)  # variance as c_var
                sigma = variance * 0.6  # csig as sigma

                # Indeed, magic
                mgv = min((window_vals[0:window_index + 1] - center) ** 2)
                if (sigma < (-mgv / np.log(0.01))):
                    sigma = -mgv / np.log(0.01)
                if (sigma < 0.000002):  # avoid dividing by 0
                    sigma = 0.000002

                window_vals[0:window_index] = np.exp(
                    -((window_vals[0:window_index] - center) ** 2) / sigma)  # use weighting funtion (2)
                window_vals[0:window_index] = window_vals[0:window_index] / np.sum(
                    window_vals[0:window_index])  # make the weighting function sum up to 1
                vals[length - window_index:length] = -window_vals[0:window_index]

            # END IF NOT COLORED

            # Add the values for the current pixel
            row_inds[length] = pixel_nr

            col_inds[length] = indices_matrix[i, j]
            vals[length] = 1
            length += 1
            pixel_nr += 1

            # END OF FOR i
    # END OF FOR j
    # ---------------------------------------------------------------------------- #
    # ------------------------ After Iteration Process --------------------------- #
    # ---------------------------------------------------------------------------- #
    # Trim to variables to the length that does not include overflow from the edges
    vals = vals[0:length]
    col_inds = col_inds[0:length]
    row_inds = row_inds[0:length]
    # ------------------------------- Sparseness --------------------------------- #
    # sys.exit('LETS NOT SPARSE IT YET')
    A = sparse.csr_matrix((vals, (row_inds, col_inds)), (pixel_nr, image_size))
    # io.mmwrite(os.path.join(dir_path, 'sparse_matrix'), A)
    b = np.zeros((A.shape[0]))
    print "b", b.shape
    colorized = np.zeros(YUV.shape)  # colorized as nI = resultant colored image
    colorized[:, :, 0] = YUV[:, :, 0]
    color_copy_for_nonzero = isColored.reshape(image_size,
                                               order='F').copy()  # We have to reshape and make a copy of the view of an array for the nonzero() to work like in MATLAB
    colored_inds = np.nonzero(color_copy_for_nonzero)  # colored_inds as lblInds
    for t in [1, 2]:
        curIm = YUV[:, :, t].reshape(image_size, order='F').copy()
        b[colored_inds] = curIm[colored_inds]
        print "b", b.shape
        print "curim", curIm.shape
        print colored_inds
        new_vals = linalg.spsolve(A,
                                  b)  # new_vals = linalg.lsqr(A, b)[0] # least-squares solution (much slower), slightly different solutions
        # lsqr returns unexpectedly (ndarray,ndarray) tuple, first is correct so:
        # use new_vals[0] for reshape if you use lsqr
        colorized[:, :, t] = new_vals.reshape(n, m, order='F')

    # ---------------------------------------------------------------------------- #
    # ------------------------------ Back to RGB --------------------------------- #
    # ---------------------------------------------------------------------------- #
    (R, G, B) = yiq_to_rgb(colorized[:, :, 0], colorized[:, :, 1], colorized[:, :, 2])
    colorizedRGB = np.zeros(colorized.shape)
    colorizedRGB[:, :, 0] = R  # colorizedRGB as colorizedIm
    colorizedRGB[:, :, 1] = G
    colorizedRGB[:, :, 2] = B
    plt.imshow(colorizedRGB)
    plt.show()


if __name__ == '__main__':
    print "i can fucking do this!"

    bw_image, marked_image= getimagesRGB('example.bmp', 'example_marked.bmp')
    marked_pixels = getMarkedPix(bw_image, marked_image)
    YUV_image = create_YUV_with_marks(bw_image, marked_image)
    # Convertwholeimg_YIQ2RGB(YUV_image)


    #my code

    colorIm=marked_pixels.copy()


    n = YUV_image.shape[0]
    m = YUV_image.shape[1]
    imgSize = n * m
    print "n, m, imgSize:", n, m, imgSize
    index_matrix = np.arange(imgSize).reshape(n, m, order='F').copy()
    wd = 1



    print index_matrix.shape
    print colorIm.shape


    window_size = imgSize * (2 * wd + 1) ** 2
    col_inds = np.zeros(window_size, dtype=np.int64)
    row_inds = np.zeros(window_size, dtype=np.int64)

    vals = np.zeros(window_size)

    print "col_inds shape: ", col_inds.shape
    print colorIm.shape

    # loops

    # print "gval shape: ", gvals.shape
    # print "gvals", gvals

    pix_num = 0
    len = 0

    for j in range(m):
        for i in range(n):
            # print consts_len, colorIm[i,j]

            if (not colorIm[i, j]):
                #    print consts_len, "NOT"
                wind_idx = 0
                window_values = np.zeros((2 * wd + 1) ** 2)
                #             print max(1, i-wd), min(i+wd,n)
                for ii in range(max(0, i - wd), min(i + wd + 1, n)):
                    for jj in range(max(0, j - wd), min(j + wd + 1, m)):
                        #                     print "ii: ", ii, "jj: ", jj
                        #                     if (ii!=i) | (jj!=j):
                        if (ii != i or jj != j):
                            #                         print ii, i, jj,j, "NOT"
                            #                         print tlen
                            row_inds[len] = pix_num
                            col_inds[len] = index_matrix[ii, jj]

                            window_values[wind_idx] = YUV_image[ii, jj, 0]
                            #                         print YUV_image[ii,jj,0]
                            len = len + 1
                            wind_idx = wind_idx + 1

                            #             print gvals
                            #             print min(gvals)
                            #             print tlen
                center_pixel = YUV_image[i, j, 0]
                #             print YUV_image[i,j,0]
                window_values[wind_idx] = center_pixel
                variance = np.mean((window_values[0:wind_idx + 1] - np.mean(window_values[0:wind_idx + 1])) ** 2)
                Sigma = variance * 0.6

                mgv = min((window_values[0:wind_idx + 1] - center_pixel) ** 2)
                if (Sigma < (-mgv / np.log(0.01))):
                    Sigma = -mgv / np.log(0.01)
                if (Sigma < 0.000002):
                    Sigma = 0.000002

                window_values[0:wind_idx] = np.exp(-(window_values[0:wind_idx] - center_pixel) ** 2 / Sigma)
                window_values[0:wind_idx] = window_values[0:wind_idx] / np.sum(window_values[0:wind_idx])
                vals[len - wind_idx:len] = -window_values[0:wind_idx]

            row_inds[len] = pix_num
            col_inds[len] = index_matrix[i, j]
            vals[len] = 1
            len = len + 1
            pix_num = pix_num + 1

vals = vals[0:len]
col_inds = col_inds[0:len]
row_inds = row_inds[0:len]

print col_inds
print row_inds
print vals

#sparse it up
print row_inds.shape, col_inds.shape, vals.shape
print pix_num, imgSize
# csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
A=csr_matrix((vals, (row_inds, col_inds)), shape=(pix_num, imgSize))
b=np.zeros((A.shape[0]))

newImage = np.zeros(YUV_image.shape)
newImage[:, :, 0] = YUV_image[:, :, 0]  # intensities are still the same; UV changes
label_indices = np.nonzero(colorIm.reshape(imgSize,order='F').copy())
# print label_indices
# print "label indices shape",label_indices.shape
print "b shape", b.shape
print A
# print b
for t in [1,2]:
    print t

    curIm=YUV_image[:,:,t].reshape(imgSize, order='F').copy()
    b[label_indices]=curIm[label_indices]
    print "curIm shape:", curIm.shape
    print "length", len
    lucky_charms= linalg.spsolve(A,b)
#     lucky_charms= np.linalg.solve(A.todense(),b)
    newImage[:,:,t]=lucky_charms.reshape(n,m,order='F')

print "the end"
plt.imshow(Convertwholeimg_YIQ2RGB(newImage))
plt.show()

dude_code()
