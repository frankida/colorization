import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy import sparse
import colorsys

def getimages(bw_image_name, marked_image_name):
    print "Get images", bw_image_name, marked_image_name
    bw_image = cv2.imread(bw_image_name)
    bw_imageRGB = cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

    plt.subplot(131)
    plt.imshow(bw_imageRGB)

    marked_image = cv2.imread(marked_image_name)
    marked_imageRGB = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB) # convert BGR to RGB

    plt.subplot(132)
    plt.imshow(marked_imageRGB)

    bw_image_norm = np.copy(bw_imageRGB).astype('float64') / 255
    marked_image_norm= np.copy(marked_imageRGB).astype('float64') / 255

    marked_locations_image = (np.sum(abs(bw_image_norm - marked_image_norm), 2) > 0.01)  # load the image and then get the cells that are different;
    # 0 not marked, 1 is marked

    marked_locations_image = marked_locations_image.astype('float64')

    print marked_locations_image.shape
    print marked_locations_image
    plt.subplot(133)
    plt.imshow(marked_locations_image, cmap="gray")

    plt.show()
    plt.savefig("0_start_images.png", bbox_inches='tight')
    return bw_imageRGB, marked_imageRGB, bw_image_norm, marked_image_norm, marked_locations_image

def create_YUV_norm(bw_image, marked_image):
    print "Create YUV image with marks"

    #testing

    (y, _, _) = colorsys.rgb_to_yiq(bw_image[:, :, 0], bw_image[:, :, 1], bw_image[:, :, 2])
    (_, u, v) = colorsys.rgb_to_yiq(marked_image[:, :, 0], marked_image[:, :, 1], marked_image[:, :, 2])


    sgI = np.zeros(g_name.shape)
    sgI[:, :, 0] = y
    sgI = sgI.astype('float64') / 255
    scI = np.zeros(c_name.shape)
    scI[:, :, 0] = y
    scI[:, :, 1] = u
    scI[:, :, 2] = v
    scI = scI.astype('float64') / 255

    YUV_image_norm = np.zeros(c_name.shape)
    YUV_image_norm[:,:,0]=sgI[:, :, 0]
    YUV_image_norm[:, :, 1] = scI[:, :, 1]
    YUV_image_norm[:, :, 2] = sgI[:, :, 2]


    # bw_image_YUV = cv2.cvtColor(bw_image, cv2.COLOR_RGB2YUV)
    # marked_image_YUV = cv2.cvtColor(marked_image, cv2.COLOR_RGB2YUV)

    # print marked_image_YUV.shape  # background on yuv https://en.wikipedia.org/wiki/YUV
    #
    # y = marked_image_YUV[:, :, 0]  # intensity
    # plt.subplot(131)
    # plt.imshow(y)
    # u = marked_image_YUV[:, :, 1]  # color
    # plt.subplot(132)
    # plt.imshow(u)
    # v = marked_image_YUV[:, :, 2]  # color
    # plt.subplot(133)
    # plt.imshow(v)
    # # plt.show()
    # YUV_image = np.zeros(g_name.shape)
    # # ones = np.ones(g_name.shape)
    # # oneblank= YUV_image[:,:,0]
    #
    # # YUV_image[:, :, 0] = ones[:,:,0]
    # YUV_image[:, :, 0] = bw_image_YUV[:, :, 0]
    # YUV_image[:, :, 1] = marked_image_YUV[:, :, 1]
    # YUV_image[:, :, 2] = marked_image_YUV[:, :, 2]
    #
    # # YUV_image[:, :, 0]  # intensity
    # plt.subplot(131)
    # plt.imshow(YUV_image[:, :, 0])
    # # u = YUV_image[:, :, 1]  # color
    # plt.subplot(132)
    # plt.imshow(YUV_image[:, :, 1])
    # # v = YUV_image[:, :, 2]  # color
    # plt.subplot(133)
    # plt.imshow(YUV_image[:, :, 2])
    # # plt.show()
    # YUV_rgb = cv2.cvtColor((255*YUV_image).astype('uint8'), cv2.COLOR_YUV2RGB)
    # # print YUV_rgb
    # plt.imshow((YUV_rgb))
    # # plt.show()
    # # YUV_image=YUV_image/255
    # YUV_image_norm= YUV_image/255
    # # print YUV_image
    return YUV_image_norm

def dude_code():
    global g_name, c_name
    g_name, c_name, gI, cI, isColored = getimages('example.bmp', 'example_marked.bmp')
    YUV = create_YUV_norm(g_name, c_name)

    def yiq_to_rgb(y, i, q):  # the code from colorsys.yiq_to_rgb is modified to work for arrays
        r = y + 0.948262 * i + 0.624013 * q
        g = y - 0.276066 * i - 0.639810 * q
        b = y - 1.105450 * i + 1.729860 * q
        r[r < 0] = 0
        r[r > 1] = 1
        g[g < 0] = 0
        g[g > 1] = 1
        b[b < 0] = 0
        b[b > 1] = 1
        return (r, g, b)

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

#
#
#     #my code
#     g_name, c_name, gI, cI, colorIm = getimages('example.bmp', 'example_marked.bmp')
#     YUV_image = create_YUV_norm(g_name, c_name)
#
#     plt.show()
#     # print YUV_image.shape
#
#     max_d = np.floor(np.log(min(YUV_image.shape[0],YUV_image.shape[1] )) / np.log(2) - 2)
#     # print max_d
#     window_function = 2 ** (max_d - 1)
#     iu = np.floor(YUV_image.shape[0] / (window_function) * (window_function)).astype('int')
#     ju = np.floor(YUV_image.shape[1] / (window_function) * (window_function)).astype('int')
#     print iu, ju
#     id = 0
#     jd = 0
#     print id, iu, jd, ju
#     colorIm = colorIm[:iu, :ju].copy()
#     YUV_image = YUV_image[:iu, :ju].copy()
#     print colorIm.shape
#
#     # Start getColorExact code
#
#     n = YUV_image.shape[0]
#     m = YUV_image.shape[1]
#     imgSize = n * m
#     print "n, m, imgSize:", n, m, imgSize
#
#     newImage = np.zeros(YUV_image.shape)
#     newImage[:, :, 0] = YUV_image[:, :, 0]  # intensities are still the same; UV changes
#     #
#     # # print newImage[:,:,0]
#     # plt.imshow(newImage[:, :, 0], cmap='gray')
#     # plt.show()
#     # newImageRGB=cv2.cvtColor((255*newImage).astype('uint8'),cv2.COLOR_YUV2BGR)
#     # plt.imshow(newImageRGB)
#     # plt.show()
#     index_matrix = np.arange(imgSize).reshape(n, m, order='F').copy()
#     print index_matrix.shape
#     print colorIm.shape
#
#     wd = 1
#
#     window_size = imgSize * (2 * wd + 1) ** 2
#     col_inds = np.zeros(window_size).astype('int64')
#     row_inds = np.zeros(window_size).astype('int64')
#
#     vals = np.zeros(window_size).astype('int64')
#
#     print "col_inds shape: ", col_inds.shape
#     print colorIm.shape
#
#     # loops
#     gvals = np.zeros((2 * wd + 1) ** 2)
#     print "gval shape: ", gvals.shape
#     print "gvals", gvals
#
#     consts_len = 0
#     len = 0
#
#     for j in range(m):
#         for i in range(n):
#             # print consts_len, colorIm[i,j]
#
#             if (not colorIm[i, j]):
#                 #    print consts_len, "NOT"
#                 tlen = 0
#                 #             print max(1, i-wd), min(i+wd,n)
#                 for ii in range(max(0, i - wd), min(i + wd + 1, n)):
#                     for jj in range(max(0, j - wd), min(j + wd + 1, m)):
#                         #                     print "ii: ", ii, "jj: ", jj
#                         #                     if (ii!=i) | (jj!=j):
#                         if (ii != i or jj != j):
#                             #                         print ii, i, jj,j, "NOT"
#                             #                         print tlen
#                             row_inds[len] = consts_len
#                             col_inds[len] = index_matrix[ii, jj]
#
#                             gvals[tlen] = YUV_image[ii, jj, 0]
#                             #                         print YUV_image[ii,jj,0]
#                             len = len + 1  # need to move - different from matlab code
#                             tlen = tlen + 1
#
#                             #             print gvals
#                             #             print min(gvals)
#                             #             print tlen
#                 t_val = YUV_image[i, j, 0]
#                 #             print YUV_image[i,j,0]
#                 gvals[tlen] = t_val
#                 c_var = np.mean((gvals[0:tlen + 1] - np.mean(gvals[0:tlen + 1])) ** 2)
#                 csig = c_var * 0.6
#
#                 mgv = min((gvals[0:tlen + 1] - t_val) ** 2)
#                 if (csig < (-mgv / np.log(0.01))):
#                     csig = -mgv / np.log(0.01)
#                 if (csig < 0.000002):
#                     csig = 0.000002
#
#                 gvals[0:tlen] = np.exp(-(gvals[0:tlen] - t_val) ** 2 / csig)
#                 gvals[0:tlen] = gvals[0:tlen] / np.sum(gvals[0:tlen])
#                 vals[len - tlen:len] = -gvals[0:tlen]
#
#             row_inds[len] = consts_len
#             col_inds[len] = index_matrix[i, j]
#             vals[len] = 1
#             len = len + 1
#             consts_len = consts_len + 1
#
# vals = vals[0:len]
# col_inds = col_inds[0:len]
# row_inds = row_inds[0:len]
#
# #sparse it up
# print row_inds.shape, col_inds.shape, vals.shape
# print consts_len, imgSize
# # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
# A=csr_matrix((vals, (row_inds, col_inds)), shape=(consts_len, imgSize))
# b=np.zeros((A.shape[0]))
#
# label_indices = np.array(np.nonzero(colorIm.flatten('F')))
# # print label_indices
# print "label indices shape",label_indices.shape
# print "b shape", b.shape
# print A
# # print b
# for t in [1,2]:
#     print t
#
#     curIm=YUV_image[:,:,t].reshape(imgSize, order='F').copy()
#     b[label_indices]=curIm[label_indices]
#     print "curIm shape:", curIm.shape
#     print "length", len
#     lucky_charms= linalg.spsolve(A,b)
# #     lucky_charms= np.linalg.solve(A.todense(),b)
#     newImage[:,:,t]=lucky_charms.reshape(n,m,order='F')
#     newImage2=newImage*255
# # print np.nonzero(newImage[:,:,1])
# # print np.nonzero(newImage[:,:,2])
#     newImage2=newImage2.astype('uint8')
#     newImageRGB=cv2.cvtColor(newImage2,cv2.COLOR_YUV2RGB)
#     plt.imshow(newImageRGB)
#     # plt.show()
#
# # print newImage
# newImage=newImage*255
# # print np.nonzero(newImage[:,:,1])
# # print np.nonzero(newImage[:,:,2])
# newImage=newImage.astype('uint8')
# newImageRGB=cv2.cvtColor(newImage,cv2.COLOR_YUV2RGB)
# plt.imshow(newImageRGB)
# plt.show()




dude_code()
