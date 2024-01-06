import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import streamlit as st

def CHE(source): # implementation according to: Krutsch, Robert, and David Tenorio (2011) "Histogram equalization." Freescale Semiconductor, Document Number AN4318, Application Note.
    hist = cv2.calcHist([source], [0], None, [256], [0, 256]) # get histogram, for some reason is 2d even though there are only singular values in the second dimension wtf
    hist = np.ndarray.flatten(hist)
    # get cumulative histogram i.e. just the sum of the current value and all previous values
    cumhist = np.cumsum(hist)
    # convert source to gray scale if it is not
    if len(source.shape) == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # all the variables were gonna need later
    m, n = source.shape
    result = np.empty([m, n], dtype=np.uint8)
    # there is something weird in the cited source here, the formula states cdfmin as the minimum value of the distribution function, but in the implementation it is simply replaced by 1.
    # That might be just because of the example use case but i am not sure
    cdfmin = cumhist[np.min(np.argwhere(cumhist!=0))] #the first non-zero element in the cumulative histogram is always going to be the minimum pixel value, so we get the minimum value of the cdf from that intensity position    levels = 256
    pxlcnt = m*n
    levels = 256

    # do the actual che.
    for x in range(m):
        for y in range(n):
            result[x, y] = round((cumhist[source[x, y]]-cdfmin)/(pxlcnt-cdfmin)*(levels-1)) # formula as given in reference
    return result

def QDHE(source): # implementation according to: Chen Hee Ooi and Nor Ashidi Mat Isa (2010), “Quadrants Dynamic Histogram Equalization for Contrast Enhancement”, IEEE Transactions on Consumer Electronics, Vol. 56, No. 4
    hist = cv2.calcHist([source], [0], None, [256], [0, 256])  # get histogram
    hist = np.ndarray.flatten(hist)

    #1 histogram partitioning, compute all the thresholds as index aka brightness value
    m0 = np.min(np.argwhere(hist!=0)) # minimum of range
    m4 = np.max(np.argwhere(hist!=0)) # maximum of range
    m2 = argmedian(hist) # median of range
    m1 = m0+argmedian(hist[m0:m2]) # median of min/med
    m3 = m2+argmedian(hist[m2:m4+1]) # median of med/max

    #2 clipping
    T = np.mean(hist) # clipping threshold
    hist[hist>T] = T # clipping process

    #3 new gray level range allocation
    L = 256
    span = np.array([m1-m0, m2-m1, m3-m2, m4-m3]) # spani in the reference
    spansum = np.sum(span) # sum of span sizes needed in ref fromula
    dynamic_range = np.zeros(4) # rangei in reference
    dynamic_span = np.zeros((4, 2)) # istart and istop in reference
    for i in range(len(dynamic_range)): # compute new ranges according to reference formula(7) step C
        dynamic_range[i] = int((L-1)*span[i]/spansum) # compute new ranges according to reference formula(7) step C
        if i == 0: # compute istart and iend according to fomrulas (8) and (9) of source
            dynamic_span[i, 1] = dynamic_range[i]
        else:
            dynamic_span[i, 0] = dynamic_span[i-1,1]+1
            dynamic_span[i, 1] = dynamic_span[i-1,1]+1+dynamic_range[i]

    #4 histogram equalization
    cdf_s1 = subhist_cdf(hist, m0, m1) # get all subhistogram cdfs for formula (10) in reference
    cdf_s2 = subhist_cdf(hist, m1, m2)
    cdf_s3 = subhist_cdf(hist, m2, m3)
    cdf_s4 = subhist_cdf(hist, m3, m4)
    if len(source.shape) == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    m, n = source.shape
    result = np.zeros((m, n), dtype=np.uint8)
    for x in range(m):
        for y in range(n):
            pxl = source[x, y]
            if pxl < m1:  # equalize for 1st histogram
                # this formula is either faulty in my refernce paper or i am stupid.
                # according to the paper dynamic_range[0] is replaced by istart-iend, i being the name of the subhistogram, this would always yield negative numbers in the first subhistogram though and that is impossible to display in 0-255 values
                result[x, y] = round(dynamic_range[0] * cdf_s1[pxl] + dynamic_span[0, 0]) # modified version of formula (10) in source, for the reason stated above
            elif pxl >= m1 and pxl < m2:  # 2nd subhistogram
                result[x, y] = round(dynamic_range[1] * cdf_s2[int(pxl - m1)] + dynamic_span[1, 0]) # modified version of formula (10) in source, for the reason stated above
            elif pxl >= m2 and pxl < m3:  # 3rd subhistogram
                result[x, y] = round(dynamic_range[2] * cdf_s3[int(pxl - m2)] + dynamic_span[2, 0]) # modified version of formula (10) in source, for the reason stated above
            else:  # sometimes theres minor inaccuracies here so i do a value check to prevent errors in the 4th subhistogram
                target = round(dynamic_range[3] * cdf_s4[int(pxl - m3)] + dynamic_span[3, 0]) # modified version of formula (10) in source, for the reason stated above
                result[x, y] = target if target <= 255 and target >= 0 else 255
    return result

def argmedian(input): # method for returning the index of the median
    sorted = np.argsort(input)
    if len(input)%2==1:
        return sorted[int(len(input)/2)+1]
    else:
        return sorted[int(len(input)/2)]

def subhist_cdf(hist, mstart, mstop): # gets the cumulative histogram function of a subsection of hist.
    indices = np.arange(256)
    cumhist = np.cumsum(hist[np.bitwise_and(indices >= mstart, indices <= mstop)])
    cdf = cumhist / np.max(cumhist, axis=0)
    return cdf

def Histogram_equalization(img):
    """
    Perform histogram equalization on the input image.

    Args:
        img (ndarray): Input image.

    Returns:
        ndarray: Equalized image.
    """
    # Histogram equalization
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[img]

def contrast_limited_adaptive_histogram_equalization_cv(img, clip_limit=3.0, grid_size=(8, 8)):
    """
    Perform contrast limited adaptive histogram equalization on the input image.

    Args:
        img (ndarray): Input image.
        clip_limit (float): Clipping limit.
        grid_size (tuple): Grid size.

    Returns:
        ndarray: Equalized image.
    """
    # Contrast limited adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return clahe.apply(img)

def MSE(im_src, im_res):
    if len(im_src.shape) == 3:
        im_src = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
    if len(im_res.shape) == 3:
        im_res = cv2.cvtColor(im_res, cv2.COLOR_BGR2GRAY)
    m, n = im_src.shape
    return np.linalg.norm(im_res-im_src,'fro')/(m*n)

def PSNR(mse):
    return (10 * math.log(255 ** 2, 10))/mse

def SD(img):
    return np.std(img)


def Generate_histogram(img):
    """
    Generate the histogram of the input image.

    Args:
        img (ndarray): Input image.

    Returns:
        tuple: Tuple containing the histogram and bins.
    """
    # Generate histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    return hist, bins

def png_to_jpg(image):
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    return rgb_im

if __name__ == '__main__':
    st.title("Image Enhancement Based on Histogtram Algorithms")
    # st.sidebar.title("Image Enhancement")
    # st.sidebar.subheader("Select the method")
    col1, col2 = st.columns(2)
    method = st.sidebar.selectbox("Select the method", ("Histogram Equalization", "Contrast Limited Adaptive Histogram Equalization", "Cumulative Histogram Equalization", "Quadratic Dynamic Histogram Equalization"))
    
    if method == "Histogram Equalization":
        image = st.sidebar.file_uploader("Upload the image", type=['jpg', 'png', 'jpeg'])
        if image is not None:
            if image.name.endswith(".png"):
                img = png_to_jpg(image)
                img = np.array(img)
            else:
                img = plt.imread(image)
            col1.image(img, caption='Uploaded Image', use_column_width=True)
            st.write("")
            if st.button('Process'):
                result = Histogram_equalization(img)
                col2.image(result, caption='Processed Image', use_column_width=True)

                # source histogram
                fig1, ax1 = plt.subplots()
                hist1, bins1 = Generate_histogram(img)
                ax1.bar(bins1[:-1], hist1, width=1)
                col1.pyplot(fig1)

                # result histogram
                hist, bins = Generate_histogram(result)
                fig, ax = plt.subplots()
                ax.bar(bins[:-1], hist, width=1)
                col2.pyplot(fig)

                # metrics
                st.write("MSE = ", MSE(img, result))
                st.write("PSNR = ", PSNR(MSE(img, result)))
                st.write("SD = ", SD(result))
    
    elif method == "Contrast Limited Adaptive Histogram Equalization":
        image = st.sidebar.file_uploader("Upload the image", type=['jpg', 'png', 'jpeg'])
        if image is not None:
            if image.name.endswith(".png"):
                img = png_to_jpg(image)
                img = np.array(img)
            else:
                img = plt.imread(image)
            col1.image(img, caption='Uploaded Image', use_column_width=True)
            st.write("")
            if st.button('Process'):
                result = contrast_limited_adaptive_histogram_equalization_cv(img)
                col2.image(result, caption='Processed Image', use_column_width=True)
                
                # source histogram
                fig1, ax1 = plt.subplots()
                hist1, bins1 = Generate_histogram(img)
                ax1.bar(bins1[:-1], hist1, width=1)
                col1.pyplot(fig1)

                # result histogram
                hist, bins = Generate_histogram(result)
                fig, ax = plt.subplots()
                ax.bar(bins[:-1], hist, width=1)
                col2.pyplot(fig)

                # metrics
                st.write("MSE = ", MSE(img, result))
                st.write("PSNR = ", PSNR(MSE(img, result)))
                st.write("SD = ", SD(result))
    
    elif method == "Cumulative Histogram Equalization":
        image = st.sidebar.file_uploader("Upload the image", type=['jpg', 'png', 'jpeg'])
        if image is not None:
            if image.name.endswith(".png"):
                img = png_to_jpg(image)
                img = np.array(img)
                else:
            img = plt.imread(image)
            col1.image(img, caption='Uploaded Image', use_column_width=True)
            st.write("")
            if st.button('Process'):
                result = CHE(img)
                col2.image(result, caption='Processed Image', use_column_width=True)
                
                # source histogram
                fig1, ax1 = plt.subplots()
                hist1, bins1 = Generate_histogram(img)
                ax1.bar(bins1[:-1], hist1, width=1)
                col1.pyplot(fig1)
                
                # result histogram
                hist, bins = Generate_histogram(result)
                fig, ax = plt.subplots()
                ax.bar(bins[:-1], hist, width=1)
                col2.pyplot(fig)

                # metrics
                st.write("MSE = ", MSE(img, result))
                st.write("PSNR = ", PSNR(MSE(img, result)))
                st.write("SD = ", SD(result))
    
    elif method == "Quadratic Dynamic Histogram Equalization":
        image = st.sidebar.file_uploader("Upload the image", type=['jpg', 'png', 'jpeg'])
        if image is not None:
            if image.name.endswith(".png"):
                img = png_to_jpg(image)
                img = np.array(img)
            else:
                img = plt.imread(image)
            col1.image(img, caption='Uploaded Image', use_column_width=True)
            st.write("")
            if st.button('Process'):
                result = QDHE(img)
                col2.image(result, caption='Processed Image', use_column_width=True)

                # source histogram
                fig1, ax1 = plt.subplots()
                hist1, bins1 = Generate_histogram(img)
                ax1.bar(bins1[:-1], hist1, width=1)
                col1.pyplot(fig1)

                # result histogram
                hist, bins = Generate_histogram(result)
                fig, ax = plt.subplots()
                ax.bar(bins[:-1], hist, width=1)
                col2.pyplot(fig)

                # metrics
                st.write("MSE = ", MSE(img, result))
                st.write("PSNR = ", PSNR(MSE(img, result)))
                st.write("SD = ", SD(result))
