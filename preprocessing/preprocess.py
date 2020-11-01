import cv2
import numpy as np
import skimage.transform as trans

def c_stretch(img, absmin, absmax):
    imgmin = img.min()
    imgmax = img.max()
    out = (img - imgmin) * ((absmax - absmin)/(imgmax - imgmin)) + absmin
    out = np.uint8(out)
    return out

def unsharp_mask(img, sigma, m_fac):
    I = np.int16(img)
    L = cv2.GaussianBlur(I, (0, 0), sigmaX=sigma)
    H = np.subtract(I, L)
    out = np.multiply(H, m_fac)
    out = np.add(I, out)
    np.clip(out, 0, 255, out)
    out = np.uint8(out)

    return out

def preprocess(image, flag=0, model=None):
    if (flag == 0):
        img = l_hela_proc(image, model)
    elif (flag == 1):
        img = s_hela_proc(image)
    else:
        img = psc_proc(image)

    return img


def l_hela_proc(image, model):
    shape = image.shape
    flat = image.flatten()
    thresh = 0.05 * shape[0] * shape[1]
    low = 0
    high = 255
    cdf = 0

    for i in range(256):
        cdf += len(flat[flat==i])
        if (cdf > thresh):
            low = i
            break
        
    cdf = 0

    for i in range(255, -1, -1):
        cdf += len(flat[flat==i])
        if (cdf > thresh):
            high = i
            break

    np.clip(image, low, high, image)
    img = cv2.GaussianBlur(image, (0, 0), 1)
    img = c_stretch(img, 0, 255)
    img = unsharp_mask(img, 1, 1.25)

    proc = img / 255
    proc = trans.resize(proc, (256, 256, 1))

    proc = np.reshape(proc, (1, )+proc.shape)


    res = model.predict(proc)
    res = res[0, :, :, 0]
    res = res * 255
    np.clip(res, 0, 255, res)
    res = np.uint8(res)
    _, res = cv2.threshold(res, 150, 255, cv2.THRESH_TOZERO)
    res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv2.THRESH_BINARY, 101, 0)

    res = trans.resize(res, shape)
    res = res * 255
    res = np.uint8(res)

    return res



def s_hela_proc(image):
    res = cv2.GaussianBlur(image, (0, 0), 2)

    res = c_stretch(res, 0, 255)
    res = cv2.morphologyEx(res, cv2.MORPH_TOPHAT, np.ones((41, 41)))

    res = c_stretch(res, 0, 255)


    res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                    101, 0)
#    kernel = 1 / 25 * np.ones((5, 5))
#    res = cv2.filter2D(res, cv2.CV_8U, kernel)

    return res 


def psc_proc(image):
    kernel = 1/9 * np.ones((3, 3))
    res = cv2.filter2D(image, cv2.CV_8U, kernel)
    res = c_stretch(res, 0, 255)
    res = cv2.morphologyEx(res, cv2.MORPH_TOPHAT, np.ones((41, 41)))

    res= c_stretch(res, 0, 255)
    _, res= cv2.threshold(res, 30, 255, cv2.THRESH_TOZERO)
#res = cv2.filter2D(res, cv2.CV_8U, kernel)
    res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                    41, 0) 

#    _, res= cv2.threshold(res, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#res = cv2.filter2D(res, cv2.CV_8U, kernel)

    return res
