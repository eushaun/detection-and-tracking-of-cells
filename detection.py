import numpy as np 
import cv2 as cv
import glob

from preprocessing.preprocess import *
from preprocessing.model import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction


def create_training(path, f_flag):
    # get mask names
    for i in range(1,3):
        for mask_path in glob.glob("{}/Sequence {} Masks/*".format(path, i)):
            # extract txxx.tif from string
            img_seq = mask_path.split()[2][6:10]

            # preprocess training images
            img = cv.imread("{}/Sequence {}/{}.tif".format(path, i, img_seq), 0)
            img = preprocess(img, f_flag)
            mask = cv.imread(mask_path, -1)

            # need to create the folders in Windows first
            # t001-1.tif = t001 from Sequence 1
            # basically just dumped all the training images into one folder
            cv.imwrite("{}/training/image/{}-{}.tif".format(path, img_seq, i), img)
            cv.imwrite("{}/training/mask/{}-{}.tif".format(path, img_seq, i), mask)

def training_augmentation(path, target_size=(256,256), batch_size=32,
                        rotation_range=30, zoom_range=0.15, width_shift_range=0.2, 
                        height_shift_range=0.2, shear_range=0.15):
    # file directory
    directory = "{}/training".format(path)

    # set arguments to augment training data
    data_gen_args = dict(
        rotation_range = rotation_range,
        zoom_range = zoom_range,
        width_shift_range = width_shift_range,
        height_shift_range = height_shift_range,
        shear_range = shear_range,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = "nearest"
    )

    # set same seed for both image and mask to ensure identical random augmentations 
    seed = 420

    # augment training data "on-the-fly"
    img_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    img_generator = img_datagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode='grayscale',
        classes = ['image'],
        class_mode=None,
        batch_size=batch_size,
        seed=seed,
        save_to_dir=None,
        save_prefix='image'
    )
    mask_generator = mask_datagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        color_mode='grayscale',
        classes = ['mask'],
        class_mode=None,
        batch_size=batch_size,
        seed=seed,
        save_to_dir=None,
        save_prefix='mask'
    )
    training_generator = zip(img_generator, mask_generator)

    return training_generator

def postprocess(img, f_flag): 
    # postprocessing now generalised for all images
    if f_flag == 0:
        maxArea = 5000
        kernel = 1 / (31 * 31) * np.ones((31, 31))
        min_dist = 25
        minArea = 500
    elif f_flag == 1:
        maxArea = 500
        kernel = 1 / (11 * 11) * np.ones((11, 11))
        min_dist = 3
        minArea = 50
    else:
        maxArea= 100
        kernel = 1 / (11 * 11) * np.ones((11, 11))
        min_dist = 3
        minArea = 20

    _, contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Mean filtered image, used to find maxima for splitting
    # undersegmented regions
    mean_img = cv.filter2D(img, cv.CV_8U, kernel)
    contour_res = []
    rects = []

    for contour in contours:
        rect = cv.boundingRect(contour)

        # if the contour is likely to be undersegmented (we may improve)
        if (cv.contourArea(contour) > maxArea):
            x, y, w, h = rect
#            sub_im = img[y:y+h, x:x+w]
            sub_mean = mean_img[y:y+h, x:x+w]
            seed_im = np.zeros(img.shape, np.uint8)
            cv.drawContours(seed_im, [contour], 0, 255, -1)
            sub_im = seed_im[y:y+h, x:x+w]

#            if (f_flag != 2):
#                for [[p_x,p_y]] in contour:
#                    seed_im[p_y-y, p_x-x] = sub_im[p_y-y, p_x-x]

#               sub_im = reconstruction(seed_im, sub_im, selem=np.ones((3, 3)))

            # Find peaks in the region's mean
            local_maxi = peak_local_max(sub_mean, indices=False,
                                        min_distance=min_dist)

            markers = ndi.label(local_maxi)[0]
            labels = watershed(-sub_mean, markers, mask=sub_im,
                       watershed_line=True)

            labels[labels > 1] = 1
            labels = np.uint8(labels)
            labels = cv.erode(labels, np.ones((3,3)))
            _, sub_cnt, _ = cv.findContours(labels, cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)


            # Add each new contour to output
            for cnt in sub_cnt:
                if (cv.contourArea(cnt) > minArea):
                    cnt = cnt + [x, y]
                    n_x,n_y,n_w,n_h = cv.boundingRect(cnt)
                    contour_res.append(cnt)
                    rects.append((n_x, n_y, n_x+n_w, n_y+n_h))
        else:
            if (cv.contourArea(contour) > minArea):
                x, y, w, h = rect
                contour_res.append(contour)
                rects.append((x, y, x+w, y+h))

#    return np.uint8(res)
    return contour_res, rects


def draw_bounding_box(img, mask, color=(0,0,255), thickness=1):
    # resize image to (256, 256)
    img = cv.resize(img, (256, 256))

    # get all contours in the mask
    _, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # returns (x, y) of one corner, width and height of each bounding box
    bounding_boxes = [cv.boundingRect(contour) for contour in contours]

    # loop and draw rectangles on the original image
    for i in range(len(bounding_boxes)):
        start = (bounding_boxes[i][0], bounding_boxes[i][1])
        end = (bounding_boxes[i][0]+bounding_boxes[i][2], bounding_boxes[i][1]+bounding_boxes[i][3])
        # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img = cv.rectangle(img, start, end, color, thickness)

    return img


def other_draw_bounding_box(img, rects, color=(0,0,255), thickness=1):
    res = img.copy()

    for x,y,w,h in rects:
        res = cv.rectangle(res, (x,y), (x+w,y+h), color, thickness)

    return res

def cell_detection(path, f_flag, model):
    for i in range(1,5):
        for img_path in glob.glob("{}/Sequence_{}/*".format(path, i)):
            # extract txxx.tif from img_path
            img_seq = img_path.split()[1][2:6]

            # read and process testing images
            ori_img = cv.imread(img_path)
            processed_img = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)
            processed_img = preprocess(processed_img, f_flag)
            processed_img = cv.resize(processed_img, (256,256))
            
            if f_flag == 0:
                # use U-Net for DIC-C2DH-HeLa
                processed_img = np.reshape(processed_img, (1,)+processed_img.shape)
                res = model.predict(processed_img)

                # post-process predicted mask
                # res = postprocess(res, f_flag)
            else: 
                res = processed_img

            regions = postprocess(res, f_flag)

            # draw bounding boxes on original cell image
            #res = draw_bounding_box(ori_img, res)
            res = other_draw_bounding_box(ori_img, rects)
            
            # save images
            if f_flag == 0:
                out_path = "output/DIC-C2DH-HeLa"
            elif f_flag == 1:
                out_path = "output/Fluo-N2DL-HeLa"
            else:
                out_path = "output/PhC-C2DL-PSC"
            # print("writing Sequence {} {}.tif...".format(i, img_seq))
            cv.imwrite("{}/Sequence {}/{}.tif".format(out_path, i, img_seq), res)
            
def detect(frame, f_flag, model):
    processed_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    processed_img = preprocess(processed_img, f_flag, model)
#    processed_img = cv.resize(processed_img, (256,256))

#    if f_flag == 0:
#        processed_img = np.reshape(processed_img, (1,)+processed_img.shape)
        
        # predict cell detection
#        res = model.predict(processed_img)
#    else:
#        res = processed_img
    
    # post-process predicted mask
    boxes = postprocess(processed_img, f_flag)
    
    #output list of bounding boxes (startX, startY, endX, endY)
    #boxes = get_bboxes(res)
    return boxes

def get_bboxes(mask, color=(0,0,255), thickness=1):
    # get all contours in the mask
    _, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # returns (x, y) of one corner, width and height of each bounding box
    bounding_boxes = [cv.boundingRect(contour) for contour in contours]
    
    boxes = [ (bounding_boxes[i][0], bounding_boxes[i][1], \
               bounding_boxes[i][0]+bounding_boxes[i][2], bounding_boxes[i][1]+bounding_boxes[i][3]) \
             for i in range(len(bounding_boxes))]

    return boxes

def get_features(img, contours):
    # Read in RGB Image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    features = []

    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv.drawContours(mask, [cnt], 0, 255, -1)
        colour = cv.mean(img, mask=mask)[0]
        features.append((x, y, x+w, y+h, area, colour, 0))

    return features

def get_cdf(img, f_flag):
    # Read in RGB Image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#    gray = c_stretch(gray, 0, 255)

    # if not DIC set, remove background from cdf calculation
    if (f_flag != 0):
        mask = preprocess(gray, f_flag)
        gray = c_stretch(gray, 0, 255)
        mask[mask > 0] = 255
        gray = np.bitwise_and(mask, gray)

    flat = gray.flatten()

    nz_size = gray.shape[0] * gray.shape[1] - len(flat[flat==0])
    cdf = np.zeros(256, dtype=np.float16)
    count = 0

    # calculate cdf
    for j in range(1, 256):
        count += (len(flat[flat==j]) / (nz_size))
        cdf[j] = count

    return cdf

def is_mitosis(img, f_flag, cdf, cnt):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#    img = c_stretch(img, 0, 255)

    # weighting
    if (f_flag == 0):
        mean_w = 0.6
        shape_w = 0.4
        area_w = 0
        mean_a = 16900
        thresh = 0.75
    elif (f_flag == 1):
        mean_w = 0.3
        shape_w = 0.4
        area_w = 0.3
        mean_a = 700
        thresh = 0.6
    else:
        mean_w = 0.4
        shape_w = 0.6
        area_w = 0
        mean_a = 200
        thresh = 0.7

    mask = np.zeros(img.shape, dtype=np.uint8)
    cv.drawContours(mask, [cnt], 0, 255, -1)
    mean = (int)(cv.mean(img, mask=mask)[0])
    perimeter = cv.arcLength(cnt, True)
    area = cv.contourArea(cnt)
    
    shape = 4 * np.pi * area / (perimeter*perimeter)
    
    if (f_flag == 1):
        if (len(cnt) >= 5):
            (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
            shape = 1 - MA / ma
        else:
            shape = 0
            
    area = 1 - area / mean_a
    
#    print(mean_w * cdf[mean] + circ_w * circularity + area_w * area)
#    print(cdf[mean], end=' ')
#    print(circularity)
#    cv.imshow("Frame", rem)
#    cv.waitKey(1000)
    return (mean_w * cdf[mean] + shape_w * shape + area_w * area > thresh)
