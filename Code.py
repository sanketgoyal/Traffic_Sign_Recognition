#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import os
import copy
from PIL import Image
from skimage.feature import hog
from skimage import feature, exposure
from sklearn import svm

Directory = "input/"

Training = [["Training/00035", 35], ["Training/00038", 38], ["Training/00045", 45]]

All_images = []
for all_img in os.listdir(Directory):
    All_images.append(all_img)
    All_images.sort()

hog_features = []
lbls = []
count = 0

for name in Training:
    val = name[0]
    lbl = name[1]
    lbl_images = [os.path.join(val, f) for f in os.listdir(val) if f.endswith('.ppm')]
    for image in range(0, len(lbl_images)):
        count += 1
        img = np.array(Image.open(lbl_images[image]))
        im_ready = cv2.resize(img, (64, 64))
        
        features, hog = feature.hog(im_ready, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1", visualise=True, multichannel=True)
        hog_features.append(features)
        lbls.append(lbl)

blue_classifier = svm.SVC(gamma='scale', decision_function_shape='ovo')
blue_classifier.fit(hog_features, lbls)

Training = [["Training/00001", 1], ["Training/00014", 14], ["Training/00017", 17], ["Training/00019", 19], 
                ["Training/00021", 21]]

hog_features = []
lbls = []
count = 0

for name in Training:
    val = name[0]
    lbl = name[1]
    lbl_images = [os.path.join(val, f) for f in os.listdir(val) if f.endswith('.ppm')]
    for image in range(0, len(lbl_images)):
        count += 1
        img = np.array(Image.open(lbl_images[image]))
        im_ready = cv2.resize(img, (64, 64))
        
        features, hog = feature.hog(im_ready, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1", visualise=True, multichannel=True)
        hog_features.append(features)
        lbls.append(lbl)

red_classifier = svm.SVC(gamma='scale', decision_function_shape='ovo')
red_classifier.fit(hog_features, lbls)


# In[9]:


#out = cv2.VideoWriter('final_RSV.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (800, 600))

def Blue_Region_Classifier(image):
    
    image1 = cv2.resize(image, (64,64))
    Test_features = []
    features, hog = feature.hog(image1, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1", visualise=True, multichannel=True)
    Test_features.append(features)
    
    predict_sign = []
    predict_sign = blue_classifier.predict(Test_features)

    if predict_sign[0] in [1, 17, 14, 19, 21, 35, 38, 45]:
        result = cv2.imread('Result/'+str(predict_sign[0])+'.PNG')
    
    return result

def Red_Region_Classifier(image):
    
    image1 = cv2.resize(image, (64,64))
    Test_features = []
    features, hog = feature.hog(image1, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1", visualise=True, multichannel=True)
    Test_features.append(features)
    
    predict_sign = []
    predict_sign = red_classifier.predict(Test_features)
    
    if predict_sign[0] in [1, 17, 14, 19, 21, 35, 38, 45]:
        output = cv2.imread('Result/'+str(predict_sign[0])+'.PNG')
    
    return output
 
def contrast_streatching(image):
    normalized_image = image - np.min(image)
    normalized_image = normalized_image/(np.max(image)-np.min(image))
    normalized_image = normalized_image * 255
    return normalized_image
    
def Box_Regions(region_of_intrest, color_detector):
    centroid_dictionary = {}
    for i in range(0, len(region_of_intrest)):
        Moment = cv2.moments(region_of_intrest[i])
        if Moment["m00"] != 0:
            centerX = int(Moment["m10"] / Moment["m00"])
            centerY = int(Moment["m01"] / Moment["m00"])
            if i == 0:
                centroid_dictionary[(centerX, centerY)] = [i]
            else:
                flag = 0
                for key in list(centroid_dictionary.keys()):
                    if (centerX - key[0])**2 + (centerY - key[1])**2 - 200**2 < 0:
                        centroid_dictionary[key].append(i)
                        flag = 1
                        break
                if flag == 0:
                    centroid_dictionary[(centerX, centerY)] = [i]
    roi = [] 
    for key in list(centroid_dictionary.keys()):
        flag = 0
        if len(centroid_dictionary[key]) > 3 and color_detector == 'b':
            for index in centroid_dictionary[key]:
                area = cv2.contourArea(region_of_intrest[index])
                if area > flag:
                    flag = area
                    main = region_of_intrest[index]
            roi.append(main)
        elif color_detector == 'r':
            for index in centroid_dictionary[key]:
                area = cv2.contourArea(region_of_intrest[index])
                if area > flag:
                    flag = area
                    main = region_of_intrest[index]
            roi.append(main)
    return roi

def modify_blue(xb,yb,wb,hb):        
    if  yb < 200 and (wb/hb) < 1.1 :
        if xb - 5 > 0 and yb - 5 > 0:
            xb = xb - 5
            yb = yb - 5
            wb = wb + 10
            hb = hb + 10
        cv2.rectangle(out_image,(xb,yb),(xb+wb,yb+hb),(255,0,0),2)
        testimage = out_image[yb:yb+hb, xb:xb+wb]
        resultimage = Blue_Region_Classifier(testimage)
        resultimageresized = cv2.resize(resultimage, (wb, hb))
        if xb-wb > 0:
            out_image[yb:yb+hb, xb-wb:xb] = resultimageresized
        else:
            out_image[yb:yb+hb, xb+wb:xb+2*wb] = resultimageresized
    return out_image

def modify_red(xr,yr,wr,hr):
    if yr < 90 and xr> 400  and (wr/hr) <= 1.1 and area > 170 and (wr/hr) >= 0.2:
        if xr - 5 > 0 and yr - 5 > 0:
            xr = xr - 5
            yr = yr - 5
            wr = wr + 10
            hr = hr + 10
        cv2.rectangle(out_image,(xr,yr),(xr+wr,yr+hr),(0,0,255),2)
        testimager = out_image[yr:yr+hr, xr:xr+wr]
        resultimager = Red_Region_Classifier(testimager)
        resultimageresizedr = cv2.resize(resultimager, (wr, hr))
        if xr-wr > 0:
            out_image[yr:yr+hr, xr-wr:xr] = resultimageresizedr
        else:
            out_image[yr:yr+hr, xr+wr:xr+2*wr] = resultimageresizedr  
    return out_image

def normalize_blue(normalized_blue, normalized_red, normalized_green):
    channel_c = (normalized_blue - normalized_red)/(normalized_blue + normalized_green + normalized_red)
    channel_c = np.where(np.invert(np.isnan(channel_c)), channel_c, 0)

    normalize_img_b = (np.maximum(temp_image, channel_c)*255).astype(np.uint8)
    normalize_img_b = np.where(normalize_img_b  > 45 , normalize_img_b, 0)
    normalize_img_b = np.where(normalize_img_b  < 150 , normalize_img_b, 0)
    return normalize_img_b

def normalize_red(normalized_blue, normalized_red, normalized_green):
    temporary1 = normalized_red - normalized_blue
    temporary2 = normalized_red - normalized_green
    temporary3 = normalized_blue + normalized_green + normalized_red
    channel_ch = np.minimum(temporary1, temporary2)/temporary3
    channel_ch = np.where(np.invert(np.isnan(channel_ch)), channel_ch, 0)

    normalize_img_r = (np.maximum(temp_image, channel_ch)*255).astype(np.uint8)
    normalize_img_r = np.where(normalize_img_r > 10, normalize_img_r, 0)
    normalize_img_r = np.where(normalize_img_r < 90, normalize_img_r, 0)
    return normalize_img_r

for index in range(1000, len(All_images)):

    img = cv2.imread("input/" + str(All_images[index]))
    images_resize = cv2.resize(img, (800,600), interpolation = cv2.INTER_AREA)
    Denoized_img = cv2.fastNlMeansDenoisingColored(images_resize, None,10,10,7,21)

    blue_channel = Denoized_img[:,:,0]
    green_channel = Denoized_img[:,:,1]
    red_channel = Denoized_img[:,:,2]

    normalized_blue = contrast_streatching(blue_channel)
    normalized_green = contrast_streatching(green_channel)
    normalized_red = contrast_streatching(red_channel)
    temp_image = np.zeros((images_resize.shape[0],images_resize.shape[1]))
    normalize_img_b = normalize_blue(normalized_blue, normalized_red, normalized_green)    
    normalize_img_r = normalize_red(normalized_blue, normalized_red, normalized_green)

    MSER_blue = cv2.MSER_create(2, 100, 1000, 0.3, 0.2, 200, 1.01, 0.003, 5)
    MSER_red = cv2.MSER_create(20, 100, 1000, 1.2, 0.2, 200, 1.01, 0.003, 5)
    
    out_image = images_resize.copy()
    b_regions, _ = MSER_blue.detectRegions(normalize_img_b)
    blueregions =  Box_Regions(b_regions, 'b')
    
    for region in blueregions:
        xb,yb,wb,hb = cv2.boundingRect(region)
        out_image = modify_blue(xb,yb,wb,hb)
    
    r_regions, _ = MSER_red.detectRegions(normalize_img_r)
    redregions =  Box_Regions(r_regions, 'r')
    
    for region1 in redregions:
        area = cv2.contourArea(region1)
        xr,yr,wr,hr = cv2.boundingRect(region1)
        out_image = modify_red(xr,yr,wr,hr)
    
    cv2.imshow('Output', out_image)
    #out.write(out_image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
#out.release()


# In[ ]:




