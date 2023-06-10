import numpy as np 
import cv2
from itertools import combinations
from bresenham import bresenham
def sharpen(src):
    blur_img = cv2.GaussianBlur(src, (3, 3), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    h, w = src.shape[:2]
    result = np.zeros([h, w, 3], dtype=src.dtype)
    result[0:h,0:w,:] = usm
    return result

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)
#def Clahe(img,cliplimit=10):
#    clahe = cv2.createCLAHE(clipLimit=cliplimit,
#                        tileGridSize=(4, 4))
#    clahe1 = clahe.apply(img)
#    return clahe1
# def gammaCorrection1(img):
#     hsv_image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     # Extract the value channel (brightness channel)
#     value_channel = hsv_image[:, :, 2].copy()
    
#     # Define the threshold value for brightness
#     up_threshold = 0  # Example threshold value
#     low_threshold = 45
#     # Create a mask for values greater than the threshold
#     mask1 = np.where(value_channel > up_threshold, 1, 0)
#     # Refine the brightness values based on the mask
#     refined_values = np.where(mask1 == 1, up_threshold, value_channel)

#     # Update the value channel with the refined values
#     hsv_image[:, :, 2] = refined_values
    
#     mask2 = np.where(value_channel < low_threshold, 1, 0)
#     # Refine the brightness values based on the mask
#     refined_values = np.where(mask2 == 1, low_threshold, value_channel)

#     # Update the value channel with the refined values
#     hsv_image[:, :, 2] = refined_values
    
    

#     # Convert the image back to RGB color space
#     result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
#     return result_image
def img_prcess(img):
    h,w,c = img.shape
    #img = gammaCorrection1(img)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gmin = np.min(hsv[:,:,2])
    gmax = np.max(hsv[:,:,2])
    if (gmin < 30 or gmax <130) and gmax < 150:
       img = gammaCorrection(img,2)
    elif gmin < 45:
       img = gammaCorrection(img,1.8)
    elif gmax > 180:
       img = gammaCorrection(img,1.2)
    else:
       img = gammaCorrection(img,1.5)
    
    #img = sharpen(img)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    #gray = Clahe(gray)
    
    kernel = np.ones((5,5), np.uint8) 
    kernel1 = np.ones((2,2), np.uint8) 
    gray = cv2.dilate(gray, kernel, iterations = 1) #膨脹mask的範圍，能夠殺更多點
    #threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    gray = cv2.erode(gray, kernel)   # 侵蝕
    gray = cv2.erode(gray, kernel1)   # 侵蝕
    ret, gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    #gray = np.float32(gray)

    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    # thresh = cv2.dilate(thresh,None)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # approx = cv2.approxPolyDP(contours[0], 3, True)
    

    
    # img2 = np.zeros((h,w),np.uint8)
    # cv2.polylines(img2, [approx], True, (0, 0, 255), 2)
    # img2 = cv2.drawContours(thresh, contours, 3, (0,255,0), 10)
    #cv2.imshow('test',img2)
    #cv2.waitKey(0)
    

  
    return gray

#def calculate_distance(point1, point2):
#    return int(np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))

# def find_adjacent_points(corners):
#     distances = []
#     for pair in combinations(corners, 2):
#         dist = np.linalg.norm(pair[0]-pair[1])
#         distances.append((pair[0], pair[1], dist))

#     distances.sort(key=lambda x: x[2])  # Sort distances in ascending order

#     black_points = set()
#     gray_points = set()
#     adjacent_points = []

#     for point1, point2, _ in distances:
#         if len(adjacent_points) > corners.shape[0]:
#         #if len(adjacent_points) > 4:
#             break
#         if tuple(point1) in black_points or tuple(point2)  in black_points:
#             continue
#         elif tuple(point1) in gray_points and tuple(point2) in gray_points:
#             adjacent_points.append((point1, point2))
#             black_points.add(tuple(point1))
#             black_points.add(tuple(point2))
#         elif tuple(point1) in gray_points and tuple(point2) not in gray_points:
#             adjacent_points.append((point1, point2))
#             black_points.add(tuple(point1))
#             gray_points.add(tuple(point2))
#         elif tuple(point1) not in gray_points and tuple(point2) in gray_points:
#             adjacent_points.append((point1, point2))
#             gray_points.add(tuple(point1))
#             black_points.add(tuple(point2))
#         else:
#             adjacent_points.append((point1, point2))
#             gray_points.add(tuple(point1))
#             gray_points.add(tuple(point2))
#     return adjacent_points

# def fill_points_between_corners(image, corners):
    
#     adjacent_points = find_adjacent_points(corners)
#     all_points = corners.copy()
#     #print(adjacent_points)
#     for point1, point2 in adjacent_points:
        
#         x1, y1 = point1
#         x2, y2 = point2
#         #print(x1, y1, x2, y2)
#         line_coordinates = np.array(list(bresenham(x1, y1, x2, y2)))
#         #print(line_coordinates)
#         #for tup in line_coordinates:
#         #    image
#         all_points = np.concatenate((all_points, line_coordinates))
#         #print(all_points)
#         #print(corners)
#         #for x, y in line_coordinates:
#         #    print(image[0][0])
#         #    image[y, x] = (255, 0, 0)  # Set the desired color for filling points

#     return  np.array(all_points)
    
def find_corner(img_ls,coord_aug):
    image_list = []
    imgs_corners = []
    img_num = len(img_ls)
    for i in range(img_num):
        copy_image = img_ls[i].copy()
        processed = img_prcess(copy_image)
        grey_3_channel = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        stack = np.concatenate((grey_3_channel,img_ls[i]/255),axis=1)
        #cv2.imshow('test',stack)
        #cv2.waitKey(0)
        dst = cv2.cornerHarris(processed,3,3,0.05)

        dst = cv2.dilate(dst,None) # corner

        #img_ls[i][dst>0.01*dst.max()]=[0,0,255]

        # dst = cv2.cornerHarris(processed,2,3,0.04)
        # dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        corners = cv2.cornerSubPix(processed,np.float32(centroids),(5,5),(-1,-1),criteria)
        corners = (np.rint(corners)).astype(int)
        #corners = fill_points_between_corners(processed,corners)
        
        #for point in corners:
        #    #print(point[0],point[1])
        #    copy_image[point[1]][point[0]] = [0,0,255]
        #image_list.append(copy_image)
        corners = corners + coord_aug[i]
        imgs_corners.append(corners.astype(int))
        #print(corners)
                # reg = gray[i]
        # print(reg.shape)
        #cv2.imshow('test',new_img)
        #cv2.waitKey(0)
        
    return image_list, imgs_corners

def combine_crop(img,corner_img,coord_aug):  ########for visualize only
    img_num = len(corner_img)
    for i in range(img_num):
        h,w,c = corner_img[i].shape
        img[coord_aug[i][1]:coord_aug[i][1]+h,coord_aug[i][0]:coord_aug[i][0]+w] = corner_img[i][:,:]
    #cv2.imshow('test',img)
    #cv2.waitKey(0)
    return img

def mask_diff(arr1, arr2):
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))
    return np.array(list(set1 - set2))

def cvshow(img):
    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def mask():
    mask =0
def find_corner1(img_ls,coord_aug):
    imgs_corners = []
    copyls = []
    img_num = len(img_ls)
    
    for i in range(img_num):
        copy_image = img_ls[i].copy()
        #processed = cv2.cvtColor(img_ls[i],cv2.COLOR_BGR2HSV)
        #print(np.max(processed[:,:,2]),np.min(processed[:,:,2]))
        processed = img_prcess(img_ls[i])
        #cvshow(processed)
        #processed = np.uint8(processed)
        img_return,cnt,hie = cv2.findContours(processed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        min_contour_size = 10
        
        # Create a blank mask image
        mask = np.zeros_like(processed)

        # Iterate over the contours and filter by size
        filtered_contours = []
        for contour in cnt:
            #if cv2.contourArea(contour) >= min_contour_size:
            if True:
                filtered_contours.append(contour)
                #cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        
        # Apply the mask to the original image to keep only the large contours
        #filtered_image = cv2.bitwise_and(img_ls[i], img_ls[i], mask=mask)
        
        # Iterate over the filtered contours and find the corner points
        corner_points = []
        for contour in filtered_contours:
            # Approximate the contour with a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get the corner points
            for point in approx:
                corner_points.append(np.array(point[0]))
        
        boundary_threshold = 0  # Distance from the image boundary
        filtered_corner_points = []
        for point in corner_points:
            x, y = point
            if x > boundary_threshold and y > boundary_threshold and x < processed.shape[1] - boundary_threshold and y < processed.shape[0] - boundary_threshold:
                filtered_corner_points.append(point)
        for point in filtered_corner_points:
            point = point + coord_aug[i]
        # Display the corner points
        #print(corner_points)
        for point in filtered_corner_points:
            cv2.circle(copy_image,(point[0],point[1]),3,[0,0,255],-1)
            #copy_image[point[1]][point[0]]= [0, 255, 0]
        copyls.append(copy_image)
        #cvshow(copy_image)
        #cv2.imshow('test',img_ls[i])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        imgs_corners.append(np.array(filtered_corner_points).astype(int))
        
        #grey_3_channel = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        #stack = np.concatenate((grey_3_channel/255,img_ls[i]/255),axis=1)
        #cv2.imshow('test',stack)
        #cv2.waitKey(0)
        #print(corners)
                # reg = gray[i]
        # print(reg.shape)
        #cv2.imshow('i',copy_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    if imgs_corners == []:
        print(imgs_corners)
    return copyls, imgs_corners



