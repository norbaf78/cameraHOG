# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 10:07:55 2018

@author: fabio.roncato
"""

import numpy as np
import json
import cv2
import pika
import time
from transform import CoordinateTransform
from reprojecter import ReprojectTo4326

def onclick(event, x, y, flags, param):
    #print("onclick !!" +  key + " " + id)
    global point_x, point_y        
    if event ==  cv2.EVENT_LBUTTONDOWN: # left button down
        print("Left click")        
        point_x = x
        point_y = y
        cv2.circle(position_frame, (x, y), 6, (255,0,0), -1)
        print("point %d, %d" %(x,y))

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 3):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
    
    
def increment_heatmap_value(img, rects, matrix_h, resize_val):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)                
        point_source = np.array([[(x+pad_w + w/2), (y+h)]], dtype='float32')
        point_source = np.array([point_source])
        point_dest = cv2.perspectiveTransform(point_source, matrix_h) 
        image_max_y_dimension,image_max_x_dimension,_ = img.shape
        point_dest[0][0][0] = point_dest[0][0][0]/resize_val
        point_dest[0][0][1] = point_dest[0][0][1]/resize_val
        new_x = max(0,point_dest[0][0][0])
        new_x = min(image_max_x_dimension,point_dest[0][0][0])
        new_x = int(new_x)
        new_y = max(0,point_dest[0][0][1])
        new_y = min(image_max_y_dimension,point_dest[0][0][1])    
        new_y =  int(new_y)                           
        img[new_y-1, new_x-1] = img[new_y-1, new_x-1]+1
        
        
def draw_homography_point(img, rects, matrix_h, thickness = 3):
    for x, y, w, h in rects:
        print(x)
        print(y)
        print(w)
        print(h)        
        pad_w, pad_h = int(0.15*w), int(0.05*h) 
        point_source = np.array([[(x+pad_w + w/2), (y+h)]], dtype='float32')
        point_source = np.array([point_source])
        print(point_source)
        point_dest = cv2.perspectiveTransform(point_source, matrix_h) 
        image_max_y_dimension,image_max_x_dimension,_ = img.shape
        new_x = max(0,point_dest[0][0][0])
        new_x = min(image_max_x_dimension,point_dest[0][0][0])
        new_y = max(0,point_dest[0][0][1])
        new_y = min(image_max_y_dimension,point_dest[0][0][1]) 
        cv2.circle(img, (new_x, new_y), 3, (0,0,255), -1 )  
        print(new_x)
        print(new_y)
        
        
def convert_homography_point(x, y, w, h, matrix_h):
    print(x)
    print(y)
    print(w)
    print(h)   
    pad_w, pad_h = int(0.15*w), int(0.05*h) 
    point_source = np.array([[(x+pad_w + w/2), (y+h)]], dtype='float32')
    point_source = np.array([point_source])
    print(point_source)
    point_dest = cv2.perspectiveTransform(point_source, matrix_h) 
    return point_dest[0][0][0], point_dest[0][0][1]         


def rescale_heatmap_image_value(img):
    # redefine the value for the heatmat when the max value tend to exced the 8bit
    if(img.max()==255):
        img = (img/(img.max()*1.0))*255.0
        img = img.astype(np.uint8)
    return img        
    


if __name__ == '__main__':

    resize_img = 3 # the resize to the acquired image. The HOG will be evaluated on the resized image
    additional_resize_point = 1.5 # the point in the image (source point for the homography) have been taken with resize_img=2. In case of
                                # other resize change this vale (example 2,1 4,2 .....)
    cell_heatmap_step = 20
    zoom_heatmap = 4.0

    transformer = CoordinateTransform()
    reprojecter = ReprojectTo4326()
    with open('configReal.json') as json_data_file:
        data = json.load(json_data_file)  
    #point in source and destination images to create the Homography transformation
    pt_0_pix_src_image_X = data['config']['pt_0_pix_src_image_X']
    pt_0_pix_src_image_Y = data['config']['pt_0_pix_src_image_Y']
    pt_0_pix_dst_image_X = data['config']['pt_0_pix_dst_image_X']
    pt_0_pix_dst_image_Y = data['config']['pt_0_pix_dst_image_Y']	
    pt_1_pix_src_image_X = data['config']['pt_1_pix_src_image_X']
    pt_1_pix_src_image_Y = data['config']['pt_1_pix_src_image_Y']
    pt_1_pix_dst_image_X = data['config']['pt_1_pix_dst_image_X']
    pt_1_pix_dst_image_Y = data['config']['pt_1_pix_dst_image_Y']	
    pt_2_pix_src_image_X = data['config']['pt_2_pix_src_image_X']
    pt_2_pix_src_image_Y = data['config']['pt_2_pix_src_image_Y']
    pt_2_pix_dst_image_X = data['config']['pt_2_pix_dst_image_X']
    pt_2_pix_dst_image_Y = data['config']['pt_2_pix_dst_image_Y']
    pt_3_pix_src_image_X = data['config']['pt_3_pix_src_image_X']
    pt_3_pix_src_image_Y = data['config']['pt_3_pix_src_image_Y']
    pt_3_pix_dst_image_X = data['config']['pt_3_pix_dst_image_X']
    pt_3_pix_dst_image_Y = data['config']['pt_3_pix_dst_image_Y']
    pt_4_pix_src_image_X = data['config']['pt_4_pix_src_image_X']
    pt_4_pix_src_image_Y = data['config']['pt_4_pix_src_image_Y']
    pt_4_pix_dst_image_X = data['config']['pt_4_pix_dst_image_X']
    pt_4_pix_dst_image_Y = data['config']['pt_4_pix_dst_image_Y']
    #configuration of the connection to rabbitMQ    
    host_rabbitmq = data['config']['host_rabbitmq']
    host_rabbitmq_username = data['config']['host_rabbitmq_username']
    host_rabbitmq_psw = data['config']['host_rabbitmq_psw']
    #configuration key and id
    key = data['config']['tile38_key']    
    id = data['config']['tile38_id']
    #image information
    image_map_max_X = float(data['config']['backgound_image_max_X'])
    image_map_max_Y = float(data['config']['backgound_image_max_Y'])

    
    print("pt_0_pix_src_image_X: " + str(pt_0_pix_src_image_X)) 
    print("pt_0_pix_src_image_Y: " + str(pt_0_pix_src_image_Y))    
    print("pt_0_pix_dst_image_X: " + str(pt_0_pix_dst_image_X))    
    print("pt_0_pix_dst_image_Y: " + str(pt_0_pix_dst_image_Y))	    
    print("pt_1_pix_src_image_X: " + str(pt_1_pix_src_image_X))    
    print("pt_1_pix_src_image_Y: " + str(pt_1_pix_src_image_Y))    
    print("pt_1_pix_dst_image_X: " + str(pt_1_pix_src_image_X))    	
    print("pt_1_pix_dst_image_Y: " + str(pt_1_pix_dst_image_Y))   
    print("pt_2_pix_src_image_X: " + str(pt_2_pix_src_image_X))   
    print("pt_2_pix_src_image_Y: " + str(pt_2_pix_src_image_Y))   
    print("pt_2_pix_dst_image_X: " + str(pt_2_pix_dst_image_X))   
    print("pt_2_pix_dst_image_Y: " + str(pt_2_pix_dst_image_Y))   
    print("pt_3_pix_src_image_X: " + str(pt_3_pix_src_image_X))   
    print("pt_3_pix_src_image_Y: " + str(pt_3_pix_src_image_Y))  
    print("pt_3_pix_dst_image_X: " + str(pt_3_pix_dst_image_X))   
    print("pt_3_pix_dst_image_Y: " + str(pt_3_pix_dst_image_Y))    
    print("pt_4_pix_src_image_X: " + str(pt_4_pix_src_image_X))    
    print("pt_4_pix_src_image_Y: " + str(pt_4_pix_src_image_Y))    
    print("pt_4_pix_dst_image_X: " + str(pt_4_pix_dst_image_X))    
    print("pt_4_pix_dst_image_Y: " + str(pt_4_pix_dst_image_Y))  
    print("host_rabbitmq: " + host_rabbitmq)
    print("host_rabbitmq_username: " + host_rabbitmq_username)    
    print("host_rabbitmq_psw: " + host_rabbitmq_psw)
    print("key: " + key)
    print("id: " + id)
    print("image_map_max_X: " + str(image_map_max_X))
    print("image_map_max_Y: " + str(image_map_max_Y))

    ####################################################  
    #connect to rabbitmq
    ####################################################
    credentials = pika.PlainCredentials(host_rabbitmq_username, host_rabbitmq_psw)
    #parameters = pika.ConnectionParameters(host_rabbitmq,5672,'/',credentials, socket_timeout=10000000, heartbeat_interval=0,blocked_connection_timeout=300)
    parameters = pika.ConnectionParameters(host_rabbitmq,5672,'/',credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()    


    img_map = cv2.imread('mapReal.png')  
    # https://zbigatron.com/mapping-camera-coordinates-to-a-2d-floor-plan/
    pts_src = np.array([[pt_0_pix_src_image_X/additional_resize_point, pt_0_pix_src_image_Y/additional_resize_point], [pt_1_pix_src_image_X/additional_resize_point, pt_1_pix_src_image_Y/additional_resize_point], [pt_2_pix_src_image_X/additional_resize_point, pt_2_pix_src_image_Y/additional_resize_point],[pt_3_pix_src_image_X/additional_resize_point, pt_3_pix_src_image_Y/additional_resize_point], [pt_4_pix_src_image_X/additional_resize_point, pt_4_pix_src_image_Y/additional_resize_point]])
    pts_dst = np.array([[pt_0_pix_dst_image_X, pt_0_pix_dst_image_Y], [pt_1_pix_dst_image_X, pt_1_pix_dst_image_Y], [pt_2_pix_dst_image_X, pt_2_pix_dst_image_Y],[pt_3_pix_dst_image_X, pt_3_pix_dst_image_Y], [pt_4_pix_dst_image_X, pt_4_pix_dst_image_Y]])
    h, status = cv2.findHomography(pts_src, pts_dst) # # calculate matrix H
    cv2.namedWindow('homography')
    #cv2.imshow('homography',img_map)  

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    cap=cv2.VideoCapture("http://root:progtrl01@192.168.208.55/mjpg/1/video.mjpg")
    #creation image where the history of the positions is visible
#    _,big_position_frame=cap.read()
    #the acquired image is the background used where draw the position points recognized
#    position_frame = cv2.resize(big_position_frame, (0,0), fx=1.0/resize_img, fy=1.0/resize_img) 
    #set the background image for the heatmap. Starting acquiring the dimension of the image where the elaboration have been made
    rows_map_frame,cols_map_frame, _ = img_map.shape 
    img_map_view = np.zeros((int(cols_map_frame/4), int(rows_map_frame/4), 3), np.uint8)
    # rows_position_frame,cols_position_frame, _ = img_map.shape
    # ne channel iamge is the starting point for the heatmap
    heatmap_gray = np.zeros((int(rows_map_frame/cell_heatmap_step), int(cols_map_frame/cell_heatmap_step), 1), dtype = np.uint8)
    # create the windows where the image will be visible and where will be possible define the points
#    cv2.namedWindow('position')
#    cv2.setMouseCallback('position', onclick)
     
 
    while True:
        _,big_frame=cap.read() # acquire a new image
        frame = cv2.resize(big_frame, (0,0), fx=1.0/resize_img, fy=1.0/resize_img) # resize the original image (this is the image ready for the elaboration)
        #https://stackoverflow.com/questions/26607418/improving-accuracy-opencv-hog-people-detector 
        found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05) # detection
        draw_detections(frame,found) # draw bounding box in one image
        draw_homography_point(img_map, found, h)
        
        increment_heatmap_value(heatmap_gray, found, h, cell_heatmap_step) # increment vale for the heatmap in the gray image       
        heatmap_gray = rescale_heatmap_image_value(heatmap_gray)        
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        heatmap_color_resize_big = cv2.resize(heatmap_color, (0,0), fx=zoom_heatmap, fy=zoom_heatmap)
        
        #convert data in geographic position and provide the data to rabbitmq
        for x, y, dim_w, dim_h  in found:             
            homographyX_in_pixel, homographyY_in_pixel = convert_homography_point(x, y, dim_w, dim_h, h)          
            homographyY_in_pixel = image_map_max_Y - homographyY_in_pixel
            metersX,metersY = transformer.pixelToMeter(homographyX_in_pixel,homographyY_in_pixel,image_map_max_X,image_map_max_Y)  
            newx,newy = transformer.transform(metersX,metersY)       
            newy,newx = reprojecter.MetersToLatLon(newx,newy)            
            # insert data in queue in rabbitmq
            body = '{"type":"Feature","geometry":{"type":"Point","coordinates":[' + str(newy) + ',' + str(newx) + ']},"properties":{"key":"' + key + '","id":"' + id + '","timestamp":"' + str(time.time()) + '"}}'
            channel.basic_publish(exchange='trilogis_exchange_pos',routing_key='trilogis_position',body=body, properties=pika.BasicProperties(delivery_mode = 2)) # make message persistent
              
        cv2.imshow('feed',frame)  
        cv2.imshow('heatmap',heatmap_color_resize_big) 
        #cv2.imshow('homography',img_map)           
        img_map_view = cv2.resize(img_map, (int(cols_map_frame/4), int(rows_map_frame/4)))
        cv2.imshow('homography',img_map_view)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    res = connection.close()
    print ('rabbitmq connection close - ' + str(res))  
    cv2.destroyAllWindows()
    
