# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:32:33 2024

@author: Sankalp Sahu
"""

import cv2
import os
#from yolo_def import Yolov8EYE 
from Yolov6EYE import Yolov6EYE
from openvino.runtime import Core, Model
from openvino.inference_engine import IECore
import numpy as np
import time
# import imageio.v2 as iio
import imageio

cls_model_dir = r"C:\Users\Sankalp Sahu\.paddleocr\whl\cls\ch_ppocr_mobile_v2.0_cls_infer"
det_model_dir = r"D:\en_PP-OCRv3_det_infer\en_PP-OCRv3_det_infer"
rec_model_dir = r"C:\Users\Sankalp Sahu\Downloads\converted_17_12_23_anpr\converted_17_12_23_anpr"
rec_char_dict_path= r"C:\Users\Sankalp Sahu\PaddleOCR\ppocr\utils\en_dict.txt" 

#target_classes = ['cycle','bike', 'vehicle']
#target_classes = ['sintex',"sintex1"]
target_classes = [ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]
#ocr = PaddleOCR(use_angle_cls=True,det_model_dir =det_model_dir,cls_model_dir = cls_model_dir,rec_model_dir= rec_model_dir, rec_char_dict_path=rec_char_dict_path)

# def image_ocr(frame):
#     result = ocr.ocr(frame)
#     print(result)
#     strings = []

#     if result is not None:
#         for sublist1 in result:
#             if sublist1 is not None:
#                 for sublist2 in sublist1:
#                     if sublist2 is not None:
#                         text = sublist2[1][0]
#                         strings.append(text)
#                     else:
#                         # Handle the case when sublist2 is None
#                         pass #strings.append("No text detected")
#                 else:
#                     # Handle the case when sublist1 is None
#                     pass #strings.append("No text detected")

#     return strings

def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2)), ((y2 + y1)/(2)), (x2 - x1), (y2 - y1)]

def infer_video(yolov8, video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    #fps = 20
    #output_path = r"E:\TATA MOTOR Vehical\egg\nesw model output_IR out 1.mp4"

    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    #writer = imageio.get_writer(output_path)
    # Process video frames
    frame_count = 0
    start_time = time.time()
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Resize the frame
        # resized_frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)

        #resized_frame = np.transpose(resized_frame, (2, 0, 1))  # Transpose to (3, 640, 640)
        #resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension (1, 3, 640, 640)

        # Perform inference on the resized frame
        objects = yolov6.inference(frame)
        print(objects)

        # Visualize the detections
        for obj in objects:
            #print("for loop ",obj,flush=True)
            xmin, ymin, xmax, ymax, classe, conf = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['class_id'], obj['confidence']
            class_name = target_classes[classe]
            frame_roi = frame[ymin:ymax, xmin:xmax]
            # print(frame.shape,"frame")

            text_class = f"{class_name}, {conf}"
            # print(text_class)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
            cv2.putText(frame, text_class, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #cv2.putText(frame, fps_str, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            #image_name = f"{output_dir}/processed_frame_{frame_count}.jpg"
            # cv2.imwrite(image_name, frame_roi)
        
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print('FPS:', fps)
            print("frame count",frame_count)
            fps_str = 'FPS: {:.2f}'.format(fps)
            text_str_size = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_position = ((width - text_str_size[0]) // 2, 30)

        #if text_str or not text_str:
            cv2.putText(frame, fps_str, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            print(fps_str,"fps in if")

        
        out.write(frame)
    #cv2.imshow("out",frame)
    #writer.close()
    # Release video capture and writer
    
    video.release()
    out.release()

    print("Inference complete. Output video saved to:", output_path)
    
def infer_image(yolov8, image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)

    # Get image dimensions
    height, width, _ = image.shape

    # Perform inference on the image
    objects = yolov6.inference(image)
    print(objects)

    # Visualize the detections
    for obj in objects:
        xmin, ymin, xmax, ymax, classe, conf = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['class_id'], obj['confidence']
        class_name = target_classes[classe]

        frame_roi = image[ymin:ymax, xmin:xmax]
        if frame_roi.size == 0:
            continue  # Skip empty region of interest

        text_class = f"{class_name}, {conf}"
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
        cv2.putText(image, text_class, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   #cv2.imshow("out", image)
    cv2.imwrite(output_path, image)
    print("Inference complete. Output image saved to:", output_path)
#model_path = "model.xml"
#model_path = "/home/clouduser/Sankalp/IR/FP32/best_openvino_model/best.xml"
#prob_threshold = 0.25 
#iou_threshold = 0.45
#video_path = "/home/clouduser/Sankalp/license_img/demo_video.mp4"
#output_path = "output11_FP32_new_ken3223.mp4"
#output_dir = "/home/clouduser/Sankalp/license_img/crop image"

#model_path = r"D:\best_ckpt_INT8 licveh_416_y6n.xml"
prob_threshold = 0.2
iou_threshold = 0.45
#video_path = r"C:\Users\Sankalp Sahu\Videos\vlc-record-2024-03-12-20h02m11s-100_2023-04-08_143918.3gp-.mp4"
#output_path = r"E:\anpr\test_output\1704252607321_Licence_100_2023-04-08_143918vehicle_versiony6n_v2_INT8.mp4"
output_path = r"D:\tata_motors\veh test\getMedia 07 out.mp4"
image_input = r"C:\Users\Sankalp Sahu\Downloads\raw_snapshot (55).png"
output_image = r"C:\Users\Sankalp Sahu\Downloads\raw_snapshot (55)_out.png"

ie = IECore()
yolov6 = Yolov6EYE(ie, model_path, prob_threshold, iou_threshold)
#infer_video(yolov6, video_path, output_path)
infer_image(yolov6, image_input, output_image)
