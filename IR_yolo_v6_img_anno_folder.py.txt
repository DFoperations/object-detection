# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:43:10 2024

@author: Sankalp Sahu
"""
import cv2
import os
#from yolo_def import Yolov8EYE 
from Yolov6EYE import Yolov6EYE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#from paddleocr import PaddleOCR, draw_ocr
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
target_classes = ['animal']

#model_path = r"D:\best_ckpt_INT8 licveh_416_y6n.xml"

prob_threshold = 0.5
iou_threshold = 0.45
#video_path = r"C:\Users\Sankalp Sahu\Videos\vlc-record-2024-03-12-20h02m11s-100_2023-04-08_143918.3gp-.mp4"
video_path = r"D:\tata_motors\animal data vids red 2\vlc-record-2024-08-02-15h35m02s-00010006571000000.mp4-.mp4"
#output_path = r"E:\anpr\test_output\1704252607321_Licence_100_2023-04-08_143918vehicle_versiony6n_v2_INT8.mp4"
output_path = r"D:\tata_motors\animal test\vlc-record-2024-08-02-15h35m02s-00010006571000000_test_out_IR_0.5.mp4" 
image_input = r"D:\tata_motors\animal test\WhatsApp Image 2024-08-24 at 12.33.24_31d25991.jpg"
output_image = r"C:\Users\Sankalp Sahu\parseq\demo_images\WhatsApp Image 2024-08-24 at 12.33.24_31d25991_new_out_IR.jpg"

ie = IECore()
yolov6 = Yolov6EYE(ie, model_path, prob_threshold, iou_threshold)
#infer_video(yolov6, video_path, output_path)
#infer_image(yolov6, image_input, output_image)
image_folder = r"H:\final_10_11_12"
output_dir = r"E:\chayyos\final_10_11_12"

image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith((".jpg", ".png"))]

for image_file in image_files:
    # Read the image using cv2
    frame = cv2.imread(image_file)

    # Process the image and get object detections
    objects = yolov6.inference(frame)
    img_height, img_width, _ = frame.shape

    # Extract the image file name without extension
    image_file_name_without_extension = os.path.splitext(os.path.basename(image_file))[0]

    # Create a unique identifier for the current image
    image_identifier = f"{image_file_name_without_extension}"

    # Save the processed image with a unique name
    processed_image_name = os.path.join(output_dir, f"{image_identifier}.jpg")
    cv2.imwrite(processed_image_name, frame)
    print(f"Saved processed image: {processed_image_name}")

    # Save bounding box coordinates to a text file with a unique name
    bbox_file_name = os.path.join(output_dir, f"{image_identifier}.txt")
    print(bbox_file_name)
    with open(bbox_file_name, "w") as bbox_file:
        for obj in objects:
            conf, cls_id, xmin, ymin, xmax, ymax = (
                obj["confidence"],
                obj["class_id"],
                obj["xmin"],
                obj["ymin"],
                obj["xmax"],
                obj["ymax"],
            )

            # Normalize the coordinates to [0, 1]
           # xmin_norm = xmin / img_width
           # ymin_norm = ymin / img_height
           # xmax_norm = xmax / img_width
           # ymax_norm = ymax / img_height
            dw = 1./(img_width)
            dh = 1./(img_height)
            center_x = (xmin + xmax) / 2 - 1
            center_y = (ymin + ymax) / 2 - 1
            center_x *= dw
            center_y *= dh
            # Calculate width and height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
    
            bbox_file.write(
                f"{cls_id} "
                f"{center_x} {center_y} {width} {height}\n"
            )
    print(f"Saved bounding box coordinates to: {bbox_file_name}")
