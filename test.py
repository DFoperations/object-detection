# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:15:42 2023

@author: Sankalp Sahu
"""

import os
import cv2
import time
import torch
import numpy as np
import torchvision
import torch.nn as nn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#from paddleocr import PaddleOCR, draw_ocr
from yolov6.layers.common import RepVGGBlock
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER, load_yaml
from typing import Tuple, Optional, Union, Callable 


char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Yolov6EYE():

    def __init__(self, weights, prob_threshold, iou_threshold, det_model_dir, cls_model_dir, rec_model_dir, rec_char_dict_path):
        self.rec_model_dir = rec_model_dir
        self.rec_char_dict_path =  rec_char_dict_path
        self.det_model_dir = det_model_dir
        self.cls_model_dir = cls_model_dir
        self.weights = weights
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        device_idx =0
        cuda = device_idx != 'cpu' and torch.cuda.is_available()
        device_name = f'cuda:{device_idx}' if cuda else 'cpu'
        self.img_size = (416,416)
        self.device = torch.device(device_name) # Choose a specific GPU device (change '0' to the desired GPU index) or 'cpu' for CPU
        self.model = DetectBackend(self.weights, device=self.device)
        self.model_switch(self.model.model, self.img_size)
        self.half = False

        # Half precision
        if self.half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False
        
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size[0], self.img_size[1]).to(self.device).type_as(next(self.model.model.parameters())))
        #self.model.eval()
        print("done")
        # self.ocr = PaddleOCR(use_angle_cls = True, 
        #                     det_model_dir = self.det_model_dir, 
        #                     cls_model_dir = self.cls_model_dir, 
        #                     rec_model_dir = self.rec_model_dir, 
        #                     rec_char_dict_path = self.rec_char_dict_path,)
        print("done intializing Paddle ocr")
        
    def model_switch(self, model, img_size): 
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

        LOGGER.info("Switch model to deploy modality.")
        
    # def image_ocr(self,frame):
    #     result = self.ocr.ocr(frame)
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
    #                         strings.append("No text detected")

    #     return strings

    def pad_resize_image(self,
        cv2_img: np.ndarray,
        new_size: Tuple[int, int] = (640, 480),
        color: Tuple[int, int, int] = (125, 125, 125)) -> np.ndarray:
        """Resize and pad image with color if necessary, maintaining orig scale

        args:
            cv2_img: numpy.ndarray = cv2 image
            new_size: tuple(int, int) = (width, height)
            color: tuple(int, int, int) = (B, G, R)
        """
        in_h, in_w = cv2_img.shape[:2]
        new_w, new_h = new_size
        # rescale down
        scale = min(new_w / in_w, new_h / in_h)
        # get new sacled widths and heights
        scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
        resized_img = cv2.resize(cv2_img, (scale_new_w, scale_new_h))
        # calculate deltas for padding
        d_w = max(new_w - scale_new_w, 0)
        d_h = max(new_h - scale_new_h, 0)
        # center image with padding on top/bottom or left/right
        top, bottom = d_h // 2, d_h - (d_h // 2)
        left, right = d_w // 2, d_w - (d_w // 2)
        pad_resized_img = cv2.copyMakeBorder(resized_img,
                                            top, bottom, left, right,
                                            cv2.BORDER_CONSTANT,
                                            value=color)
        return pad_resized_img

    def non_max_suppression(self,
        prediction: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[torch.Tensor] = None,
        agnostic: bool = False,
        multi_label: bool = False,
        labels: Tuple[str] = ()) -> torch.Tensor:
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # (pixels) maximum and minimum box width and height
        max_wh = 4096  # min_wh = 2
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)
                ] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lxi = labels[xi]
                v = torch.zeros((len(lxi), nc + 5), device=x.device)
                v[:, :4] = lxi[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lxi)), lxi[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                # sort by confidence
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float(
                ) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output

    def xywh2xyxy(self, x: torch.Tensor) -> torch.Tensor:
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(
            x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def scale_coords(self,img1_shape: Tuple[int, int], coords: np.ndarray, img0_shape: Tuple[int, int], ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0],
                    img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self,
        boxes: Union[torch.Tensor, np.ndarray],
        img_shape: Tuple[int, int]) -> None:
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):
            boxes[:, 0].clamp_(0, img_shape[1])  # x1
            boxes[:, 1].clamp_(0, img_shape[0])  # y1
            boxes[:, 2].clamp_(0, img_shape[1])  # x2
            boxes[:, 3].clamp_(0, img_shape[0])  # y2
        else:  # np.array
            boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
            boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
            boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
            boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


    def inference(self,image):
        
           # Provide the path to your YOLOv6 model weights file
        
        
        
        prob_threshold = 0.40
        iou_threshold = 0.45
        # Load an image using OpenCV
        
        orig_img = image[..., ::-1]
        
        resized = self.pad_resize_image(orig_img,  (416, 416))
        img = torch.from_numpy(resized).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0).to(self.device)  # Add batch dimension and move to the selected device
        
        # Perform inference
        with torch.no_grad():
            detections = self.model(img)
            print(detections.shape)
            detections = self.non_max_suppression(detections, conf_thres=prob_threshold, iou_thres=iou_threshold, agnostic=False)
            detections = detections[0]
            image_src = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            labels = detections[..., -1].cpu().numpy()
            boxs = detections[..., :4].cpu().numpy()
            confs = detections[..., 4].cpu().numpy()
            h, w = image_src.shape[:2]
            boxs[:, :] = self.scale_coords((416, 416), boxs[:, :], (h, w)).round()
            
            objects = []
        
            #confidence , class_id, xmin, ymin, xmax, ymax
            for i, box in enumerate(boxs):
                x1, y1, x2, y2 = map(int, box)
                # print(x1,y1,x2,y2)
                objects.append(
                    {
                        "confidence":confs[i],
                        "class_id":int(labels[i]),
                        "xmin":x1,
                        "ymin":y1,
                        "xmax":x2,
                        "ymax":y2 
                    }
                )
        
            return objects

def gamma_correction(image, gamma):
    # Apply gamma correction
    corrected_image = np.power(image / 255.0, gamma) * 255.0
    corrected_image = np.uint8(corrected_image)
    return corrected_image



def infer_video(yolov6, video_path, output_path,output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    #fps = 20
    #output_path = "outputlatest.mp4" 
    #target_classes = ['cycle','bike', 'vehicle']
    target_classes = ['animals']
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    #writer = imageio.get_writer(output_path)
    # Process video frames  1920 by 1080
    prev_text_str = "" 
    frame_count =0
    start_time=time.time()
    while True:
        ret, frame = video.read()
        if not ret:
            print("Breaking bcoz not ret", flush = True)
            break

        # Resize the frame
        # resized_frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)

        #resized_frame = np.transpose(resized_frame, (2, 0, 1))  # Transpose to (3, 640, 640)
        #resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension (1, 3, 640, 640)

        # Perform inference on the resized frame
        print(frame.shape)
        objects = yolov6.inference(frame)
        #objects = yolov4.inference((frame, 0, 0))
        print(objects)

        # Visualize the detections
        for obj in objects:
            conf,cls_id, xmin, ymin, xmax, ymax = obj['confidence'], obj['class_id'] ,obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
            class_name = target_classes[cls_id]
            # x, y, w, h = pascal_voc_to_yolo(xmin, ymin, xmax, ymax, width, height)
            # print(x,y,w,h)
            # x, y, w, h = int(x), int(y), int(w), int(h)
            
            #x, y, w, h = int(x * width), int(y * height), int(w * width), int(h * height)
            frame_roi = frame[ymin:ymax, xmin:xmax]
            gamma_value = 2
            frame_roi = gamma_correction(frame_roi, gamma_value)
            # img = frame_roi/255.0
            # img = cv2.resize(img,(96,48))
            # print(img.shape,"shape img")
            # img = np.array(img, dtype= np.float32)
            # ie = Core()
            # model = ie.read_model(nvidia_ocr)

            # compiled_model = ie.compile_model(model=model, device_name="CPU")

            # input_key = compiled_model.input(0)
            # output_key = compiled_model.output(0)
            # network_input_shape = input_key.shape
            # input_img = np.expand_dims(img, axis=0)
            # result = compiled_model(input_img)[output_key]
            # input_length = tf.constant([result.shape[1]])
            # print(input_length,"input len")
            # outs = K.get_value(K.ctc_decode(result, input_length=input_length, greedy=True)[0][0])
            # print(outs.shape, "pred shape")
            # print(outs)
            # text_str = ""
            # for row in outs:
            #     for value in row:
            #         if value != -1:
            #             text_str +=char_list[value]
            # print(frame.shape,"frame")
            # print(text_str)
            # print(frame_roi.shape,"frame_roi")
            if frame_roi.size == 0:
            
                continue  # Skip empty region of interest
            # print("ok")
           
#            image_name = f"{output_dir}/_processed_frame_{frame_count}.jpg"
#            frame_identifier = f"{video_file_name_without_extension}_frame_{frame_count}"
#            image_name = os.path.join(output_dir, f"{frame_identifier}.jpg")
#            cv2.imwrite(image_name, frame_roi)
#            print(f"Saved processed image: {image_name}")
            #frame_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)  # Convert to RGB format
            text = yolov6.image_ocr(frame_roi)
            print(text,flush=True)
            #text = image_IR_ocr(frame)
            text_str = ' ' 
            if text:
                text_str = ' '.join(text)
                print(text_str)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_color = (0, 255, 0)  # Green color
            background_color = (0, 255, 255)
            
            text_size = cv2.getTextSize(text_str, font, font_scale, font_thickness)[0]
            text_x = xmin
            text_y = ymin - 10
            
            background_x = xmin
            background_y = ymin - text_size[1] - 20  # Adjust for text size and spacing
            background_width = text_size[0] + 20  # Adjust for text size and spacing
            background_height = text_size[1] + 20  # Adjust for text size and spacing
            
            cv2.rectangle(frame, (background_x, background_y), (background_x + background_width, background_y + background_height), background_color, -1)
            
            # Draw the text on top of the background
            cv2.putText(frame, text_str, (text_x, text_y), font, font_scale, text_color, font_thickness)

            #labels = ["license plate"]
            #class_name = labels[cls_id]
            #text = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            #cv2.putText(frame,text_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            text_class = f"{class_name}, {conf}"
            cv2.putText(frame, text_class, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            #writer.append_data(frame)
            print(text_str)
        # Write the frame with detections to the output video
        
        print(frame_count)
        frame_count+=1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print('FPS:', fps)
        print("frame count",frame_count)
        fps_str = 'FPS: {:.2f}'.format(fps)
        text_str_size = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_position = ((width - text_str_size[0]) // 2, 60)
        #if text_str or not text_str:
        cv2.putText(frame, fps_str, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        video_file_name_without_extension = "anpr (1)"
        image_name = f"{output_dir}/_processed_frame_{frame_count}.jpg"
        frame_identifier = f"{video_file_name_without_extension}_frame_{frame_count}"
        image_name = os.path.join(output_dir, f"{frame_identifier}.jpg")
        cv2.imwrite(image_name, frame)
        print(f"Saved processed image: {image_name}")
        out.write(frame)
        #time.sleep(90)
        #cv2.imshow("output",frame)
        print(fps_str,"fps in if")

    #writer.close()
    # Release video capture and writer
    #import time
    #time.sleep(90)
    video.release()
    out.release()





prob_threshold = 0.70
iou_threshold = 0.65


#image_path = r'C:\Users\Sankalp Sahu\OneDrive\Pictures\Screenshots\Screenshot (176).png'

cls_model_dir = r"C:\Users\Sankalp Sahu\.paddleocr\whl\cls\ch_ppocr_mobile_v2.0_cls_infer"
#det_model_dir = r"C:\Users\Sankalp Sahu\PaddleOCR\output\det_db_inference" # Trained Model
det_model_dir = r"D:\en_PP-OCRv3_det_infer\en_PP-OCRv3_det_infer" # pretrained model
#rec_model_dir = r"C:\Users\Sankalp Sahu\OneDrive\Desktop\new anpr paddle model\best_model_25_09_2023"
rec_model_dir=r"C:\Users\Sankalp Sahu\Downloads\converted_17_12_23_anpr\converted_17_12_23_anpr"
rec_char_dict_path=r"C:\Users\Sankalp Sahu\PaddleOCR\ppocr\utils\en_dict.txt"
#weights= r"path"
#vid_path = r"D:\tata motors\D08_20240516065447.mp4"
vid_path = r"path"
output_path =  r"path"
output_dir = r"path"



yolov6 = Yolov6EYE(weights = weights, prob_threshold = prob_threshold, iou_threshold = iou_threshold, det_model_dir = det_model_dir,cls_model_dir=cls_model_dir, rec_model_dir = rec_model_dir, rec_char_dict_path = rec_char_dict_path)
#while True :
    
print("A")
infer_video(yolov6, vid_path, output_path,output_dir=output_dir)
print("B")






