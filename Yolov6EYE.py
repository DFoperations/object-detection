import os
import time
import numpy as np
import torchvision
from typing import Tuple, Optional, Union, Callable
import cv2
import torch



class Yolov6EYE():

    def __init__(self, ie, model, prob_threshold, iou_threshold):

        self.ie = ie 
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold

        self.OVNet = self.ie.read_network(
            model = model,
            weights = os.path.splitext(model)[0] + ".bin"
        )
        self.ie.set_config(config={"CPU_THROUGHPUT_STREAMS": "16"}, device_name="CPU")
        self.OVExec = self.ie.load_network(self.OVNet, "CPU")

        self.inputLayer = next(iter(self.OVNet.input_info))
        self.outputLayer = list(self.OVNet.outputs)[-1]
        self.N, self.C, self.H, self.W = self.OVNet.input_info[self.inputLayer].input_data.shape

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



    # def inference(self, image, request_id, cur_request_id):
    def inference(self, image_):
        image= image_

        # ret_val, orig_input = cap.read()
        # print("sdjkfsdf",image.shape)
        orig_img = image[..., ::-1]
        
        resized = self.pad_resize_image(orig_img,  (self.W, self.H))
        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)
        img_in /= 255.0
        model_input = np.expand_dims(img_in, axis=0)
        results = self.OVExec.infer(inputs={self.inputLayer: model_input})
        detections = results[self.outputLayer]
        # print(detections.shape)
        detections = torch.from_numpy(detections)
        detections = self.non_max_suppression(detections, conf_thres=self.prob_threshold, iou_thres=self.iou_threshold, agnostic=False)
        # print(detections)
        detections = detections[0]
        image_src = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        labels = detections[..., -1].numpy()
        boxs = detections[..., :4].numpy()
        confs = detections[..., 4].numpy()

        h, w = image_src.shape[:2]
        boxs[:, :] = self.scale_coords((self.H, self.W), boxs[:, :], (h, w)).round()
        
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





