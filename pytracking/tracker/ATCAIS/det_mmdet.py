from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.datasets.transforms import ImageTransform
from mmdet.datasets import to_tensor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

import cv2
import torch
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from .bounding_box import BoxList,boxlist_iou

class Detector(object):
    def __init__(self,
        config_file = '/home/tangjiuqi097/vot/pytracking/pytracking/mmdetection/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py',
        checkpoint_file = '/home/tangjiuqi097/vot/pytracking/pytracking/mmdetection/checkpoints/epoch_3.pth',
        *args, **kwargs ):
        self.model = init_detector(config_file, checkpoint_file)
        self.model.CLASSES=['Object']

        # self.model.test_cfg['rcnn']['nms']['iou_thr']=0.7


    def initialize(self, image, pos, target_sz):
        target_bbox = torch.cat(
            (pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, pos[[1, 0]] + (target_sz[[1, 0]] - 1) / 2))
        image_sz = image.shape[0:2]

        predictions = self.run_on_opencv_image(image,score_thr=0.001)
        # self.show_prediction(image,predictions)

        iou = self.compute_iou(predictions.bbox, target_bbox)
        iou = iou.reshape(-1)
        if len(iou) > 0:
            ious, inds = torch.sort(iou, descending=True)

            if ious[0] < 0.6:
                self.first_frame_distractors_boxes = predictions
            else:
                self.first_frame_distractors_boxes = predictions[inds][1:]

            self.distractors_boxes=self.first_frame_distractors_boxes

        first_frame_state = BoxList(target_bbox, [image_sz[1], image_sz[0]], mode="xyxy")
        first_frame_prediction = self.compute_mask_from_box(image, first_frame_state)
        self.first_frame_prediction=first_frame_prediction
        # self.show_prediction(image,first_frame_prediction)


        if len(first_frame_prediction)>0:
            box_from_mask, [L, S], flag = self.compute_box_from_mask(first_frame_prediction[0])
            iou_mask = self.compute_iou(self.xywh2xyxy(box_from_mask), target_bbox)
            if iou_mask > 0.65:
                self.first_frame_L = L
                self.first_frame_S = S
                self.first_frame_aspect_ratio=L/S
                self.last_L = L
                self.last_S = S
                self.L_max = 2*L
                self.L_min = 0.5*L
                self.S_max = 2*S
                self.S_min = 0.5*S
                self.weak_det = False
            else:
                self.weak_det = True
        else:
            if len(iou) > 0 and ious[0]>0.7:
                self.first_frame_prediction = predictions[inds[0]]

                box_from_mask, [L, S], flag = self.compute_box_from_mask(predictions[inds[0]])
                iou_mask = self.compute_iou(self.xywh2xyxy(box_from_mask), target_bbox)
                if iou_mask > 0.7:
                    self.first_frame_L = L
                    self.first_frame_S = S
                    self.first_frame_aspect_ratio = L / S
                    self.last_L = L
                    self.last_S = S
                    self.L_max = 2 * L
                    self.L_min = 0.5 * L
                    self.S_max = 2 * S
                    self.S_min = 0.5 * S
                    self.weak_det = False
            else:
                self.weak_det = True




    def update_distractors_boxes(self, image, pos, target_sz,predictions):
        target_bbox = torch.cat(
            (pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, pos[[1, 0]] + (target_sz[[1, 0]] - 1) / 2))
        iou = self.compute_iou(predictions.bbox, target_bbox)
        iou = iou.reshape(-1)
        if len(iou) > 0:
            inds=iou<0.01
            distractors_boxes=predictions[inds]

            self.distractors_boxes=self.distractors_boxes.concat(distractors_boxes)
            # self.show_prediction(image,self.distractors_boxes)


    def trackcing_by_detection(self, image, pos, target_sz, last_pos, last_target_sz, first_frame_target_sz, visulize):

        predictions = self.run_on_opencv_image(image)
        self.show_prediction(image, predictions)

        target_bbox = torch.cat(
            (pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, pos[[1, 0]] + (target_sz[[1, 0]] - 1) / 2))
        iou = self.compute_iou(predictions.bbox, target_bbox)
        iou = iou.reshape(-1)


        state, flag_det = self.compute_det_convince(predictions, pos, target_sz)

        return state, flag_det

    def compute_det_convince(self, predictions, pos, target_sz):

        if len(predictions) == 0:
            return None, "not_found"

        target_bbox = torch.cat(
            (pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, pos[[1, 0]] + (target_sz[[1, 0]] - 1) / 2))

        bbox = predictions.bbox

        iou = self.compute_iou(bbox, target_bbox)
        iou = iou.reshape(-1)
        iou_max, ind = torch.max(iou, dim=0)

        bbox_c = [(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2, bbox[:, 2] - bbox[:, 0],
                  bbox[:, 3] - bbox[:, 1]]
        bbox_c = torch.stack(bbox_c, dim=1)

        pos_c = bbox_c[:, [0, 1]]
        target_sz_det = bbox_c[:, [2, 3]]
        area_box = target_sz_det[:, 0] * target_sz_det[:, 1]

        masks = predictions.get_field("mask")
        masks = masks.squeeze(1)
        area_mask = masks.sum(dim=1).sum(dim=1)

        prediction = predictions[ind]
        box_from_mask, [L, S], flag = self.compute_box_from_mask(prediction)
        iou_mask = self.compute_iou(self.xywh2xyxy(box_from_mask), target_bbox)

        # if self.weak_det:  # if not dectect the target at first frame, use very strict condition: if the iou is low, detector will not be valid
        #
        #     if iou_max < 0.6:
        #         return None, "not_found"
        #
        #     if max(iou_max, iou_mask) < 0.7:
        #         return None, "not_found"
        #
        #     if iou_mask > iou_max:
        #         return box_from_mask, None
        #     return prediction.convert("xywh").bbox, None

        thr_S = max(0.1, 5 / self.last_S)
        thr_L = max(0.1, 5 / self.last_L)

        if L / self.last_L > 1 - 2 * thr_L and L / self.last_L < 1 + 2 * thr_L:
            if S / self.last_S > 1 - 2 * thr_S and S / self.last_S < 1 + 2 * thr_S:
                self.last_L = L
                self.last_S = S
                self.L_max = min(max(L, self.L_max), 2 * self.first_frame_L)
                self.L_min = max(min(L, self.L_min), 0.5 * self.first_frame_L)
                self.S_max = min(max(S, self.S_max), 2 * self.first_frame_S)
                self.S_min = max(min(S, self.S_min), 0.5 * self.first_frame_S)

                return box_from_mask, None

        # try to refind
        thr_S = max(0.1, 5 / self.first_frame_S)
        thr_L = max(0.1, 5 / self.first_frame_L)

        if L / self.first_frame_L > 1 - 2 * thr_L and L / self.first_frame_L < 1 + 2 * thr_L:
            if S / self.first_frame_S > 1 - 2 * thr_S and S / self.first_frame_S < 1 + 2 * thr_S:
                self.last_L = L
                self.last_S = S
                self.L_max = min(max(L, self.L_max), 2 * self.first_frame_L)
                self.L_min = max(min(L, self.L_min), 0.5 * self.first_frame_L)
                self.S_max = min(max(S, self.S_max), 2 * self.first_frame_S)
                self.S_min = max(min(S, self.S_min), 0.5 * self.first_frame_S)

                return box_from_mask, None

        iou_sorted, inds = torch.sort(iou, dim=0, descending=True)
        convinces = []
        box_from_masks = []
        LSs = []
        for iou_, ind in zip(iou_sorted, inds):
            if iou_ <= 0:
                break

            convince, box_from_mask, LS = self.compute_per_det_convince(predictions[ind])
            convinces.append(convince)
            box_from_masks.append(box_from_mask)
            LSs.append(LS)
        convinces = torch.tensor(convinces)
        inds = torch.nonzero(convinces == 1).squeeze(1)
        if len(inds) > 1:
            return None, "uncertain"
        elif len(inds) == 1:
            self.last_L, self.last_S = LSs[inds]
            return box_from_masks[inds], None

        return None, "not_found"

    def choose_prediction(self, predictions, pos, target_sz):

        if len(predictions) == 0:
            return None

        target_bbox = torch.cat(
            (pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, pos[[1, 0]] + (target_sz[[1, 0]] - 1) / 2))

        bbox = predictions.bbox

        iou = self.compute_iou(bbox, target_bbox)
        iou = iou.reshape(-1)
        iou_max, ind = torch.max(iou, dim=0)
        if iou_max>0.3:
            return ind


        if self.weak_det:
            if iou_max>0.7:
                return ind
            else:
                return None

        bbox_c = [(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2, bbox[:, 2] - bbox[:, 0],
                  bbox[:, 3] - bbox[:, 1]]
        bbox_c = torch.stack(bbox_c, dim=1)

        pos_c = bbox_c[:, [0, 1]]
        target_sz_det = bbox_c[:, [2, 3]]
        area_box = target_sz_det[:, 0] * target_sz_det[:, 1]

        masks = predictions.get_field("mask")
        masks = masks.squeeze(1)
        area_mask = masks.sum(dim=1).sum(dim=1)

        prediction = predictions[ind]
        box_from_mask, [L, S], flag = self.compute_box_from_mask(prediction)
        iou_mask = self.compute_iou(self.xywh2xyxy(box_from_mask), target_bbox)

        if L is not None and S is not None:
            thr_S = max(0.1, 5 / self.last_S)
            thr_L = max(0.1, 5 / self.last_L)

            if L / self.last_L > 1 - 2 * thr_L and L / self.last_L < 1 + 2 * thr_L:
                if S / self.last_S > 1 - 2 * thr_S and S / self.last_S < 1 + 2 * thr_S:
                    if self.L_min<L<self.L_max and self.S_min<S<self.S_max:

                        self.last_L = L
                        self.last_S = S
                        self.L_max = min(max(L, self.L_max), 2 * self.first_frame_L)
                        self.L_min = max(min(L, self.L_min), 0.5 * self.first_frame_L)
                        self.S_max = min(max(S, self.S_max), 2 * self.first_frame_S)
                        self.S_min = max(min(S, self.S_min), 0.5 * self.first_frame_S)

                        return ind

            # try to refind
            thr_S = max(0.1, 5 / self.first_frame_S)
            thr_L = max(0.1, 5 / self.first_frame_L)

            if L / self.first_frame_L > 1 - 2 * thr_L and L / self.first_frame_L < 1 + 2 * thr_L:
                if S / self.first_frame_S > 1 - 2 * thr_S and S / self.first_frame_S < 1 + 2 * thr_S:
                    if self.L_min < L < self.L_max and self.S_min < S < self.S_max:
                        self.last_L = L
                        self.last_S = S
                        self.L_max = min(max(L, self.L_max), 2 * self.first_frame_L)
                        self.L_min = max(min(L, self.L_min), 0.5 * self.first_frame_L)
                        self.S_max = min(max(S, self.S_max), 2 * self.first_frame_S)
                        self.S_min = max(min(S, self.S_min), 0.5 * self.first_frame_S)

                    return ind

        iou_sorted, inds = torch.sort(iou, dim=0, descending=True)
        convinces = []
        box_from_masks = []
        LSs = []
        for iou_, ind in zip(iou_sorted, inds):
            if iou_ <= 0:
                break

            convince, box_from_mask, LS = self.compute_per_det_convince(predictions[ind])
            convinces.append(convince)
            box_from_masks.append(box_from_mask)
            LSs.append(LS)
        convinces = torch.tensor(convinces)
        inds = torch.nonzero(convinces == 1).squeeze(1)
        if len(inds) > 1:
            return None
        elif len(inds) == 1:
            self.last_L, self.last_S = LSs[inds]
            return inds

        return None


    def compute_per_det_convince(self, prediction):

        box_from_mask, [L, S], flag = self.compute_box_from_mask(prediction)
        if flag == "mask_not_confidence":
            return False, None, [None, None]

        if L is not None and S is not None:

            thr_S = max(0.1, 5 / self.last_S)
            thr_L = max(0.1, 5 / self.last_L)

            if L / self.last_L > 1 - thr_L and L / self.last_L < 1 + thr_L:
                if S / self.last_S > 1 - thr_S and S / self.last_S < 1 + thr_S:
                    return True, box_from_mask, [L, S]

            # try to refind
            thr_S = max(0.1, 5 / self.first_frame_S)
            thr_L = max(0.1, 5 / self.first_frame_L)

            if L / self.first_frame_L > 1 - 2 * thr_L and L / self.first_frame_L < 1 + 2 * thr_L:
                if S / self.first_frame_S > 1 - 2 * thr_S and S / self.first_frame_S < 1 + 2 * thr_S:
                    return True, box_from_mask, [L, S]

            if self.L_min <= L <= self.L_max:
                if self.S_min <= S < self.S_max:
                    if L / self.first_frame_L / (S / self.first_frame_S) > 0.9 and L / self.first_frame_L / (
                            S / self.first_frame_S) < 1.1:
                        return True, box_from_mask, [L, S]

        return False, None, [None, None]


    def remove_easy_distractors(self,prediction,distractors):
        if prediction is None or len(prediction)==0 or distractors is None or len(distractors)==0:
            return prediction

        iou=boxlist_iou(prediction,distractors)
        iou_max,_=torch.max(iou,dim=1)
        keep=iou_max<0.6

        return  prediction[keep]

######################################
#   detection and visualization
    def detection(self,img):

        result = inference_detector(self.model, img)
        show_result(img, result, self.model.CLASSES,score_thr=0.3)
        predictions=self.toBoxlist(result)
        return predictions

    def parepare_img(self, img):
        if self.new_image:
            cfg = self.model.cfg
            img_transform = ImageTransform(
                size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

            device = next(self.model.parameters()).device
            img = mmcv.imread(img)

            ori_shape = img.shape
            img, img_shape, pad_shape, scale_factor = img_transform(
                img,
                scale=cfg.data.test.img_scale,
                keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
            img = to_tensor(img).to(device).unsqueeze(0)
            img_meta = [
                dict(
                    ori_shape=ori_shape,
                    img_shape=img_shape,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    flip=False)
            ]
            self.img=img
            self.img_meta=img_meta

    def prepare_features(self):
        if self.new_image:
            with torch.no_grad():
                self.model.extract_feat_pre(self.img)
            self.new_image=False

    def prepare_proposals(self,state):
        device = next(self.model.parameters()).device
        img_shape=self.img_meta[0]['img_shape']
        image_sz = img_shape[1], img_shape[0]
        state = state.resize(image_sz).to(device)
        confidence = torch.ones(len(state)).to(device)
        proposals = [torch.cat([state.bbox, confidence[:, None]], dim=1)]
        self.proposals=proposals

    def post_detection(self,ues_proposals=False):
        if ues_proposals:
            data = dict(img_meta=self.img_meta,proposals=self.proposals)
        else:
            data = dict(img_meta=self.img_meta)

        with torch.no_grad():
            result=self.model.simple_test_post(rescale=True,**data)
        return result

    def toBoxlist(self,result,score_thr=0.3):

        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        inds = np.where(bboxes[:, -1] > score_thr)[0]
        bboxes=bboxes[inds]

        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)

            if len(segms)==0:
                return None

            masks=[]
            for i in inds:
                mask = maskUtils.decode(segms[i]).astype(np.uint8)
                masks.append(mask)

        bbox=torch.tensor(bboxes[:,:4]).reshape(-1,4)
        im_sz=segms[0]['size'][::-1]
        predictions=BoxList(bbox,im_sz,"xyxy")

        masks=torch.tensor(masks)[:,None]
        predictions.add_field("mask",masks)

        scores=torch.tensor(bboxes[:,4]).reshape(-1)
        predictions.add_field("scores",scores)
        assert len(bboxes)==len(masks)==len(scores),"should have the same size"
        return predictions

    def compute_mask_from_box(self, image, state):
        self.parepare_img(image)
        self.prepare_features()
        self.prepare_proposals(state)
        result=self.post_detection(ues_proposals=True)
        predictions=self.toBoxlist(result,score_thr=0.01)
        return predictions

    def run_on_opencv_image(self, image,score_thr=0.3):
        self.parepare_img(image)
        self.prepare_features()
        result=self.post_detection(ues_proposals=False)
        predictions=self.toBoxlist(result,score_thr=score_thr)
        return predictions

    def show_prediction(self,image,predictions,fig_num=101):
        if predictions is None:
            print("predictions is none")
            return

        result = image.copy()
        result = self.overlay_boxes(result, predictions)
        result = self.overlay_scores(result, predictions)
        result = self.overlay_mask(result, predictions)
        self.show_image(result,fig_num=fig_num)
        return result

    def show_bbox(self,image,predictions,fig_num=101):
        if predictions is None:
            print("predictions is none")
            return

        result = image.copy()
        result = self.overlay_boxes(result, predictions)
        self.show_image(result,fig_num=fig_num)

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        # labels = predictions.get_field("labels")
        boxes = predictions.bbox

        for box in boxes:
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), (0,0,255), 2
            )

        return image

    def overlay_scores(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        boxes = predictions.bbox

        template = "{:.2f}"
        for box, score in zip(boxes, scores):
            x, y = box[:2]
            s = template.format(score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()

        for mask in masks:
            mask=mask.astype(np.bool).squeeze(0)
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)

            image[mask] = image[mask] * 0.5 + color_mask * 0.5
        composite = image

        return composite

    def show_image(self, a, fig_num=None, title=None):

        if a is None:
            # print("do not show image")
            return

        a_np = a[:, :, ::-1]
        plt.figure(fig_num)
        plt.tight_layout()
        plt.cla()
        plt.imshow(a_np)
        plt.axis('off')
        plt.axis('equal')
        if title is not None:
            plt.title(title)
        plt.draw()
        plt.pause(0.001)

######################################
#   utils
    def compute_iou(self, box1, box2):

        box1 = torch.tensor(box1).reshape(-1, 4)
        box2 = torch.tensor(box2).reshape(-1, 4)
        N = box1.shape[0]
        M = box2.shape[0]
        area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
        area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

        wh = (rb - lt + 1).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / ((area1 + area2).reshape(N, M) - inter)
        return iou

    def compute_box_from_mask(self, prediction,use_mask=False):
        """

        :param prediction: Boxlist mode xyxy
        :return: box_from_mask,   (x,y,w,h)
        [L,S],
        flag
        """

        L, S = None, None

        box_crop = prediction.bbox.squeeze().long()

        mask = prediction.get_field("mask").squeeze(0)

        mask = mask[box_crop[1]:box_crop[3], box_crop[0]:box_crop[2]]

        mask_sz = mask.shape
        kenel_sz = round(min(mask_sz) * 0.3)
        kenel_sz = (kenel_sz + 1) % 2 + kenel_sz

        # mask_close = cv2.morphologyEx(mask.numpy(), cv2.MORPH_CLOSE, np.ones([11, 11]), iterations=1)
        #
        # mask_open = cv2.morphologyEx(mask.numpy(), cv2.MORPH_OPEN, np.ones([kenel_sz,kenel_sz]), iterations=1)

        contours, _ = cv2.findContours(
            mask.numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  # cv2.RETR_TREE
        )

        area = []
        boxes = []
        for contour in contours:
            (x, y), (w, h), t = cv2.minAreaRect(contour)
            area.append(w * h)
            boxes.append([x, y, w, h, t])
        area = torch.tensor(area)
        boxes = torch.tensor(boxes)

        if len(area) == 0:
            return prediction.convert("xywh").bbox, [L, S], "mask_not_confidence"

        area_max, ind = torch.max(area, dim=0)

        X, Y, W, H, T = boxes[ind]

        rbox = cv2.boxPoints(((X, Y), (W, H), T)).reshape(1, 8)

        L, S = max(W, H), min(W, H)

        selection = np.array(rbox).reshape(-1)
        cx = np.mean(selection[0::2])
        cy = np.mean(selection[1::2])
        x1 = np.min(selection[0::2])
        x2 = np.max(selection[0::2])
        y1 = np.min(selection[1::2])
        y2 = np.max(selection[1::2])
        A1 = np.linalg.norm(selection[0:2] - selection[2:4]) * np.linalg.norm(selection[2:4] - selection[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        # s = max(s, 0.9)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        box_from_mask = torch.tensor([cx - (w - 1) / 2, cy - (h - 1) / 2, w, h])

        box_from_mask[0:2] = box_from_mask[0:2] + box_crop[0:2].float()

        box_det = prediction.convert("xywh").bbox

        iou = self.compute_iou(self.xywh2xyxy(box_det), self.xywh2xyxy(box_from_mask))

        flag = None
        if iou < 0.6:
            flag = "not_confidence"
            # return box_det,[L,S],  flag
        if use_mask:
            return box_from_mask, [L, S], flag
        else:
            return box_det,[L,S],  "use_box"

    def compute_disp(self,prediction,pos):
        bbox = prediction.bbox
        bbox_c = [(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2, bbox[:, 2] - bbox[:, 0],
                  bbox[:, 3] - bbox[:, 1]]
        bbox_c = torch.stack(bbox_c, dim=1)

        pos_c = bbox_c[:, [1, 0]]

        disp1 = pos_c - pos[None,]
        disp_norm1 = torch.sqrt(torch.sum(disp1 ** 2, dim=1))
        return  disp_norm1

    def xywh2xyxy(self, box):
        box = box.reshape(-1, 4).clone()
        box[:, 2:4] = box[:, 0:2] + box[:, 2:4]

        return box


if __name__ == "__main__":

    detector=Detector()
    root_path ='/home/tangjiuqi097/data/vot2019/vot_rgbd/sequences'
    save_root_path='/media/tangjiuqi097/ext/det_rgbd_first_frame'
    seq_names=os.listdir(root_path)
    for seq_name in seq_names:
        # if seq_name!='backpack_blue':
        #     continue

        seq_path=os.path.join(root_path,seq_name,'color')
        try:
            im_names=sorted(os.listdir(seq_path))
        except:
            continue
        print('\n')
        print(seq_name)

        for im_name in tqdm(im_names):
            if im_name!='00000001.jpg':
                continue
            detector.new_image=True
            save_path = os.path.join(save_root_path, seq_name, im_name)
            # if os.path.exists(save_path):
            #     continue

            im_path=os.path.join(seq_path,im_name)
            img=cv2.imread(im_path)


            # detector.detection(im_path)
            predictions= detector.run_on_opencv_image(img)
            # show_image(result_image, 1, "detection results")
            result_image=detector.show_prediction(img[:,:,::-1], predictions, fig_num=101)

            if  not os.path.exists(os.path.join(save_root_path,seq_name)):
                os.makedirs(os.path.join(save_root_path,seq_name),exist_ok=True)
            cv2.imwrite(save_path, result_image)


    print("done")