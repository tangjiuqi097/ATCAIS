
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import os


import numpy as np
import cv2
import torch
from torchvision import transforms as T
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from tqdm import tqdm
from demo.predictor import COCODemo


def show_image(a, fig_num = None, title = None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a[:,:,::-1]
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

class Detector(COCODemo):
    CATEGORIES = [
        "__background",
        "object",
    ]

    def __init__(self,config_file, confidence_threshold=-1,show_mask_heatmaps=False,
        masks_per_dim=2,min_image_size=448):

        cfg.merge_from_file(config_file)

        # cfg.freeze()
        super(Detector,self).__init__(cfg,confidence_threshold,show_mask_heatmaps,
        masks_per_dim,min_image_size)


    def initialize(self,image,pos,target_sz):
        target_bbox = torch.cat(
            (pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, pos[[1, 0]] + (target_sz[[1, 0]] - 1) / 2))
        image_sz=image.shape[0:2]
        first_frame_state=BoxList(target_bbox,[image_sz[1],image_sz[0]],mode="xyxy")
        # first_frame_prediction,result_image=self.compute_mask_from_box( image, first_frame_state)
        # # self.show_image(result_image, fig_num=2)
        #
        # box_from_mask, [L, S], flag = self.compute_box_from_mask(first_frame_prediction[0])
        # iou_mask = self.compute_iou(self.xywh2xyxy(box_from_mask), target_bbox)
        # if iou_mask>0.7:
        #     self.first_frame_L = L
        #     self.first_frame_S = S
        #     self.last_L = L
        #     self.last_S = S
        #     self.L_max=L
        #     self.L_min=L
        #     self.S_max=S
        #     self.S_min=S
        #     self.weak_det = False
        # else :
        #     self.weak_det = True

        top_predictions,result_image=self.run_on_opencv_image(image,visualize=True)
        self.show_image(result_image, fig_num=2)

        iou = self.compute_iou(top_predictions.bbox, target_bbox)
        iou = iou.reshape(-1)
        if len(iou)>0:
            ious, inds = torch.sort(iou, descending=True)

            if ious[0] < 0.6:
                self.first_frame_distractors_boxes = top_predictions
            else:
                self.first_frame_distractors_boxes = top_predictions[inds][1:]

    def compute_mask_from_box(self, original_image,first_frame_state,visualize=True):

        # apply pre-processing to image
        image = self.transforms(original_image)

        image_sz=image.shape[2],image.shape[1]
        first_frame_state=first_frame_state.resize(image_sz).to(self.device)

        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        # compute predictions
        with torch.no_grad():
            predictions = self.model.compute_mask_from_box(image_list,first_frame_state)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        top_predictions=prediction
        result=None
        if visualize:
            result = original_image.copy()
            if self.show_mask_heatmaps:
                return self.create_mask_montage(result, top_predictions)
            result = self.overlay_boxes(result, top_predictions)
            if self.cfg.MODEL.MASK_ON:
                result = self.overlay_mask(result, top_predictions)
            if self.cfg.MODEL.KEYPOINT_ON:
                result = self.overlay_keypoints(result, top_predictions)
            result = self.overlay_class_names(result, top_predictions)


        return prediction,result

    def trackcing_by_detection(self,image,pos,target_sz,last_pos,last_target_sz,first_frame_target_sz,visulize):
        
        top_predictions, result_image = self.run_on_opencv_image(image, visualize=visulize)
        self.show_image(result_image, fig_num=2)
        target_bbox = torch.cat(
            (pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, pos[[1, 0]] + (target_sz[[1, 0]] - 1) / 2))
        iou = self.compute_iou(top_predictions.bbox, target_bbox)
        iou = iou.reshape(-1)

        predictions = top_predictions

        state, flag_det = self.compute_det_convince(predictions, pos, target_sz, last_pos,
                                                             last_target_sz, first_frame_target_sz)

        return state,flag_det

    def compute_det_convince(self,predictions,pos,target_sz,last_pos,last_target_sz,first_frame_target_sz):

        if len(predictions)==0:
            return None, "not_found"

        target_bbox = torch.cat(
            (pos[[1, 0]] - (target_sz[[1, 0]] - 1) / 2, pos[[1, 0]] + (target_sz[[1, 0]] - 1) / 2))


        bbox = predictions.bbox

        iou=self.compute_iou(bbox,target_bbox)
        iou=iou.reshape(-1)
        iou_max, ind = torch.max(iou, dim=0)


        bbox_c = [(bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2, bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]]
        bbox_c = torch.stack(bbox_c, dim=1)

        pos_c = bbox_c[:, [0, 1]]
        target_sz_det = bbox_c[:, [2, 3]]
        area_box=target_sz_det[:,0]*target_sz_det[:,1]

        masks=predictions.get_field("mask")
        masks=masks.squeeze(1)
        area_mask=masks.sum(dim=1).sum(dim=1)


        prediction = predictions[ind]
        box_from_mask,[L,S], flag = self.compute_box_from_mask(prediction)
        iou_mask = self.compute_iou(self.xywh2xyxy(box_from_mask), target_bbox)


        if self.weak_det:  #if not dectect the target at first frame, use very strict condition: if the iou is low, detector will not be valid

            if iou_max < 0.6:
                return None, "not_found"

            if max(iou_max,iou_mask)<0.7:
                return None, "not_found"

            if iou_mask>iou_max:
                return box_from_mask,None
            return prediction.convert("xywh").bbox,None


        thr_S=max(0.1,5/self.last_S)
        thr_L=max(0.1,5/self.last_L)

        if L/self.last_L > 1-2*thr_L and L/self.last_L < 1+2*thr_L:
            if S/self.last_S > 1-2*thr_S and S/self.last_S<1+2*thr_S:
                self.last_L=L
                self.last_S=S
                self.L_max = min(max(L,self.L_max),2*self.first_frame_L)
                self.L_min = max(min(L,self.L_min),0.5*self.first_frame_L)
                self.S_max = min(max(S,self.S_max),2*self.first_frame_S)
                self.S_min = max(min(S,self.S_min),0.5*self.first_frame_S)

                return box_from_mask, None


        # try to refind
        thr_S=max(0.1,5/self.first_frame_S)
        thr_L=max(0.1,5/self.first_frame_L)

        if L/self.first_frame_L > 1-2*thr_L and L/self.first_frame_L < 1+2*thr_L:
            if S/self.first_frame_S > 1-2*thr_S and S/self.first_frame_S<1+2*thr_S:
                self.last_L=L
                self.last_S=S
                self.L_max = min(max(L,self.L_max),2*self.first_frame_L)
                self.L_min = max(min(L,self.L_min),0.5*self.first_frame_L)
                self.S_max = min(max(S,self.S_max),2*self.first_frame_S)
                self.S_min = max(min(S,self.S_min),0.5*self.first_frame_S)

                return box_from_mask, None


        iou_sorted,inds = torch.sort(iou,dim=0,descending=True)
        convinces=[]
        box_from_masks=[]
        LSs=[]
        for iou_,ind in zip(iou_sorted,inds):
            if iou_<=0:
                break

            convince,box_from_mask,LS=self.compute_per_det_convince(predictions[ind])
            convinces.append(convince)
            box_from_masks.append(box_from_mask)
            LSs.append(LS)
        convinces=torch.tensor(convinces)
        inds=torch.nonzero(convinces==1).squeeze(1)
        if len(inds)>1:
            return None, "uncertain"
        elif len(inds)==1:
            self.last_L,self.last_S=LSs[inds]
            return box_from_masks[inds],None

        return None,"not_found"

    def compute_per_det_convince(self,prediction):

        box_from_mask, [L, S], flag = self.compute_box_from_mask(prediction)
        if flag=="mask_not_confidence":
            return False, None, [None, None]


        thr_S=max(0.1,5/self.last_S)
        thr_L=max(0.1,5/self.last_L)

        if L/self.last_L > 1-thr_L and L/self.last_L < 1+thr_L:
            if S/self.last_S > 1-thr_S and S/self.last_S<1+thr_S:

                return True,box_from_mask, [L,S]


        # try to refind
        thr_S=max(0.1,5/self.first_frame_S)
        thr_L=max(0.1,5/self.first_frame_L)

        if L/self.first_frame_L > 1-2*thr_L and L/self.first_frame_L < 1+2*thr_L:
            if S/self.first_frame_S > 1-2*thr_S and S/self.first_frame_S<1+2*thr_S:

                return True,box_from_mask, [L,S]

        if self.L_min<=L<=self.L_max:
            if self.S_min<=S<self.S_max:
                if L/self.first_frame_L / (S/self.first_frame_S) > 0.9 and L/self.first_frame_L / (S/self.first_frame_S) <1.1:

                    return True,box_from_mask, [L,S]


        return False,None,[None,None]

    def run_on_opencv_image(self, image,visualize=True):

        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        result=None
        if visualize:
            result = image.copy()
            if self.show_mask_heatmaps:
                return self.create_mask_montage(result, top_predictions)
            result = self.overlay_boxes(result, top_predictions)
            if self.cfg.MODEL.MASK_ON:
                result = self.overlay_mask(result, top_predictions)
            if self.cfg.MODEL.KEYPOINT_ON:
                result = self.overlay_keypoints(result, top_predictions)
            result = self.overlay_class_names(result, top_predictions)

        return top_predictions,result

    def show_image(self,a, fig_num=None, title=None):
        """Display a 2D tensor.
        args:
            fig_num: Figure number.
            title: Title of figure.
        """
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

    def change_min_image_size(self,min_image_size):
        self.min_image_size=min_image_size
        self.transforms=self.build_transform()

    def change_confidence_threshold(self,confidence_threshold):
        self.confidence_threshold=confidence_threshold

    def compute_iou(self,box1, box2):

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

        iou = inter / ((area1 + area2).reshape(N,M) - inter)
        return iou

    def compute_box_from_mask(self,prediction):
        """

        :param prediction: Boxlist mode xyxy
        :return: box_from_mask,   (x,y,w,h)
        [L,S],
        flag
        """


        L,S=None,None

        box_crop = prediction.bbox.squeeze().long()

        mask=prediction.get_field("mask").squeeze(0)

        mask=mask[box_crop[1]:box_crop[3],box_crop[0]:box_crop[2]]

        mask_sz=mask.shape
        kenel_sz=round(min(mask_sz)*0.3)
        kenel_sz=(kenel_sz+1)%2+kenel_sz

        # mask_close = cv2.morphologyEx(mask.numpy(), cv2.MORPH_CLOSE, np.ones([11, 11]), iterations=1)
        #
        # mask_open = cv2.morphologyEx(mask.numpy(), cv2.MORPH_OPEN, np.ones([kenel_sz,kenel_sz]), iterations=1)

        contours, _ = cv2.findContours(
            mask.numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE  # cv2.RETR_TREE
        )

        area=[]
        boxes=[]
        for contour in contours:
            (x, y), (w, h), t = cv2.minAreaRect(contour)
            area.append(w*h)
            boxes.append([x,y,w,h,t])
        area=torch.tensor(area)
        boxes=torch.tensor(boxes)

        if len(area)==0:
            return prediction.convert("xywh").bbox,[L,S], "mask_not_confidence"

        area_max,ind=torch.max(area,dim=0)


        X,Y,W,H,T=boxes[ind]

        rbox = cv2.boxPoints(((X, Y), (W, H), T)).reshape(1,8)

        L,S = max(W,H),min(W,H)

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
        s= max(s,0.9)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        box_from_mask = torch.tensor([cx - (w - 1) / 2, cy - (h - 1) / 2, w, h])

        box_from_mask[0:2]=box_from_mask[0:2]+box_crop[0:2].float()

        box_det=prediction.convert("xywh").bbox

        iou=self.compute_iou(self.xywh2xyxy(box_det),self.xywh2xyxy(box_from_mask))

        flag=None
        if iou<0.6:
            flag= "not_confidence"
            # return box_det,[L,S],  flag
        return box_from_mask,[L,S],flag

    def xywh2xyxy(self,box):
        box=box.reshape(-1,4).clone()
        box[:, 2:4] = box[:, 0:2] + box[:, 2:4]

        return box





if __name__ == '__main__':

    # detector=Detector('/home/tangjiuqi097/research/pytracking/pytracking/detection/configs/e2e_mask_rcnn_R_50_FPN_1x_test.yaml')
    # detector=Detector('/home/tangjiuqi097/research/pytracking/pytracking/detection/configs/e2e_mask_rcnn_R_50_FPN_1x_test.yaml')
    detector = Detector(
        '/home/tangjiuqi097/research/pytracking/pytracking/detection/configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x_test.yaml')



    # im_path = '/home/tangjiuqi097/data/vot2019/vot_rgbd/sequences/backpack_blue/color/00000001.jpg'
    # # img=cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    #
    # detector.change_min_image_size(448)
    # detector.change_confidence_threshold(0.5)
    # det,result_image=detector.run_on_opencv_image(cv2.imread(im_path))
    # show_image(result_image,1,"detection results")
    # cv2.imwrite('test.jpg',result_image)
    # # cv2.imshow("Detections",result_image)
    # # cv2.waitKey(0)

    root_path ='/home/tangjiuqi097/data/vot2019/vot_rgbd/sequences'
    save_root_path='/media/tangjiuqi097/ext/det_rgbd'
    seq_names=os.listdir(root_path)
    for seq_name in seq_names:
        if seq_name!='backpack_blue':
            continue

        seq_path=os.path.join(root_path,seq_name,'color')
        try:
            im_names=sorted(os.listdir(seq_path))
        except:
            continue
        print('\n')
        print(seq_name)

        for im_name in tqdm(im_names):
            save_path = os.path.join(save_root_path, seq_name, im_name)
            # if os.path.exists(save_path):
            #     continue

            im_path=os.path.join(seq_path,im_name)
            img=cv2.imread(im_path)

            det, result_image = detector.run_on_opencv_image(img)
            show_image(result_image, 1, "detection results")


            # if  not os.path.exists(os.path.join(save_root_path,seq_name)):
            #     os.makedirs(os.path.join(save_root_path,seq_name),exist_ok=True)
            # cv2.imwrite(save_path, result_image)


    print("done")


