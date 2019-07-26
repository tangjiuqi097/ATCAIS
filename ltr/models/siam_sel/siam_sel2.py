import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.apis import init_detector, inference_detector, show_result
from pytracking import TensorDict
import numpy as np

import pycocotools.mask as maskUtils
from ltr.train_settings.siam_sel.bounding_box import BoxList
import mmcv


class SiamSelNet(nn.Module):
    def __init__(self):
        super(SiamSelNet, self).__init__()
        config_file = '/home/tangjiuqi097/vot/pytracking/pytracking/mmdetection/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
        checkpoint_file = '/home/tangjiuqi097/vot/pytracking/pytracking/mmdetection/checkpoints/epoch_3.pth'
        self.detector = init_detector(config_file, checkpoint_file)
        self.detector.CLASSES=['Object']
        self.selector=SelNet()
        self.top_k=7

    def forward(self, *args,**kwargs):
        data=TensorDict(kwargs)
        data = data.apply(lambda x: x[0] if isinstance(x, torch.Tensor) else x)
        img_meta=data['train_img_meta'][0].copy()
        for key in img_meta:
            values = img_meta[key]
            for i in range(len(values)):
                if len(values)>1:
                    img_meta[key][i]=img_meta[key][i].item()
                else:
                    img_meta[key] = img_meta[key][i].item()

        with torch.no_grad():
            self.detector.extract_feat_pre(data['train_images'])
            result=self.detector.simple_test_post(rescale=True,img_meta=[img_meta])
            prediction=self.toBoxlist(result,0.)
            img_shape = img_meta['img_shape']
            box_gt = data['train_anno'].to("cpu")
            if prediction is None:
                roi_box = torch.cat([box_gt])
                gt_bboxes=[torch.cat([box_gt]).to("cuda")]
                gt_bboxes_ignore=None
            else:
                scores=prediction.get_field("scores")
                gt_bboxes_ind=scores>=0.3
                gt_bboxes_ignore_ind=scores<0.3
                gt_bboxes_prediction=prediction[gt_bboxes_ind]
                gt_bboxes_ignore_prediction=prediction[gt_bboxes_ignore_ind]
                gt_bboxes_from_det=gt_bboxes_prediction.resize([img_shape[1], img_shape[0]]).bbox
                gt_bboxes=[torch.cat([box_gt, gt_bboxes_from_det]).to("cuda")]
                gt_bboxes_ignore=[gt_bboxes_ignore_prediction.resize([img_shape[1], img_shape[0]]).bbox.to("cuda")]
                prediction=prediction[gt_bboxes_ind]
                num_pred=len(prediction)
                top_k=min(self.top_k,num_pred)
                scores=prediction.get_field("scores")
                _,ind=torch.topk(scores,top_k)
                boxes = prediction[ind].resize([img_shape[1], img_shape[0]]).bbox
                roi_box=torch.cat([box_gt,boxes])

            gt_labels = [torch.ones(len(gt_bboxes[0]),dtype=torch.int64).to("cuda")]

            iou=self.compute_iou(roi_box,box_gt)
            roi_ind=torch.zeros([len(roi_box),1])

            roi1=torch.cat([roi_ind,roi_box],dim=1).to("cuda")
            labels1 = (iou > 0.7).squeeze().float().to("cuda")
            roi_fea1=self.detector.extract_roi_featrue(roi1)

        loss1=self.detector.forward_train_on_trackingdata(img_meta=[img_meta],gt_bboxes=gt_bboxes,gt_labels=gt_labels,gt_bboxes_ignore=gt_bboxes_ignore)


        img_meta=data['test_img_meta'][0].copy()
        for key in img_meta:
            values = img_meta[key]
            for i in range(len(values)):
                if len(values)>1:
                    img_meta[key][i]=img_meta[key][i].item()
                else:
                    img_meta[key] = img_meta[key][i].item()

        with torch.no_grad():
            self.detector.extract_feat_pre(data['test_images'])
            result=self.detector.simple_test_post(rescale=True,img_meta=[img_meta])
            prediction=self.toBoxlist(result,0.)

            img_shape = img_meta['img_shape']
            box_gt = data['test_anno'].to("cpu")
            if prediction is None:
                roi_box = torch.cat([box_gt])
                gt_bboxes=[torch.cat([box_gt]).to("cuda")]
                gt_bboxes_ignore=None
            else:
                scores=prediction.get_field("scores")
                gt_bboxes_ind=scores>=0.3
                gt_bboxes_ignore_ind=scores<0.3
                gt_bboxes_prediction=prediction[gt_bboxes_ind]
                gt_bboxes_ignore_prediction=prediction[gt_bboxes_ignore_ind]
                gt_bboxes_from_det=gt_bboxes_prediction.resize([img_shape[1], img_shape[0]]).bbox
                gt_bboxes=[torch.cat([box_gt, gt_bboxes_from_det]).to("cuda")]
                gt_bboxes_ignore=[gt_bboxes_ignore_prediction.resize([img_shape[1], img_shape[0]]).bbox.to("cuda")]
                prediction=prediction[gt_bboxes_ind]
                num_pred=len(prediction)
                top_k=min(self.top_k,num_pred)
                scores=prediction.get_field("scores")
                _,ind=torch.topk(scores,top_k)
                boxes = prediction[ind].resize([img_shape[1], img_shape[0]]).bbox
                roi_box=torch.cat([box_gt,boxes])

            gt_labels = [torch.ones(len(gt_bboxes[0]),dtype=torch.int64).to("cuda")]

            iou=self.compute_iou(roi_box,box_gt)
            roi_ind=torch.zeros([len(roi_box),1])

            roi2=torch.cat([roi_ind,roi_box],dim=1).to("cuda")
            labels2 = (iou > 0.7).squeeze().float().to("cuda")
            roi_fea2 = self.detector.extract_roi_featrue(roi2)
        loss2=self.detector.forward_train_on_trackingdata(img_meta=[img_meta],gt_bboxes=gt_bboxes,gt_labels=gt_labels,gt_bboxes_ignore=gt_bboxes_ignore)


        predict_scores1=  self.selector(roi_fea2[0][None,],roi_fea1)
        predict_scores2 = self.selector(roi_fea1[0][None,], roi_fea2)

        return predict_scores1,labels1,predict_scores2,labels2,loss1,loss2

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


class SelNet(nn.Module):
    def __init__(self):
        super(SelNet, self).__init__()

        self.conv1=nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1,bias=False)
        self.fc1 = nn.Linear(6272, 1024)
        self.fc2 = nn.Linear(1024, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, reference_feature,candidate_features):
        assert len(reference_feature)==1
        num_candidate=len(candidate_features)
        reference_features=reference_feature.repeat([num_candidate,1,1,1])
        all_features=torch.cat([reference_features,candidate_features],dim=1)

        x=self.conv1(all_features)
        x=F.relu(x)

        x=x.view(x.size(0), -1)
        x=self.fc1(x)
        x=F.relu(x)

        x=self.fc2(x)
        return x