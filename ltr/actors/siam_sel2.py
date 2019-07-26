from . import BaseActor
import torch
from collections import OrderedDict
class SiamSelActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        predict_scores1,labels1,predict_scores2,labels2,loss_det1,loss_det2 = self.net(**data)


        # # Compute loss
        loss1 = self.objective(predict_scores1.reshape(-1), labels1.reshape(-1))
        loss2 = self.objective(predict_scores2.reshape(-1), labels2.reshape(-1))

        loss_cls1,_=self.parse_losses(loss_det1)
        loss_cls2,_=self.parse_losses(loss_det2)
        # loss_det=0
        # for key in loss_det1:
        #     if key.find("loss")>=0 and key.find("cls")>=0:
        #         if isinstance(loss_det1[key],torch.Tensor):
        #             loss_det+=loss_det1[key]
        #         else:
        #             for value in loss_det1[key]:
        #                 loss_det+=value
        #
        # for key in loss_det2:
        #     if key.find("loss")>=0 and key.find("cls")>=0:
        #         if isinstance(loss_det2[key],torch.Tensor):
        #             loss_det+=loss_det2[key]
        #         else:
        #             for value in loss_det2[key]:
        #                 loss_det+=value

        loss=0.1*loss_cls1+0.1*loss_cls2+loss1+loss2

        stats={'loss_sum':loss.item(),'loss_cls':0.1*loss_cls1.item()+0.1*loss_cls2.item(),'loss_sel':loss1.item()+loss2.item()}

        return loss, stats

    def parse_losses(self,losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'cls' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars