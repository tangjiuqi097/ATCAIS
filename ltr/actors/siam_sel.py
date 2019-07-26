from . import BaseActor


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
        predict_scores1,labels1,predict_scores2,labels2 = self.net(**data)


        # # Compute loss
        loss1 = self.objective(predict_scores1.reshape(-1), labels1.reshape(-1))
        loss2 = self.objective(predict_scores2.reshape(-1), labels2.reshape(-1))
        loss=loss1+loss2

        stats={'loss':loss}

        return loss, stats