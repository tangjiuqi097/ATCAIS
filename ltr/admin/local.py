class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './checkpoint'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.lasot_dir = '/media/tangjiuqi097/ext/LaSOTBenchmark'
        self.trackingnet_dir = '/media/tangjiuqi097/ext/TrackingNet'
        self.coco_dir = '/home/tangjiuqi097/data/coco'

