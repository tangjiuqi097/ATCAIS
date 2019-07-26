import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


def VOTDataset():
    return VOTDatasetClass().get_sequence_list()


class VOTDatasetClass(BaseDataset):
    """VOT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot_path
        self.sequence_list = self._get_sequence_list2019_rgbd()
        # self.sequence_list = self._get_sequence_list2019()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        # if ground_truth_rect.shape[1] > 4:
        #     gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
        #     gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]
        #
        #     x1 = np.amin(gt_x_all, 1).reshape(-1,1)
        #     y1 = np.amin(gt_y_all, 1).reshape(-1,1)
        #     x2 = np.amax(gt_x_all, 1).reshape(-1,1)
        #     y2 = np.amax(gt_y_all, 1).reshape(-1,1)
        #
        #     ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
        return Sequence(sequence_name, frames, ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['ants1',
                         'ants3',
                         'bag',
                         'ball1',
                         'ball2',
                         'basketball',
                         'birds1',
                         'blanket',
                         'bmx',
                         'bolt1',
                         'bolt2',
                         'book',
                         'butterfly',
                         'car1',
                         'conduction1',
                         'crabs1',
                         'crossing',
                         'dinosaur',
                         'drone_across',
                         'drone_flip',
                         'drone1',
                         'fernando',
                         'fish1',
                         'fish2',
                         'fish3',
                         'flamingo1',
                         'frisbee',
                         'girl',
                         'glove',
                         'godfather',
                         'graduate',
                         'gymnastics1',
                         'gymnastics2',
                         'gymnastics3',
                         'hand',
                         'handball1',
                         'handball2',
                         'helicopter',
                         'iceskater1',
                         'iceskater2',
                         'leaves',
                         'matrix',
                         'motocross1',
                         'motocross2',
                         'nature',
                         'pedestrian1',
                         'rabbit',
                         'racing',
                         'road',
                         'shaking',
                         'sheep',
                         'singer2',
                         'singer3',
                         'soccer1',
                         'soccer2',
                         'soldier',
                         'tiger',
                         'traffic',
                         'wiper',
                         'zebrafish1']

        return sequence_list

    def _get_sequence_list2019(self):
        self.base_path='/home/tangjiuqi097/data/vot2019/vot_short/sequences'
        sequence_list = ['agility',
                         'ants1',
                         'ball2',
                         'ball3',
                         'basketball',
                         'birds1',
                         'bolt1',
                         'book',
                         'butterfly',
                         'car1',
                         'conduction1',
                         'crabs1',
                         'dinosaur',
                         'dribble',
                         'drone1',
                         'drone_across',
                         'drone_flip',
                         'fernando',
                         'fish1',
                         'fish2',
                         'flamingo1',
                         'frisbee',
                         'girl',
                         'glove',
                         'godfather',
                         'graduate',
                         'gymnastics1',
                         'gymnastics2',
                         'gymnastics3',
                         'hand',
                         'hand2',
                         'handball1',
                         'handball2',
                         'helicopter',
                         'iceskater1',
                         'iceskater2',
                         'lamb',
                         'leaves',
                         'marathon',
                         'matrix',
                         'monkey',
                         'motocross1',
                         'nature',
                         'pedestrian1',
                         'polo',
                         'rabbit',
                         'rabbit2',
                         'road',
                         'rowing',
                         'shaking',
                         'singer2',
                         'singer3',
                         'soccer1',
                         'soccer2',
                         'soldier',
                         'surfing',
                         'tiger',
                         'wheel',
                         'wiper',
                         'zebrafish1']

        return sequence_list

    def _get_sequence_list2019_rgbd(self):
        sequence_list = [
            'backpack_blue',
            'backpack_robotarm_lab_occ',
            'backpack_room_noocc_1',
            'bag_outside',
            'bicycle2_outside',
            'bicycle_outside',
            'bottle_box',
            'bottle_room_noocc_1',
            'bottle_room_occ_1',
            'box1_outside',
            'box_darkroom_noocc_1',
            'box_darkroom_noocc_10',
            'box_darkroom_noocc_2',
            'box_darkroom_noocc_3',
            'box_darkroom_noocc_4',
            'box_darkroom_noocc_5',
            'box_darkroom_noocc_6',
            'box_darkroom_noocc_7',
            'box_darkroom_noocc_8',
            'box_darkroom_noocc_9',
            'box_humans_room_occ_1',
            'box_room_noocc_1',
            'box_room_noocc_2',
            'box_room_noocc_3',
            'box_room_noocc_4',
            'box_room_noocc_5',
            'box_room_noocc_6',
            'box_room_noocc_7',
            'box_room_noocc_8',
            'box_room_noocc_9',
            'box_room_occ_1',
            'box_room_occ_2',
            'boxes_backpack_room_occ_1',
            'boxes_humans_room_occ_1',
            'boxes_office_occ_1',
            'boxes_room_occ_1',
            'cart_room_occ_1',
            'cartman',
            'cartman_robotarm_lab_noocc',
            'case',
            'container_room_noocc_1',
            'dog_outside',
            'human_entry_occ_1',
            'human_entry_occ_2',
            'humans_corridor_occ_1',
            'humans_corridor_occ_2_A',
            'humans_corridor_occ_2_B',
            'humans_longcorridor_staricase_occ_1',
            'humans_shirts_room_occ_1_A',
            'humans_shirts_room_occ_1_B',
            'jug',
            'mug_ankara',
            'mug_gs',
            'paperpunch',
            'person_outside',
            'robot_corridor_noocc_1',
            'robot_corridor_occ_1',
            'robot_human_corridor_noocc_1_A',
            'robot_human_corridor_noocc_1_B',
            'robot_human_corridor_noocc_2',
            'robot_human_corridor_noocc_3_A',
            'robot_human_corridor_noocc_3_B',
            'robot_lab_occ',
            'teapot',
            'tennis_ball',
            'thermos_office_noocc_1',
            'thermos_office_occ_1',
            'toy_office_noocc_1',
            'toy_office_occ_1',
            'trashcan_room_occ_1',
            'trashcans_room_occ_1_A',
            'trashcans_room_occ_1_B',
            'trendNetBag_outside',
            'trendNet_outside',
            'trophy_outside',
            'trophy_room_noocc_1',
            'trophy_room_occ_1',
            'two_mugs',
            'two_tennis_balls',
            'XMG_outside',
        ]

        return sequence_list

    def _get_sequence_list2019_long(self):
        sequence_list = [
            'ballet',
            'bicycle',
            'bike1',
            'bird1',
            'boat',
            'bull',
            'car1',
            'car3',
            'car6',
            'car8',
            'car9',
            'car16',
            'carchase',
            'cat1',
            'cat2',
            'deer',
            'dog',
            'dragon',
            'f1',
            'following',
            'freesbiedog',
            'freestyle',
            'group1',
            'group2',
            'group3',
            'helicopter',
            'horseride',
            'kitesurfing',
            'liverRun',
            'longboard',
            'nissan',
            'parachute',
            'person2',
            'person4',
            'person5',
            'person7',
            'person14',
            'person17',
            'person19',
            'person20',
            'rollerman',
            'sitcom',
            'skiing',
            'sup',
            'tightrope',
            'uav1',
            'volkswagen',
            'warmup',
            'wingsuit',
            'yamaha',
        ]

        return sequence_list
