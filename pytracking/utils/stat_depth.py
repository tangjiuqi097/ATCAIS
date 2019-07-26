
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.patches as patches
import numpy as np
import os
import PIL.Image as Image
from pytracking.utils.plotting import show_tensor
import pandas
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
def stat_depth(path):
    seq_names = os.listdir(path)
    for seq_name in seq_names:

        print(seq_name)
        seq_path=os.path.join(path,seq_name)
        gt_path=os.path.join(seq_path,'groundtruth.txt')
        p_occ_gt_path=os.path.join(seq_path,'partial-occlusion.tag')
        f_occ_gt_path = os.path.join(seq_path, 'full-occlusion.tag')
        try:
            gts = pandas.read_csv(gt_path, delimiter=',', header=None, dtype=np.float32, na_filter=True,
                                 low_memory=False).values
            p_occ_gts= pandas.read_csv(p_occ_gt_path, delimiter=',', header=None, dtype=np.float32, na_filter=True,
                                 low_memory=False).values
            f_occ_gts= pandas.read_csv(f_occ_gt_path, delimiter=',', header=None, dtype=np.float32, na_filter=True,
                                 low_memory=False).values
        except:

            continue
        im_names=sorted(os.listdir(os.path.join(seq_path,'depth')))
        for i,im_name in enumerate(im_names):
            if p_occ_gts[i]:
                continue
            if f_occ_gts[i]:
                continue
            if np.isnan(gts[i].sum()):
                continue

            gt =(gts[i].round()).astype(np.int64)


            w,h=gt[2:]
            area=w*h

            if i==0:
                first_fram_area=area
                first_r=w/h
                area_max=area
                area_min=area
                r_max=w/h
                r_min=w/h


            if area>area_max:
                area_max=area
            if area<area_min:
                area_min=area
            if w/h>r_max:
                r_max=w/h
            if w/h<r_min:
                r_min=w/h


            #
            # im_path=os.path.join(seq_path,'depth',im_name)
            # depth_img=Image.open(im_path)
            #
            # depth=np.array(depth_img,dtype=np.float64)
            #
            # pos = [gt[0] + gt[2] / 2, gt[1] + gt[3] / 2]
            # target_size = [gt[2] * 0.5, gt[3] * 0.5]
            #
            # gt[0]=pos[0]-target_size[0]/2
            # gt[1]=pos[1]-target_size[1]/2
            # gt[2]=target_size[0]
            # gt[3] = target_size[1]
            #
            # inside_depth=depth[gt[1]:gt[1]+gt[3], gt[0]:gt[0]+gt[2]]
            # object_depth=np.median(inside_depth[inside_depth>0])
            # if i==0:
            #     first_fram_depth=object_depth
            #     depth_max=object_depth
            #     depth_min=object_depth
            #
            # if object_depth>depth_max:
            #     depth_max=object_depth
            # if object_depth<depth_min:
            #     depth_min=object_depth
            #
            # depth_map = np.array((depth > 0.9 * object_depth) & (depth < 1.1 * object_depth), dtype=np.uint8)
            # show_tensor(torch.tensor(depth), 6, title='depth')
            # show_tensor(torch.tensor(depth_map), 7, title='depth_map')


            # figure=plt.figure(1)
            # plt.tight_layout()
            # plt.cla()
            # plt.imshow(depth)
            # rect = patches.Rectangle((gt[0], gt[1]), gt[2], gt[3], linewidth=1, edgecolor='r',
            #                          facecolor='none')
            # figure.axes[0].add_patch(rect)
            # plt.axis('off')
            # plt.axis('equal')
            #
            # figure=plt.figure(2)
            # plt.tight_layout()
            # plt.cla()
            # plt.imshow(depth_map)
            # rect = patches.Rectangle((gt[0], gt[1]), gt[2], gt[3], linewidth=1, edgecolor='r',
            #                          facecolor='none')
            # figure.axes[0].add_patch(rect)
            # plt.axis('off')
            # plt.axis('equal')
            #
            # plt.draw()
            # plt.pause(0.001)

        # print("first depth {}, max/first {}, min/first {}".format(first_fram_depth,depth_max/first_fram_depth,depth_min/first_fram_depth))
        # print("first area {:.2f},  max_area {:.2f}, min_area {:.2f},  first r {:.2f},   r_max {:.2f},  r_min {:.2f}"
        #       .format(first_fram_area,area_max,area_min,first_r,r_max,r_min))
        print("{}".format(area_max/first_fram_area))


if __name__ == '__main__':
    path='/home/tangjiuqi097/data/vot2019/vot_rgbd/sequences'
    stat_depth(path)

