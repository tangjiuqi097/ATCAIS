import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch.nn.functional as F
def show_tensor(a: torch.Tensor, fig_num = None, title = None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    # min_sz=min(a.shape[-2:])
    # scale=400/min_sz
    # a=F.interpolate(a, scale_factor=scale, mode='bilinear')


    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))

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


    # img=a_np
    # img=cv2.putText(img, title, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    # img=cv2.imshow("show tensor {}".format(fig_num),img)
    # cv2.waitKey(1)



def plot_graph(a: torch.Tensor, fig_num = None, title = None):
    """Plot graph. Data is a 1D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim > 1:
        raise ValueError
    plt.figure(fig_num)
    # plt.tight_layout()
    plt.cla()
    plt.plot(a_np)
    if title is not None:
        plt.title(title)
    plt.draw()
    plt.pause(0.001)
