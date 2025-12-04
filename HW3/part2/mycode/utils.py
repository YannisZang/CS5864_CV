import random

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def reset_seed(number):
    random.seed(number)
    torch.manual_seed(number)
    return


def tensor_to_image(tensor):
    """
    Convert a tensor into a numpy ndarray.
    
    Inputs:
    - tensor: range [0, 1]
    
    Returns:
    - ndarr: uint8 numpy array 
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def vis_dataset(X_data, y_data, samples_per_class, class_list):
    img_half_width = X_data.shape[2] // 2
    samples = []
    for y, cls in enumerate(class_list):
        plt.text(
            -4,
            (img_half_width * 2 + 2) * y + (img_half_width + 2),
            cls,
            ha="right",
        )
        idxs = (y_data == y).nonzero().view(-1)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(X_data[idx])

    img = make_grid(samples, nrow=samples_per_class)
    return tensor_to_image(img)


def det_vis(img, idx_to_class, bbox=None, pred=None, points=None):
    if isinstance(img, torch.Tensor):
        img = (img * 255).permute(1, 2, 0)
    img_copy = np.array(img).astype("uint8")
    _, ax = plt.subplots(frameon=False)
    ax.axis("off")
    ax.imshow(img_copy)
    if points is not None:
        points_x = [t[0] for t in points]
        points_y = [t[1] for t in points]
        ax.scatter(points_x, points_y, color="yellow", s=24)
    if bbox is not None:
        for single_bbox in bbox:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(1.0, 0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                ax.text(
                    x0, y0, obj_cls, size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )

    if pred is not None:
        for single_bbox in pred:
            x0, y0, x1, y1 = single_bbox[:4]
            width = x1 - x0
            height = y1 - y0

            ax.add_patch(
                mpl.patches.Rectangle(
                    (x0, y0), width, height, fill=False, edgecolor=(0, 1.0, 0),
                    linewidth=4, linestyle="solid",
                )
            )
            if len(single_bbox) > 4:  # if class info provided
                obj_cls = idx_to_class[single_bbox[4].item()]
                conf_score = single_bbox[5].item()
                ax.text(
                    x0, y0 + 15, f"{obj_cls}, {conf_score:.2f}",
                    size=18, family="sans-serif",
                    bbox={
                        "facecolor": "black", "alpha": 0.8,
                        "pad": 0.7, "edgecolor": "none"
                    },
                    verticalalignment="top",
                    color=(1, 1, 1),
                    zorder=10,
                )
    plt.show()
