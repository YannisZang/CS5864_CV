# -*- coding: utf-8 -*-
from tarfile import DIRTYPE
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


class DetBackboneFPN(nn.Module):
    r"""
    RegNet + FPN. Takes in batches of input images with
    shape `(B, 3, H, W)` from FPN levels
        - level p3: (3, H /  8, W /  8)      total stride =  8
        - level p4: (3, H / 16, W / 16)      total stride = 16
        - level p5: (3, H / 32, W / 32)      total stride = 32
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        _cnn = models.regnet_x_400mf(pretrained=True)

        # By default, torchvision models provide only the final-level features.
        # However, detector backbones that use FPN need access to multi-scale
        # intermediate outputs. To handle this, we wrap the ConvNet using
        # torchvision’s feature extractor utility. This allows us to retrieve
        # feature maps labeled as (c3, c4, c5), which align in stride with the
        # (p3, p4, p5) levels mentioned earlier.

        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )
        

        # Send a placeholder batch of input images through the network to determine the shapes of (c3, c4, c5).
        # The 'features' dictionary contains entries with the same keys as described earlier.
        # Each value is a batch of tensors in NHWC format, representing intermediate feature maps
        # produced by the backbone model.

        d_out = self.backbone(torch.randn(2, 3, 224, 224))
        d_out_shapes = [(key, value.shape) for key, value in d_out.items()]

        print("For dummy input with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in d_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Set up extra convolution layers for the feature pyramid.     #
        # TIP: Refer to `output_shapes` to determine channel configuration.  #
        ######################################################################
        # Using ModuleDict lets PyTorch track the learnable parameters inside
        # even though we’re building a dictionary-like container.
        

        self.fpn_params = nn.ModuleDict()

        # C3 = None # TO DO!
        # C4 = None # TO DO!
        # C5 = None # TO DO!

        ### IMPLEMENT ME !!!!!!!
        C3_channels = d_out["c3"].shape[1]
        C4_channels = d_out["c4"].shape[1]
        C5_channels = d_out["c5"].shape[1]
        
        C3 = nn.Conv2d(C3_channels, out_channels, kernel_size=1)
        C4 = nn.Conv2d(C4_channels, out_channels, kernel_size=1)
        C5 = nn.Conv2d(C5_channels, out_channels, kernel_size=1)

        self.fpn_params["c3"] = C3
        self.fpn_params["c4"] = C4
        self.fpn_params["c5"] = C5

        ################################################################
        #                      END                                     #
        ################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to FPN level.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Feature maps at different resolutions, stored with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Construct outputs (p3, p4, p5) using       #
        # the corresponding RegNet intermediate maps (c3, c4, c5) along      #
        # with the FPN-specific convolutional layers you've defined above.  #
        # TIP: To scale the spatial dimensions appropriately, make use of   #
        # the `F.interpolate` function for upsampling.                      #
        ######################################################################
        
        ### IMPLEMENT ME!!!!!!!!!!!!!!!!
        c3_out = self.fpn_params["c3"](backbone_feats["c3"])
        c4_out = self.fpn_params["c4"](backbone_feats["c4"])
        c5_out = self.fpn_params["c5"](backbone_feats["c5"])
        
    
        p5 = c5_out
        p4 = c4_out + F.interpolate(p5, size=c4_out.shape[-2:], mode="nearest")
        p3 = c3_out + F.interpolate(p4, size=c3_out.shape[-2:], mode="nearest")
        
        fpn_feats["p3"] = p3
        fpn_feats["p4"] = p4
        fpn_feats["p5"] = p5


        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        return fpn_feats


def get_my_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    For each feature map level in the FPN, compute the corresponding coordinates
    in the original input image for every location on that level’s grid. These
    coordinates correspond to the center of each location's receptive field and
    are used to unify the spatial representation across all levels and with 
    ground truth boxes.

    Args:
        shape_per_fpn_level: A dictionary mapping FPN level identifiers 
            (e.g., "p3", "p4", "p5") to their feature map shapes, where each 
            shape is a tuple of the form (B, C, H, W).
        strides_per_fpn_level: A dictionary using the same keys, where each value 
            is an integer representing how much the spatial resolution has been 
            reduced from the input image to that feature map level. Refer to 
            `backbone.py` for how these strides are determined.

    Returns:
        Dict[str, torch.Tensor]
            A dictionary keyed by FPN levels, each value being a tensor of shape 
            (H * W, 2). Each tensor contains the (x, y) coordinates of the 
            center points, mapped back onto the input image’s coordinate space.
    """

    # These should be (N, 2) tensors with image-space center coordinates.

    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################

        # IMPLEMENT ME!!!!!!!!!!!!!!    
        # H, W = int(feat_shape[2]), int(feat_shape[3])
        B, C, H, W = feat_shape
        w_coords = torch.arange(W, dtype=dtype, device=device)
        h_coords = torch.arange(H, dtype=dtype, device=device)
        
        y, x = torch.meshgrid(h_coords, w_coords, indexing="ij")
        x = (x + 0.5) * level_stride
        y = (y + 0.5) * level_stride
        
        
        location_coords[level_name] = torch.stack([x, y], dim=-1)
        location_coords[level_name] = location_coords[level_name].reshape(-1, 2)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Apply non-maximum suppression to filter out overlapping detection boxes.

    Arguments:
        boxes: A tensor of shape (N, 4), where each row defines a bounding box 
            using its top-left and bottom-right coordinates.
        scores: A tensor of shape (N,) representing the confidence scores 
            associated with each bounding box.
        iou_threshold: A float representing the IoU cutoff — any box with 
            IoU greater than this value (with respect to a higher scored box) 
            will be removed.

    Returns:
        keep: A torch.long tensor containing the indices of the boxes that remain 
            after suppression, ordered by descending confidence score.
            Shape: [number of boxes retained]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    # keep = None
    #############################################################################
    # TASK: Develop a NMS that repeatedly applies:   #
    #       1. From the remaining boxes, identify the one with the top score   #
    #          that hasn’t been picked yet                                     #
    #       2. Discard all other boxes that have significant overlap           #
    #          (i.e., IoU exceeds the given threshold)                         #
    #       3. If there are still boxes left after pruning, repeat the steps   #
    # HINT: CHECK THIS OUT
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    
    # IMPLEMENT ME!!!!!!
    keep = []
    scores_sorted = scores.argsort(descending=True)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while scores_sorted.numel() > 0:
        i = scores_sorted[0].item()
        keep.append(i)
        
        if scores_sorted.numel() == 1:
            break
        xx1 = torch.max(x1[i], x1[scores_sorted[1:]])
        yy1 = torch.max(y1[i], y1[scores_sorted[1:]])
        xx2 = torch.min(x2[i], x2[scores_sorted[1:]])
        yy2 = torch.min(y2[i], y2[scores_sorted[1:]])
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[scores_sorted[1:]] - inter)
        keep_indices = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
        scores_sorted = scores_sorted[keep_indices + 1]
        
    keep = torch.tensor(keep, dtype=torch.long)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_specific_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
