# -*- coding: utf-8 -*-
from locale import T_FMT
import os
from pydoc import stripid
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import math
from typing import Dict, List, Optional

import torch
from fcos_starter import *
from common import DetBackboneFPN, class_specific_nms, get_my_fpn_location_coords
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops import generalized_box_iou_loss

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


class MyFCOSNetwork(nn.Module):
    """
    FCOS prediction net produces three types of predictions at each spatial location: bounding box offsets,
    object category scores, and a centerness value. This module consists of a series of shared
    convolutional layers (the "stem") followed by separate output layers for each prediction type.

    Refer to the right-hand side of Figure 2 in the FCOS paper for an illustration:
    https://arxiv.org/abs/1904.01355

    In this implementation, we work with feature maps corresponding to levels P3, P4, and P5
    from the FPN, while levels P6 and P7 are not used.
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Total count of target categories that the model should distinguish.
            in_channels: Number of input feature channels. This should match the channel 
                count from the FPN outputs, as this component processes those features directly.
            stem_channels: A list where each entry specifies the output channel size for 
                corresponding layers within the initial convolutional block (stem).
        """

        super().__init__()

        ##########################################################################
        # TASK: Build a sequence (or "stem") made up of alternating 3x3 
        # convolutional layers followed by ReLU activations. There will be two 
        # such sequences: one for classification (`stem_cls`) and one for 
        # bounding box prediction (`stem_box`). These sequences are structurally 
        # identical, but operate separately. 
        #
        # Refer back to the FCOS architecture diagram for clarity—each stem follows 
        # the same structure.
        #
        # When constructing these layers, use the provided `in_channels` and 
        # `stem_channels` to define the number of input and output channels 
        # respectively. Each convolution should have a kernel size of 3x3, stride 1, 
        # and padding that preserves the spatial resolution of the feature map. This 
        # is important since predictions must be made at every position in the 
        # feature map—no locations should be dropped.
        #
        # All convolution layers should have their weights initialized from a normal 
        # distribution with a mean of 0 and a standard deviation of 0.01. Biases 
        # should be initialized to zero.
        ##########################################################################

        # Fill these.
        stem_cls = []
        stem_box = []

        # IMPLEMENT ME !!!!!!!!!!!
        in_dim = in_channels
        for out_dim in stem_channels:
            stem_cls.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
            stem_cls.append(nn.ReLU())
            
            stem_box.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))
            stem_box.append(nn.ReLU())
            
            in_dim = out_dim
        
        # stem_cls.append(nn.Conv2d(in_dim, num_classes, kernel_size=3, stride=1, padding=1))
        
        # stem_box.append(nn.Conv2d(in_dim, 4, kernel_size=3, stride=1, padding=1))
    
        
        
        # Wrap layers defined by you into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        ######################################################################
        # Design three separate convolutional layers (each with a 3x3 kernel)
        # that will make predictions for three different outputs at each
        # location of the input feature map:
        #
        #     1. Raw class scores (`num_classes` values)
        #     2. Bounding box adjustments (4 values representing LTRB offsets)
        #     3. Centerness scores (1 value)
        #
        # NOTE:
        # These layers should directly output raw logits.
        # Do NOT apply any activation functions like sigmoid inside this module.
        # The raw logits should be passed as-is, since PyTorch loss functions are
        # optimized to work directly with them for numerical stability.
        # Sigmoid should be applied at inference time and that should happen
        # OUTSIDE of this module !!!!!! Check out Python documentation on losses for details
        #
        ######################################################################


        # Replace these lines with your code, keep variable names unchanged !!!!!!!!!
        # self.pred_cls = None  # This is the class prediction convolution
        # self.pred_box = None  # This is the box regression convolution
        # self.pred_ctr = None  # This is the centerness convolution

        # IMPLEMENT ME!!!!!!!!!!!
        self.pred_cls = nn.Conv2d(in_dim, num_classes, kernel_size=3, stride=1, padding=1)
        self.pred_box = nn.Conv2d(in_dim, 4, kernel_size=3, stride=1, padding=1)
        self.pred_ctr = nn.Conv2d(in_dim, 1, kernel_size=3, stride=1, padding=1)
        
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # NOTE FOR STUDENTS: When defining `pred_cls`, include a negative bias term.
        # This adjustment helps keep training stable and prevents it from diverging.
        # You don't need to worry about the underlying reasons—just make sure to apply it.
        # This is a trick used especially in object detection or classification tasks 
        # where the number of background (negative) examples hugely outnumbers the positive ones.
        # Basically this helps keep early training stable and avoids exploding gradients or super-wrong predictions.
        
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))  ## DONE FOR YOU!

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Processes FPN-derived features to produce output tensors for each spatial location.
        These outputs are rearranged such that the channel dimension comes last, and the
        spatial (H, W) dimensions are flattened. This layout is useful for both computing
        losses and making predictions.

        Args:
            feats_per_fpn_level: A dictionary of tensors from different FPN layers,
                with keys {"p3", "p4", "p5"}. Each tensor follows the shape
                `(batch_size, fpn_channels, H, W)`. For input images of size (224, 224),
                the spatial resolutions are (28, 14, 7) for levels (p3, p4, p5) respectively.

        Returns:
            A list where each element is a dictionary containing outputs for {"p3", "p4", "p5"}:
            - Classification scores: `(batch_size, H * W, num_classes)`
            - Box offset predictions: `(batch_size, H * W, 4)`
            - Centerness values:      `(batch_size, H * W, 1)`
        """

        ######################################################################
        # TODO: Loop through each level in the FPN pyramid and generate outputs
        # using the prediction heads defined earlier. Recall that the box
        # regression and centerness heads operate on the result of `stem_box`,
        # while the classification head uses the output from `stem_cls`.
        #
        # NOTE: Unlike the original FCOS implementation, which shares the same
        # stem for classification and centerness, this setup aligns with more
        # modern approaches where centerness and box regression share a stem.
        #
        # IMPORTANT: Do NOT apply sigmoid activation to either the centerness
        # or classification outputs at this stage.
        ######################################################################
        # Expected output keys: {"p3", "p4", "p5"} — match the structure of the input.

        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        # IMPLEMENT ME!!!!!
        class_logits["p3"] = self.pred_cls(self.stem_cls(feats_per_fpn_level["p3"]))
        class_logits["p4"] = self.pred_cls(self.stem_cls(feats_per_fpn_level["p4"]))
        class_logits["p5"] = self.pred_cls(self.stem_cls(feats_per_fpn_level["p5"]))
        #  flatten h and w
        class_logits["p3"] = class_logits["p3"].permute(0, 2, 3, 1).flatten(1, 2)
        class_logits["p4"] = class_logits["p4"].permute(0, 2, 3, 1).flatten(1, 2)
        class_logits["p5"] = class_logits["p5"].permute(0, 2, 3, 1).flatten(1, 2)
        
        boxreg_deltas["p3"] = self.pred_box(self.stem_box(feats_per_fpn_level["p3"]))
        boxreg_deltas["p4"] = self.pred_box(self.stem_box(feats_per_fpn_level["p4"]))
        boxreg_deltas["p5"] = self.pred_box(self.stem_box(feats_per_fpn_level["p5"]))
        
        boxreg_deltas["p3"] = boxreg_deltas["p3"].permute(0, 2, 3, 1).flatten(1, 2)
        boxreg_deltas["p4"] = boxreg_deltas["p4"].permute(0, 2, 3, 1).flatten(1, 2)
        boxreg_deltas["p5"] = boxreg_deltas["p5"].permute(0, 2, 3, 1).flatten(1, 2)
        
        centerness_logits["p3"] = self.pred_ctr(self.stem_box(feats_per_fpn_level["p3"]))
        centerness_logits["p4"] = self.pred_ctr(self.stem_box(feats_per_fpn_level["p4"]))
        centerness_logits["p5"] = self.pred_ctr(self.stem_box(feats_per_fpn_level["p5"]))
        
        centerness_logits["p3"] = centerness_logits["p3"].permute(0, 2, 3, 1).flatten(1, 2)
        centerness_logits["p4"] = centerness_logits["p4"].permute(0, 2, 3, 1).flatten(1, 2)
        centerness_logits["p5"] = centerness_logits["p5"].permute(0, 2, 3, 1).flatten(1, 2)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_my_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    for level_name, centers in locations_per_fpn_level.items():
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_the_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Calculate offset distances from feature map points to the edges of
    corresponding ground-truth boxes. These offsets, referred to as
    "deltas", follow the `(left, top, right, bottom)` or `LTRB` convention.
    The input coordinates for both the feature points and the boxes are in
    absolute image space.

    These computed values act as learning targets for the FCOS model's
    bounding box regression and centerness prediction heads. All output
    deltas should be divided by the stride of the FPN level from which
    the features originate (see `get_my_fpn_location_coords`). For entries
    representing background regions, the deltas must be filled with
    `(-1, -1, -1, -1)`.

    NOTE: This routine should work regardless of whether class information
    is included in the GT boxes. It should support input GT tensors shaped
    either `(N, 4)` or `(N, 5)`. Background boxes may also appear in either
    format with all entries as `-1`.

    Args:
        locations: A tensor with shape `(N, 2)` containing the `(x, y)`
            coordinates of the feature map points.
        gt_boxes: A tensor of shape `(N, 4)` or `(N, 5)` representing
            ground-truth boxes, with or without a class label.
        stride: Integer representing the stride value of the associated
            feature level in the FPN.

    Returns:
        torch.Tensor:
            A tensor of shape `(N, 4)` containing the normalized LTRB deltas
            for each feature point.
    """
    ##########################################################################
    # TODO: Below you need to write code to get deltas from feature locs        #
    ##########################################################################
    # Set to a tensor of (N, 4) to get deltas (left, top, right, bottom)
    # from the locs to the GT edges. Make sure to normalize by FPN stride!!
    deltas = torch.zeros((locations.size(0), 4), device=locations.device)

    #### IMPLEMENT ME !!!!!!!!!!!!
    gt_boxes = gt_boxes[..., :4]
    bg_boxes = (gt_boxes == -1).all(dim=1)
    deltas[bg_boxes] = -1
    
    loc = locations[~bg_boxes]
    gt = gt_boxes[~bg_boxes]
    
    x, y = loc[:, 0], loc[:, 1]
    # non bg ground truth boxes
    x1, y1, x2, y2 = gt[:, 0], gt[:,1 ], gt[:, 2], gt[:, 3]
    
    l = (x - x1) / stride
    t = (y - y1) / stride
    r = (x2 - x) / stride
    b = (y2 - y) / stride
    
    non_bg_deltas = torch.stack([l, t, r, b], dim=1)
    
    deltas[~bg_boxes] = non_bg_deltas
    

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return deltas


def fcos_apply_the_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    This function reverses the effect of `fcos_get_the_deltas_from_locations`.

    You are provided with offset values (left, top, right, bottom) and the
    corresponding pixel-level coordinates from a Feature Pyramid Network (FPN).
    Your task is to recover the predicted bounding boxes by adjusting the
    provided coordinates using the offsets.

    During training, deltas were normalized using the feature stride. At
    inference time, we need to do the opposite—scale them back up using the
    stride value, since the input locations are already in the image's
    coordinate space.

    Args:
        deltas (torch.Tensor): A tensor with shape `(N, 4)` representing the
            offsets for each edge of the box.
        locations (torch.Tensor): A tensor with shape `(N, 2)` giving the
            base coordinates to apply the offsets to.
        stride (int): The stride value associated with the current feature map
            level.

    Returns:
        torch.Tensor:
            A tensor with the same shape as `deltas` and `locations` where
            each row represents the bounding box in `(x1, y1, x2, y2)` format,
            all in absolute image coordinates.
    """
    ##########################################################################
    # TODO: Implement the code to get boxes.                 #
    #                                                                        #
    # NOTE: Clip any negatives to zero! (It needs to lie inside, so neg isn't ok!)
    ##########################################################################

    # IMPLEMENT ME!!!!
    output_boxes = torch.zeros((locations.size(0), 4), device=locations.device)

    # background mask
    bg = (deltas == -1).all(dim=1)

    # background output = (x, y, x, y)
    output_boxes[bg, 0] = locations[bg, 0]
    output_boxes[bg, 1] = locations[bg, 1]
    output_boxes[bg, 2] = locations[bg, 0]
    output_boxes[bg, 3] = locations[bg, 1]

    deltas_fg = deltas[~bg]
    loc_fg = locations[~bg]
    
    l, t, r, b = deltas_fg[:, 0], deltas_fg[:, 1], deltas_fg[:, 2], deltas_fg[:, 3]
    
    l *= stride
    t *= stride
    r *= stride
    b *= stride
    # may need to modified, as the location shape is [N, 4]
    x, y = loc_fg[:, 0], loc_fg[:, 1]
    
    x1 = x - l
    y1 = y - t
    x2 = x + r
    y2 = y + b
    
    x1 = torch.clamp(x1, min=0)  
    y1 = torch.clamp(y1, min=0)
    x2 = torch.clamp(x2, min=0)
    y2 = torch.clamp(y2, min=0)
    
    output_boxes[~bg] = torch.stack([x1, y1, x2, y2], dim=1)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return output_boxes


def fcos_make_my_centerness_targets(deltas: torch.Tensor):
    """
    Computes the target values for centerness prediction based on LTRB-style 
    deltas derived from ground-truth boxes. For context on how these deltas 
    are formed, refer to the `fcos_get_the_deltas_from_locations` method. 

    If a location does not correspond to any valid object (i.e., is background), 
    it is represented by a delta of `(-1, -1, -1, -1)`, and the resulting 
    centerness target should be `-1`.

    For the formula used, see Equation 3 in the FCOS:
    https://arxiv.org/abs/1904.01355

    Args:
        deltas: A `(N, 4)` tensor where each row corresponds to the 
                [left, top, right, bottom] distances from a point to the 
                edges of a ground-truth box.

    Returns:
        torch.Tensor: A 1D tensor of length `N` containing centerness targets 
                      for each input box.
    """
    ##########################################################################
    # TODO: Implement the centerness calc.                      #
    ##########################################################################

    # IMPLEMENT ME!!!
    centerness = torch.zeros(deltas.size(0), device=deltas.device)
    
    bg = (deltas == -1).all(dim=1)
    
    
    l, t, r, b = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    
    l, t, r, b = l[~bg], t[~bg], r[~bg], b[~bg]
    
    min_lr = torch.min(l, r)
    max_lr = torch.max(l, r)
    
    min_tb = torch.min(t, b)
    max_tb = torch.max(t, b)
    
    eps = 1e-8
    cen = torch.sqrt((min_lr / (max_lr + eps)) * (min_tb / (max_tb + eps)))
    
    centerness[bg] = -1
    centerness[~bg] = cen
    
    
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return centerness


class FCOS(nn.Module):

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # TODO: Init backbone and network.  #
        ######################################################################
        # Feel free to delete these two lines: (but keep variable names same)
        # self.backbone = None
        # self.pred_net = None
        
        ## IMPLEMENT ME!!!!!!!!
        self.backbone = DetBackboneFPN(fpn_channels)
        self.pred_net = MyFCOSNetwork(num_classes=num_classes, in_channels=fpn_channels, stem_channels=stem_channels)
        
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Smoothing coefficient used for computing training loss;
        # serves as an exponential moving average over detected foreground positions.
        # STUDENTS: Refer to how this is utilized inside the `forward` method 
        # while working on the loss computation section.
        
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: A collection of image tensors with shape `(B, C, H, W)`,
                where `B` is the batch size, `C` is the number of channels,
                and `H`, `W` are spatial dimensions.

            gt_boxes: Optional annotation data for training, formatted as a tensor
                of shape `(B, N, 5)`. Each `gt_boxes[i, j]` corresponds to an object
                in `images[i]` and is structured as `(x1, y1, x2, y2, C)`. Here,
                `(x1, y1)` and `(x2, y2)` define the corners of a bounding rectangle,
                with coordinates in the continuous range `[0, H]` and `[0, W]`.
                The final value `C` is a class index identifying the object category.
                This argument is omitted during evaluation.

            test_score_thresh: A threshold used at test time to suppress predictions
                with confidence scores below this level. Not relevant for training.

            test_nms_thresh: Non-maximum suppression threshold for overlap
                (IoU) when making predictions. This is only used during inference.

        Returns:
            If training, returns loss values. If evaluating, returns predicted outputs.
        """

        ########################################################################
        # TODO: Run the input image through the backbone network, the FPN,     #
        # and the detection head in order to compute the predictions.          #
        # You should return dictionaries keyed by {"p3", "p4", "p5"}, where    #
        # each value contains the predicted class logits, box deltas, and      #
        # centerness scores.                                                   #
        ########################################################################
        # pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = None, None, None
        
        
        # IMPLEMENT ME!!!!
        fpn_feats = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(fpn_feats)
        

        ######################################################################
        # TODO: Compute the absolute center coordinates `(xc, yc)` for all
        # positions across the FPN layers.
        #
        # NOTE: You've already built the required components—this step
        # simply involves invoking them correctly.
        ######################################################################
        # You may remove this comment when done (keep variable names unchanged)
        
        # IMPLEMENT ME !!!!!!!!
        # locations_per_fpn_level = get_my_fpn_location_coords(fpn_feats, self.backbone.fpn_strides)
        
        locations_per_fpn_level = get_my_fpn_location_coords(
            {level: feat.shape for level, feat in fpn_feats.items()},
            self.backbone.fpn_strides,
            dtype=images.dtype,
            device=images.device
        )
        
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # During inference, skip rest of the forward pass.
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
        
        ##############################################################################
        # TODO: Link each location on the feature maps to the appropriate 
        # ground-truth bounding box. Use the helper routine `fcos_match_my_locations_to_gt`
        # to carry this out. Note: this operation should be done individually 
        # for each image—do not attempt to batch process.
        ##############################################################################

        # You'll need to populate the following list. It's a list of dictionaries,
        # one for each image, with keys {"p3", "p4", "p5"} mapping to the matched
        # boxes at each pyramid level.
        matched_gt_boxes = []

        # IMPLEMENT ME !!!!
    #     matched_gt_boxes = {
    #     level_name: None for level_name in locations_per_fpn_level.keys()
    # }
        for i in range(images.size(0)):
            matched_gt = fcos_match_my_locations_to_gt(
                locations_per_fpn_level,
                self.backbone.fpn_strides,
                gt_boxes[i]
            )
            matched_gt_boxes.append(matched_gt)


        # Next, compute the target box regression deltas for the matched boxes.
        # The structure of this list should mirror that of `matched_gt_boxes`.
        
    
        # IMPLEMENT ME!!!!!
        matched_gt_deltas = []
    
        for i in range(images.size(0)):
            deltas_per_image = {} 
            for level_name in ["p3","p4","p5"]:
                deltas = fcos_get_the_deltas_from_locations(
                    locations_per_fpn_level[level_name],
                    matched_gt_boxes[i][level_name],
                    stride=self.backbone.fpn_strides[level_name]
                )
                deltas_per_image[level_name] = deltas
            matched_gt_deltas.append(deltas_per_image)

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Aggregate per-sample lists of dictionaries into a dictionary of batched tensors.
        # Input format: {"p3", "p4", "p5"} → tensors shaped like (batch_size, num_locs, 5 or 4)
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)
        # gt_centerness = default_collate(gt_centerness)

        # Stitch together predictions and targets from all FPN stages.
        # Resulting tensor shape: (batch_size, total_fpn_locations, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        # gt_centerness = self._cat_across_fpn_levels(gt_centerness)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Smoothly update normalizer using a moving average of foreground sample counts.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image


        #######################################################################
        # TODO: Compute the loss values per spatial point for classification, 
        # bounding box adjustment, and center-ness scoring. Be sure to ignore 
        # locations marked as background when computing box and center-ness losses!
        #######################################################################

        # loss_cls, loss_box, loss_ctr = None, None, None
        
        # IMPLEMENT ME!!!!!!
        gt_label = matched_gt_boxes[..., 4].long() # B, N
        t_mask = gt_label != -1 ## non back ground
        
        gt_classes_onehot = F.one_hot(
            gt_label.clamp(min=0),  # 
            num_classes=self.num_classes
        ).float()
        
        gt_classes_onehot[~t_mask] = 0.0
        
        # focal loss（torchvision）
        loss_cls = sigmoid_focal_loss(
            pred_cls_logits,      # (B, N, C)
            gt_classes_onehot,    # (B, N, C)
            reduction="none"
        )  # (B, N, C)
        
        loss_cls = loss_cls.sum(dim=-1)
        
        gt_centerness = fcos_make_my_centerness_targets(matched_gt_deltas.reshape(-1, 4))
        gt_centerness = gt_centerness.reshape(matched_gt_deltas.shape[:2])  # (B, N)
        ## new added
        pred_ctr_logits = pred_ctr_logits.squeeze(-1)
        
        ctr_fg_mask = gt_centerness != -1


        #####
        loss_ctr = F.binary_cross_entropy_with_logits(
            pred_ctr_logits[ctr_fg_mask], 
            gt_centerness[ctr_fg_mask],
            reduction="none"
        )
        
        ### ctr loss
        # GIoU loss

        fg = t_mask  # (B, N)

        deltas_fg   = pred_boxreg_deltas[fg]       
        gt_boxes_fg = matched_gt_boxes[fg][..., :4]

        # cat locations
        all_locations = torch.cat([
            locations_per_fpn_level["p3"],
            locations_per_fpn_level["p4"],
            locations_per_fpn_level["p5"],
        ], dim=0).unsqueeze(0).expand(images.size(0), -1, 2)
        locs_fg = all_locations[fg]

        # strides
        all_strides = torch.cat([
            torch.full((locations_per_fpn_level["p3"].shape[0],), self.backbone.fpn_strides["p3"]),
            torch.full((locations_per_fpn_level["p4"].shape[0],), self.backbone.fpn_strides["p4"]),
            torch.full((locations_per_fpn_level["p5"].shape[0],), self.backbone.fpn_strides["p5"]),
        ], dim=0).to(images.device)
        all_strides = all_strides.unsqueeze(0).expand(images.size(0), -1)
        stride_fg = all_strides[fg]

        # decode delta → bbox
        pred_boxes_fg = fcos_apply_the_deltas_to_locations(
            deltas_fg,
            locs_fg,
            stride_fg
        )


        # GIoU loss
        loss_box = generalized_box_iou_loss(
            pred_boxes_fg,
            gt_boxes_fg,
            reduction="none"
        )


        # loss_box = F.l1_loss(
        #     pred_boxreg_deltas[t_mask],
        #     matched_gt_deltas[t_mask],
        #     reduction="none"
        # ).sum(dim=-1)  
        ######################################################################
        #                            END OF YOUR CODE                             #
        ######################################################################

        # Sum all locations and average by the EMA .
        # In training, we add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Execute prediction logic for a single image input (assumes batch size of 1).
        Input parameters align with those in the `forward` method and should not be
        invoked from outside that context.

        Outputs:
            A trio of tensors:
                - pred_boxes: A tensor shaped `(N, 4)` representing *absolute* XYXY
                  coordinates for the predicted bounding boxes.

                - pred_classes: A tensor of shape `(N,)` indicating the predicted
                  class index for each box (each entry corresponds to one of the
                  `num_classes`, and should exclude background class, i.e., no -1s).

                - pred_scores: A tensor of shape `(N,)` containing the confidence
                  values for each prediction. These are calculated as:
                  `sqrt(class_prob * ctrness)`, where both components result from
                  applying a sigmoid function to their respective logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locs and preds from a single level.
            # Index preds by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # TODO : This method combines the class probability with the 
            # centerness score using their geometric mean, as done in FCOS. 
            # This approach helps suppress boxes that are far from the center 
            # of objects.
            #
            # Do the following steps:
            #   1. Determine the class with the highest score and retrieve 
            #      that score for each box using level_pred_scores: 
            #      shape (N, num_classes) => (N,)
            #   2. Filter out predictions that do not exceed the confidence 
            #      threshold provided as an argument.
            #   3. Decode the final bounding boxes from the predicted deltas 
            #      and anchor locations.
            #   4. Ensure the resulting boxes stay within image boundaries by 
            #      clipping box coordinates that extend beyond the image’s 
            #      height and width.
            ##################################################################

            # level_pred_boxes, level_pred_classes, level_pred_scores = (
            #     None,
            #     None,
            #     None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            # )
            
            # IMPLEMENT ME !!!!!!!!!!!!!
            cls_probs = torch.sigmoid(level_cls_logits) 
            cls_scores, cls_targets = cls_probs.max(dim=1)
            
            ctr_probs = torch.sigmoid(level_ctr_logits.squeeze(1))
            
            final_scores = torch.sqrt(cls_scores * ctr_probs)
            
            keep = final_scores > test_score_thresh
            
            final_scores = final_scores[keep]
            cls_targets = cls_targets[keep]
            level_deltas = level_deltas[keep]
            level_locations = level_locations[keep]
            
            stride =  self.backbone.fpn_strides[level_name]
            level_pred_boxes = fcos_apply_the_deltas_to_locations(
                level_deltas, 
                level_locations, 
                stride
            )
            
            H, W = images.shape[2], images.shape[3]
            level_pred_boxes[:, 0::2] = level_pred_boxes[:, 0::2].clamp(min=0, max=W)
            level_pred_boxes[:, 1::2] = level_pred_boxes[:, 1::2].clamp(min=0, max=H)

            level_pred_classes = cls_targets
            level_pred_scores = final_scores
            
            
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine preds and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        # Note that this depends on your implementation of NMS!!! 
        keep = class_specific_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
