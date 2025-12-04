import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the IOU of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h
        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Take note that x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = x/self.S - 0.5*w, y/self.S - 0.5*h
        x2, y2 = x/self.S + 0.5*w, y/self.S + 0.5*h
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 4) ...]
        box_target : (tensor)  size (-1, 5) ??? [-1, 4]

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Key pointers to consider:

        1. Calculate the Intersection over Union (IoU) for the two bounding boxes within each grid cell of each image.
        2. Use the 'compute_iou' function to compute the IoU values efficiently.
        3. If you need to, apply 'xywh2xyxy' to convert the bounding box format. Note that the initial representation involves 'x' and 'y' as the center of the box, while 'w' and 'h' represent width and height. This transformation is crucial for aligning the correct coordinates to the bounding box format.
        """

        ### CODE ###
        # Your code here
        
        # pred_box_coor = self.xywh2xyxy(pred_box_list)
        target_box_coor = self.xywh2xyxy(box_target[:, :4]) # [n, 4]
        ious = []
        
        for pred_box in pred_box_list:
            # pred_box_coor = self.xywh2xyxy(pred_box[:, :4]) # [n, 4]
            pred_box_coor = self.xywh2xyxy(pred_box)  # [N,4]
            iou = compute_iou(pred_box_coor, target_box_coor)
            
            iou = iou.diag()  # shape [n] [[1, 1], [2, 2], [3, 3] ...]
            ious.append(iou.unsqueeze(1)) # [n, 1]
        
        ious_new = torch.cat(ious, dim=1) # [n, b]
        best_ious, best_idx = torch.max(ious_new, dim=1, keepdim=True) # [n, 1]
        
        best_boxes = torch.zeros((box_target.size(0), 5), device=box_target.device)
        # best_conf = torch.zeros((box_target.size(0), 1))

        for i in range(box_target.size(0)):
            b = int(best_idx[i].item())                         # box index
            best_boxes[i, :4] = pred_box_list[b][i, :4]    # x,y,w,h
            best_boxes[i, 4] = pred_box_list[b][i, 4]        # confidence
 
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        objects = has_object_map.unsqueeze(-1).to(classes_target.device)
        objects = objects.expand_as(classes_target).to(classes_target.device)
        
        pred = classes_pred[objects]
        target = classes_target[objects]
        
        loss = F.mse_loss(pred, target, reduction='sum')
        
        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Some tips:

        1. Calculate the loss solely for cells that do not contain an object.
        2. Compute the loss for all predictions listed within 'pred_boxes_list'.
        3. For non-object cells, it's acceptable to presume that the ground truth confidence is 0.
    
        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        
        objects = has_object_map.unsqueeze(-1).to(pred_boxes_list[0].device)
        objects = objects.expand_as(pred_boxes_list[0]).to(pred_boxes_list[0].device)
        
        loss = 0.0
        for pred_box in pred_boxes_list:
            pred_box = pred_box[~objects]
            target_box = torch.zeros_like(pred_box)
            loss += self.l_noobj * F.mse_loss(pred_box[..., 4], target_box[..., 4], reduction='sum')

        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        Treat box_target_conf as the ground truth.

        """
        ### CODE
        # your code here
        box_target_conf = box_target_conf.to(box_pred_conf.device)
        loss = F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        box_target_response = box_target_response.to(box_pred_response.device)
        reg_loss = self.l_coord * F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum')
        pred_sqrt_wh = torch.sqrt(box_pred_response[:, 2:])
        target_sqrt_wh = torch.sqrt(box_target_response[:, 2:])
        
        reg_loss += self.l_coord * F.mse_loss(pred_sqrt_wh, target_sqrt_wh, reduction='sum')
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        # print("pred_tensor.device:", pred_tensor.device)
        # print("target_boxes.device:", target_boxes.device)
        # print("target_cls.device:", target_cls.device)
        # print("has_object_map.device:", has_object_map.device)
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split pred_tensor to separate tensors:
        # -- pred_boxes_list: this is a list with all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (contains all classification prediction)
        pred_boxes_list = []
        r = 0
        for i in range(self.B):
            pred_boxes_list.append(pred_tensor[..., r:r+5])
            r += 5
    
        pred_cls = pred_tensor[..., self.B*5:]

        # compute classification loss
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        # You'll need to reshape boxes in pred_boxes_list and target_boxes so that:
        # 1) you only keep those cells that have objects
        # 2) It's recommended to vectorize all the dimensions except for the last one so the computation is faster
        
        objects = has_object_map.unsqueeze(-1).to(pred_tensor.device) # (N,S,S,1)
        
        pred_boxes_list_vec = []
        for i in range(self.B):
            pred_boxes_vec = pred_boxes_list[i][objects.expand_as(pred_boxes_list[i]).to(pred_boxes_list[i].device)]
            pred_boxes_vec = pred_boxes_vec.view(-1, 5)
            pred_boxes_list_vec.append(pred_boxes_vec)
        
        box_target_vec = target_boxes[objects.expand_as(target_boxes)].to(target_boxes.device).view(-1, 4)

        # Next, find the best boxes among those 2 (or self.B) (number of predicted boxes) predicted boxes and the IOU
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list_vec, box_target_vec)

        # compute the regression loss between the best bbox that you found and the GT bbox for all the cells containing objects
        reg_loss = self.get_regression_loss(best_boxes[..., :4], box_target_vec)

        # compute the contain_object_loss
        box_pred_conf = best_boxes[..., 4:5]

        # conf = best IoU
        box_target_conf = best_ious.view(-1, 1).to(box_pred_conf.device)

        contain_obj_loss = self.get_contain_conf_loss(box_pred_conf, box_target_conf)

        # compute the final loss
        
        total_loss = reg_loss + contain_obj_loss + no_obj_loss + cls_loss

        # return loss_dict
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=contain_obj_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss,
        )
        return loss_dict
