import extension2d
import numpy as np
import torch
import math

def nms_rotated_base(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.

    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.

    As an extreme example, consider a single character v and the square box around it.

    If the angle is 0 degree, the object (text) would be read as 'v';

    If the angle is 90 degrees, the object (text) would become '>';

    If the angle is 180 degrees, the object (text) would become '^';

    If the angle is 270/-90 degrees, the object (text) would become '<'

    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.

    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)

    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)

    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """

    return extension2d.nms_rotated(boxes, scores, iou_threshold)

def trans_pred2dst_batch(pred_result, area_ratio=0.85):
    # pred_result (n, 10), x1,y1,x2,y2,score1, score2, cls, g1, g2, g3
    pred_boxes = pred_result[:, :4]
    pred_cls = pred_result[:, 6]
    pred_scores = pred_result[:, 4] * pred_result[:, 5]
    pred_glides = pred_result[:, -3:]
    post_points = get_rotate_points_batch(pred_boxes,
                                          pred_glides,
                                          area_ratio=area_ratio)
    rboxes = convert_points2rbox(post_points)

    return rboxes, pred_cls, pred_scores

def nms_rotated(pred_result, area_ratio=0.85, iou_threshold=0.25):
    # pred_result (n, 10), x1,y1,x2,y2,score1, score2, cls, g1, g2, g3
    res, pred_cls, pred_scores = trans_pred2dst_batch(pred_result, area_ratio)
    keep = nms_rotated_base(res, pred_scores, iou_threshold)
    return keep


def batched_nms_rotated(pred_result, area_ratio=0.85, iou_threshold=0.25):
    # pred_result (n, 10), x1,y1,x2,y2,score1, score2, cls, g1, g2, g3
    res, pred_cls, pred_scores = trans_pred2dst_batch(pred_result, area_ratio)
    keep = batched_nms_rotated_base(res, pred_scores, pred_cls, iou_threshold)
    return keep

def get_rotate_points_batch(boxes, glides, area_ratio=0.85):

    gw0 = boxes[:, 0]
    gh0 = boxes[:, 1]
    gw1 = boxes[:, 2]
    gh1 = boxes[:, 3]
    gw = gw1 - gw0
    gh = gh1 - gh0
    glides_1 = glides[:, 0]
    glides_2 = glides[:, 1]
    glides_3 = glides[:, 2]

    idx = glides_3 >= area_ratio
    glides_1[idx] = 0.0
    glides_2[idx] = 0.0

    dst_points = torch.zeros((boxes.shape[0], 4, 2), device=boxes.device)
    dst_points[:, 0, 0] = gw0 + glides_1 * gw
    dst_points[:, 0, 1] = gh0
    dst_points[:, 1, 0] = gw1
    dst_points[:, 1, 1] = gh0 + glides_2 * gh

    dst_points[:, 2, 0] = gw1 - glides_1 * gw
    dst_points[:, 2, 1] = gh1
    dst_points[:, 3, 0] = gw0
    dst_points[:, 3, 1] = gh1 - glides_2 * gh

    return dst_points

def convert_points2rbox(post_points):
    # post_points: [N, 4, 2]

    p0, p1, p2, p3 = post_points[:,
                                 0], post_points[:,
                                                 1], post_points[:,
                                                                 2], post_points[:,
                                                                                 3]
    d1 = torch.sqrt((p0[:, 0] - p1[:, 0])**2 + (p0[:, 1] - p1[:, 1])**2)  # w
    d2 = torch.sqrt((p0[:, 0] - p3[:, 0])**2 + (p0[:, 1] - p3[:, 1])**2)  # h

    angle_1 = torch.zeros(d1.shape[0], device=d1.device)
    angle_2 = torch.zeros(d2.shape[0], device=d1.device)

    idx = (d1 > 0) & (d2 > 0)
    cos_1 = (p1[idx][:, 0] - p0[idx][:, 0]) / d1[idx]
    cos_2 = (p3[idx][:, 1] - p0[idx][:, 1]) / d2[idx]
    angle_1[idx] = torch.acos(cos_1)
    angle_2[idx] = torch.acos(cos_2)
    assert torch.any(angle_1.isnan()) == False
    assert torch.any(angle_2.isnan()) == False
    angle = torch.mean(torch.stack([angle_1, angle_2], dim=1),
                       dim=1) * 180 / 3.1415926
    x_center = torch.mean(post_points[:, :, 0], dim=1)
    y_center = torch.mean(post_points[:, :, 1], dim=1)
    rboxes = torch.stack([x_center, y_center, d1, d2, -angle], dim=1)

    return rboxes
def batched_nms_rotated_base(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0, ), dtype=torch.int64, device=boxes.device)
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    max_coordinate = (torch.max(boxes[:, 0], boxes[:, 1]) +
                      torch.max(boxes[:, 2], boxes[:, 3]) / 2).max()
    min_coordinate = (torch.min(boxes[:, 0], boxes[:, 1]) -
                      torch.max(boxes[:, 2], boxes[:, 3]) / 2).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone(
    )  # avoid modifying the original values in boxes
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nms_rotated_base(boxes_for_nms, scores, iou_threshold)
    return keep

def trans_result_with_angle_mod(result: torch.Tensor,
                                area_ratio: float = 0.85):
    if result is None or len(result) == 0:
        return result
    np_result = result.cpu().numpy()
    pred_boxes = np_result[:, :4]
    pred_scores = np_result[:, 4]
    pred_cls = np_result[:, 5]
    pred_glides = np_result[:, 6:]
    Box_Infor = []
    angle_infor = []
    for k in range(len(pred_boxes)):
        if pred_glides[k][2] < 0:
            continue
        cls = pred_cls[k]
        score = pred_scores[k]
        post_points = get_rotate_boxes(pred_boxes[k], pred_glides[k])
        if pred_glides[k][2] < area_ratio:
            dst_box, angle = get_retangle_boxes_mod(post_points)
        else:
            x_min = np.min(post_points[:, 0])
            y_min = np.min(post_points[:, 1])
            x_max = np.max(post_points[:, 0])
            y_max = np.max(post_points[:, 1])
            dst_box = [x_min, y_min, x_max, y_max]
            angle = 0.0
        dst_box.append(cls)
        dst_box.append(score)
        Box_Infor.append(dst_box)
        angle_infor.append(angle)
    Box_Infor = np.array(Box_Infor).reshape(-1, 6)
    angle_infor = np.array(angle_infor).reshape(-1, 1)
    all_result = np.concatenate([Box_Infor, angle_infor], axis=1)
    return all_result
# 根据网络输出的最小外接正框和角点的偏移量算出斜四边形。
def get_rotate_boxes(boxes1, glides1, area_ratio=0.85):

    # gh0 = boxes1[0]
    # gw0 = boxes1[1]
    # gh1 = boxes1[2]
    # gw1 = boxes1[3]
    gh0 = boxes1[1]  # y1
    gw0 = boxes1[0]  # x1
    gh1 = boxes1[3]  # y2
    gw1 = boxes1[2]  # x2

    if glides1[2] >= area_ratio:
        gp1 = (gw0, gh0)
        gp2 = (gw1, gh0)
        gp3 = (gw1, gh1)
        gp4 = (gw0, gh1)
    else:
        gw = gw1 - gw0
        gh = gh1 - gh0
        gs1 = glides1[0] * gw
        gs2 = glides1[1] * gh
        gs3 = glides1[0] * gw
        gs4 = glides1[1] * gh
        gp1 = (gw0 + gs1, gh0)
        gp2 = (gw1, gh0 + gs2)
        gp3 = (gw1 - gs3, gh1)
        gp4 = (gw0, gh1 - gs4)
    points = np.array([gp1, gp2, gp3, gp4])  # w,h,w,h
    return points


def get_retangle_boxes_mod(boxes):
    p0 = boxes[0]
    p1 = boxes[1]
    p3 = boxes[3]
    d1 = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    d2 = np.sqrt((p0[0] - p3[0])**2 + (p0[1] - p3[1])**2)
    base_vec = np.array([0.0, 1.0])
    v1 = np.array([p1[0] - p0[0], p1[1] - p0[1]])
    v1 = v1 / (d1 + 1e-5)
    v2 = np.array([p3[0] - p0[0], p3[1] - p0[1]])
    v2 = v2 / (d2 + 1e-5)
    cos_angles = v1.dot(v2)  # 计算与y轴的夹角
    angle_v12 = math.acos(cos_angles)
    angle_v12 = angle_v12 * 180. / 3.1416

    max_length = max(d1, d2)
    min_length = min(d1, d2)
    if angle_v12 > 90:
        angle_v12 = 180 - angle_v12
    long_side = max_length + math.cos(angle_v12 * 3.1416 / 180) * min_length
    short_side = math.sin(angle_v12 * 3.1416 / 180) * min_length
    x_min = boxes[:, 0].min()
    y_min = boxes[:, 1].min()
    x_max = boxes[:, 0].max()
    y_max = boxes[:, 1].max()
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    if d1 >= d2:
        cos_angles = v1.dot(base_vec)  # 计算与y轴的夹角
        angle = math.acos(cos_angles)
        angle = angle * 180. / 3.1416
        if angle > 45:  # 若大于45°则取横向框，顺时针转；若小于45°则取纵向框,逆时针转
            angle = 90 - angle
            angle = -1 * angle
            w = long_side
            h = short_side
        else:
            h = long_side
            w = short_side
    else:
        cos_angles = v2.dot(base_vec)
        angle = math.acos(cos_angles)
        angle = angle * 180. / 3.1416
        if angle > 45:  # 若大于45°则取横向框，逆时针转(>0)；若小于45°则取纵向框,顺时针转(<0)
            angle = 90 - angle
            # angle = -1 * angle
            w = long_side
            h = short_side
        else:
            angle = angle * -1
            h = long_side
            w = short_side
    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2
    box = [x1, y1, x2, y2]
    return box, angle