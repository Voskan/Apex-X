from __future__ import annotations

import torch
from torch import Tensor


def bbox_iou(
    box1: Tensor,
    box2: Tensor,
    *,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    MPDIoU: bool = False,
    eps: float = 1e-7,
) -> Tensor:
    """Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1: bounding boxes
        box2: bounding boxes
        xywh: if True, box format is [x, y, w, h], else [x1, y1, x2, y2]
        GIoU: if True, calculate Generalized IoU
        DIoU: if True, calculate Distance IoU
        CIoU: if True, calculate Complete IoU
        MPDIoU: if True, calculate Minimum Point Distance IoU (SOTA)
        eps: epsilon to avoid division by zero

    Returns:
        IoU tensor
    """
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if GIoU or DIoU or CIoU or MPDIoU:
        # Convex hull (smallest enclosing box)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        if MPDIoU:
            # MPDIoU: Minimum Point Distance IoU
            # d1: distance between top-left corners
            # d2: distance between bottom-right corners
            d1 = (b1_x1 - b2_x1) ** 2 + (b1_y1 - b2_y1) ** 2
            d2 = (b1_x2 - b2_x2) ** 2 + (b1_y2 - b2_y2) ** 2
            return iou - d1 / (w1 ** 2 + h1 ** 2 + eps) - d2 / (w1 ** 2 + h1 ** 2 + eps)
        
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / (3.1415926535 ** 2)) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    return iou
