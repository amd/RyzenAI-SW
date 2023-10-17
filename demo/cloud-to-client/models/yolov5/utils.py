# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import numpy as np

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    eps = 1e-7
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + eps)
        h = np.maximum(0.0, yy2 - yy1 + eps)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    max_wh = 640  # (pixels) minimum and maximum box width and height

    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])
        j = np.expand_dims(x[:,5:].argmax(1), axis=1)
        conf = np.take_along_axis(x[:,5:], j, axis=1)
        x = np.concatenate((box, conf, j.astype('float32')), 1)[conf.reshape(-1) > conf_thres]


        c = x[:, 5:6] * max_wh  # classes
        
        x[:,:4] += c
        i = np.array(py_cpu_nms(x, iou_thres)).astype('uint')
        x[:,:4] -= c
        output.append(x[i])

    return output
