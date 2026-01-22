import os
import onnxruntime
import numpy as np
import math


def tiling_inference(session, lr, overlapping, patch_size):
    _, _, h, w = lr.shape
    sr = np.zeros((1, 3, 2*h, 2*w))
    n_h = math.ceil(h / float(patch_size[0] - overlapping))
    n_w = math.ceil(w / float(patch_size[1] - overlapping))
    #every tilling input has same size of patch_size
    for ih in range(n_h):
        h_idx = ih * (patch_size[0] - overlapping)
        h_idx = h_idx if h_idx + patch_size[0] <= h else h - patch_size[0]
        for iw in range(n_w):
            w_idx = iw * (patch_size[1] - overlapping)
            w_idx = w_idx if w_idx + patch_size[1] <= w else w - patch_size[1]

            tilling_lr = lr[..., h_idx: h_idx+patch_size[0], w_idx: w_idx+patch_size[1]]
            sr_tiling = session.run(None, {session.get_inputs()[0].name: tilling_lr.transpose(0,2,3,1)})[0].transpose(0,3,1,2)

            left, right, top, bottom = 0, patch_size[1], 0, patch_size[0]
            left += overlapping//2 
            right -= overlapping//2
            top += overlapping//2
            bottom -= overlapping//2
            #processing edge pixels
            if w_idx == 0:
                left -= overlapping//2
            if h_idx == 0:
                top -= overlapping//2
            if h_idx+patch_size[0]>=h:
                bottom += overlapping//2
            if w_idx+patch_size[1]>=w:
                right += overlapping//2
            
            #get preditions
            sr[... , 2*(h_idx+top): 2*(h_idx+bottom), 2*(w_idx+left): 2*(w_idx+right)] = sr_tiling[..., 2*top:2*bottom, 2*left:2*right]   
    return sr