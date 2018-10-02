"""Detector functions with different imread methods"""

import ctypes
from darknet_libwrapper import *


def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(ctypes.c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def _detector(net, meta, image, thresh=.5, hier=.5, nms=.45):
    cuda_set_device(0)
    num = ctypes.c_int(0)
    num_ptr = ctypes.pointer(num)
    network_predict_image(net, image)
    dets = get_network_boxes(net, image.w, image.h, thresh, hier, None, 0, num_ptr)
    num = num_ptr[0]
    if (nms):
         do_nms_sort(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                # Notice: in Python3, mata.names[i] is bytes array from c_char_p instead of string
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res

# Darknet
net = load_network("cfg/yolov3.cfg", "yolov3.weights", 0)
meta = get_metadata("cfg/coco.data")
im = load_image_color('data/traffic.jpg', 0, 0)
result = _detector(net, meta, im)
free_image(im)
print ('Darknet:\n', result)



