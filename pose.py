import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
import math

def distance(a,b):
    return math.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None): #将坐标映射到原图im0上，要求coords为torch而非numpy
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[0::3] -= pad[0]  # x padding
    coords[1::3] -= pad[1]  # y padding
    coords[0::3] /= gain
    coords[1::3] /= gain
    clip_coords(coords, img0_shape)
    return coords
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[0::3].clamp_(0, img_shape[1])
    for i in range(len(boxes)):
        if i % 3 == 0:
            boxes[i].clamp_(0, img_shape[1])  # x<->img_shape[1]:w
        elif i % 3 == 1:
            boxes[i].clamp_(0, img_shape[0])  # y<->img_shape[0]:h


def pose_detect(im0,device):
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()
    if torch.cuda.is_available():
        model.half().to(device)

    # image = cv2.imread(im0)
    image = letterbox(im0, 960, stride=64)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)

    output, _ = model(image)
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True) #非极大值抑制
    with torch.no_grad():
        output = output_to_keypoint(output)# target format [batch_id, class_id, x, y, w, h, conf]

    possible_pts=[] #可能打电话的耳朵位置
    for idx in range(output.shape[0]): # every person
        kpts = scale_coords(image.shape[2:],torch.from_numpy(output[idx, 7:].T), im0.shape).round() # Rescale points from img_size to im0 size
        steps = 3   # every point:(x,y,conf)
        # l_ear=3; l_shoulder=5; l_wrist=9;  r_ear=4; r_shoulder=6; r_wrist=10 ;
        l_ear = (int(kpts[3 * steps]), int(kpts[3 * steps + 1]))
        r_ear = (int(kpts[4 * steps]), int(kpts[4 * steps + 1]))

        l_shoulder=(int(kpts[5 * steps]), int(kpts[5 * steps + 1]))
        r_shoulder=(int(kpts[6 * steps]), int(kpts[6 * steps + 1]))

        l_wrist = (int(kpts[9 * steps]), int(kpts[9 * steps + 1]))
        r_wrist = (int(kpts[10 * steps]), int(kpts[10 * steps + 1]))

        # 肩膀到耳朵的距离作为l
        l=max( distance(l_shoulder,l_ear), distance(r_shoulder,r_ear) )
        if abs(l_ear[0]-l_wrist[0])<1.2*l and ((l_wrist[1]>l_shoulder[1] and l_wrist[1]-l_shoulder[1]<=0.65*l) or (l_wrist[1]<=l_shoulder[1] and l_shoulder[1]-l_wrist[1]<1.1*l)):
            possible_pts.append(((l_ear),l_wrist[1]))
        if abs(r_ear[0]-r_wrist[0])<1.2*l and ((r_wrist[1]>r_shoulder[1] and r_wrist[1]-r_shoulder[1]<=0.65*l) or (r_wrist[1]<=r_shoulder[1] and r_shoulder[1]-r_wrist[1]<1.1*l)):
            possible_pts.append(((r_ear),r_wrist[1]))
    return possible_pts
