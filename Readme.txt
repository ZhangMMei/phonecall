用yolov7实现打电话识别的检测

train.py：手机检测部分的训练，使用yolo数据集实现训练，训练结果保存至"runs\train\exp"。
	  参数修改：--cfg 'cfg/training/yolov7-phone.yaml'  ；  --data 'data/phone.yaml'

pose.py:用于人体姿态检测，并将耳朵可疑点的坐标映射到原图。
	使用参数yolov7-w6-pose.pt，根据手腕，肩膀，耳朵的距离分析，返回可疑点数组，数组中的每个元素为[ (X_ear,Y_ear),Y_wrist]

detect.py:用于效果展示，使用参数"runs\train\exp\weights\best.pt"进行手机检测，再根据手机与可疑点数组的距离判断是否为打电话的行为，结果存至runs/detect