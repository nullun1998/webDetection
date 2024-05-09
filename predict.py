
import time
from pathlib import Path


import cv2
import torch
from models.experimental import attempt_load
from numpy import random
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import time_synchronized,select_device
from utils.datasets import  LoadImages
from backend.flask_id2name import id2name
from utils.plots import plot_one_box,plot_one_box_linebot
import time
from random import randint



def predict(opt, model, img):
    out,weights, source, view_img, save_img, save_txt, imgsz = \
        opt['output'],opt['weights'], opt['source'], opt['view_img'], opt['save_img'], opt['save_txt'], opt['imgsz']

    #webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    randomNumber = randint(1000,9999)
    LLt = time.localtime()
    T_tfnT = time.strftime("%Y%m%d-%H%M%S",LLt)
    randomdate = T_tfnT+'-'+str(randomNumber)
    pathSave = 'C:\yolov7\Savapath\{}.jpg'.format(randomdate)
    pathSaveTXT = 'C:\yolov7\Savapath\{}'.format(randomdate)


    # Initialize
    device = select_device(opt['device']) # 選擇設備
    """
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    """
    half = device.type != 'cpu'  # half precision only supported on CUDA

    im0_shape = img.shape # 記下原圖片W,H
    print('im0_shape = %s \n' % str(im0_shape))

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(opt['source'], img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1 # init img
    #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt['augment'])[0]


        # Inference
        t1 = time_synchronized()
        
        # 前向推理
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt['augment'])[0]
        # Apply NMS（非極大抑制）
        t2 = time_synchronized()
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
       
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name) # 保存路徑
            txt_path = str(Path(out) / randomdate) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            cv2.imwrite(pathSave, im0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                boxes_detected = [] #檢測結果
                detectName = []
                for *xyxy, conf, cls in det:
                    xywh2 = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist() 
                    detectName.append({"classes":int(cls.item()),"name": id2name[int(cls.item())]})
                    boxes_detected.append({"name": id2name[int(cls.item())],
                                    "conf": str(conf.item()),
                                    "bbox": [int(xywh2[0]), int(xywh2[1]), int(xywh2[2]), int(xywh2[3])]
                                    })
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        with open(pathSaveTXT + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t3 - t1))
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

    if len(det) == 0:
        return 0
    else:
        resultsBbox = {"results": boxes_detected,"imgName":randomdate}
        classesName=detectName
        return resultsBbox,classesName



#########linebot#############


def predict_linbot(opt, model, img_path):
    out,weights, source, view_img, save_img, save_txt, imgsz = \
        opt['output'],opt['weights'], opt['source'], opt['view_img'], opt['save_img'], opt['save_txt'], opt['imgsz']

    #webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    randomNumber = randint(1000,9999)
    LLt = time.localtime()
    T_tfnT = time.strftime("%Y%m%d-%H%M%S",LLt)
    randomdate = T_tfnT+'-'+str(randomNumber)
    pathSave = 'C:\yolov7\linebot_imgSave\{}.jpg'.format(randomdate)
    pathSaveTXT = 'C:\yolov7\linebot_imgSave\{}'.format(randomdate)

    source_lineget = img_path
    # Initialize
    device = select_device(opt['device']) # 選擇設備
    """
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    """
    
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(source_lineget, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1 # init img
    #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt['augment'])[0]


        # Inference
        t1 = time_synchronized()
        
        # 前向推理
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt['augment'])[0]
        # Apply NMS（非極大抑制）
        t2 = time_synchronized()
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
       
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name) # 保存路徑
            txt_path = str(Path(out) / randomdate) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            cv2.imwrite(pathSave, im0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                boxes_detected = [] #檢測結果
                detectName = []

                for *xyxy, conf, cls in det:
                    xywh2 = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist() 
                    detectName.append({"classes":int(cls.item()),"name": id2name[int(cls.item())]})
                    boxes_detected.append({"name": id2name[int(cls.item())],
                                    "conf": str(conf.item()),
                                    "bbox": [int(xywh2[0]), int(xywh2[1]), int(xywh2[2]), int(xywh2[3])]
                                    })
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        with open(pathSaveTXT + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box_linebot(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t3 - t1))

            cv2.imwrite(save_path, im0)
            # Save results (image with detections)

    linbot_getimgpath = "C:\\yolov7\\"+save_path

    if len(det) == 0:
        return 0
    else:
        resultsBbox = {"results": detectName,"imgName":randomdate,"savepath":linbot_getimgpath}       
        return resultsBbox

    