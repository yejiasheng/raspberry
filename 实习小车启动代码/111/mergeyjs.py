# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import sys
sys.path.append("/home/lzk/samples/common")
sys.path.append("../")
import os
import numpy as np
import acl
import time
import socket
import cv2 
import traceback
from PIL import Image, ImageDraw, ImageFont
import atlas_utils.constants as const
from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource
import atlas_utils.utils as utils
from atlas_utils.acl_dvpp import Dvpp
from atlas_utils.acl_image import AclImage
import threading

app = Flask(__name__)

camera = cv2.VideoCapture('rtsp://192.168.10.24/test')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

labels =["hand"]
MODEL_PATH = "/home/YJS/model/yolov3_me.om"
MODEL_WIDTH = 416
MODEL_HEIGHT = 416
class_num = 3
stride_list = [32, 16, 8]
anchors_3 = np.array([[12, 16], [19, 36], [40, 28]]) / stride_list[2]
anchors_2 = np.array([[36, 75], [76, 55], [72, 146]]) / stride_list[1]
anchors_1 = np.array([[142, 110], [192, 243], [459, 401]]) / stride_list[0]
anchor_list = [anchors_1, anchors_2, anchors_3]
conf_threshold = 0.8
iou_threshold = 0.3
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
threads=[]
# Initialization
acl_resource = AclResource()
acl_resource.init()
#model = Model("/home/YJS/model/yolov3_me.om")


def preprocess(image):#cv         
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    img_h = image.size[1] #360
    img_w = image.size[0] #640
    net_h = MODEL_HEIGHT  #416
    net_w = MODEL_WIDTH   #416

    scale = min(float(net_w) / float(img_w), float(net_h) / float(img_h)) #416/640
    new_w = int(img_w * scale) #416
    new_h = int(img_h * scale) #234
    #delta = (MODEL_HEIGHT - int(image.size[1] * scale)) // 2 
    shift_x = (net_w - new_w) // 2       #0
    shift_y = (net_h - new_h) // 2       #91
    shift_x_ratio = (net_w - new_w) / 2.0 / net_w #0
    shift_y_ratio = (net_h - new_h) / 2.0 / net_h #0.21875

    image_ = image.resize((new_w, new_h))
    new_image = np.zeros((net_h, net_w, 3), np.uint8)
    new_image[shift_y: new_h + shift_y, shift_x: new_w + shift_x, :] = np.array(image_)
    new_image = new_image.astype(np.float32)
    new_image = new_image / 255
    #print('new_image.shape', new_image.shape)
    new_image = new_image.transpose(2, 0, 1).copy() 
    return new_image, image


def overlap(x1, x2, x3, x4):
    left = max(x1, x3)
    right = min(x2, x4)
    return right - left


def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w <= 0 or h <= 0:
        return 0
    inter_area = w * h
    union_area = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
    return inter_area * 1.0 / union_area


def apply_nms(all_boxes, thres):
    res = []

    for cls in range(class_num):
        cls_bboxes = all_boxes[cls]
        sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

        p = dict()
        for i in range(len(sorted_boxes)):
            if i in p:
                continue

            truth = sorted_boxes[i]
            for j in range(i + 1, len(sorted_boxes)):
                if j in p:
                    continue
                box = sorted_boxes[j]
                iou = cal_iou(box, truth)
                if iou >= thres:
                    p[j] = 1

        for i in range(len(sorted_boxes)):
            if i not in p:
                res.append(sorted_boxes[i])
    return res


def _sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def decode_bbox(conv_output, anchors, img_w, img_h, x_scale, y_scale, shift_x_ratio, shift_y_ratio):
    print('conv_output.shape', conv_output.shape)
    _, _, h, w = conv_output.shape
    conv_output = conv_output.transpose(0, 2, 3, 1)
    pred = conv_output.reshape((h * w, 3, 5 + class_num))
    pred[..., 4:] = _sigmoid(pred[..., 4:])
    pred[..., 0] = (_sigmoid(pred[..., 0]) + np.tile(range(w), (3, h)).transpose((1, 0))) / w
    pred[..., 1] = (_sigmoid(pred[..., 1]) + np.tile(np.repeat(range(h), w), (3, 1)).transpose((1, 0))) / h
    pred[..., 2] = np.exp(pred[..., 2]) * anchors[:, 0:1].transpose((1, 0)) / w
    pred[..., 3] = np.exp(pred[..., 3]) * anchors[:, 1:2].transpose((1, 0)) / h

    bbox = np.zeros((h * w, 3, 4))
    bbox[..., 0] = np.maximum((pred[..., 0] - pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, 0)  # x_min
    bbox[..., 1] = np.maximum((pred[..., 1] - pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, 0)  # y_min
    bbox[..., 2] = np.minimum((pred[..., 0] + pred[..., 2] / 2.0 - shift_x_ratio) * x_scale * img_w, img_w)  # x_max
    bbox[..., 3] = np.minimum((pred[..., 1] + pred[..., 3] / 2.0 - shift_y_ratio) * y_scale * img_h, img_h)  # y_max
    # print('bbox', bbox)
    pred[..., :4] = bbox
    pred = pred.reshape((-1, 5 + class_num))
    # pred[:, 4] = np.max(pred[:, 5:], axis=-1)
    pred[:, 4] = pred[:, 4] * pred[:, 5:].max(1)
    pred[:, 5] = np.argmax(pred[:, 5:], axis=-1)
    pred = pred[pred[:, 4] >= 0.2]
    print('pred[:, 5]', pred[:, 5])
    print('pred[:, 5] shape', pred[:, 5].shape)
    # pred = pred[pred[:, 4] >= conf_threshold]

    all_boxes = [[] for ix in range(class_num)]
    for ix in range(pred.shape[0]):
        box = [int(pred[ix, iy]) for iy in range(4)]
        box.append(int(pred[ix, 5]))
        box.append(pred[ix, 4])
        all_boxes[box[4] - 1].append(box)
    # print('all_boxes', all_boxes)
    return all_boxes


def convert_labels(label_list):
    if isinstance(label_list, np.ndarray):
        label_list = label_list.tolist()
        label_names = [labels[int(index)] for index in label_list]
    return label_names

def construct_image_info():
    """construct image info"""
    image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT,
                           MODEL_WIDTH, MODEL_HEIGHT],
                           dtype = np.float32)
    return image_info

def post_process(infer_output, origin_img):
    """postprocess"""
    print("post process")
    box_num = infer_output[1][0, 0]
    print(infer_output[1][0, 0])
    print("box num ", box_num)
    box_info = infer_output[0].flatten()
    scalex = origin_img.width / MODEL_WIDTH
    delta = (MODEL_HEIGHT - int(origin_img.height * 416/640)) // 2 #91
    # print(delta)
    scaley = origin_img.height / MODEL_HEIGHT
    # if scalex > scaley:
    #     scaley = scalex
    draw = ImageDraw.Draw(origin_img)
    font = ImageFont.load_default()
    signal=0
    for n in range(int(box_num)):
        if box_num==1:
            ids = int(box_info[5 * int(box_num) + n])
            label = labels[ids]
            score = box_info[4 * int(box_num)+n]
            print(score) 
            top_left_x = box_info[0 * int(box_num)+n] * scalex
            top_left_y = (box_info[1 * int(box_num)+n]-delta)/234*360
            bottom_right_x = box_info[2 * int(box_num) + n] * scalex
            bottom_right_y = (box_info[3 * int(box_num) + n]-delta)/234*360
            x_length=bottom_right_x-top_left_x
            y_length=bottom_right_y-top_left_y
            area=(x_length*y_length)
            if area>=8000:
                draw.line([(top_left_x, top_left_y), (bottom_right_x, top_left_y), (bottom_right_x, bottom_right_y), \
                (top_left_x, bottom_right_y), (top_left_x, top_left_y)], fill=(0, 200, 100), width=5)
                draw.text((top_left_x, top_left_y), label, font=font, fill=255)
            else:
                draw.line([(top_left_x, top_left_y), (bottom_right_x, top_left_y), (bottom_right_x, bottom_right_y), \
                (top_left_x, bottom_right_y), (top_left_x, top_left_y)], fill=(255, 0, 0), width=5)
                draw.text((top_left_x, top_left_y), label, font=font, fill=255)
                signal=signal-1
        else:
            ids = int(box_info[5 * int(box_num) + n])
            label = labels[ids]
            score = box_info[4 * int(box_num)+n]
            print(score) 
            top_left_x = box_info[0 * int(box_num)+n] * scalex
            top_left_y = (box_info[1 * int(box_num)+n]-delta)/234*360
            bottom_right_x = box_info[2 * int(box_num) + n] * scalex
            bottom_right_y = (box_info[3 * int(box_num) + n]-delta)/234*360
            draw.line([(top_left_x, top_left_y), (bottom_right_x, top_left_y), (bottom_right_x, bottom_right_y), \
            (top_left_x, bottom_right_y), (top_left_x, top_left_y)], fill=(255, 0, 0), width=5)
            draw.text((top_left_x, top_left_y), label, font=font, fill=255)

    num=0
    if box_num==1 and signal==0:
        xpt=(top_left_x+bottom_right_x)/2#获取绿框的中心点
        ypt=(top_left_y+bottom_right_y)/2#获取绿框的中心点
        w = origin_img.size[0]  # 图片长度
        h = origin_img.size[1]  # 图片宽度
        # print(w)
        # print(h)
        if 0<=ypt<(1/3)*h and ypt < (h/w)*xpt and ypt < -(h/w)*xpt+h:
            print("前进！")
            # print(f"数字信号{num}")
            #draw.text((xpt, ypt), "前进", font=font, fill=255)
            num=0
        elif 0 <= xpt < (1/3)*w and (h/w)*xpt <= ypt <= -(h/w)*xpt+h:
            print("右转！")
            # print(f"数字信号{num}")
            #draw.text((xpt, ypt), "左转", font=font, fill=255)
            num=1
        elif ypt > (h/w)*xpt and ypt>-(h/w)*xpt+h and (2/3)*h < ypt <= h:
            print("后退！")
            # print(f"数字信号{num}")
            #draw.text((xpt, ypt), "后退", font=font, fill=255)
            num=2
        elif (2/3)*w < xpt <= w and -(h/w)*xpt+h <= ypt <= (h/w)*xpt:
            print("左转！")
            # print(f"数字信号{num}")
            #draw.text((xpt, ypt), "右转", font=font, fill=255)
            num=3  
        elif (1/3)*w <= xpt <= (2/3)*w and (1/3)*h <= ypt <= (2/3)*h:
            print("停止！")
            # print(f"数字信号{num}")
            #draw.text((xpt, ypt), "停止", font=font, fill=255)
            num=4
        else :
            print("error")
    else:
        print("未成功识别")
        # print(f"数字信号{num}")
        num=4
    return origin_img,num

def frameprocessing(model,frame):
    w=640
    h=360
    # frame == cv2.flip(frame,1)
    image_info = construct_image_info()
    data, orig = preprocess(frame)
    # print("here is ok")

    # model = Model("/home/YJS/model/yolov3_me.om")
    result_list = model.execute([data,image_info])
    # ret = acl.rt.synchronize_stream(0)
    # print(result_list)
    image1 = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    afterframe,num = post_process(result_list,image1)
    afterframe = cv2.cvtColor(np.asarray(afterframe),cv2.COLOR_RGB2BGR)
    a = int(w/3)#长三分之一处
    b = int(2*w/3)#长三分之二处
    c = int(h/3)  # 宽三分之一处
    d = int(2*h/3)  # 宽三分之二处
    cv2.line(afterframe, (0,0), (a,c), (0, 0, 255), 2)
    cv2.line(afterframe, (a,c), (b,c), (0, 0, 255), 2)
    cv2.line(afterframe, (b,c), (w,0), (0, 0, 255), 2)
    cv2.line(afterframe, (a,c), (a,d), (0, 0, 255), 2)
    cv2.line(afterframe, (a,d), (0,h), (0, 0, 255), 2)
    cv2.line(afterframe, (a,d), (b,d), (0, 0, 255), 2)
    cv2.line(afterframe, (b,d), (w,h), (0, 0, 255), 2)
    cv2.line(afterframe, (b,c), (b,d), (0, 0, 255), 2)#以上八行为区域判定

    # client = socket.socket()  # 声明socket类型，同时生成socket连接对象 clinet is 201, it uses the car
    # client.connect(('192.168.43.250', 6666))  # 链接服务器的ip + 端口
    

    # t=threading.Thread(target=testt,args=('4',))
    # t.start()
    # t.terminate()
    # t = threading.Thread(target=testt(num))
    # t.start()
    # print("111")
    # t.join()


    # client.send(str(num).encode("utf-8"))
    # print(f"numlist[{num}]={numlist[num]}")

    # if max(numlist) > 10:
    #     #将这个数返回给树莓派让小车 
    #     ch = numlist.index(max(numlist))
    #     print(f"去噪音后：{ch}")
    #     client.send(str(ch).encode("utf-8"))
    #     numlist = [0,0,0,0,0]

    return num, afterframe

def gen_frames():  # generate frame by frame from camera
    cont, ret = acl.rt.create_context(0)
    # print(ret)
    # print(cont)
    # cont, ret = acl.rt.get_context()
    # print(cont)
    # print(ret)
    model = Model("/home/YJS/model/yolov3_me.om")

    numlist = [0,0,0,0,0]

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        success = 1
        # frame, num=ff.frameprocessing(frame)
        if not success:
            break
        else:
            # print('1')
            # frame = cv2.imread('/home/YJS/111/1.jpg')
            # print("2")
            signal,frame =frameprocessing(model,frame) ###############
            client = socket.socket()  # 声明socket类型，同时生成socket连接对象 clinet is 201, it uses the car
            client.connect(('192.168.43.250', 6666))  # 链接服务器的ip + 端口
            client.send(str(signal).encode("utf-8"))

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')



# def tttt():
#     fram = cv2.imread('/home/YJS/111/1.jpg')
#     frame=frameprocessing(fram) ############### 
#     cv2.imwrite('/home/YJS/111/4.jpg',frame)





if __name__ == '__main__':
    # tttt()
    os.system("source /home/lzk/hand_detection/venv2/bin/activate")
    # client = socket.socket()  # 声明socket类型，同时生成socket连接对象 clinet is 201, it uses the car
    # client.connect(('192.168.43.250', 6666))  # 链接服务器的ip + 端口
    # t=threading.Thread(target=testt,args=('4',))
    # t.start()
    app.run(host="0.0.0.0",debug=True)

    # while True:
    #     client.send(str(1).encode("utf-8"))
    # client.send(str(1).encode("utf-8"))
    # fram = cv2.imread('/home/YJS/111/1.jpg')
    # frame=frameprocessing(fram) ############### 
    # cv2.imwrite('/home/YJS/111/3.jpg',frame)