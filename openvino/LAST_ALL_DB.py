### 웹캠 보여주기만
import numpy as np
from collections import namedtuple
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import os
from openvino.inference_engine import IENetwork, IECore
from Tracker import TrackerIoU, TrackerOKS, TRACK_COLORS
import pandas as pd
from keras.models import load_model
from datetime import datetime
import mysql.connector
from db_connect import insert_db_data
from db_connect import insert_visit
from db_connect import insert_vio


model1 = load_model('C:/Last_Project/openvino/pred_model/buy.h5')
model2 = load_model('C:/Last_Project/openvino/pred_model/compare.h5')
model4 = load_model('C:/Last_Project/openvino/pred_model/jeon.h5')
model5 = load_model('C:/Last_Project/openvino/pred_model/select.h5')
model6 = load_model('C:/Last_Project/openvino/pred_model/smoke_last.h5')
model7 = load_model('C:/Last_Project/openvino/pred_model/theft.h5')
model8 = load_model('C:/Last_Project/openvino/pred_model/yugi.h5')
model9 = load_model('C:/Last_Project/openvino/pred_model/violence.h5')


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models/movenet_multipose_lightning_256x256_FP32.xml"


# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
    'head': 17
}


def compute_head_position(keypoints):
    relevant_keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
    total_x, total_y, count = 0, 0, 0
    for keypoint in relevant_keypoints:
        idx = KEYPOINT_DICT[keypoint]
        if keypoints[idx] is not None:
            x, y = keypoints[idx]
            total_x += x
            total_y += y
            count += 1
    if count == 0:
        return None
    return (total_x / count, total_y / count)

# LINES_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram

LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
            [10,8],[8,6],[6,5],[5,7],[7,9],
            [6,12],[12,11],[11,5],
            [12,14],[14,16],[11,13],[13,15]]

class Body:
    def __init__(self, score, xmin, ymin, xmax, ymax, keypoints_score, keypoints, keypoints_norm):
        self.score = score # global score
        # xmin, ymin, xmax, ymax : bounding box
        self.stop_frame_count_dict = {}
        self.stop_frame_count = 0
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoints_score = keypoints_score# scores of the keypoints
        self.keypoints_norm = keypoints_norm # keypoints normalized ([0,1]) coordinates (x,y) in the input image
        self.keypoints = keypoints # keypoints coordinates (x,y) in pixels in the input image
        
        
    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

    def str_bbox(self):
        return f"xmin={self.xmin} xmax={self.xmax} ymin={self.ymin} ymax={self.ymax}"

# Padding (all values are in pixel) :
# w (resp. h): horizontal (resp. vertical) padding on the source image to make its ratio same as Movenet model input. 
#               The padding is done on one side (bottom or right) of the image.
# padded_w (resp. padded_h): width (resp. height) of the image after padding
Padding = namedtuple('Padding', ['w', 'h', 'padded_w',  'padded_h'])

class MovenetMPOpenvino:
    def __init__(self, input_src=None,
                xml=DEFAULT_MODEL, 
                device="CPU",
                tracking="oks",
                score_thresh=0.25,
                output=None):
        self.visited_tracks = {} 
        self.user_id = 'a001'
        self.shop_id = '#001'
        self.vio_time = None
        self.predicted_label = {}
        self.count_predicted = {}
        self.predict_violence = None
        self.time_data = {}
        self.db_data = []
        self.prev_keypoints = {}
        self.temp_array_dict = {}
        self.array_list = []
        self.stop_frame_count_dict = {}
        self.stop_frame_count = 0
        self.frame_counter = 0
        self.temp_array = np.array([])
        self.score_thresh = score_thresh
        self.tracking = tracking
        if tracking is None:
            self.tracking = False
        elif tracking == "iou":
            self.tracking = True
            self.tracker = TrackerIoU()
        elif tracking == "oks":
            self.tracking = True
            self.tracker = TrackerOKS()
         
        if input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
            
        else:
            self.input_type = "video"
            if input_src.isdigit(): 
                input_type = "webcam"
                input_src = int(input_src) #2
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video FPS:", self.video_fps)
    
        # Load Openvino models
        self.load_model(xml, device)     

        # Rendering flags
        self.show_fps = True
        self.show_bounding_box = False

        if output is None: 
            self.output = None
        else:
            if self.input_type == "image":
                # For an source image, we will output one image (and not a video) and exit
                self.output = output
            else:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.output = cv2.VideoWriter(output,fourcc,self.video_fps,(self.img_w, self.img_h)) 

        # Define the padding
        # Note we don't center the source image. The padding is applied
        # on the bottom or right side. That simplifies a bit the calculation
        # when depadding
        if self.img_w / self.img_h > self.pd_w / self.pd_h:
            pad_h = int(self.img_w * self.pd_h / self.pd_w - self.img_h)
            self.padding = Padding(0, pad_h, self.img_w, self.img_h + pad_h)
        else:
            pad_w = int(self.img_h * self.pd_w / self.pd_h - self.img_w)
            self.padding = Padding(pad_w, 0, self.img_w + pad_w, self.img_h)
        print("Padding:", self.padding)
        
    def load_model(self, xml_path, device):

        print("Loading Inference Engine")
        self.ie = IECore()
        print("Device info:")
        versions = self.ie.get_versions(device)
        print("{}{}".format(" "*8, device))
        print("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[device].major, versions[device].minor))
        print("{}Build ........... {}".format(" "*8, versions[device].build_number))

        name = os.path.splitext(xml_path)[0]
        bin_path = name + '.bin'
        print("Pose Detection model - Reading network files:\n\t{}\n\t{}".format(xml_path, bin_path))
        self.pd_net = self.ie.read_network(model=xml_path, weights=bin_path)
        # Input blob: input:0 - shape: [1, 3, 256, 256] (lightning)
        # Output blob: Identity - shape: [1, 6, 56]
        self.pd_input_blob = next(iter(self.pd_net.input_info))
        print(f"Input blob: {self.pd_input_blob} - shape: {self.pd_net.input_info[self.pd_input_blob].input_data.shape}")
        _,_,self.pd_h,self.pd_w = self.pd_net.input_info[self.pd_input_blob].input_data.shape
        for o in self.pd_net.outputs.keys():
            print(f"Output blob: {o} - shape: {self.pd_net.outputs[o].shape}")
        self.pd_kps = "Identity"
        print("Loading pose detection model into the plugin")
        self.pd_exec_net = self.ie.load_network(network=self.pd_net, num_requests=1, device_name=device)

        self.infer_nb = 0
        self.infer_time_cumul = 0

    def pad_and_resize(self, frame):
        """ Pad and resize the image to prepare for the model input."""

        padded = cv2.copyMakeBorder(frame, 
                                        0, 
                                        self.padding.h,
                                        0, 
                                        self.padding.w,
                                        cv2.BORDER_CONSTANT)

        padded = cv2.resize(padded, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)

        return padded

    def pd_postprocess(self, inference):
        result = np.squeeze(inference[self.pd_kps]) # 6x56
        bodies = []
        for i in range(6):
            kps = result[i][:51].reshape(17,-1)
            bbox = result[i][51:55].reshape(2,2)          
            score = result[i][55]
            if score > self.score_thresh:
                ymin, xmin, ymax, xmax = (bbox * [self.padding.padded_h, self.padding.padded_w]).flatten().astype(int)
                kp_xy =kps[:,[1,0]]
                keypoints = kp_xy * np.array([self.padding.padded_w, self.padding.padded_h])

                body = Body(score=score, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, 
                            keypoints_score = kps[:,2], 
                            keypoints = keypoints.astype(int),
                            keypoints_norm = keypoints / np.array([self.img_w, self.img_h]))
                bodies.append(body)
        return bodies
    
### 웹캠에 표시하는 부분
    def pd_render(self, frame, bodies):
        thickness = 3 
        color_skeleton = (255, 180, 90)
        color_box = (0,255,255)
        for body in bodies:
            if self.tracking:
                color_skeleton = color_box = TRACK_COLORS[body.track_id % len(TRACK_COLORS)]
            
            head_position = compute_head_position(body.keypoints)
            if head_position:
                x, y = head_position
                if x is not None and y is not None:
                    body.keypoints = np.vstack((body.keypoints, np.array([head_position])))
                    assert len(body.keypoints) == 18, f"Expected 18 keypoints, but got {len(body.keypoints)}"
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 255), -11)  # 보라색으로 'head' 표시    
            
            
            # 사람의 포즈를 표시하기 위한 선을 그리는 부분
            lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if body.keypoints_score[line[0]] > self.score_thresh and body.keypoints_score[line[1]] > self.score_thresh]
            lines = np.array(lines, dtype=np.int32)
            cv2.polylines(frame, lines, False, color_skeleton, 2, cv2.LINE_AA)
            
            # Keypoints(관절) 위치에 원을 그리는 부분
            for i, (x_y, score) in enumerate(zip(body.keypoints, body.keypoints_score)):
                if score > self.score_thresh:
                    if i % 2 == 1:
                        color = (0, 255, 0)
                    elif i == 0:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(frame, (int(x_y[0]), int(x_y[1])), 4, color, -11)


                    
            # 바운딩 박스를 그리는 부분
            if self.show_bounding_box:
                cv2.rectangle(frame, (body.xmin, body.ymin), (body.xmax, body.ymax), color_box, thickness)
                
            # 추적이 활성화된 경우 바운딩 박스의 중앙에 추적 ID를 표시하는 부분
            if self.tracking:
                # Display track_id at the center of the bounding box
                x = (body.xmin + body.xmax) // 2
                y = (body.ymin + body.ymax) // 2
                cv2.putText(frame, str(body.track_id), (x,y), cv2.FONT_HERSHEY_PLAIN, 4, color_box, 3)
            
            # 정지시간 알림
            if len(bodies) > 0:
                current_keypoints = body.keypoints
                if body.track_id in self.prev_keypoints:  # 해당 바디의 이전 키포인트를 가져옵니다.
                    prev_keypoints_for_body = self.prev_keypoints[body.track_id]
                    total_movement = np.sum(np.abs(current_keypoints - prev_keypoints_for_body))
                    if total_movement < 15 * 17:
                        # 여기서 해당 track_id가 stop_frame_count_dict에 없으면 초기화해줍니다.
                        if body.track_id not in self.stop_frame_count_dict:
                            self.stop_frame_count_dict[body.track_id] = 0
                        
                        self.stop_frame_count_dict[body.track_id] += 1  # 해당 body의 정지 프레임 카운트를 증가시킵니다.
                        if self.stop_frame_count_dict[body.track_id] >= 30:
                            stopped_time = self.stop_frame_count_dict[body.track_id] / self.video_fps
                            text_position = (body.xmin, body.ymin - 10)  # 바운딩 박스의 왼쪽 상단에 텍스트를 표시
                            cv2.putText(frame, f'Stopped: {stopped_time:.2f} sec', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                    else:
                        self.stop_frame_count_dict[body.track_id] = 0  # 움직임이 감지되면 카운트를 리셋합니다.
                self.prev_keypoints[body.track_id] = current_keypoints  # 현재 키포인트를 저장합니다.
                
            
            now_time = datetime.now()
            now_time = now_time.replace(microsecond=0)
            
            
            if body.track_id not in self.temp_array_dict:
                self.temp_array_dict[body.track_id] = np.array([])
                
            if body.track_id not in self.predicted_label and len(self.temp_array_dict[body.track_id]) >= 200:
                self.predicted_label[body.track_id] = [None, None, None, None, None, None, None, None]
                
            if body.track_id not in self.time_data and len(self.temp_array_dict[body.track_id]) >= 200:
                self.time_data[body.track_id] = [[], [], [], [], [], [], [], []]
                
            if body.track_id not in self.count_predicted and len(self.temp_array_dict[body.track_id]) >= 200:
                self.count_predicted[body.track_id] = [0, 0, 0, 0, 0, 0, 0, 0]
            

################################################# 모델##############################################
            
            # buy Refund
            if len(self.temp_array_dict[body.track_id]) >= 200 and self.frame_counter % 270 == 0:
                input_data = self.temp_array_dict[body.track_id].copy()
                
                input_data = input_data[1:-1]
                
                input_data[:, 0] = 1
                            
                # 패딩 추가
                padding = np.zeros((610 - input_data.shape[0], input_data.shape[1]))
                
                input_data = np.vstack((input_data, padding))
                
                # 마지막 열에 인덱스 추가
                index_array = np.arange(input_data.shape[0]).reshape(-1, 1)
                input_data = np.hstack((input_data, index_array))
                
                # 데이터 형변환
                input_data = input_data.astype(np.float32)
                
                input_data = np.array([input_data])
                
                
                prediction = model1.predict(input_data)
                
                
                
                if np.argmax(prediction) == 0:
                    self.predicted_label[body.track_id][0] = 'NO_buy'
                elif np.argmax(prediction) == 1:
                    self.predicted_label[body.track_id][0] = 'YES_buy'
                    
                    if len(self.time_data[body.track_id][0]) == 0:
                        self.time_data[body.track_id][0] = [now_time]
                        data = [self.user_id ,self.shop_id, body.track_id, now_time, 1]
                        insert_db_data(data)
                    elif len(self.time_data[body.track_id][0]) != 0:
                        if (now_time - self.time_data[body.track_id][0][0]).seconds >= 30:
                            self.time_data[body.track_id][0] = [now_time]
                            data = [self.user_id ,self.shop_id, body.track_id, now_time, 1]
                            insert_db_data(data)
                    
                            
            # compare
            if len(self.temp_array_dict[body.track_id]) >= 200 and self.frame_counter % 270 == 30:
                input_data = self.temp_array_dict[body.track_id].copy()
                
                input_data = input_data[1:-1]
                
                input_data[:, 0] = 1
                            
                # 패딩 추가
                padding = np.zeros((610 - input_data.shape[0], input_data.shape[1]))
                
                input_data = np.vstack((input_data, padding))
                
                # 마지막 열에 인덱스 추가
                index_array = np.arange(input_data.shape[0]).reshape(-1, 1)
                input_data = np.hstack((input_data, index_array))
                
                # 데이터 형변환
                input_data = input_data.astype(np.float32)
                
                input_data = np.array([input_data])
                
                prediction = model2.predict(input_data)
                
                
                
                if np.argmax(prediction) == 0:
                    self.predicted_label[body.track_id][1] = 'NO_compare'
                elif np.argmax(prediction) == 1:
                    self.predicted_label[body.track_id][1] = 'YES_compare'
                    
                    if len(self.time_data[body.track_id][1]) == 0:
                        self.time_data[body.track_id][1] = [now_time]
                        data = [self.user_id ,self.shop_id, body.track_id, now_time, 2]
                        insert_db_data(data)
                    elif len(self.time_data[body.track_id][1]) != 0:
                        if (now_time - self.time_data[body.track_id][1][0]).seconds >= 30:
                            self.time_data[body.track_id][1] = [now_time]
                            data = [self.user_id ,self.shop_id, body.track_id, now_time, 2]
                            insert_db_data(data)
                    
    
                    
                    
            # jeon
            if len(self.temp_array_dict[body.track_id]) >= 200 and self.frame_counter % 270 == 90:
                input_data = self.temp_array_dict[body.track_id].copy()
                
                input_data = input_data[1:-1]
                
                input_data[:, 0] = 1
                            
                # 패딩 추가
                padding = np.zeros((610 - input_data.shape[0], input_data.shape[1]))
                
                input_data = np.vstack((input_data, padding))
                
                # 마지막 열에 인덱스 추가
                index_array = np.arange(input_data.shape[0]).reshape(-1, 1)
                input_data = np.hstack((input_data, index_array))
                
                # 데이터 형변환
                input_data = input_data.astype(np.float32)
                
                input_data = np.array([input_data])
                
                prediction = model4.predict(input_data)
                
                
                
                if np.argmax(prediction) == 0:
                    self.predicted_label[body.track_id][3] = 'NO_jeon'
                elif np.argmax(prediction) == 1:
                    self.count_predicted[body.track_id][3] + 1
                    
                    if self.count_predicted[body.track_id][3] >= 3:  
                        self.predicted_label[body.track_id][3] = 'YES_jeon'
                        
                        if len(self.time_data[body.track_id][3]) == 0:
                            self.time_data[body.track_id][3] = [now_time]
                            data = [self.user_id ,self.shop_id, body.track_id, now_time, 4]
                            insert_db_data(data)
                        elif len(self.time_data[body.track_id][3]) != 0:
                            if (now_time - self.time_data[body.track_id][3][0]).seconds >= 30:
                                self.time_data[body.track_id][3] = [now_time]
                                data = [self.user_id ,self.shop_id, body.track_id, now_time, 4]
                                insert_db_data(data)
                        if self.count_predicted[body.track_id][3] >= 5:  
                            self.count_predicted[body.track_id][3] = 0
                    
                            
            # select
            if len(self.temp_array_dict[body.track_id]) >= 200 and self.frame_counter % 270 == 120:
                input_data = self.temp_array_dict[body.track_id].copy()
                
                input_data = input_data[1:-1]
                
                input_data[:, 0] = 1
                            
                # 패딩 추가
                padding = np.zeros((610 - input_data.shape[0], input_data.shape[1]))
                
                input_data = np.vstack((input_data, padding))
                
                # 마지막 열에 인덱스 추가
                index_array = np.arange(input_data.shape[0]).reshape(-1, 1)
                input_data = np.hstack((input_data, index_array))
                
                # 데이터 형변환
                input_data = input_data.astype(np.float32)
                
                input_data = np.array([input_data])
                
                prediction = model5.predict(input_data)
                
                
                
                if np.argmax(prediction) == 0:
                    self.predicted_label[body.track_id][4] = 'NO_select'
                elif np.argmax(prediction) == 1:
                    self.predicted_label[body.track_id][4] = 'YES_select'
                    
                    if len(self.time_data[body.track_id][4]) == 0:
                        self.time_data[body.track_id][4] = [now_time]
                        data = [self.user_id ,self.shop_id, body.track_id, now_time, 5]
                        insert_db_data(data)
                    elif len(self.time_data[body.track_id][4]) != 0:
                        if (now_time - self.time_data[body.track_id][4][0]).seconds >= 30:
                            self.time_data[body.track_id][4] = [now_time]
                            data = [self.user_id ,self.shop_id, body.track_id, now_time, 5]
                            insert_db_data(data)
                    
                    
            # smoke
            if len(self.temp_array_dict[body.track_id]) >= 200 and self.frame_counter % 270 == 150:
                input_data = self.temp_array_dict[body.track_id].copy()
                
                input_data = input_data[1:-1]
                
                input_data[:, 0] = 1
                            
                # 패딩 추가
                padding = np.zeros((610 - input_data.shape[0], input_data.shape[1]))
                
                input_data = np.vstack((input_data, padding))
                
                # 마지막 열에 인덱스 추가
                index_array = np.arange(input_data.shape[0]).reshape(-1, 1)
                input_data = np.hstack((input_data, index_array))
                
                # 데이터 형변환
                input_data = input_data.astype(np.float32)
                
                input_data = np.array([input_data])
                
                prediction = model6.predict(input_data)
                
                
                
                if np.argmax(prediction) == 0:
                    self.predicted_label[body.track_id][5] = 'NO_smoke'
                elif np.argmax(prediction) == 1:
                    self.predicted_label[body.track_id][5] = 'YES_smoke'
                    
                    if len(self.time_data[body.track_id][5]) == 0:
                        self.time_data[body.track_id][5] = [now_time]
                        data = [self.user_id ,self.shop_id, body.track_id, now_time, 6]
                        insert_db_data(data)
                    elif len(self.time_data[body.track_id][5]) != 0:
                        if (now_time - self.time_data[body.track_id][5][0]).seconds >= 30:
                            self.time_data[body.track_id][5] = [now_time]
                            data = [self.user_id ,self.shop_id, body.track_id, now_time, 6]
                            insert_db_data(data)
                    
                    
            # theft
            if len(self.temp_array_dict[body.track_id]) >= 200 and self.frame_counter % 270 == 180:
                input_data = self.temp_array_dict[body.track_id].copy()
                
                input_data = input_data[1:-1]
                
                input_data[:, 0] = 1
                            
                # 패딩 추가
                padding = np.zeros((610 - input_data.shape[0], input_data.shape[1]))
                
                input_data = np.vstack((input_data, padding))
                
                # 마지막 열에 인덱스 추가
                index_array = np.arange(input_data.shape[0]).reshape(-1, 1)
                input_data = np.hstack((input_data, index_array))
                
                # 데이터 형변환
                input_data = input_data.astype(np.float32)
                
                input_data = np.array([input_data])
                
                prediction = model7.predict(input_data)
                
                
                
                if np.argmax(prediction) == 0:
                    self.predicted_label[body.track_id][6] = 'NO_theft'
                elif np.argmax(prediction) == 1:
                    self.predicted_label[body.track_id][6] = 'YES_theft'
                    
                    if len(self.time_data[body.track_id][6]) == 0:
                        self.time_data[body.track_id][6] = [now_time]
                        data = [self.user_id ,self.shop_id, body.track_id, now_time, 7]
                        insert_db_data(data)
                    elif len(self.time_data[body.track_id][6]) != 0:
                        if (now_time - self.time_data[body.track_id][6][0]).seconds >= 30:
                            self.time_data[body.track_id][6] = [now_time]
                            data = [self.user_id ,self.shop_id, body.track_id, now_time, 7]
                            insert_db_data(data)
                    
                    
            
             # yugi
            if len(self.temp_array_dict[body.track_id]) >= 200 and self.frame_counter % 270 == 210:
                input_data = self.temp_array_dict[body.track_id].copy()
                
                input_data = input_data[1:-1]
                
                input_data[:, 0] = 1
                            
                # 패딩 추가
                padding = np.zeros((610 - input_data.shape[0], input_data.shape[1]))
                
                input_data = np.vstack((input_data, padding))
                
                # 마지막 열에 인덱스 추가
                index_array = np.arange(input_data.shape[0]).reshape(-1, 1)
                input_data = np.hstack((input_data, index_array))
                
                # 데이터 형변환
                input_data = input_data.astype(np.float32)
                
                input_data = np.array([input_data])
                
                prediction = model8.predict(input_data)
                
                if np.argmax(prediction) == 0:
                    self.predicted_label[body.track_id][7] = 'NO_yugi'
                elif np.argmax(prediction) == 1:
                    self.predicted_label[body.track_id][7] = 'YES_yugi'
                    
                    if len(self.time_data[body.track_id][7]) == 0:
                        self.time_data[body.track_id][7] = [now_time]
                        data = [self.user_id ,self.shop_id, body.track_id, now_time, 8]
                        insert_db_data(data)
                    elif len(self.time_data[body.track_id][7]) != 0:
                        if (now_time - self.time_data[body.track_id][7][0]).seconds >= 30:
                            self.time_data[body.track_id][7] = [now_time]
                            data = [self.user_id ,self.shop_id, body.track_id, now_time, 8]
                            insert_db_data(data)
            
             # violence
            if len(self.temp_array_dict[body.track_id]) >= 200 and self.frame_counter % 270 == 240:
                input_data = self.temp_array_dict[body.track_id].copy()
                
                input_data = input_data[1:-1]
                
                input_data[:, 0] = 1
                            
                # 패딩 추가
                padding = np.zeros((610 - input_data.shape[0], input_data.shape[1]))
                
                input_data = np.vstack((input_data, padding))
                
                # 마지막 열에 인덱스 추가
                index_array = np.arange(input_data.shape[0]).reshape(-1, 1)
                input_data = np.hstack((input_data, index_array))
                
                # 데이터 형변환
                input_data = input_data.astype(np.float32)
                
                input_data = np.array([input_data])
                
                prediction = model9.predict(input_data)
                
                if np.argmax(prediction) == 0:
                    self.predicted_label[body.track_id][8] = 'NO_violence'
                elif np.argmax(prediction) == 1:
                    self.predicted_label[body.track_id][8] = 'YES_violence' 
                    
                    if len(self.time_data[body.track_id][8]) == 0:
                        self.time_data[body.track_id][8] = [now_time]
                        data = [self.user_id ,self.shop_id, body.track_id, now_time, 8]
                        insert_db_data(data)
                    elif len(self.time_data[body.track_id][8]) != 0:
                        if (now_time - self.time_data[body.track_id][8][0]).seconds >= 30:
                            self.time_data[body.track_id][8] = [now_time]
                            data = [self.user_id ,self.shop_id, body.track_id, now_time, 8]
                            insert_db_data(data)
                    
                    

            if self.predicted_label is not None and body.track_id in self.predicted_label:
                text_position_1 = (body.xmin, body.ymin+30)
                text_position_2 = (body.xmin, body.ymin+60)
                text_position_3 = (body.xmin, body.ymin+90)
                text_position_4 = (body.xmin, body.ymin+120)
                cv2.putText(frame, "{}".format(self.predicted_label[body.track_id][0:2]), text_position_1, cv2.FONT_HERSHEY_PLAIN, 2, color_box, 3)
                cv2.putText(frame, "{}".format(self.predicted_label[body.track_id][2:4]), text_position_2, cv2.FONT_HERSHEY_PLAIN, 2, color_box, 3)
                cv2.putText(frame, "{}".format(self.predicted_label[body.track_id][4:6]), text_position_3, cv2.FONT_HERSHEY_PLAIN, 2, color_box, 3)
                cv2.putText(frame, "{}".format(self.predicted_label[body.track_id][6:8]), text_position_4, cv2.FONT_HERSHEY_PLAIN, 2, color_box, 3)
                
                
    def save_to_array(self, bodies):
        if not hasattr(self, 'temp_array_dict'):
            self.temp_array_dict = {}
        if not hasattr(self, 'prev_joint_positions'):
            self.prev_joint_positions = {}

        for body in bodies:
            ### 610 보다 크면 앞에 부분 자르기 
            if len(self.temp_array_dict[body.track_id]) >= 200:
                self.temp_array_dict[body.track_id] = self.temp_array_dict[body.track_id][1:]
            head_position = compute_head_position(body.keypoints)
            
            if head_position:
                body.keypoints[KEYPOINT_DICT['head']] = head_position

            data_row = [len(bodies)]
            COLUMN_ORDER = [
                'left_shoulder', 'left_elbow', 'left_wrist',
                'right_shoulder', 'right_elbow', 'right_wrist',
                'left_hip', 'left_knee', 'left_ankle',
                'right_hip', 'right_knee', 'right_ankle',
                'head']

            for column in COLUMN_ORDER:
                joint_index = KEYPOINT_DICT[column.lower()]
                data_row.extend([body.keypoints[joint_index][0], body.keypoints[joint_index][1]])
            
            # Compute the difference between the current and previous joint positions
            if body.track_id in self.prev_joint_positions:
                data_minus = self.prev_joint_positions[body.track_id] - np.array(data_row)
            else:
                data_minus = np.zeros_like(np.array(data_row))  # or some other initialization
                
            # Update the previous joint positions with the current data_row
            self.prev_joint_positions[body.track_id] = np.array(data_row)
                
                
            if body.track_id not in self.temp_array_dict or len(self.temp_array_dict[body.track_id]) == 0:
                self.temp_array_dict[body.track_id] = np.array([data_minus])
            else:
                self.temp_array_dict[body.track_id] = np.vstack((self.temp_array_dict[body.track_id], data_minus))
            
            
        # # 형태 확인하기    
        # for track_id, array in self.temp_array_dict.items():
        #     print(f"Shape for track_id {track_id}: {array.shape}")
        
        # # 데이터 확인하기 
        #     for track_id, array in self.temp_array_dict.items():
        #         print(track_id, array)    
        
        # # 데이터 확인하기 
        #     for track_id, array in self.prev_joint_positions.items():
        #         print(track_id, array)   
    
    
    
    def run(self):

        self.fps = FPS()
        nb_pd_inferences = 0
        glob_pd_rtrip_time = 0
        
        # 카메라 해상도 설정
        if hasattr(self, 'cap'):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while True:
            
            self.frame_counter += 1
            
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    break

            padded = self.pad_and_resize(frame)
            # cv2.imshow("Padded", padded)
                
            frame_nn = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)[None,] 
            pd_rtrip_time = now()
            inference = self.pd_exec_net.infer(inputs={self.pd_input_blob: frame_nn})
            glob_pd_rtrip_time += now() - pd_rtrip_time
            bodies = self.pd_postprocess(inference)
            if self.tracking:
                bodies = self.tracker.apply(bodies, now())
            self.pd_render(frame, bodies)
            nb_pd_inferences += 1
            
            # 10프레임 마다 저장
            if self.frame_counter % 3 == 0:  # 10프레임마다 조건을 확인
                self.save_to_array(bodies)
                

            self.fps.update()               

            if self.show_fps:
                self.fps.draw(frame, orig=(50,50), size=1, color=(270,180,100))
            cv2.imshow("Movenet", frame)

            if self.output:
                if self.input_type == "image":
                    cv2.imwrite(self.output, frame)
                    break
                else:
                    self.output.write(frame)

            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('f'):
                self.show_fps = not self.show_fps
            elif key == ord('b'):
                self.show_bounding_box = not self.show_bounding_box
            


        # Print some stats
        if nb_pd_inferences > 1:
            global_fps, nb_frames = self.fps.get_global()

            print(f"FPS : {global_fps:.1f} f/s (# frames = {nb_frames})")
            print(f"# pose detection inferences : {nb_pd_inferences}")
            print(f"Pose detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")

        if self.output and self.input_type != "image":
            self.output.release()
            
    
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='0', 
                        help="Path to video or image file to use as input (default=%(default)s)")
    # parser.add_argument("-p", "--precision", type=int, choices=[16, 32], default=32,
    #                     help="Precision (default=%(default)i")                    
    parser.add_argument("--xml", type=str,
                        help="Path to an .xml file for model")
    parser.add_argument("-r", "--res", default="256x256", choices=["192x192", "192x256", "256x256", "256x320", "320x320", "480x640", "736x1280"])
    # parser.add_argument("-d", "--device", default='CPU', type=str,
    #                     help="Target device to run the model (default=%(default)s)") 
    parser.add_argument("-t", "--tracking", choices=["iou", "oks"], default="oks",
                        help="Enable tracking and specify method")
    parser.add_argument("-s", "--score_threshold", default=0.25, type=float,
                        help="Confidence score (default=%(default)f)")                     
    parser.add_argument("-o","--output",
                        help="Path to output video file")
    
    args = parser.parse_args()

    
    # if args.device == "MYRIAD" or args.device == "GPU":
    #     args.precision = 16
    if not args.xml:
        args.xml = SCRIPT_DIR / f"models/movenet_multipose_lightning_{args.res}_FP32.xml"

    pd = MovenetMPOpenvino(input_src=args.input,
                    xml=args.xml,
                    # device=args.device, 
                    tracking=args.tracking,
                    score_thresh=args.score_threshold,
                    output=args.output)
    pd.run()
