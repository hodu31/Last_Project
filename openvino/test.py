import cv2
import csv
import argparse

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
    'right_ankle': 16
}

DEFAULT_MODEL = "models/movenet_multipose_lightning_256x256_FP32.xml"  # Assuming a default path

# Simplifying the code to set parameters directly in the script

# Set the parameters directly
INPUT_VIDEO = '/sample.mp4'
MODEL_XML = DEFAULT_MODEL  
CSV_OUTPUT = 'C:/Users/de31/Documents/output.csv'


class MovenetMPSimple:
    
    def __init__(self, input_src, xml=DEFAULT_MODEL, csv_output=None):
        self.cap = cv2.VideoCapture(input_src)
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup for CSV output
        if csv_output:
            self.csv_file = open(csv_output, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=["NUMOFBODIES", "ID"] + 
                                             [f"{name.upper()}_X" for name in KEYPOINT_DICT.keys()] + 
                                             [f"{name.upper()}_Y" for name in KEYPOINT_DICT.keys()] + 
                                             [f"{name.upper()}_CONFIDENCE_LEVEL" for name in KEYPOINT_DICT.keys()] + 
                                             ["TIMESTAMP", "FRAME_NUM"])
            self.csv_writer.writeheader()
        else:
            self.csv_file = None

    def run(self):
        frame_num = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            # TODO: Add the inference and post-processing steps to get the bodies

            # Save to CSV
            if self.csv_file:
                save_to_csv(bodies, self.csv_writer, frame_num)

            frame_num += 1  # Increment the frame number

        # Close the CSV file
        if self.csv_file:
            self.csv_file.close()


    

class MovenetMPSimpleInternal:
    
    def __init__(self, input_src, xml=DEFAULT_MODEL, csv_output=None):
        self.cap = cv2.VideoCapture(input_src)
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup for CSV output
        if csv_output:
            self.csv_file = open(csv_output, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=["NUMOFBODIES", "ID"] + 
                                             [f"{name.upper()}_X" for name in KEYPOINT_DICT.keys()] + 
                                             [f"{name.upper()}_Y" for name in KEYPOINT_DICT.keys()] + 
                                             [f"{name.upper()}_CONFIDENCE_LEVEL" for name in KEYPOINT_DICT.keys()] + 
                                             ["TIMESTAMP", "FRAME_NUM"])
            self.csv_writer.writeheader()
        else:
            self.csv_file = None

    def run(self):
        frame_num = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            # TODO: Add the inference and post-processing steps to get the bodies

            # Save to CSV
            if self.csv_file:
                save_to_csv(bodies, self.csv_writer, frame_num)

            frame_num += 1  # Increment the frame number

        # Close the CSV file
        if self.csv_file:
            self.csv_file.close()

# Run the program with the given internal parameters
mp_simple = MovenetMPSimpleInternal(input_src=INPUT_VIDEO, xml=MODEL_XML, csv_output=CSV_OUTPUT)
mp_simple.run()
