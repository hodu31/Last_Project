# openvino-movenet-action-lstm

## 🇰🇷 소개
편의점에서의 고객 행동 분석을 위한 프로젝트입니다. 이 프로젝트는 관절 위치 데이터를 활용하여 LSTM 모델을 훈련시키고, 실시간으로 웹캠을 통한 행동 분석을 제공합니다.

### 📁 `people` 폴더
- **기능**: 관절 위치 데이터를 활용하여 LSTM 모델 학습.
- **내용**: LSTM 모델을 학습시키는 코드 포함.


### openvino 폴더 
여기에는 학습된 모델을 활용하여 웹캠을 통해 실시간 행동 분석을 수행하는 코드가 있습니다. 이를 통해 사용자는 실시간으로 행동을 분석하며 결과를 화면에 표시할 수 있습니다.
이러한 구조를 통해 편의점에서의 고객 행동 패턴을 정확하게 파악하고 분석할 수 있습니다. \
최소 33초 이상 찍혀야 행동을 판단합니다.\
- **실행 가능한 파일**:
  - 📄 `LAST_TEST`: 웹캠을 통해 개별 행동 예측 표시.
  - 📄 `LAST_ALL_TEST`: 웹캠을 통해 여러 행동 예측 표시.
  - 📄 `LAST_ALL_DB`: 웹캠을 통한 여러 행동 예측을 DB에 저장.
### 📁 `pred_model` 폴더
- **내용**: 학습된 LSTM 모델 저장 위치.

### 학습 원리
관절의 위치를 계속 저장하고 과거의 관절 위치 마이너스 현재의 관절위치를 해서 사람의 움직임을 학습 시켰습니다.
여기서 학습시킨 영상의 초당 프레임이 3프레임이였고 따라서 웹캠에서 관절이 저장되는 시점이 중요합니다. 

---

이 프로젝트를 통해 편의점에서의 고객 행동 패턴을 정확하게 파악하고 분석하는 것이 목표입니다.



# openvino-movenet-action-lstm

## 🇬🇧 Introduction
This project is designed for analyzing customer behaviors in convenience stores. It utilizes joint position data to train an LSTM model and provides real-time action analysis through a webcam.

### 📁 `people` folder
- **Function**: Train the LSTM model using joint position data.
- **Content**: Includes code for training the LSTM model.

### 📁 `openvino` folder
This folder contains code that utilizes the trained model to perform real-time action analysis through a webcam. Through this, users can analyze actions in real-time and display the results on the screen. This setup allows for precise understanding and analysis of customer behavior patterns in convenience stores. \
Actions are determined when recorded for at least 33 seconds.
- **Executable Files**:
  - 📄 `LAST_TEST`: Display individual action prediction via webcam.
  - 📄 `LAST_ALL_TEST`: Display multiple action predictions via webcam.
  - 📄 `LAST_ALL_DB`: Store multiple action predictions in a database through the webcam.

### 📁 `pred_model` folder
- **Content**: Location where the trained LSTM model is stored.

### Training Principle
The positions of the joints are continuously saved, and the movement of the person is learned by subtracting the current joint position from the past joint position. The videos used for training had a frame rate of 3 frames per second, so the point at which the joint is saved in the webcam is crucial.

---
