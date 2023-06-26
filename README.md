# Lane Detection Project 2022

### 차량 주행 영상 기반 도로 차선 검출 프로젝트
![Sliding window](https://github.com/shinjian/Lane-Detection/assets/75853990/b8c49c95-adde-4239-a8fa-1bdda54f66cf)

https://github.com/shinjian/Lane-Detection/assets/75853990/5be685a9-aadc-4c28-9fd9-2d2c79a95aa2


<img src="Sliding window2.gif" width="45%" height="45%"/>

---

## Introduction
- 프로젝트 제작 기간
    - 2022.10.09 ~ 2022.10.23 (14일)
    - 순수 제작 시간 9h 32m

이번 프로젝트는 차량의 곡선 부분까지 검출할 수 있는 알고리즘을 적용했습니다.

예전에 제작한 차선 검출 프로그램은 하프 변환 알고리즘을 기반으로 하기 때문에 곡선 부분의 차선은 검출하기 힘들었던 단점이 있었습니다. 이를 보완하고자 Sliding Window 알고리즘을 적용시켜 곡선 차선까지 검출하는데 성공했습니다.

추후 여기서 더 나아가 YOLO v5 등을 이용하여 영상 속에 있는 차량들까지 감지해서 거리 계산 알고리즘 및 보행자 인식까지 기대할 수 있을 것 같습니다.

---

## Development environment
- Visual Studio Code
    - https://code.visualstudio.com
- Python 3.9.11
    - https://www.python.org/downloads/release/python-3911
- OpencCV 4.5.5
    - ```pip install opencv-python```
    - ```python -m pip install opencv-python```
- Numpy Library
    - ```pip install numpy```

---

## Sliding Window Algorithm
|Sliding Window[.](http://poj.org/problem?id=2823)|
|:---:|
![Alt text](image.png)

Sliding Window 기법은 위 그림처럼 세 칸의 윈도우 크기(이하 "margin")를 정해주고 일정한 방향으로 윈도우(창)를 이동시켜가며 Maximum value를 중식으로 원하는 값이 제일 많이 분포하는 윈도우를 찾아가는 알고리즘이다.


## 1. Sliding Window를 위한 전처리 과정
<img src="1.jpg" width="40%" height="40%"/><img src="1111111111111111.png" width="40%" height="40%"/>

## 2. 이미지에 대한 Histogram으로 Window의 시작점을 정함
<img src="21.png" width="40%" height="40%"/>
<img src="Figure_1.png" width="32.56%" height="32%"/>

## 3. Sliding Window Algorithm 적용
<img src="Sliding window2_Moment.jpg" width="40%" height="40%"/>
<img src="Sliding window_Moment.jpg" width="40%" height="40%"/>

## 도로 주행영상 차선 검출 결과
<img src="Lane detection_Moment.jpg" width="60%" height="60%"/>



---

## Reference
- Navigational Path Detection Using Fuzzy Binarization and
Hough Transfor - [Korea Science](http://koreascience.or.kr/article/JAKO201408739560207.page)
- Sliding Window Algorithm - [POJ](http://poj.org/problem?id=2823)
