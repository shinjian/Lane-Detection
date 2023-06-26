# Lane Detection Project 2022

### 차량 주행 영상 기반 도로 차선 검출 프로젝트

<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/081364e6-9214-487e-93f2-c0a83ed73c8b" width="45%" height="45%"/>
<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/f67627e0-30db-4d74-ba96-ce3bd7f3eb6d" width="45%" height="45%"/>
<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/07e75318-b179-4ea5-a12d-0e31e5a88579" width="45%" height="45%"/>

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
- Numpy Library 1.22.1
    - ```pip install numpy```

---

## Sliding Window Algorithm
|Sliding Window[.](http://poj.org/problem?id=2823)|
|:---:|
![image](https://github.com/shinjian/Lane-Detection/assets/75853990/1dbbc61f-7fc7-4032-b8a2-0aab2e35986c)

Sliding Window 기법은 위 그림처럼 세 칸의 윈도우 크기(이하 "margin")를 정해주고 일정한 방향으로 윈도우(창)를 이동시켜가며 Maximum value를 중식으로 원하는 값이 제일 많이 분포하는 윈도우를 찾아가는 알고리즘입니다.


## 1. Sliding Window를 위한 전처리 과정
<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/ca35dcef-ccce-4a0b-b7c7-dce0884e585e" width="40%" height="40%"/><img src="https://github.com/shinjian/Lane-Detection/assets/75853990/ad28aef2-add3-4f76-a553-29b449a22eb3" width="40%" height="40%"/>

## 2. 이미지에 대한 Histogram으로 Window의 시작점을 정함
<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/4ca9192b-2250-4086-a030-d2652943daea" width="40%" height="40%"/>
<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/3fc032fd-46f5-448d-a228-631b55c70920" width="32.56%" height="32%"/>

## 3. Sliding Window Algorithm 적용
<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/b9fc4b52-b4a0-4edb-b43e-89f8efa534f7" width="40%" height="40%"/>
<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/384ad8ac-3d1f-4b76-ba3a-e34826883cb9" width="40%" height="40%"/>

## 도로 주행영상 차선 검출 결과
<img src="https://github.com/shinjian/Lane-Detection/assets/75853990/33ed103e-edb9-446c-8fcb-45cfbdd7e077" width="60%" height="60%"/>



---

## References
- Navigational Path Detection Using Fuzzy Binarization and
Hough Transfor - [Korea Science](http://koreascience.or.kr/article/JAKO201408739560207.page)
- Sliding Window Algorithm - [POJ](http://poj.org/problem?id=2823)
