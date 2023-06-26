# Lane Detection Project 2022

차량 주행 영상 기반 도로 차선 검출 프로젝트

---

## Introduction
- 프로젝트 제작 기간
    - 2022.10.09 ~ 2022.10.23 (14일)
    - 순수 제작 시간 9h 32m

이번 프로젝트는 차량의 곡선 부분까지 검출할 수 있는 알고리즘을 적용했습니다.

예전에 제작한 차선 검출 프로그램은 하프 변환 알고리즘을 기반으로 하기 때문에 곡선 부분의 차선은 검출하기 힘들었던 단점이 있었습니다. 이를 보완하고자 Sliding Window 알고리즘을 적용시켜 곡선 차선까지 검출하는데 성공했습니다.

추후 여기서 더 나아가 YOLO v5 등을 이용하여 영상 속에 있는 차량들까지 감지해서 거리 계산 알고리즘 및 보행자 인식까지 적용시켜볼 계획입니다.

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
