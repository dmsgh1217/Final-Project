import numpy as np

def reference_loc(x, y):
    x3 = (x[5] + x[17] + x[0]) / 3
    y3 = (y[5] + y[17] + y[0]) / 3
    return [x3, y3]

def sol_length(x1, y1, x2, y2):  # solution_length, 두 점사이의 길이
  return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def sol_ratio(x1, y1, x2, y2, x3, y3):  # solution_ratio
  len_a = sol_length(x1, y1, x3, y3)
  len_b = sol_length(x2, y2, x3, y3)
  return len_a / len_b

def __angle_between(p1, p2):  # 두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    res = np.rad2deg((ang1 - ang2) % (2 * np.pi))  # radius 값 -> degree 로 바꾼 값에 2파이로 나눈 나머지를 구함 // 360도 넘어가지 않도록 설정
    return res

def getAngle3P(p1, p2, p3, direction="CW"):  # 세점 사이의 각도 1->2->3, CW = 시계 방향
    pt1 = (p1[0] - p2[0], p1[1] - p2[1])
    pt2 = (p3[0] - p2[0], p3[1] - p2[1])  # p2 좌표를 0,0으로 재설정
    res = __angle_between(pt1, pt2)
    res = (res + 360) % 360
    if direction == "CCW":  # 반시계 방향
        res = (360 - res) % 360
    return res

def convert_ratio(x, y, normalization=2.5):  # 비율에 대한 DataFrame 생성 함수, 손바닥 끝~손가락 가장 끝마디/손바닥 끝~손가락 가장 안쪽마디 = 2.5 배
    fingers = [[4, 5, 17], [8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
    result = []

    for i in fingers:
        result.append(sol_ratio(x[i[0]], y[i[0]], x[i[1]], y[i[1]], x[i[2]], y[i[2]]) / normalization)

    return result


def convert_angle(x, y, normalization=360):  # 각도에 대한 DataFrame 생성 함수
    fingers = [[4, 3, 2, 1], [8, 7, 6, 5, 9], [12, 11, 10, 9, 13], [16, 15, 14, 13, 17], [20, 19, 18, 17, 0]]  # 엄지~새끼손가락 순
    result = []

    for i in range(5):  # 손가락 개수 (엄지~새끼손가락 순)
        count = len(fingers[i]) - 2 # 한 리스트에서 세 점을 순차적으로 추출해서 계산 ex) 엄지 = [4,3,2], [3,2,1]

        for j in range(count):
            result.append(
                getAngle3P([x[fingers[i][j]], y[fingers[i][j]]],
                           [x[fingers[i][j + 1]], y[fingers[i][j + 1]]],
                           [x[fingers[i][j + 2]], y[fingers[i][j + 2]]]) / normalization)

    return result

