import os,numpy as np
import cv2 as cv
import argparse
def func(x):
    pass
# читаем исходное изображение
def run(
        hong_img_path=r'C:\Users\Dmitry\Desktop\M8\M8MachineLearningLabsAndHomeworks\Sem3\Lab5\hw\hong.png',  # dataset.yaml path
):
    img = cv.cvtColor(cv.imread(hong_img_path), cv.COLOR_BGR2GRAY)
    # меняем его размер, иначе оно на пол экрана
    width = int(img.shape[1] * 0.5)
    height = int(img.shape[0] * 0.5)
    img = cv.resize(img,(width, height))
    cv.namedWindow('canny')
    #ползунки
    cv.createTrackbar('threshold1','canny',0,1000,func)
    cv.createTrackbar('threshold2','canny',0,1000,func)
    while(1):
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        #получаем значения
        threshold1 = cv.getTrackbarPos('threshold1','canny')
        threshold2 = cv.getTrackbarPos('threshold2','canny')
        #обрабатываем
        edges = cv.Canny(img, threshold1, threshold2)
        cv.imshow('canny', edges)
    cv.destroyAllWindows()
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hong_img_path', nargs='+', type=str, default=r'C:\Users\Dmitry\Desktop\M8\M8MachineLearningLabsAndHomeworks\Sem3\Lab5\hw\hong.png', help='Path to image')
    opt = parser.parse_args()
    return opt
def main(opt):
    run(**vars(opt))
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)