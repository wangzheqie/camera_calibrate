#!/user/bin/env python2.7
# -*- coding: utf-8 -*- 

__author__ = "wang zhe"
__date__ = "18-6-10, 9:36"
__copyright__ = "copyright PI, 2018"
__version__ = "1.0"

import sys
import os
import numpy as np
import cv2
import glob

is_debug = False


def SingleCalibrate(path, cam, patchSize):
    mtx_name = "matrix_" + cam + ".txt"
    dist_name = "dist_" + cam + ".txt"
    rvecs_name = "rvecs_" + cam + ".txt"
    tvecs_name = "tvecs_" + cam + ".txt"
    objp_name = "objp_" + cam + ".txt"
    imgp_name = "imgp_" + cam + ".txt"

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w = patchSize[1]
    h = patchSize[0]
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objPoints = []
    imgPoints = []
    images = glob.glob(path + "*.png")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, centers = cv2.findCirclesGrid(gray, (w, h), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        if ret == True:
            objPoints.append(objp)
            imgPoints.append(centers)
            if is_debug:
                cv2.drawChessboardCorners(img, (w, h), centers, ret)
                cv2.imshow("img", img)
                cv2.waitKey(delay=0)
                cv2.destroyWindow("img  ")

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    np.savetxt(path + mtx_name, mtx, fmt="%.6e", delimiter=" ")
    np.savetxt(path + dist_name, dist, fmt="%.6e", delimiter=" ")
    np.savetxt(path + rvecs_name, np.array(rvecs).reshape(-1, 3), fmt="%.6e", delimiter=" ")
    np.savetxt(path + tvecs_name, np.array(tvecs).reshape(-1, 3), fmt="%.6e", delimiter=" ")
    np.savetxt(path + objp_name, np.array(objPoints).reshape(-1, 3), fmt="%.6e", delimiter=" ")
    np.savetxt(path + imgp_name, np.array(imgPoints).reshape(-1, 2), fmt="%.6e", delimiter=" ")

    if is_debug:
        print("the rms is : ", rms)
    return rms


def StereoCalibrate(path_l, path_r, path_stereo, patternSize, imgSize):
    obj = np.loadtxt(path_l + "objp_left.txt", np.float32)
    imgpL = np.loadtxt(path_l + "imgp_left.txt", np.float32)
    imgpR = np.loadtxt(path_r + "imgp_right.txt", np.float32)
    mtxL = np.loadtxt(path_l + "matrix_left.txt", np.float32)
    mtxR = np.loadtxt(path_r + "matrix_right.txt", np.float32)
    distL = np.loadtxt(path_l + "dist_left.txt", np.float32)
    distR = np.loadtxt(path_r + "dist_right.txt", np.float32)

    t = patternSize[0] * patternSize[1]
    obj = obj.reshape(-1, t, 3)
    imgpL = imgpL.reshape(-1, t, 2)
    imgpR = imgpR.reshape(-1, t, 2)
    print distL
    print tuple(distL)
    ret, AL, AR, DL, DR, R, T, E, F = cv2.stereoCalibrate(obj, imgpL, imgpR, cameraMatrix1=mtxL, distCoeffs1=distL,
                                                          cameraMatrix2=mtxR, distCoeffs2=distR, imageSize=imgSize)
    np.savetxt(path_stereo + "AL.txt", AL)
    np.savetxt(path_stereo + "AR.txt", AR)
    np.savetxt(path_stereo + "DL.txt", DL)
    np.savetxt(path_stereo + "DR.txt", DR)
    np.savetxt(path_stereo + "R.txt", R)
    np.savetxt(path_stereo + "T.txt", T)
    np.savetxt(path_stereo + "E.txt", E)
    np.savetxt(path_stereo + "F.txt", F)

    print(ret)


def stereoRectify(path_l, path_r, path_stereo, alpha, imgSize):
    """

    :param path_l:
    :param path_r:
    :param path_stereo:
    :param alpha: 0: retrieve only sensible pixels, 1: keep all pixels
    :return:
    """
    mtxL = np.loadtxt(path_l + "matrix_left.txt")
    mtxR = np.loadtxt(path_r + "matrix_right.txt")
    distL = np.loadtxt(path_l + "dist_left.txt")
    distR = np.loadtxt(path_r + "dist_right.txt")
    R = np.loadtxt(path_stereo + "R.txt")
    T = np.loadtxt(path_stereo + "T.txt")

    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, imgSize, R, T, alpha=0)
    np.savetxt(path_stereo + "RL.txt", RL)
    np.savetxt(path_stereo + "RR.txt", RR)
    np.savetxt(path_stereo + "PL.txt", PL)
    np.savetxt(path_stereo + "PR.txt", PR)
    np.savetxt(path_stereo + "Q.txt", Q)


def stereoUnDist(path_l, path_r, path_stereo, imgSize):
    mtxL = np.loadtxt(path_l + "matrix_left.txt")
    mtxR = np.loadtxt(path_r + "matrix_right.txt")
    distL = np.loadtxt(path_l + "dist_left.txt")
    distR = np.loadtxt(path_r + "dist_right.txt")
    RL = np.loadtxt(path_stereo + "RL.txt")
    PL = np.loadtxt(path_stereo + "PL.txt")
    RR = np.loadtxt(path_stereo + "RR.txt")
    PR = np.loadtxt(path_stereo + "PR.txt")
    mapxL, mapyL = cv2.initUndistortRectifyMap(mtxL, distL, RL, PL, imgSize, cv2.CV_32FC1)
    mapxR, mapyR = cv2.initUndistortRectifyMap(mtxR, distR, RR, PR, imgSize, cv2.CV_32FC1)
    imagesL = glob.glob(path_l + "*.png")
    imagesR = glob.glob(path_r + "*.png")
    for fname in imagesL:
        imgL = cv2.imread(fname)
        r_imgL = cv2.remap(imgL, mapxL, mapyL, cv2.INTER_LINEAR)
        path, name = os.path.split(fname)
        # shotname, extension = os.path.splitext(name)
        cv2.imwrite(path + "/rectified/rectify_" + name, r_imgL)

    for fname in imagesR:
        imgR = cv2.imread(fname)
        r_imgR = cv2.remap(imgR, mapxR, mapyR, cv2.INTER_LINEAR)
        path, name = os.path.split(fname)
        cv2.imwrite(path + "/rectified/rectify_" + name, r_imgR)


def unDist( img, path, cam, alpha=0, is_debug=False):
    """

    :param img:
    :param alpha: 0: retrieve only sensible pixels, 1: keep all pixels
    :return:
    """
    mtx_name = "matrix_" + cam + ".txt"
    dist_name = "dist_" + cam + ".txt"

    row, col = img.shape[:2]
    mtx = np.loadtxt(path + mtx_name)
    dist = np.loadtxt(path + dist_name)
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (row, col), alpha, (row, col))
    dst = cv2.undistort(img, mtx, dist, None, new_mtx)
    y, x, h, w = roi
    dst = dst[y:y + h, x:x + w]
    if is_debug:
        cv2.imwrite("dst.png", dst)
    return dst, roi


def reProjectError( path, cam, objPoints, imgPoints):
    total_error = 0

    mtx_name = "matrix_" + cam + ".txt"
    dist_name = "dist_" + cam + ".txt"
    rvecs_name = "rvecs_" + cam + ".txt"
    tvecs_name = "tvecs_" + cam + ".txt"
    mtx = np.loadtxt(path + mtx_name)
    dist = np.loadtxt(path + dist_name)
    rvecs = np.loadtxt(path + rvecs_name)
    tvecs = np.loadtxt(path + tvecs_name)

    for i in xrange(len(objPoints)):
        imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgPoints[i], imgPoints2, cv2.NORM_L2) / len(imgPoints2)
        total_error += error
        print("total error: ", total_error)
    return total_error


def main():
    path_l = "input/cam_left/"
    path_r = "input/cam_right/"
    stereo_path = "output/stereo/"

    patternSize = (7, 7)
    imgSize = (1280, 1024)
    # SingleCalibrate(path_l, "left", patternSize)
    # SingleCalibrate(path_r, "right", patternSize)

    # StereoCalibrate(path_l, path_r, stereo_path, patternSize, imgSize)
    # stereoRectify(path_l, path_r, stereo_path, 0, imgSize)

    stereoUnDist(path_l, path_r, stereo_path, imgSize)


if __name__ == "__main__":
    main()
