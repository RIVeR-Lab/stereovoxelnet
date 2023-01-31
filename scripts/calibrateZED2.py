import cv2
import numpy as np

if __name__ == "__main__":
    FRAME_WIDTH = int(2560/2)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    left_imgpoints = [] # 2d points in image plane.
    right_imgpoints = [] # 2d points in image plane.

    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    FRAME_WIDTH = int(2560/2)
    while True:
        _, img = vid.read()

        # right frame
        right_img = cv2.rotate(img[:,FRAME_WIDTH:], cv2.cv2.ROTATE_180)

        # left frame
        left_img = cv2.rotate(img[:,:FRAME_WIDTH], cv2.cv2.ROTATE_180)

        # DrivingStereo resolution
        right_img = cv2.resize(right_img, None, None, 0.6875, 0.6875, interpolation = cv2.INTER_LANCZOS4)
        left_img = cv2.resize(left_img, None, None, 0.6875, 0.6875, interpolation = cv2.INTER_LANCZOS4)
        # right_img = cv2.resize(right_img, None, None, 0.6875, 0.6875, interpolation = cv2.INTER_AREA)
        # left_img = cv2.resize(left_img, None, None, 0.6875, 0.6875, interpolation = cv2.INTER_AREA)

        right_gray = cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)
        left_gray = cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('right img',right_gray)
        cv2.imshow('left img',left_gray)

        # Find the chess board corners
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, (6,9),None)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, (6,9),None)

        # If found, add object points, image points (after refining them)
        if left_ret == True and right_ret == True:
            objpoints.append(objp)

            left_corners = cv2.cornerSubPix(left_gray,left_corners,(11,11),(-1,-1),criteria)
            left_imgpoints.append(left_corners)

            right_corners = cv2.cornerSubPix(right_gray,right_corners,(11,11),(-1,-1),criteria)
            right_imgpoints.append(right_corners)

            # Draw and display the corners
            left_img = cv2.drawChessboardCorners(left_img, (6,9), left_corners,left_ret)
            cv2.imshow('left img',left_img)

            right_img = cv2.drawChessboardCorners(right_img, (6,9), right_corners,right_ret)
            cv2.imshow('right img',right_img)

            cv2.waitKey(1000)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    left_focal = np.array([[366.61,0,448.69386382],[0, 366.77,254.89438038],[0,0,1]])
    right_focal = np.array([[366.34,0,441.43815104],[0, 366.47,252.22637952],[0,0,1]])
    left_dist = np.array([-0.0538111, 0.0255988, 5.81597e-05, -0.000453786, -0.00971802])
    right_dist = np.array([-0.0531113, 0.0251762, 0.000169659, 3.1518e-05, -0.00977131])

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, None, None,
                                                         None, None, left_gray.shape[::-1])