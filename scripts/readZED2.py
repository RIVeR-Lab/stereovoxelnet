# import the opencv library
import cv2
import configparser
from matplotlib import image
import numpy as np
import os
import logging
import coloredlogs
import time
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

REVERSE = True

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

CURR_DIR = os.path.dirname(__file__)

LEFT_CAMERA_INFO = None
RIGHT_CAMERA_INFO = None

def read_calib(calib_file="SN28281527.conf"):
    config = configparser.ConfigParser()
    config.read(os.path.join(CURR_DIR, calib_file))

    baseline = float(config["STEREO"]["Baseline"])
    T = np.zeros((3,1), dtype=float)
    T[0,0] = baseline
    T[1,0] = float(config["STEREO"]["TY"])
    T[2,0] = float(config["STEREO"]["TZ"])

    left_cam_cx = float(config["LEFT_CAM_DS"]["cx"])
    left_cam_cy = float(config["LEFT_CAM_DS"]["cy"])
    left_cam_fx = float(config["LEFT_CAM_DS"]["fx"])
    left_cam_fy = float(config["LEFT_CAM_DS"]["fy"])
    left_cam_k1 = float(config["LEFT_CAM_DS"]["k1"])
    left_cam_k2 = float(config["LEFT_CAM_DS"]["k2"])
    left_cam_p1 = float(config["LEFT_CAM_DS"]["p1"])
    left_cam_p2 = float(config["LEFT_CAM_DS"]["p2"])
    left_cam_k3 = float(config["LEFT_CAM_DS"]["k3"])

    right_cam_cx = float(config["RIGHT_CAM_DS"]["cx"])
    right_cam_cy = float(config["RIGHT_CAM_DS"]["cy"])
    right_cam_fx = float(config["RIGHT_CAM_DS"]["fx"])
    right_cam_fy = float(config["RIGHT_CAM_DS"]["fy"])
    right_cam_k1 = float(config["RIGHT_CAM_DS"]["k1"])
    right_cam_k2 = float(config["RIGHT_CAM_DS"]["k2"])
    right_cam_p1 = float(config["RIGHT_CAM_DS"]["p1"])
    right_cam_p2 = float(config["RIGHT_CAM_DS"]["p2"])
    right_cam_k3 = float(config["RIGHT_CAM_DS"]["k3"])

    R_zed = np.zeros((1,3), dtype=float)
    R_zed[0,0] = float(config["STEREO"]["RX_DS"])
    R_zed[0,1] = float(config["STEREO"]["CV_DS"])
    R_zed[0,2] = float(config["STEREO"]["RZ_DS"])

    R,_ = cv2.Rodrigues(R_zed)

    cameraMatrix_left = np.zeros((3,3), dtype=float)
    cameraMatrix_left[0,0] = left_cam_fx
    cameraMatrix_left[0,2] = left_cam_cx
    cameraMatrix_left[1,1] = left_cam_fy
    cameraMatrix_left[1,2] = left_cam_cy
    cameraMatrix_left[2,2] = 1.0

    distCoeffs_left = np.zeros((5,1), dtype=float)
    distCoeffs_left[0,0] = left_cam_k1
    distCoeffs_left[1,0] = left_cam_k2
    distCoeffs_left[2,0] = left_cam_p1
    distCoeffs_left[3,0] = left_cam_p2
    distCoeffs_left[4,0] = left_cam_k3

    cameraMatrix_right = np.zeros((3,3), dtype=float)
    cameraMatrix_right[0,0] = right_cam_fx
    cameraMatrix_right[0,2] = right_cam_cx
    cameraMatrix_right[1,1] = right_cam_fy
    cameraMatrix_right[1,2] = right_cam_cy
    cameraMatrix_right[2,2] = 1.0

    distCoeffs_right = np.zeros((5,1), dtype=float)
    distCoeffs_right[0,0] = right_cam_k1
    distCoeffs_right[1,0] = right_cam_k2
    distCoeffs_right[2,0] = right_cam_p1
    distCoeffs_right[3,0] = right_cam_p2
    distCoeffs_right[4,0] = right_cam_k3

    if REVERSE:
        _tmp_mat = cameraMatrix_left
        _tmp_dist = distCoeffs_left
        cameraMatrix_left = cameraMatrix_right
        distCoeffs_left = distCoeffs_right
        cameraMatrix_right = _tmp_mat
        distCoeffs_right = _tmp_dist
    
    # image_size = (int(2560/2),720)
    image_size = (880, 495)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, image_size, R, T)
    # change unit
    P2[0][-1] /= -1000

    rospy.logdebug(f"Camera Matrix L: {cameraMatrix_left}")
    rospy.logdebug(f"Camera Matrix R: {cameraMatrix_right}")

    left_cam_info = CameraInfo()
    left_cam_info.width = image_size[0]
    left_cam_info.height = image_size[1]
    left_cam_info.D = distCoeffs_left.T.tolist()[0]
    left_cam_info.K = cameraMatrix_left.reshape(-1,9).tolist()[0]
    left_cam_info.R = R1.reshape(-1,9).tolist()[0]
    left_cam_info.P = P1.reshape(-1,12).tolist()[0]
    left_cam_info.distortion_model = "plumb_bob"
    left_cam_info.header = Header()
    left_cam_info.header.stamp = rospy.Time.now()
    left_cam_info.header.frame_id = "zed_left"

    right_cam_info = CameraInfo()
    right_cam_info.width = image_size[0]
    right_cam_info.height = image_size[1]
    right_cam_info.D = distCoeffs_right.T.tolist()[0]
    right_cam_info.K = cameraMatrix_right.reshape(-1,9).tolist()[0]
    right_cam_info.R = R2.reshape(-1,9).tolist()[0]
    right_cam_info.P = P2.reshape(-1,12).tolist()[0]
    right_cam_info.distortion_model = "plumb_bob"
    right_cam_info.header = Header()
    right_cam_info.header.stamp = rospy.Time.now()
    right_cam_info.header.frame_id = "zed_right"

    global LEFT_CAMERA_INFO
    global RIGHT_CAMERA_INFO
    LEFT_CAMERA_INFO = left_cam_info
    RIGHT_CAMERA_INFO = right_cam_info

def main(namespace=""):
    rate = rospy.Rate(10)

    # read ZED2 factory calibration file
    read_calib()

    # define a video capture object
    frame = None
    while frame is None:
        rospy.logwarn("Camera Stream not ready yet!")
        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        _, frame = vid.read()

        # avoid crazy reading
        rate.sleep()
    
    rospy.logwarn("Camera Stream ready!")
    FRAME_WIDTH = int(2560/2)

    bridge = CvBridge()
    raw_left_image_pub = rospy.Publisher(namespace+"left/image_raw", Image, queue_size=10)
    raw_right_image_pub = rospy.Publisher(namespace+"right/image_raw", Image, queue_size=10)
    left_camear_info_pub = rospy.Publisher(namespace+"left/camera_info", CameraInfo, queue_size=10)
    right_camear_info_pub = rospy.Publisher(namespace+"right/camera_info", CameraInfo, queue_size=10)
    
    while not rospy.is_shutdown():    
        # Capture the video frame
        # by frame
        _, frame = vid.read()

        if not REVERSE:
            left_frame = frame[:,:FRAME_WIDTH]
            right_frame = frame[:,FRAME_WIDTH:]
        else:
            # camera is upside-down
            left_frame = cv2.rotate(frame[:,FRAME_WIDTH:], cv2.cv2.ROTATE_180)
            right_frame = cv2.rotate(frame[:,:FRAME_WIDTH], cv2.cv2.ROTATE_180)
        
        # resize to near DrivingStereo resolution
        left_frame = cv2.resize(left_frame, None, None, 0.6875, 0.6875, interpolation = cv2.INTER_AREA)
        right_frame = cv2.resize(right_frame, None, None, 0.6875, 0.6875, interpolation = cv2.INTER_AREA)

        left_img = bridge.cv2_to_imgmsg(left_frame,"bgr8")
        right_img = bridge.cv2_to_imgmsg(right_frame,"bgr8")

        right_img.header.stamp = left_img.header.stamp
        raw_left_image_pub.publish(left_img)
        raw_right_image_pub.publish(right_img)

        if LEFT_CAMERA_INFO is not None:
            LEFT_CAMERA_INFO.header.stamp = left_img.header.stamp
            left_camear_info_pub.publish(LEFT_CAMERA_INFO)

        if RIGHT_CAMERA_INFO is not None:
            RIGHT_CAMERA_INFO.header.stamp = left_img.header.stamp
            right_camear_info_pub.publish(RIGHT_CAMERA_INFO)

        rate.sleep()

    
    # After the loop release the cap object
    vid.release()

if __name__ == "__main__":
    try:
        rospy.init_node("readZED2", log_level=rospy.DEBUG)
        rospy.loginfo(f"Node Namespace is {rospy.get_namespace()}")
        main(rospy.get_namespace())
    except rospy.ROSInterruptException:
        pass