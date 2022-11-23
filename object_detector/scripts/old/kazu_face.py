import pyrealsense2 as rs
import numpy as np
import cv2

cascade_path = "/catkin_ws/src/ytlab_whill_modules/config/opencv_face_detector/haarcascade_eye.xml"

def detect_human(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(50, 50))
    color = (255, 255, 255)
    if len(facerect) > 0:
        #検出した顔を囲む矩形の作成
        for rect in facerect:
            cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
        return image
    else:
        return image

# ストリーム(Color/Depth)の設定
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

try:
    while True:
        # フレーム待ち
        frames = pipeline.wait_for_frames()

        #RGB
        RGB_frame = frames.get_color_frame()
        RGB_image = np.asanyarray(RGB_frame.get_data())

        #depyh
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)


        # 表示
        RGB_image=detect_human(RGB_image)
        images = np.hstack((RGB_image, depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        if cv2.waitKey(1) & 0xff == 27:#ESCで終了
            cv2.destroyAllWindows()
            break

finally:
    # ストリーミング停止
    pipeline.stop()