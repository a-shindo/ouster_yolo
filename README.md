# ouster_yolo

# launch
    <arg name="topic_name_nearir" default="/ouster/nearir_image"/>
    <arg name="topic_name_range" default="/ouster/range_image"/>
    <arg name="topic_name_reflec" default="/ouster/reflec_image"/>
    <arg name="topic_name_signal" default="/ouster/signal_image"/>


# しきい値
detectron2_core_shindo.py
 line 37 self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.4     40%