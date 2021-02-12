#!/usr/bin/env python3
import os

import numpy as np
import torch.utils.data
import rospy
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from cv_bridge import CvBridge

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps


class Global:
    use_depth = rospy.get_param("use_depth", True)
    use_rgb = rospy.get_param("use_rgb", True)
    n_grasps = rospy.get_param("n_grasps", 3)
    force_cpu = rospy.get_param("force_cpu", True)
    network = rospy.get_param("network", os.path.join(
        os.path.dirname(__file__),
        "trained-models",
        "cornell-randsplit-rgbd-grconvnet3-drop1-ch32",
        "epoch_19_iou_0.98",
    ))
    cvbridge = None
    rgb = None
    depth = None
    rgb_ready = False
    depth_ready = False
    publish_topics = [
        "center_x",
        "center_y",
        "angle",
        "quality",
        "length",
        "width",
    ]


def rgb_callback(data):
    Global.rgb = Global.cvbridge.imgmsg_to_cv2(data, "passthrough")
    Global.rgb_ready = True


def depth_callback(data):
    Global.depth = Global.cvbridge.imgmsg_to_cv2(data, "passthrough")
    Global.depth = np.expand_dims(Global.depth, axis=2)
    Global.depth_ready = True


if __name__ == '__main__':
    rospy.init_node('antipodal_robotic_grasping', anonymous=True)
    rate = rospy.Rate(30)
    Global.cvbridge = CvBridge()

    # Setup Camera
    rospy.loginfo('Setting up the camera...')
    if Global.use_depth:
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)
    if Global.use_rgb:
        rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    cam_data = CameraData(include_depth=Global.use_depth, include_rgb=Global.use_rgb)

    # Load Network
    rospy.loginfo('Loading model...')
    if Global.force_cpu:
        net = torch.load(Global.network, map_location=torch.device('cpu'))
    else:
        net = torch.load(Global.network)
    rospy.loginfo('Loading model...Done')

    # Get the compute device
    device = get_device(Global.force_cpu)

    # Publishers
    publishers = [
        rospy.Publisher(
            "{:s}".format(topic),
            Float64MultiArray,
            queue_size=1
        )
        for topic in Global.publish_topics
    ]

    publisher = rospy.Publisher("grasp", Float64MultiArray, queue_size=1)

    # Layout for the message
    layout = MultiArrayLayout()
    layout.dim.append(MultiArrayDimension())
    layout.dim.append(MultiArrayDimension())
    layout.dim[0].label = "grasp_data"
    layout.dim[0].size = 6
    layout.dim[0].stride = 6*Global.n_grasps
    layout.dim[1].label = "n_grasps"
    layout.dim[1].size = Global.n_grasps
    layout.dim[1].stride = Global.n_grasps
    layout.data_offset = 0

    while (not Global.rgb_ready) or (not Global.depth_ready):
        pass

    while not rospy.is_shutdown():
        x, depth_img, rgb_img = cam_data.get_data(rgb=Global.rgb, depth=Global.depth)
        with torch.no_grad():
            xc = x.to(device)
            pred = net.predict(xc)
            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

            grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=Global.n_grasps)

            msg = Float64MultiArray()
            msg.layout = layout
            for g in grasps:
                msg.data.extend([
                    g.center[0],
                    g.center[1],
                    g.angle,
                    g.length,
                    g.width,
                    g.quality,
                ])

            # for topic in Global.publish_topics:
            #     if topic == "center_x":
            #         pub_msgs.append(Float64MultiArray(
            #             data=[g.center[0] for g in grasps]
            #         ))
            #     elif topic == "center_y":
            #         pub_msgs.append(Float64MultiArray(
            #             data=[g.center[1] for g in grasps]
            #         ))
            #     else:
            #         pub_msgs.append(Float64MultiArray(
            #             data=[getattr(g, topic) for g in grasps]
            #         ))

            # [publishers[i].publish(pub_msgs[i]) for i in range(len(Global.publish_topics))]
            publisher.publish(msg)
            rate.sleep()
