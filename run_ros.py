#!/usr/bin/env python3
import os

import numpy as np
import torch.utils.data
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from cv_bridge import CvBridge

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps


class Global:
    use_depth = None
    use_rgb = None
    n_grasps = None
    force_cpu = None
    sim_mode = None
    network = None
    cvbridge = None
    rgb = None
    depth = None
    rgb_ready = False
    depth_ready = False


# ----- Callbacks -----
def rgb_callback(data):
    """Convert and save RGB Image

    Args:
        data (Image): RGB Image from ROS
    """
    Global.rgb = Global.cvbridge.imgmsg_to_cv2(data, "passthrough")
    Global.rgb_ready = True


def depth_callback(data):
    """Convert and save Depth Image

    Args:
        data (Image): Depth Image from ROS
    """
    if Global.sim_mode:
        # In simulation, NaN can be recieved. So, `16UC1` encoding is needed.
        Global.depth = Global.cvbridge.imgmsg_to_cv2(data, "16UC1")
    else:
        Global.depth = Global.cvbridge.imgmsg_to_cv2(data, "passthrough")
    Global.depth = np.expand_dims(Global.depth, axis=2)
    Global.depth_ready = True


# ----- Main -----
if __name__ == '__main__':
    rospy.init_node('antipodal_robotic_grasping', anonymous=True)

    # Load Parameters
    Global.use_depth = rospy.get_param("~use_depth", True)
    Global.use_rgb = rospy.get_param("~use_rgb", True)
    Global.n_grasps = rospy.get_param("~n_grasps", 3)
    Global.force_cpu = rospy.get_param("~force_cpu", True)
    Global.sim_mode = rospy.get_param("~sim_mode", False)
    Global.network = rospy.get_param("~network", os.path.join(
        os.path.dirname(__file__),
        "trained-models",
        "cornell-randsplit-rgbd-grconvnet3-drop1-ch32",
        "epoch_19_iou_0.98",
    ))

    rate = rospy.Rate(30)
    Global.cvbridge = CvBridge()

    # Setup Camera
    rospy.loginfo('Setting up the camera...')
    if Global.use_depth:
        rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)
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

    # Publisher
    publisher = rospy.Publisher("grasp", Float64MultiArray, queue_size=1)

    # Layout for the message
    layout = MultiArrayLayout()
    layout.dim.append(MultiArrayDimension())
    layout.dim.append(MultiArrayDimension())
    layout.dim[0].label = "n_grasps"
    layout.dim[0].size = Global.n_grasps
    layout.dim[0].stride = Global.n_grasps*7
    layout.dim[1].label = "grasp_data"
    layout.dim[1].size = 7
    layout.dim[1].stride = 7
    layout.data_offset = 0

    while (not Global.rgb_ready) or (not Global.depth_ready):
        pass

    if Global.sim_mode:
        rospy.loginfo('Running in simulation mode...')

    while not rospy.is_shutdown():
        depth = Global.depth
        x, depth_img, rgb_img = cam_data.get_data(rgb=Global.rgb, depth=depth)
        with torch.no_grad():
            # Predict
            xc = x.to(device)
            pred = net.predict(xc)
            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

            # Extract grasps
            grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=Global.n_grasps)

            # Publish grasp message
            msg = Float64MultiArray()
            msg.layout = layout
            for g in grasps:
                x = g.center[1] + cam_data.top_left[1]
                y = g.center[0] + cam_data.top_left[0]
                if Global.sim_mode:
                    z = depth[y, x]
                else:
                    z = depth[y, x] / 1000.0  # Convert to [m]
                msg.data.extend([
                    x,
                    y,
                    z,
                    g.angle,
                    g.length,
                    g.width,
                    g.quality,
                ])
            publisher.publish(msg)
            rate.sleep()
