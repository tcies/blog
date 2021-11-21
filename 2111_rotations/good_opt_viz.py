import IPython
import numpy as np
import pyquaternion
import scipy.optimize
import time

import geometry_msgs.msg
import rospy
import tf2_ros
import visualization_msgs.msg

import bad_opt
import good_opt


def make_generic_marker(frame, rgba, stamp=None, marker_id=0):
    if stamp is None:
        stamp = rospy.Time.now()
    marker = visualization_msgs.msg.Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = frame
    marker.id = marker_id
    marker.action = marker.ADD
    marker.pose.orientation.w = 1.
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]
    marker.scale.x = 1.
    marker.scale.y = 1.
    marker.scale.z = 1.
    if len(rgba) == 4:
        marker.color.a = rgba[3]
    else:
        marker.color.a = 1.
    return marker


def make_point_message(x):
    msg = geometry_msgs.msg.Point()
    msg.x = x[0]
    msg.y = x[1]
    msg.z = x[2]
    return msg


def make_point_marker(xyz, marker_id):
    marker = make_generic_marker('world', [0., 0., 1.], marker_id=marker_id)
    marker.type = marker.SPHERE
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.pose.position = make_point_message(xyz)
    # marker.points = [make_point_message(i) for i in xyz]
    return marker


buf = None
listener = None
broadcaster = None


def initializeIfIsnt():
    global buf
    if buf is not None:
        return
    buf = tf2_ros.Buffer()
    global listener
    listener = tf2_ros.TransformListener(buf)
    global broadcaster
    broadcaster = tf2_ros.TransformBroadcaster()
    time.sleep(0.2)


def toTransformMsg(R, t):
    msg = geometry_msgs.msg.Transform()
    msg.translation.x = t[0]
    msg.translation.y = t[1]
    msg.translation.z = t[2]
    q = pyquaternion.Quaternion(matrix=R).unit.q
    msg.rotation.w = q[0]
    msg.rotation.x = q[1]
    msg.rotation.y = q[2]
    msg.rotation.z = q[3]
    return msg


def sendPose(A, B, R_A_B, t_A_B, stamp=None):
    if stamp is None:
        stamp = rospy.Time.now()
    initializeIfIsnt()
    global broadcaster
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = A
    t.child_frame_id = B
    t.transform = toTransformMsg(R_A_B, t_A_B)
    broadcaster.sendTransform(t)


def main():
    rospy.init_node('good_opt_viz')
    pub = rospy.Publisher('points', visualization_msgs.msg.Marker, queue_size=10)
    time.sleep(0.5)

    p = bad_opt.Problem()
    for i, pt in zip(range(len(p.points)), p.points):
        pub.publish(make_point_marker(pt, i))
        time.sleep(0.1)
        pub.publish(make_point_marker(pt, i))
    IPython.embed()

    R_opt = scipy.optimize.minimize(lambda x: p.error(good_opt.R_from_rpy(x)), np.zeros(3))
    while not rospy.is_shutdown():
        sendPose('world', 'opt', good_opt.R_from_rpy(R_opt.x), np.zeros(3))
        time.sleep(0.1)


if __name__ == '__main__':
    main()
