#!/usr/bin/env python
import argparse
from robot import Robot
import numpy as np

def main(args):
    workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])
    # Initialize pick-and-place system (camera and robot)
    robot = Robot(workspace_limits=workspace_limits,
                 tcp_host_ip=args.tcp_host_ip, tcp_port=args.tcp_port, rtc_host_ip=args.rtc_host_ip, rtc_port=args.rtc_port )

    robot.move_joints(joint_configuration=args.joint_configuration)
if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',
                        help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,
                        help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',
                        help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,
                        help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--joint_configuration', type=list,
                        help='joint_configuration 6 * 1 list (UR5)')
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)