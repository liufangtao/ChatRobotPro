#!/usr/bin/env python3

import rospy
from ros_operator_command.srv import RosCommand

def main():
    rospy.init_node("ros_operator_command")
    command_client = rospy.ServiceProxy("ros_operator/command", RosCommand)
    print('wait command server.')
    rospy.wait_for_service("ros_operator/command")
    print('connected server.')

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        try:
            command = input('input command: \npause(暂停)\nstart(启动)\nreset(复位)\nquit(退出)\n:')
            result = command_client(command)
            if not result:
                print('command error.')
            else:
                print('command success.')
        except rospy.ServiceException as e:
            print('service call fail: %s' %e)

if __name__ == '__main__':
    main()