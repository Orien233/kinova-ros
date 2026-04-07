#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Kinova 示教/夹爪控制脚本（ROS Melodic 友好版）。

功能：
1. 键盘控制夹爪开合。
2. 在 trajectory / torque(gravity) 模式之间切换。
3. 提供 FingerTeleopModeSwitch 类，供其他脚本复用。

兼容性说明：
- 使用 Python 2/3 兼容写法，适合 ROS Melodic。
- 不依赖 Python 3 专属语法（f-string、type hint、pathlib 等）。
"""

import select
import sys
import termios
import tty

import actionlib
import rospy
import rosservice
from actionlib_msgs.msg import GoalStatus
from kinova_msgs.msg import FingerPosition, SetFingersPositionAction, SetFingersPositionGoal
from kinova_msgs.srv import SetTorqueControlMode, SetTorqueControlModeRequest


def normalize_finger_tuple(values):
    vals = list(values)
    if len(vals) == 2:
        vals.append(0.0)
    if len(vals) != 3:
        raise ValueError('Finger position must have length 2 or 3.')
    return tuple(float(v) for v in vals)


class FingerTeleopModeSwitch(object):
    def __init__(self):
        self.robot_prefix = rospy.get_param('~robot_prefix', 'j2s6s300_driver')
        self.open_pos = normalize_finger_tuple(rospy.get_param('~open_position', [0, 0, 0]))
        self.close_pos = normalize_finger_tuple(rospy.get_param('~close_position', [7000, 7000, 0]))
        self.switch_delay = float(rospy.get_param('~switch_delay', 0.35))
        self.action_timeout = float(rospy.get_param('~action_timeout', 5.0))

        self.action_name = rospy.get_param(
            '~finger_action_name',
            '/{}/fingers_action/finger_positions'.format(self.robot_prefix)
        )
        self.torque_service_name = rospy.get_param(
            '~torque_service_name',
            self._discover_torque_mode_service()
        )

        rospy.loginfo('[finger_mode_switch] finger action: %s', self.action_name)
        rospy.loginfo('[finger_mode_switch] torque mode service: %s', self.torque_service_name)
        rospy.loginfo('[finger_mode_switch] open_position=%s', str(self.open_pos))
        rospy.loginfo('[finger_mode_switch] close_position=%s', str(self.close_pos))

        self.client = actionlib.SimpleActionClient(self.action_name, SetFingersPositionAction)
        rospy.loginfo('[finger_mode_switch] waiting for finger action server...')
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            raise RuntimeError('Finger action server not available.')

        rospy.loginfo('[finger_mode_switch] waiting for torque mode service...')
        rospy.wait_for_service(self.torque_service_name, timeout=5.0)
        self.torque_mode_srv = rospy.ServiceProxy(self.torque_service_name, SetTorqueControlMode)

        probe = SetTorqueControlModeRequest()
        self.mode_field = None
        for name in ('state', 'enable', 'flag', 'mode'):
            if hasattr(probe, name):
                self.mode_field = name
                break
        if self.mode_field is None:
            raise RuntimeError('Cannot determine request field for SetTorqueControlModeRequest.')

        rospy.loginfo('[finger_mode_switch] torque mode request field: %s', self.mode_field)

    def _discover_torque_mode_service(self):
        default_name = '/{}/in/set_torque_control_mode'.format(self.robot_prefix)
        try:
            services = rosservice.get_service_list()
            candidates = [s for s in services if self.robot_prefix in s and 'torque_control_mode' in s]
            if candidates:
                return candidates[0]
        except Exception:
            pass
        return default_name

    def _call_torque_mode(self, enable_torque):
        req = SetTorqueControlModeRequest()
        setattr(req, self.mode_field, 1 if enable_torque else 0)
        try:
            resp = self.torque_mode_srv(req)
            rospy.loginfo('[finger_mode_switch] set torque mode -> %s, resp=%s', enable_torque, resp)
            return True
        except Exception as exc:
            rospy.logerr('[finger_mode_switch] torque mode switch failed (enable_torque=%s): %s',
                         enable_torque, exc)
            return False

    def switch_to_trajectory(self):
        rospy.loginfo('[finger_mode_switch] switching to trajectory/kinematics mode...')
        ok = self._call_torque_mode(False)
        if ok:
            rospy.sleep(self.switch_delay)
        return ok

    def switch_to_torque(self):
        rospy.loginfo('[finger_mode_switch] switching back to torque/gravity mode...')
        ok = self._call_torque_mode(True)
        if ok:
            rospy.sleep(self.switch_delay)
        return ok

    def _make_goal(self, pos):
        goal = SetFingersPositionGoal()
        goal.fingers = FingerPosition()
        goal.fingers.finger1 = float(pos[0])
        goal.fingers.finger2 = float(pos[1])
        goal.fingers.finger3 = float(pos[2])
        return goal

    def send_finger_goal_blocking(self, pos, label=''):
        goal = self._make_goal(pos)
        rospy.loginfo('[finger_mode_switch] sending finger goal %s: %s', label, str(pos))
        self.client.cancel_goal()
        self.client.send_goal(goal)

        ok = self.client.wait_for_result(rospy.Duration(self.action_timeout))
        if not ok:
            rospy.logwarn('[finger_mode_switch] finger action timeout, canceling goal.')
            self.client.cancel_goal()
            return False

        state = self.client.get_state()
        result = self.client.get_result()
        rospy.loginfo('[finger_mode_switch] finger result state=%s, result=%s', str(state), str(result))
        return state == GoalStatus.SUCCEEDED

    def _do_finger_action(self, pos, label):
        if not self.switch_to_trajectory():
            return False

        ok_action = self.send_finger_goal_blocking(pos, label=label)
        ok_back = self.switch_to_torque()

        if ok_action and ok_back:
            rospy.loginfo('[finger_mode_switch] %s done.', label)
            return True

        rospy.logwarn('[finger_mode_switch] %s finished with issues: action_ok=%s, back_to_torque_ok=%s',
                      label, ok_action, ok_back)
        return False

    def do_open(self):
        return self._do_finger_action(self.open_pos, 'OPEN')

    def do_close(self):
        return self._do_finger_action(self.close_pos, 'CLOSE')


def get_key_nonblocking():
    if select.select([sys.stdin], [], [], 0.02)[0]:
        return sys.stdin.read(1)
    return None


def main():
    rospy.init_node('finger_teleop_mode_switch', anonymous=True)
    teleop = FingerTeleopModeSwitch()

    print(
        '\nFinger teleop with mode switch\n'
        '  o : open fingers  (trajectory -> finger -> torque)\n'
        '  c : close fingers (trajectory -> finger -> torque)\n'
        '  t : switch to trajectory mode only\n'
        '  g : switch to torque/gravity mode only\n'
        '  q : quit\n'
    )

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    try:
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            k = get_key_nonblocking()
            if k is None:
                rate.sleep()
                continue

            k = k.lower()
            if k == 'o':
                teleop.do_open()
            elif k == 'c':
                teleop.do_close()
            elif k == 't':
                teleop.switch_to_trajectory()
            elif k == 'g':
                teleop.switch_to_torque()
            elif k == 'q':
                break

            rate.sleep()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':
    main()
