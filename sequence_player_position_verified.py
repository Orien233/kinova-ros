#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sequence player (position-control / pose-action version) for Kinova J2S6S300.

这一版在原来的 position player 基础上，做了两件关键增强：
1. 支持读取 absolute pose 序列文件：
       Episode=1, t=24, Pose=[x, y, z, roll, pitch, yaw, gripper_state]
   这样每一步都直接回到一个绝对位姿，减少 delta 累积误差。
2. 支持动作完成后的到位复核与重发：
   在 pose action 返回 success 后，再读取 /out/tool_pose 检查终点误差；
   若误差仍超阈值，可自动重发 1~N 次，提升终点精度与重复性。

同时保留对旧格式 delta action 的支持：
       Episode=1, t=24, Action=[dx, dy, dz, droll, dpitch, dyaw, gripper_state]

推荐用法：
- position 回放优先使用 recorder 生成的 sequence_pose.txt
- 若文件中是 Pose=，将直接按绝对位姿播放
- 若文件中是 Action=，则按 action_mode 参数解释（默认 delta）
"""

from __future__ import print_function

import math
import os

import rospy
import actionlib
import geometry_msgs.msg
import std_msgs.msg
from geometry_msgs.msg import PoseStamped

import kinova_msgs.msg

try:
    import tf.transformations as tft
except Exception as exc:
    raise RuntimeError('无法导入 tf.transformations，请确认 ROS tf 已正确安装: %s' % str(exc))

try:
    from kinova_msgs.msg import SetFingersPositionAction, SetFingersPositionGoal
    HAS_GRIPPER_ACTION = True
except Exception:
    SetFingersPositionAction = None
    SetFingersPositionGoal = None
    HAS_GRIPPER_ACTION = False


class PositionSequencePlayer(object):
    def __init__(self, robot_type='j2s6s300'):
        rospy.init_node('sequence_player_position_verified', anonymous=True)

        self.robot_type = rospy.get_param('~robot_type', robot_type)
        self.sequence = []
        self.current_step = 0

        # -------------------------
        # 执行参数
        # -------------------------
        self.pause_between_steps = float(rospy.get_param('~pause_between_steps', 0.0))
        self.step_action_timeout = float(rospy.get_param('~step_action_timeout', 8.0))
        self.wait_pose_timeout = float(rospy.get_param('~wait_pose_timeout', 3.0))
        self.driver_ready_sleep = float(rospy.get_param('~driver_ready_sleep', 1.0))

        # -------------------------
        # 文件与动作解释参数
        # -------------------------
        # auto | absolute_pose | delta_action
        self.sequence_kind = rospy.get_param('~sequence_kind', 'auto')
        # delta | absolute
        self.action_mode = rospy.get_param('~action_mode', 'delta')
        self.translation_scale = float(rospy.get_param('~translation_scale', 1.0))
        self.rotation_scale = float(rospy.get_param('~rotation_scale', 1.0))
        self.use_orientation = bool(rospy.get_param('~use_orientation', True))

        # -------------------------
        # 到位复核参数
        # -------------------------
        self.verify_goal = bool(rospy.get_param('~verify_goal', True))
        self.verify_retries = int(rospy.get_param('~verify_retries', 1))
        self.verify_settle_sec = float(rospy.get_param('~verify_settle_sec', 0.15))
        self.goal_pos_tolerance = float(rospy.get_param('~goal_pos_tolerance', 5e-4))
        self.goal_ori_tolerance = float(rospy.get_param('~goal_ori_tolerance', 0.02))

        self.base_link = rospy.get_param('~base_link', self.robot_type + '_link_base')

        # -------------------------
        # tool_pose 订阅缓存
        # -------------------------
        self.latest_tool_pose = None
        self.tool_pose_sub = rospy.Subscriber(
            '/%s_driver/out/tool_pose' % self.robot_type,
            PoseStamped,
            self._tool_pose_cb,
            queue_size=1
        )

        # -------------------------
        # pose action client
        # -------------------------
        self.pose_action_address = '/%s_driver/pose_action/tool_pose' % self.robot_type
        self.pose_client = actionlib.SimpleActionClient(
            self.pose_action_address,
            kinova_msgs.msg.ArmPoseAction
        )

        # -------------------------
        # gripper（可选）
        # -------------------------
        self.enable_gripper = bool(rospy.get_param('~enable_gripper', True))
        self.gripper_mode = rospy.get_param('~gripper_mode', 'binary')
        self.gripper_threshold = float(rospy.get_param('~gripper_threshold', 0.5))
        self.gripper_open_turn = float(rospy.get_param('~gripper_open_turn', 0.0))
        self.gripper_closed_turn = float(rospy.get_param('~gripper_closed_turn', 6800.0))
        self.gripper_action_timeout = float(rospy.get_param('~gripper_action_timeout', 2.0))
        self.gripper_blocking = bool(rospy.get_param('~gripper_blocking', False))
        self.gripper_change_tolerance = float(rospy.get_param('~gripper_change_tolerance', 1e-6))
        self.finger_action_address = '/%s_driver/fingers_action/finger_positions' % self.robot_type
        self.last_gripper_state = None
        self.gripper_client = None
        if self.enable_gripper and HAS_GRIPPER_ACTION:
            self.gripper_client = actionlib.SimpleActionClient(
                self.finger_action_address,
                SetFingersPositionAction
            )
        elif self.enable_gripper:
            rospy.logwarn('未能导入 SetFingersPositionAction，夹爪状态将被忽略。')

        rospy.loginfo('PositionSequencePlayer initialized.')
        rospy.loginfo('robot_type=%s', self.robot_type)
        rospy.loginfo('sequence_kind=%s action_mode=%s', self.sequence_kind, self.action_mode)
        rospy.loginfo('translation_scale=%.6f rotation_scale=%.6f',
                      self.translation_scale, self.rotation_scale)
        rospy.loginfo('use_orientation=%s', self.use_orientation)
        rospy.loginfo('verify_goal=%s retries=%d pos_tol=%.6f ori_tol=%.6f',
                      self.verify_goal, self.verify_retries,
                      self.goal_pos_tolerance, self.goal_ori_tolerance)
        rospy.loginfo('pose_action_address=%s', self.pose_action_address)
        rospy.loginfo('base_link=%s', self.base_link)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _tool_pose_cb(self, msg):
        self.latest_tool_pose = msg

    # ------------------------------------------------------------------
    # 数学 / 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _clip(value, low, high):
        return max(low, min(high, value))

    @staticmethod
    def _normalize_angle_diff(curr, prev):
        d = curr - prev
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    @staticmethod
    def _norm3(vec3):
        return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

    @staticmethod
    def _pose_to_pos_quat(pose_msg):
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        pos = [float(p.x), float(p.y), float(p.z)]
        quat = [float(q.x), float(q.y), float(q.z), float(q.w)]
        return pos, quat

    @staticmethod
    def _quat_to_rpy(quat_xyzw):
        roll, pitch, yaw = tft.euler_from_quaternion(quat_xyzw)
        return [float(roll), float(pitch), float(yaw)]

    @staticmethod
    def _rpy_to_quat(roll, pitch, yaw):
        quat = tft.quaternion_from_euler(roll, pitch, yaw)
        return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]

    # ------------------------------------------------------------------
    # 读取当前末端位姿
    # ------------------------------------------------------------------
    def wait_for_pose(self, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = self.wait_pose_timeout
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.latest_tool_pose is not None:
                return True
            if rospy.Time.now().to_sec() - t0 > timeout_sec:
                return False
            rate.sleep()
        return False

    def get_current_tool_pose(self):
        if not self.wait_for_pose(timeout_sec=self.wait_pose_timeout):
            raise RuntimeError('等待 /%s_driver/out/tool_pose 超时。' % self.robot_type)
        msg = self.latest_tool_pose
        pos, quat = self._pose_to_pos_quat(msg)
        rpy = self._quat_to_rpy(quat)
        return pos, quat, rpy

    # ------------------------------------------------------------------
    # 解析 sequence 文件
    # ------------------------------------------------------------------
    def parse_sequence_file(self, filename):
        sequences = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    kind = None
                    payload = None
                    if 'Pose=' in line:
                        kind = 'absolute_pose'
                        payload = line.split('Pose=')[1].strip().strip('[]')
                    elif 'Action=' in line:
                        kind = 'delta_action' if self.action_mode == 'delta' else 'absolute_pose'
                        payload = line.split('Action=')[1].strip().strip('[]')
                    else:
                        continue

                    if self.sequence_kind == 'absolute_pose':
                        kind = 'absolute_pose'
                    elif self.sequence_kind == 'delta_action':
                        kind = 'delta_action'

                    values = [float(x.strip()) for x in payload.split(',')]
                    if len(values) < 6:
                        rospy.logwarn('Skip invalid line: %s', line)
                        continue

                    entry = {
                        'kind': kind,
                        'pose_or_delta': values[:6],
                        'gripper': values[6] if len(values) >= 7 else None,
                    }
                    sequences.append(entry)
            return sequences
        except Exception as exc:
            rospy.logerr('Error parsing file %s: %s', filename, exc)
            return []

    # ------------------------------------------------------------------
    # 夹爪辅助
    # ------------------------------------------------------------------
    def wait_for_gripper_server(self):
        if self.gripper_client is None:
            return False
        rospy.loginfo('Waiting for gripper action server: %s', self.finger_action_address)
        ok = self.gripper_client.wait_for_server(rospy.Duration(5.0))
        if not ok:
            rospy.logwarn('Gripper action server not available: %s', self.finger_action_address)
        return ok

    def gripper_state_to_turn(self, gripper_state):
        if gripper_state is None:
            return None
        if self.gripper_mode == 'continuous':
            alpha = self._clip(float(gripper_state), 0.0, 1.0)
            return self.gripper_open_turn + alpha * (self.gripper_closed_turn - self.gripper_open_turn)
        return self.gripper_closed_turn if float(gripper_state) > self.gripper_threshold else self.gripper_open_turn

    def maybe_execute_gripper(self, gripper_state):
        if not self.enable_gripper or gripper_state is None or self.gripper_client is None:
            return True
        if self.last_gripper_state is not None:
            if abs(float(gripper_state) - float(self.last_gripper_state)) <= self.gripper_change_tolerance:
                return True
        target_turn = self.gripper_state_to_turn(gripper_state)
        if target_turn is None:
            return True
        try:
            goal = SetFingersPositionGoal()
            goal.fingers.finger1 = float(target_turn)
            goal.fingers.finger2 = float(target_turn)
            goal.fingers.finger3 = 0.0
            rospy.loginfo('Send gripper goal | state=%.4f target_turn=%.2f', float(gripper_state), target_turn)
            self.gripper_client.send_goal(goal)
            if self.gripper_blocking:
                finished = self.gripper_client.wait_for_result(rospy.Duration(self.gripper_action_timeout))
                if not finished:
                    rospy.logwarn('Gripper action timeout after %.2f sec', self.gripper_action_timeout)
            self.last_gripper_state = float(gripper_state)
            return True
        except Exception as exc:
            rospy.logwarn('Failed to execute gripper command: %s', exc)
            return False

    # ------------------------------------------------------------------
    # 构造目标位姿
    # ------------------------------------------------------------------
    def build_step_target(self, start_pos, start_rpy, entry):
        raw = list(entry['pose_or_delta'])
        pos_part = [self.translation_scale * float(v) for v in raw[:3]]
        rpy_part = [self.rotation_scale * float(v) for v in raw[3:6]]

        if entry['kind'] == 'absolute_pose':
            target_pos = list(pos_part)
            if self.use_orientation:
                target_rpy = list(rpy_part)
            else:
                target_rpy = list(start_rpy)
        else:
            target_pos = [
                start_pos[0] + pos_part[0],
                start_pos[1] + pos_part[1],
                start_pos[2] + pos_part[2],
            ]
            if self.use_orientation:
                target_rpy = [
                    start_rpy[0] + rpy_part[0],
                    start_rpy[1] + rpy_part[1],
                    start_rpy[2] + rpy_part[2],
                ]
            else:
                target_rpy = list(start_rpy)

        target_quat = self._rpy_to_quat(target_rpy[0], target_rpy[1], target_rpy[2])
        return target_pos, target_quat, target_rpy, pos_part + rpy_part

    # ------------------------------------------------------------------
    # Pose action 执行与复核
    # ------------------------------------------------------------------
    def wait_for_pose_server(self):
        rospy.loginfo('Waiting for pose action server: %s', self.pose_action_address)
        ok = self.pose_client.wait_for_server(rospy.Duration(5.0))
        if not ok:
            rospy.logerr('Pose action server not available: %s', self.pose_action_address)
        return ok

    def send_pose_goal(self, target_pos, target_quat, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = self.step_action_timeout
        goal = kinova_msgs.msg.ArmPoseGoal()
        goal.pose.header = std_msgs.msg.Header(frame_id=self.base_link)
        goal.pose.pose.position = geometry_msgs.msg.Point(
            x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2])
        )
        goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
            x=float(target_quat[0]), y=float(target_quat[1]), z=float(target_quat[2]), w=float(target_quat[3])
        )
        rospy.loginfo('Send pose goal | pos=%s quat=%s',
                      [round(v, 6) for v in target_pos], [round(v, 6) for v in target_quat])
        self.pose_client.send_goal(goal)
        finished = self.pose_client.wait_for_result(rospy.Duration(timeout_sec))
        if not finished:
            self.pose_client.cancel_all_goals()
            rospy.logwarn('Pose action timeout after %.2f sec', timeout_sec)
            return False
        state = self.pose_client.get_state()
        rospy.loginfo('Pose action finished | state=%s', str(state))
        return True

    def verify_pose_goal(self, target_pos, target_rpy):
        if not self.verify_goal:
            return True, None, None, None
        if self.verify_settle_sec > 0.0:
            rospy.sleep(self.verify_settle_sec)
        current_pos, current_quat, current_rpy = self.get_current_tool_pose()
        pos_err_vec = [
            target_pos[0] - current_pos[0],
            target_pos[1] - current_pos[1],
            target_pos[2] - current_pos[2],
        ]
        pos_err = self._norm3(pos_err_vec)
        ori_err = 0.0
        if self.use_orientation:
            ori_err_vec = [
                self._normalize_angle_diff(target_rpy[0], current_rpy[0]),
                self._normalize_angle_diff(target_rpy[1], current_rpy[1]),
                self._normalize_angle_diff(target_rpy[2], current_rpy[2]),
            ]
            ori_err = self._norm3(ori_err_vec)
        ok = (pos_err <= self.goal_pos_tolerance) and ((not self.use_orientation) or (ori_err <= self.goal_ori_tolerance))
        return ok, current_pos, pos_err, ori_err

    def send_pose_goal_with_verification(self, target_pos, target_quat, target_rpy, timeout_sec=None):
        attempts = max(self.verify_retries, 0) + 1
        for attempt in range(attempts):
            success = self.send_pose_goal(target_pos, target_quat, timeout_sec=timeout_sec)
            if not success:
                continue
            ok, current_pos, pos_err, ori_err = self.verify_pose_goal(target_pos, target_rpy)
            if ok:
                rospy.loginfo('Goal verified | pos_err=%.6f ori_err=%.6f', pos_err or 0.0, ori_err or 0.0)
                return True
            rospy.logwarn('Goal verification failed (attempt %d/%d) | pos_err=%.6f ori_err=%.6f current_pos=%s',
                          attempt + 1, attempts, pos_err or 0.0, ori_err or 0.0,
                          [round(v, 6) for v in current_pos] if current_pos is not None else None)
        return False

    # ------------------------------------------------------------------
    # 单步执行
    # ------------------------------------------------------------------
    def execute_step(self, step_index):
        if step_index >= len(self.sequence):
            rospy.loginfo('Sequence execution completed')
            return False

        entry = self.sequence[step_index]
        payload = entry['pose_or_delta']
        gripper_state = entry['gripper']

        rospy.loginfo('Step %d/%d | kind=%s | payload=%s | gripper=%s',
                      step_index + 1, len(self.sequence), entry['kind'],
                      [round(v, 6) for v in payload], str(gripper_state))

        self.maybe_execute_gripper(gripper_state)

        start_pos, start_quat, start_rpy = self.get_current_tool_pose()
        target_pos, target_quat, target_rpy, used_payload = self.build_step_target(start_pos, start_rpy, entry)

        rospy.loginfo('  start_pos=%s start_rpy=%s',
                      [round(v, 6) for v in start_pos], [round(v, 6) for v in start_rpy])
        rospy.loginfo('  used_payload=%s', [round(v, 6) for v in used_payload])
        rospy.loginfo('  target_pos=%s', [round(v, 6) for v in target_pos])
        if self.use_orientation:
            rospy.loginfo('  target_rpy=%s', [round(v, 6) for v in target_rpy])

        success = self.send_pose_goal_with_verification(target_pos, target_quat, target_rpy,
                                                        timeout_sec=self.step_action_timeout)
        if success:
            rospy.loginfo('Step %d success', step_index + 1)
        else:
            rospy.logwarn('Step %d failed', step_index + 1)
        return success

    # ------------------------------------------------------------------
    # 整段执行
    # ------------------------------------------------------------------
    def run_sequence(self, filename, start_step=0):
        self.sequence = self.parse_sequence_file(filename)
        if not self.sequence:
            rospy.logerr('Unable to load sequence file or no valid sequence data found: %s', filename)
            return False

        rospy.loginfo('Loaded %d steps from %s', len(self.sequence), os.path.abspath(filename))
        if not self.wait_for_pose(timeout_sec=5.0):
            rospy.logerr('Timeout waiting for /%s_driver/out/tool_pose', self.robot_type)
            return False
        if not self.wait_for_pose_server():
            return False
        if self.enable_gripper and self.gripper_client is not None:
            self.wait_for_gripper_server()
        rospy.sleep(self.driver_ready_sleep)

        self.current_step = start_step
        total_steps = len(self.sequence)
        rospy.loginfo('Start position-control sequence execution | total_steps=%d', total_steps)
        while self.current_step < total_steps and not rospy.is_shutdown():
            success = self.execute_step(self.current_step)
            self.current_step += 1
            if not success:
                rospy.logwarn('Skipping failed step %d', self.current_step)
            if self.pause_between_steps > 0.0:
                rospy.sleep(self.pause_between_steps)

        rospy.loginfo('Sequence execution completed')
        return True


def main():
    player = PositionSequencePlayer(robot_type='j2s6s300')
    sequence_file = rospy.get_param('~sequence_file', 'sequence_pose.txt')
    start_step = int(rospy.get_param('~start_step', 0))

    try:
        player.run_sequence(sequence_file, start_step=start_step)
    except KeyboardInterrupt:
        rospy.loginfo('Sequence execution interrupted by user')
    except Exception as exc:
        rospy.logerr('Sequence execution error: %s', exc)


if __name__ == '__main__':
    main()
