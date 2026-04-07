#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cumulative position player for Kinova Cartesian pose actions.

Core idea:
1. Read a delta sequence (or absolute sequence).
2. Capture one startup reference pose pose_0.
3. Pre-accumulate all targets relative to pose_0 before execution.
4. Execute precomputed absolute targets step by step.

Benefits:
- No separate pose file required when using delta sequences.
- Reduces per-step residual error accumulation.
- Compatible with pose-action / position-control pipelines.

Data convention:
- First 3 dims: translation delta in meters.
- Next 3 dims: orientation delta in radians (Euler XYZ / RPY).
- 7th dim: optional gripper state.

Recommendation:
- First test with _use_orientation:=false for position-only verification.
- Enable orientation after confirming rotational semantics.
"""

from __future__ import print_function

import os
import math
import rospy
import actionlib

import geometry_msgs.msg
import std_msgs.msg
from geometry_msgs.msg import PoseStamped

import kinova_msgs.msg

try:
    import tf.transformations as tft
except Exception as exc:
    raise RuntimeError('Cannot import tf.transformations; verify ROS tf is installed correctly: %s' % str(exc))

try:
    from kinova_msgs.msg import SetFingersPositionAction, SetFingersPositionGoal
    HAS_GRIPPER_ACTION = True
except Exception:
    SetFingersPositionAction = None
    SetFingersPositionGoal = None
    HAS_GRIPPER_ACTION = False


class CumulativePositionSequencePlayer(object):
    def __init__(self, robot_type='j2s6s300'):
        rospy.init_node('sequence_player_position_cumulative', anonymous=True)

        self.robot_type = rospy.get_param('~robot_type', robot_type)
        self.sequence = []
        self.targets = []
        self.current_step = 0

        # -------------------------
        # Basic execution parameters
        # -------------------------
        self.pause_between_steps = float(rospy.get_param('~pause_between_steps', 0.0))
        self.step_action_timeout = float(rospy.get_param('~step_action_timeout', 8.0))
        self.wait_pose_timeout = float(rospy.get_param('~wait_pose_timeout', 5.0))
        self.driver_ready_sleep = float(rospy.get_param('~driver_ready_sleep', 1.0))

        # -------------------------
        # Action interpretation parameters
        # -------------------------
        # delta    : first 6 values are interpreted as delta actions (default)
        # absolute : first 6 values are interpreted as absolute pose (reserved extension)
        self.action_mode = rospy.get_param('~action_mode', 'delta')

        # Scale translation and rotation independently; keep 1.0 for physical-unit inputs
        self.translation_scale = float(rospy.get_param('~translation_scale', 1.0))
        self.rotation_scale = float(rospy.get_param('~rotation_scale', 1.0))

        # Whether to include orientation deltas in target construction
        self.use_orientation = bool(rospy.get_param('~use_orientation', True))

        # Pose action reference frame (typically base link)
        self.base_link = rospy.get_param('~base_link', self.robot_type + '_link_base')

        # Whether to precompute all targets before start (default and recommended)
        self.cumulative_from_initial = bool(rospy.get_param('~cumulative_from_initial', True))

        # -------------------------
        # tool_pose subscription cache
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
        # gripper (optional)
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
            rospy.logwarn('SetFingersPositionAction import failed; gripper state will be ignored.')

        rospy.loginfo('CumulativePositionSequencePlayer initialized.')
        rospy.loginfo('robot_type=%s', self.robot_type)
        rospy.loginfo('action_mode=%s', self.action_mode)
        rospy.loginfo('translation_scale=%.6f rotation_scale=%.6f',
                      self.translation_scale, self.rotation_scale)
        rospy.loginfo('use_orientation=%s', self.use_orientation)
        rospy.loginfo('cumulative_from_initial=%s', self.cumulative_from_initial)
        rospy.loginfo('pose_action_address=%s', self.pose_action_address)
        rospy.loginfo('base_link=%s', self.base_link)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _tool_pose_cb(self, msg):
        self.latest_tool_pose = msg

    # ------------------------------------------------------------------
    # Math / utility functions
    # ------------------------------------------------------------------
    @staticmethod
    def _clip(value, low, high):
        return max(low, min(high, value))

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

    @staticmethod
    def _quat_to_matrix(quat_xyzw):
        return tft.quaternion_matrix(quat_xyzw)

    @staticmethod
    def _matrix_to_quat(mat44):
        quat = tft.quaternion_from_matrix(mat44)
        return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]

    @staticmethod
    def _round_list(values, ndigits=6):
        return [round(v, ndigits) for v in values]

    # ------------------------------------------------------------------
    # Current end-effector pose
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
        """
        Read current end-effector pose.

        Returns:
            pos  : [x, y, z]，meters
            quat : [qx, qy, qz, qw]
            rpy  : [roll, pitch, yaw]，radians
        """
        if not self.wait_for_pose(timeout_sec=self.wait_pose_timeout):
            raise RuntimeError('Timeout while waiting for /%s_driver/out/tool_pose.' % self.robot_type)

        msg = self.latest_tool_pose
        pos, quat = self._pose_to_pos_quat(msg)
        rpy = self._quat_to_rpy(quat)
        return pos, quat, rpy

    # ------------------------------------------------------------------
    # Parse sequence file
    # ------------------------------------------------------------------
    def parse_sequence_file(self, filename):
        """
        Supported formats:
            Episode=1, t=24, Action=[dx, dy, dz, dr, dp, dy, g]

        Returns:
            [
              {
                'delta_pose': [6 values],
                'gripper': optional float
              },
              ...
            ]
        """
        sequences = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if 'Action=' not in line:
                        continue
                    action_str = line.split('Action=')[1].strip().strip('[]')
                    values = [float(x.strip()) for x in action_str.split(',')]
                    if len(values) < 6:
                        rospy.logwarn('Skip invalid action line: %s', line.strip())
                        continue

                    entry = {
                        'delta_pose': values[:6],
                        'gripper': values[6] if len(values) >= 7 else None,
                    }
                    sequences.append(entry)
            return sequences
        except Exception as e:
            rospy.logerr('Error parsing file %s: %s', filename, e)
            return []

    # ------------------------------------------------------------------
    # Gripper helpers
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

        # binary
        return self.gripper_closed_turn if float(gripper_state) > self.gripper_threshold else self.gripper_open_turn

    def maybe_execute_gripper(self, gripper_state):
        """
        Send a gripper command only when the state changes.
        """
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

            rospy.loginfo('Send gripper goal | state=%.4f target_turn=%.2f',
                          float(gripper_state), target_turn)
            self.gripper_client.send_goal(goal)

            if self.gripper_blocking:
                finished = self.gripper_client.wait_for_result(rospy.Duration(self.gripper_action_timeout))
                if not finished:
                    rospy.logwarn('Gripper action timeout after %.2f sec', self.gripper_action_timeout)

            self.last_gripper_state = float(gripper_state)
            return True
        except Exception as e:
            rospy.logwarn('Failed to execute gripper command: %s', e)
            return False

    # ------------------------------------------------------------------
    # Pre-accumulated targets
    # ------------------------------------------------------------------
    def build_targets_from_initial_pose(self, initial_pos, initial_quat, initial_rpy):
        """
        Use a single startup reference pose and pre-accumulate the full delta sequence,
        to construct absolute targets for each step.

        For delta mode:
            target_1 = pose_0 + delta_1
            target_2 = pose_0 + delta_1 + delta_2
            ...

        For absolute mode:
            Treat the first 6 sequence values directly as absolute target.

        Returns:
            [
              {
                'target_pos': [x, y, z],
                'target_quat': [qx, qy, qz, qw],
                'target_rpy': [r, p, y],
                'used_delta': [6],
                'gripper': optional float,
              },
              ...
            ]
        """
        targets = []

        # Accumulate both position and rotation from the initial reference pose
        cumulative_pos = [float(initial_pos[0]), float(initial_pos[1]), float(initial_pos[2])]
        cumulative_rot = self._quat_to_matrix(initial_quat)

        for idx, entry in enumerate(self.sequence):
            raw_delta = list(entry['delta_pose'])
            gripper_state = entry['gripper']

            # Scale translation and rotation separately
            delta_pos = [self.translation_scale * float(v) for v in raw_delta[:3]]
            delta_rpy = [self.rotation_scale * float(v) for v in raw_delta[3:6]]
            used_delta = delta_pos + delta_rpy

            if self.action_mode == 'absolute':
                # absolute mode: interpret values as absolute target pose
                target_pos = [delta_pos[0], delta_pos[1], delta_pos[2]]
                if self.use_orientation:
                    target_quat = self._rpy_to_quat(delta_rpy[0], delta_rpy[1], delta_rpy[2])
                    target_rot = self._quat_to_matrix(target_quat)
                else:
                    target_quat = list(initial_quat)
                    target_rot = self._quat_to_matrix(target_quat)
                target_rpy = self._quat_to_rpy(target_quat)

                # Synchronize accumulators to current absolute target for consistent start_step semantics
                cumulative_pos = list(target_pos)
                cumulative_rot = target_rot.copy()
            else:
                # delta mode: accumulate from initial pose
                cumulative_pos[0] += delta_pos[0]
                cumulative_pos[1] += delta_pos[1]
                cumulative_pos[2] += delta_pos[2]
                target_pos = list(cumulative_pos)

                if self.use_orientation:
                    # Use rotation-matrix accumulation; more stable than direct RPY summation
                    delta_rot = tft.euler_matrix(delta_rpy[0], delta_rpy[1], delta_rpy[2])
                    cumulative_rot = tft.concatenate_matrices(cumulative_rot, delta_rot)
                target_quat = self._matrix_to_quat(cumulative_rot)
                target_rpy = self._quat_to_rpy(target_quat)

            targets.append({
                'target_pos': target_pos,
                'target_quat': target_quat,
                'target_rpy': target_rpy,
                'used_delta': used_delta,
                'gripper': gripper_state,
            })

            rospy.logdebug('Prebuilt target step=%d pos=%s rpy=%s',
                           idx + 1,
                           self._round_list(target_pos, 6),
                           self._round_list(target_rpy, 6))

        return targets

    # ------------------------------------------------------------------
    # Pose action execution
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
            x=float(target_pos[0]),
            y=float(target_pos[1]),
            z=float(target_pos[2])
        )
        goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
            x=float(target_quat[0]),
            y=float(target_quat[1]),
            z=float(target_quat[2]),
            w=float(target_quat[3])
        )

        rospy.loginfo('Send pose goal | pos=%s quat=%s',
                      self._round_list(target_pos, 6),
                      self._round_list(target_quat, 6))

        self.pose_client.send_goal(goal)

        finished = self.pose_client.wait_for_result(rospy.Duration(timeout_sec))
        if not finished:
            self.pose_client.cancel_all_goals()
            rospy.logwarn('Pose action timeout after %.2f sec', timeout_sec)
            return False

        state = self.pose_client.get_state()
        rospy.loginfo('Pose action finished | state=%s', str(state))
        return True

    # ------------------------------------------------------------------
    # Single-step execution
    # ------------------------------------------------------------------
    def execute_step(self, step_index):
        if step_index >= len(self.targets):
            rospy.loginfo('Sequence execution completed')
            return False

        target_entry = self.targets[step_index]
        gripper_state = target_entry['gripper']

        rospy.loginfo('Step %d/%d', step_index + 1, len(self.targets))
        rospy.loginfo('  used_delta=%s', self._round_list(target_entry['used_delta'], 6))
        rospy.loginfo('  target_pos=%s', self._round_list(target_entry['target_pos'], 6))
        rospy.loginfo('  target_rpy=%s', self._round_list(target_entry['target_rpy'], 6))
        rospy.loginfo('  gripper=%s', str(gripper_state))

        # Execute gripper first (if enabled and changed)
        self.maybe_execute_gripper(gripper_state)

        success = self.send_pose_goal(
            target_entry['target_pos'],
            target_entry['target_quat'],
            timeout_sec=self.step_action_timeout
        )

        if success:
            rospy.loginfo('Step %d success', step_index + 1)
        else:
            rospy.logwarn('Step %d failed', step_index + 1)
        return success

    # ------------------------------------------------------------------
    # Full sequence execution
    # ------------------------------------------------------------------
    def run_sequence(self, filename, start_step=0):
        self.sequence = self.parse_sequence_file(filename)
        if not self.sequence:
            rospy.logerr('Unable to load sequence file or no valid sequence data found: %s', filename)
            return False

        rospy.loginfo('Loaded %d steps from %s', len(self.sequence), os.path.abspath(filename))
        rospy.loginfo('Waiting for current tool pose...')
        if not self.wait_for_pose(timeout_sec=5.0):
            rospy.logerr('Timeout waiting for /%s_driver/out/tool_pose', self.robot_type)
            return False

        if not self.wait_for_pose_server():
            return False

        if self.enable_gripper and self.gripper_client is not None:
            self.wait_for_gripper_server()

        rospy.sleep(self.driver_ready_sleep)

        # Capture one reference pose before playback starts
        initial_pos, initial_quat, initial_rpy = self.get_current_tool_pose()
        rospy.loginfo('Initial pose captured before sequence execution:')
        rospy.loginfo('  initial_pos=%s', self._round_list(initial_pos, 6))
        rospy.loginfo('  initial_rpy=%s', self._round_list(initial_rpy, 6))

        # Pre-accumulate full target pose list
        if self.cumulative_from_initial:
            self.targets = self.build_targets_from_initial_pose(initial_pos, initial_quat, initial_rpy)
        else:
            # Compatibility note: when cumulative_from_initial is disabled, the old per-step dynamic path is intentionally not implemented.
            # Raise directly to avoid ambiguous semantics.
            raise RuntimeError('Current version implements only cumulative_from_initial=True.')

        self.current_step = start_step
        total_steps = len(self.targets)
        rospy.loginfo('Start cumulative position-control sequence execution | total_steps=%d', total_steps)

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
    player = CumulativePositionSequencePlayer(robot_type='j2s6s300')
    sequence_file = rospy.get_param('~sequence_file', 'sequence.txt')
    start_step = int(rospy.get_param('~start_step', 0))

    try:
        player.run_sequence(sequence_file, start_step=start_step)
    except KeyboardInterrupt:
        rospy.loginfo('Sequence execution interrupted by user')
    except Exception as e:
        rospy.logerr('Sequence execution error: %s', e)


if __name__ == '__main__':
    main()
