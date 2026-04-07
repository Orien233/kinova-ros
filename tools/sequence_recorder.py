#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sequence recorder for Kinova Gen2 J2S6S300.

This version supports two playback pipelines:
1. velocity/Jacobian playback: outputs raw delta and normalized delta.
2. position/pose-action playback: additionally outputs absolute pose sequence to reduce delta accumulation error.

Current alignment updates:
- All default output files are written under the script directory instead of CWD.
- teach/auto still respect explicitly provided output paths.
- By default it generates:
  sequence.txt / sequence_raw.txt / sequence_pose.txt / sequence_debug.csv / sequence_stats.json
"""

from __future__ import print_function

import csv
import json
import math
import os

import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

try:
    from kinova_msgs.msg import FingerPosition
    HAS_FINGER_POSITION_MSG = True
except Exception:
    FingerPosition = None
    HAS_FINGER_POSITION_MSG = False

try:
    from std_srvs.srv import Empty
    HAS_EMPTY_SRV = True
except Exception:
    Empty = None
    HAS_EMPTY_SRV = False


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_output_path(filename):
    return os.path.join(THIS_DIR, filename)


def _resolve_output_path(path_value):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(THIS_DIR, path_value)


class SequenceRecorder(object):
    ACTION_NAMES = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
    POSE_NAMES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

    def __init__(self):
        rospy.init_node('sequence_recorder_pose', anonymous=True)

        self.robot_type = rospy.get_param('~robot_type', 'j2s6s300')
        self.episode_id = int(rospy.get_param('~episode_id', 1))

        self.output_path = _resolve_output_path(
            rospy.get_param('~output_path', _default_output_path('sequence.txt'))
        )
        self.raw_action_path = _resolve_output_path(
            rospy.get_param('~raw_action_path', _default_output_path('sequence_raw.txt'))
        )
        self.absolute_pose_path = _resolve_output_path(
            rospy.get_param('~absolute_pose_path', _default_output_path('sequence_pose.txt'))
        )
        self.raw_csv_path = _resolve_output_path(
            rospy.get_param('~raw_csv_path', _default_output_path('sequence_debug.csv'))
        )
        self.stats_json_path = _resolve_output_path(
            rospy.get_param('~stats_json_path', _default_output_path('sequence_stats.json'))
        )

        self.write_raw_action_file = bool(rospy.get_param('~write_raw_action_file', True))
        self.write_absolute_pose_file = bool(rospy.get_param('~write_absolute_pose_file', True))
        self.write_raw_csv = bool(rospy.get_param('~write_raw_csv', True))
        self.write_initial_zero = bool(rospy.get_param('~write_initial_zero', False))

        self.player_step_duration = float(rospy.get_param('~player_step_duration', 0.1))
        default_sample_rate = 1.0 / self.player_step_duration if self.player_step_duration > 1e-9 else 10.0
        self.sample_rate_hz = float(rospy.get_param('~sample_rate_hz', default_sample_rate))
        self.start_delay_sec = float(rospy.get_param('~start_delay_sec', 2.0))
        self.record_duration_sec = float(rospy.get_param('~record_duration_sec', -1.0))
        self.wait_for_topics_timeout = float(rospy.get_param('~wait_for_topics_timeout', 10.0))

        self.pose_topic = rospy.get_param('~pose_topic', '/%s_driver/out/tool_pose' % self.robot_type)
        self.finger_topic = rospy.get_param('~finger_topic', '/%s_driver/out/finger_position' % self.robot_type)
        self.joint_state_topic = rospy.get_param('~joint_state_topic', '/%s_driver/out/joint_state' % self.robot_type)

        self.gripper_mode = rospy.get_param('~gripper_mode', 'binary')
        self.finger_open_turn = float(rospy.get_param('~finger_open_turn', 0.0))
        self.finger_closed_turn = float(rospy.get_param('~finger_closed_turn', 6800.0))
        self.gripper_close_threshold_turn = float(rospy.get_param('~gripper_close_threshold_turn', 3400.0))
        self.joint_finger_open_value = float(rospy.get_param('~joint_finger_open_value', 0.0))
        self.joint_finger_closed_value = float(rospy.get_param('~joint_finger_closed_value', 1.2))
        self.joint_gripper_close_threshold = float(rospy.get_param('~joint_gripper_close_threshold', 0.35))

        player_max_linear_vel = float(rospy.get_param('~player_max_linear_vel', 0.05))
        player_max_angular_vel = float(rospy.get_param('~player_max_angular_vel', 0.5))
        default_max_linear_delta = player_max_linear_vel * self.player_step_duration
        default_max_angular_delta = player_max_angular_vel * self.player_step_duration

        self.max_dx = float(rospy.get_param('~max_dx', default_max_linear_delta))
        self.max_dy = float(rospy.get_param('~max_dy', default_max_linear_delta))
        self.max_dz = float(rospy.get_param('~max_dz', default_max_linear_delta))
        self.max_drx = float(rospy.get_param('~max_drx', default_max_angular_delta))
        self.max_dry = float(rospy.get_param('~max_dry', default_max_angular_delta))
        self.max_drz = float(rospy.get_param('~max_drz', default_max_angular_delta))

        self.deadband_linear = float(rospy.get_param('~deadband_linear', 1e-4))
        self.deadband_angular = float(rospy.get_param('~deadband_angular', 1e-3))

        self.auto_force_control = bool(rospy.get_param('~auto_force_control', False))
        self.start_force_control_srv = '/%s_driver/in/start_force_control' % self.robot_type
        self.stop_force_control_srv = '/%s_driver/in/stop_force_control' % self.robot_type

        self.latest_pose_msg = None
        self.latest_finger_msg = None
        self.latest_joint_state_msg = None

        self.prev_pose = None
        self.step_idx = 0
        self.record_started_time = None

        self.norm_sequence_fp = None
        self.raw_action_fp = None
        self.absolute_pose_fp = None
        self.raw_csv_fp = None
        self.raw_csv_writer = None

        self.raw_action_history = []
        self.norm_action_history = []
        self.absolute_pose_history = []
        self.gripper_history = []

        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(self.joint_state_topic, JointState, self._joint_state_cb, queue_size=1)
        if HAS_FINGER_POSITION_MSG:
            rospy.Subscriber(self.finger_topic, FingerPosition, self._finger_cb, queue_size=1)
        else:
            rospy.logwarn('kinova_msgs/FingerPosition is unavailable; gripper state will be estimated from joint_state only.')

        rospy.loginfo('SequenceRecorder initialized.')
        rospy.loginfo('robot_type           = %s', self.robot_type)
        rospy.loginfo('pose_topic           = %s', self.pose_topic)
        rospy.loginfo('finger_topic         = %s', self.finger_topic)
        rospy.loginfo('joint_state_topic    = %s', self.joint_state_topic)
        rospy.loginfo('normalized output    = %s', self.output_path)
        rospy.loginfo('raw action output    = %s', self.raw_action_path)
        rospy.loginfo('absolute pose output = %s', self.absolute_pose_path)
        rospy.loginfo('stats json output    = %s', self.stats_json_path)
        rospy.loginfo('sample_rate_hz       = %.6f', self.sample_rate_hz)
        rospy.loginfo('player_step_duration = %.6f', self.player_step_duration)

    def _pose_cb(self, msg):
        self.latest_pose_msg = msg

    def _finger_cb(self, msg):
        self.latest_finger_msg = msg

    def _joint_state_cb(self, msg):
        self.latest_joint_state_msg = msg

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))

    @staticmethod
    def _safe_div(numerator, denominator):
        if abs(denominator) < 1e-12:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _normalize_angle_diff(curr, prev):
        d = curr - prev
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    def _apply_deadband(self, value, threshold):
        return 0.0 if abs(value) < threshold else value

    def _pose_msg_to_xyzrpy(self, pose_msg):
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = tft.euler_from_quaternion(quat)
        return [float(p.x), float(p.y), float(p.z), float(roll), float(pitch), float(yaw)]

    def _normalize_delta(self, delta, limit):
        norm = self._safe_div(delta, limit)
        return self._clamp(norm, -1.0, 1.0)

    @staticmethod
    def _quantile(sorted_values, q):
        if not sorted_values:
            return 0.0
        if q <= 0.0:
            return sorted_values[0]
        if q >= 1.0:
            return sorted_values[-1]
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]
        pos = (n - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_values[lo]
        weight = pos - lo
        return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight

    def _series_stats(self, values):
        if not values:
            return {
                'count': 0,
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'q01': 0.0,
                'q99': 0.0,
                'abs_max': 0.0,
            }
        count = len(values)
        vmin = min(values)
        vmax = max(values)
        mean = sum(values) / float(count)
        var = sum((v - mean) ** 2 for v in values) / float(count)
        std = math.sqrt(var)
        sorted_values = sorted(values)
        q01 = self._quantile(sorted_values, 0.01)
        q99 = self._quantile(sorted_values, 0.99)
        abs_max = max(abs(vmin), abs(vmax))
        return {
            'count': count,
            'min': vmin,
            'max': vmax,
            'mean': mean,
            'std': std,
            'q01': q01,
            'q99': q99,
            'abs_max': abs_max,
        }

    def _compute_vector_stats(self, history, names):
        per_dim = {}
        for dim_idx, name in enumerate(names):
            series = [row[dim_idx] for row in history]
            per_dim[name] = self._series_stats(series)
        return per_dim

    def _finger_turn_average(self):
        if self.latest_finger_msg is None:
            return None
        vals = []
        for attr in ('finger1', 'finger2', 'finger3'):
            if hasattr(self.latest_finger_msg, attr):
                vals.append(float(getattr(self.latest_finger_msg, attr)))
        if not vals:
            return None
        return sum(vals) / float(len(vals))

    def _joint_finger_average(self):
        if self.latest_joint_state_msg is None:
            return None
        msg = self.latest_joint_state_msg
        if not msg.name or not msg.position:
            return None
        finger_vals = []
        for idx, name in enumerate(msg.name):
            if 'finger' in name.lower() and idx < len(msg.position):
                finger_vals.append(float(msg.position[idx]))
        if not finger_vals and len(msg.position) >= 9:
            finger_vals = [float(v) for v in msg.position[-3:]]
        if not finger_vals:
            return None
        return sum(finger_vals) / float(len(finger_vals))

    def _gripper_from_turn_value(self, avg_turn):
        if self.gripper_mode == 'continuous':
            value = self._safe_div(avg_turn - self.finger_open_turn,
                                   self.finger_closed_turn - self.finger_open_turn)
            return self._clamp(value, 0.0, 1.0)
        return 1.0 if avg_turn >= self.gripper_close_threshold_turn else 0.0

    def _gripper_from_joint_value(self, avg_joint_val):
        if self.gripper_mode == 'continuous':
            value = self._safe_div(avg_joint_val - self.joint_finger_open_value,
                                   self.joint_finger_closed_value - self.joint_finger_open_value)
            return self._clamp(value, 0.0, 1.0)
        return 1.0 if avg_joint_val >= self.joint_gripper_close_threshold else 0.0

    def _current_gripper_state(self):
        avg_turn = self._finger_turn_average()
        if avg_turn is not None:
            return self._gripper_from_turn_value(avg_turn), avg_turn, 'finger_position'
        avg_joint = self._joint_finger_average()
        if avg_joint is not None:
            return self._gripper_from_joint_value(avg_joint), avg_joint, 'joint_state'
        return 0.0, None, 'default'

    def _wait_until_ready(self):
        rospy.loginfo('Waiting for pose and gripper feedback topics...')
        start_t = rospy.Time.now().to_sec()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            pose_ok = (self.latest_pose_msg is not None)
            grip_ok = (self.latest_finger_msg is not None) or (self.latest_joint_state_msg is not None)
            if pose_ok and grip_ok:
                rospy.loginfo('Recorder topics are ready.')
                return True
            if rospy.Time.now().to_sec() - start_t > self.wait_for_topics_timeout:
                rospy.logerr('Timeout waiting for topics. pose_ok=%s, grip_ok=%s', pose_ok, grip_ok)
                return False
            rate.sleep()
        return False

    @staticmethod
    def _ensure_output_dir(path):
        folder = os.path.dirname(os.path.abspath(path))
        if folder and (not os.path.exists(folder)):
            os.makedirs(folder)

    def _open_outputs(self):
        self._ensure_output_dir(self.output_path)
        self.norm_sequence_fp = open(self.output_path, 'w')

        if self.write_raw_action_file:
            self._ensure_output_dir(self.raw_action_path)
            self.raw_action_fp = open(self.raw_action_path, 'w')

        if self.write_absolute_pose_file:
            self._ensure_output_dir(self.absolute_pose_path)
            self.absolute_pose_fp = open(self.absolute_pose_path, 'w')

        if self.write_raw_csv:
            self._ensure_output_dir(self.raw_csv_path)
            self.raw_csv_fp = open(self.raw_csv_path, 'w')
            self.raw_csv_writer = csv.writer(self.raw_csv_fp)
            self.raw_csv_writer.writerow([
                'episode', 't', 'stamp',
                'x', 'y', 'z', 'roll', 'pitch', 'yaw',
                'dx_phys', 'dy_phys', 'dz_phys', 'drx_phys', 'dry_phys', 'drz_phys',
                'dx_norm', 'dy_norm', 'dz_norm', 'drx_norm', 'dry_norm', 'drz_norm',
                'gripper_state', 'gripper_raw', 'gripper_source'
            ])

    def _close_outputs(self):
        if self.norm_sequence_fp is not None:
            self.norm_sequence_fp.flush()
            self.norm_sequence_fp.close()
            self.norm_sequence_fp = None
        if self.raw_action_fp is not None:
            self.raw_action_fp.flush()
            self.raw_action_fp.close()
            self.raw_action_fp = None
        if self.absolute_pose_fp is not None:
            self.absolute_pose_fp.flush()
            self.absolute_pose_fp.close()
            self.absolute_pose_fp = None
        if self.raw_csv_fp is not None:
            self.raw_csv_fp.flush()
            self.raw_csv_fp.close()
            self.raw_csv_fp = None
            self.raw_csv_writer = None

    @staticmethod
    def _format_action_line(episode_id, t_idx, action7):
        return 'Episode={0}, t={1}, Action=[{2}, {3}, {4}, {5}, {6}, {7}, {8}]\n'.format(
            episode_id, t_idx,
            repr(action7[0]), repr(action7[1]), repr(action7[2]),
            repr(action7[3]), repr(action7[4]), repr(action7[5]), repr(action7[6])
        )

    @staticmethod
    def _format_pose_line(episode_id, t_idx, pose7):
        return 'Episode={0}, t={1}, Pose=[{2}, {3}, {4}, {5}, {6}, {7}, {8}]\n'.format(
            episode_id, t_idx,
            repr(pose7[0]), repr(pose7[1]), repr(pose7[2]),
            repr(pose7[3]), repr(pose7[4]), repr(pose7[5]), repr(pose7[6])
        )

    def _write_norm_sequence_line(self, t_idx, action7):
        if self.norm_sequence_fp is not None:
            self.norm_sequence_fp.write(self._format_action_line(self.episode_id, t_idx, action7))

    def _write_raw_action_line(self, t_idx, action7):
        if self.raw_action_fp is not None:
            self.raw_action_fp.write(self._format_action_line(self.episode_id, t_idx, action7))

    def _write_absolute_pose_line(self, t_idx, pose7):
        if self.absolute_pose_fp is not None:
            self.absolute_pose_fp.write(self._format_pose_line(self.episode_id, t_idx, pose7))

    def _write_raw_row(self, t_idx, stamp_sec, pose_xyzrpy,
                       phys_delta, norm_delta, gripper_state, gripper_raw, gripper_source):
        if self.raw_csv_writer is None:
            return
        self.raw_csv_writer.writerow([
            self.episode_id, t_idx, stamp_sec,
            pose_xyzrpy[0], pose_xyzrpy[1], pose_xyzrpy[2],
            pose_xyzrpy[3], pose_xyzrpy[4], pose_xyzrpy[5],
            phys_delta[0], phys_delta[1], phys_delta[2],
            phys_delta[3], phys_delta[4], phys_delta[5],
            norm_delta[0], norm_delta[1], norm_delta[2],
            norm_delta[3], norm_delta[4], norm_delta[5],
            gripper_state, gripper_raw, gripper_source
        ])

    def _write_stats_json(self):
        self._ensure_output_dir(self.stats_json_path)
        normalization_limits = {
            'dx': self.max_dx,
            'dy': self.max_dy,
            'dz': self.max_dz,
            'drx': self.max_drx,
            'dry': self.max_dry,
            'drz': self.max_drz,
        }
        payload = {
            'episode_id': self.episode_id,
            'robot_type': self.robot_type,
            'sample_rate_hz': self.sample_rate_hz,
            'player_step_duration': self.player_step_duration,
            'step_count': self.step_idx,
            'normalization': {
                'type': 'fixed_limit_clip',
                'normalized_range': [-1.0, 1.0],
                'limits': normalization_limits,
            },
            'topics': {
                'pose_topic': self.pose_topic,
                'finger_topic': self.finger_topic,
                'joint_state_topic': self.joint_state_topic,
            },
            'gripper': {
                'mode': self.gripper_mode,
                'finger_open_turn': self.finger_open_turn,
                'finger_closed_turn': self.finger_closed_turn,
                'gripper_close_threshold_turn': self.gripper_close_threshold_turn,
                'joint_finger_open_value': self.joint_finger_open_value,
                'joint_finger_closed_value': self.joint_finger_closed_value,
                'joint_gripper_close_threshold': self.joint_gripper_close_threshold,
                'observed_state_stats': self._series_stats(self.gripper_history),
            },
            'deadband': {
                'linear': self.deadband_linear,
                'angular': self.deadband_angular,
            },
            'files': {
                'normalized_sequence': os.path.abspath(self.output_path),
                'raw_sequence': os.path.abspath(self.raw_action_path) if self.write_raw_action_file else None,
                'absolute_pose_sequence': os.path.abspath(self.absolute_pose_path) if self.write_absolute_pose_file else None,
                'raw_csv': os.path.abspath(self.raw_csv_path) if self.write_raw_csv else None,
            },
            'observed_action_stats': {
                'raw_physical': self._compute_vector_stats(self.raw_action_history, self.ACTION_NAMES),
                'normalized': self._compute_vector_stats(self.norm_action_history, self.ACTION_NAMES),
                'absolute_pose': self._compute_vector_stats(self.absolute_pose_history, self.POSE_NAMES),
            }
        }
        with open(self.stats_json_path, 'w') as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)

    def _maybe_switch_force_control(self, enable):
        if not self.auto_force_control:
            return
        if not HAS_EMPTY_SRV:
            rospy.logwarn('std_srvs/Empty is unavailable; cannot auto-switch force control.')
            return
        service_name = self.start_force_control_srv if enable else self.stop_force_control_srv
        try:
            rospy.loginfo('Waiting for service: %s', service_name)
            rospy.wait_for_service(service_name, timeout=3.0)
            proxy = rospy.ServiceProxy(service_name, Empty)
            proxy()
            rospy.loginfo('Service call success: %s', service_name)
        except Exception as exc:
            rospy.logwarn('Service call failed: %s, error=%s', service_name, exc)

    def _record_step(self, stamp_sec, pose_xyzrpy, phys_delta, norm_delta, gripper_state, gripper_raw, gripper_source):
        norm_action7 = [norm_delta[0], norm_delta[1], norm_delta[2],
                        norm_delta[3], norm_delta[4], norm_delta[5],
                        float(gripper_state)]
        raw_action7 = [phys_delta[0], phys_delta[1], phys_delta[2],
                       phys_delta[3], phys_delta[4], phys_delta[5],
                       float(gripper_state)]
        pose7 = [pose_xyzrpy[0], pose_xyzrpy[1], pose_xyzrpy[2],
                 pose_xyzrpy[3], pose_xyzrpy[4], pose_xyzrpy[5],
                 float(gripper_state)]

        self._write_norm_sequence_line(self.step_idx, norm_action7)
        self._write_raw_action_line(self.step_idx, raw_action7)
        self._write_absolute_pose_line(self.step_idx, pose7)
        self._write_raw_row(self.step_idx, stamp_sec, pose_xyzrpy,
                            phys_delta, norm_delta, gripper_state, gripper_raw, gripper_source)

        self.raw_action_history.append(list(phys_delta))
        self.norm_action_history.append(list(norm_delta))
        self.absolute_pose_history.append(list(pose_xyzrpy))
        self.gripper_history.append(float(gripper_state))

    def run(self):
        if not self._wait_until_ready():
            return False

        self._open_outputs()
        self._maybe_switch_force_control(True)

        rospy.loginfo('Recorder will start in %.3f seconds...', self.start_delay_sec)
        rospy.sleep(self.start_delay_sec)

        self.record_started_time = rospy.Time.now().to_sec()
        current_pose = self._pose_msg_to_xyzrpy(self.latest_pose_msg)
        gripper_state, gripper_raw, gripper_source = self._current_gripper_state()
        self.prev_pose = list(current_pose)

        if self.write_initial_zero:
            phys_delta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            norm_delta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self._record_step(self.record_started_time, current_pose, phys_delta, norm_delta,
                              gripper_state, gripper_raw, gripper_source)
            self.step_idx += 1

        rate = rospy.Rate(self.sample_rate_hz)
        rospy.loginfo('Recording started. Press Ctrl+C to stop.')
        try:
            while not rospy.is_shutdown():
                now_sec = rospy.Time.now().to_sec()
                if self.record_duration_sec > 0.0 and (now_sec - self.record_started_time >= self.record_duration_sec):
                    rospy.loginfo('record_duration_sec reached, stopping recorder.')
                    break

                if self.latest_pose_msg is None:
                    rate.sleep()
                    continue

                current_pose = self._pose_msg_to_xyzrpy(self.latest_pose_msg)
                gripper_state, gripper_raw, gripper_source = self._current_gripper_state()

                dx = self._apply_deadband(current_pose[0] - self.prev_pose[0], self.deadband_linear)
                dy = self._apply_deadband(current_pose[1] - self.prev_pose[1], self.deadband_linear)
                dz = self._apply_deadband(current_pose[2] - self.prev_pose[2], self.deadband_linear)
                drx = self._apply_deadband(self._normalize_angle_diff(current_pose[3], self.prev_pose[3]), self.deadband_angular)
                dry = self._apply_deadband(self._normalize_angle_diff(current_pose[4], self.prev_pose[4]), self.deadband_angular)
                drz = self._apply_deadband(self._normalize_angle_diff(current_pose[5], self.prev_pose[5]), self.deadband_angular)
                phys_delta = [dx, dy, dz, drx, dry, drz]

                norm_delta = [
                    self._normalize_delta(dx, self.max_dx),
                    self._normalize_delta(dy, self.max_dy),
                    self._normalize_delta(dz, self.max_dz),
                    self._normalize_delta(drx, self.max_drx),
                    self._normalize_delta(dry, self.max_dry),
                    self._normalize_delta(drz, self.max_drz),
                ]

                self._record_step(now_sec, current_pose, phys_delta, norm_delta,
                                  gripper_state, gripper_raw, gripper_source)

                if (self.step_idx % max(int(self.sample_rate_hz), 1)) == 0:
                    rospy.loginfo(
                        'Recorded t=%d pose=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f] raw=[%.5f, %.5f, %.5f, %.5f, %.5f, %.5f] g=%.2f',
                        self.step_idx,
                        current_pose[0], current_pose[1], current_pose[2], current_pose[3], current_pose[4], current_pose[5],
                        phys_delta[0], phys_delta[1], phys_delta[2], phys_delta[3], phys_delta[4], phys_delta[5],
                        float(gripper_state)
                    )

                self.prev_pose = list(current_pose)
                self.step_idx += 1
                rate.sleep()

        except rospy.ROSInterruptException:
            pass
        finally:
            try:
                self._write_stats_json()
            finally:
                self._close_outputs()
                self._maybe_switch_force_control(False)
                rospy.loginfo('Normalized sequence saved to: %s', os.path.abspath(self.output_path))
                if self.write_raw_action_file:
                    rospy.loginfo('Raw action sequence saved to: %s', os.path.abspath(self.raw_action_path))
                if self.write_absolute_pose_file:
                    rospy.loginfo('Absolute pose sequence saved to: %s', os.path.abspath(self.absolute_pose_path))
                if self.write_raw_csv:
                    rospy.loginfo('Debug CSV saved to: %s', os.path.abspath(self.raw_csv_path))
                rospy.loginfo('Stats JSON saved to: %s', os.path.abspath(self.stats_json_path))

        return True


def main():
    recorder = SequenceRecorder()
    ok = recorder.run()
    if not ok:
        rospy.logerr('Sequence recorder exited with failure.')


if __name__ == '__main__':
    main()
