#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Teaching-session recording orchestrator.

Current alignment updates:
1. Default output root is tools/teach_sessions.
2. Each teaching run creates a unique timestamped folder; no overwrite.
3. Calls sequence_recorder.py and explicitly saves:
   sequence.txt / sequence_raw.txt / sequence_pose.txt / sequence_debug.csv / sequence_stats.json
4. Restore logic remains: initial joints first, pose fallback, then home-and-retry.
"""

import argparse
import io
import json
import math
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RECORDER = os.path.join(THIS_DIR, 'sequence_recorder.py')
DEFAULT_OUTPUT_ROOT = os.path.join(THIS_DIR, 'teach_sessions')


def _find_executable(name):
    for folder in os.environ.get('PATH', '').split(os.pathsep):
        if not folder:
            continue
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def maybe_reexec_to_python2_for_melodic():
    distro = os.environ.get('ROS_DISTRO', '').strip().lower()
    if sys.version_info[0] < 3:
        return
    if distro != 'melodic':
        return
    py2 = _find_executable('python2') or _find_executable('python')
    if not py2:
        sys.stderr.write('[ERROR] Detected ROS Melodic + python3, but python2/python was not found.\n')
        sys.exit(2)
    sys.stderr.write('[WARN] Detected ROS Melodic + python3; re-executing with %s.\n' % py2)
    os.execvp(py2, [py2, os.path.abspath(__file__)] + sys.argv[1:])


maybe_reexec_to_python2_for_melodic()

import actionlib
import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

try:
    from std_srvs.srv import Empty
    HAS_EMPTY = True
except Exception:
    Empty = None
    HAS_EMPTY = False

try:
    from kinova_msgs.msg import (
        ArmJointAnglesAction,
        ArmJointAnglesGoal,
        ArmPoseAction,
        ArmPoseGoal,
        FingerPosition,
        SetFingersPositionAction,
        SetFingersPositionGoal,
    )
    HAS_ACTIONS = True
except Exception:
    ArmJointAnglesAction = None
    ArmJointAnglesGoal = None
    ArmPoseAction = None
    ArmPoseGoal = None
    FingerPosition = None
    SetFingersPositionAction = None
    SetFingersPositionGoal = None
    HAS_ACTIONS = False

from finger_telep_oc import FingerTeleopModeSwitch


def parse_args():
    parser = argparse.ArgumentParser(description='Teach and record Kinova sequence with one-key start/stop.')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT, help='Teach session root directory; relative paths are resolved from this script directory')
    parser.add_argument('--session-prefix', default='teach', help='Session folder prefix')
    parser.add_argument('--recorder-python', default='auto', help='Interpreter for sequence_recorder.py')
    parser.add_argument('--recorder-script', default=DEFAULT_RECORDER)

    parser.add_argument('--robot-type', default='j2s6s300')
    parser.add_argument('--robot-prefix', default='j2s6s300_driver')
    parser.add_argument('--base-link', default='j2s6s300_link_base')

    parser.add_argument('--record-sample-rate', type=float, default=8.0)
    parser.add_argument('--record-player-step-duration', type=float, default=0.0)
    parser.add_argument('--record-gripper-mode', default='binary', choices=['binary', 'continuous'])
    parser.add_argument('--record-write-initial-zero', action='store_true')

    parser.add_argument('--startup-wait', type=float, default=0.2)
    parser.add_argument('--shutdown-wait', type=float, default=3.0)
    parser.add_argument('--wait-ready-timeout', type=float, default=8.0)

    parser.add_argument('--restore-joint-timeout', type=float, default=12.0)
    parser.add_argument('--restore-pose-timeout', type=float, default=12.0)
    parser.add_argument('--restore-gripper-timeout', type=float, default=5.0)

    parser.add_argument('--open-position', nargs='*', type=float, default=[0, 0, 0])
    parser.add_argument('--close-position', nargs='*', type=float, default=[7000, 7000, 0])
    parser.add_argument('--switch-delay', type=float, default=0.35)
    parser.add_argument('--action-timeout', type=float, default=5.0)
    parser.add_argument('--torque-service-name', default=None)
    return parser.parse_args()


def resolve_local_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(THIS_DIR, path_value)


def resolve_child_python(requested):
    if requested and requested != 'auto':
        return requested
    distro = os.environ.get('ROS_DISTRO', '').strip().lower()
    if distro == 'melodic':
        return _find_executable('python2') or _find_executable('python') or 'python'
    return sys.executable


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def make_unique_session_dir(root, prefix):
    ensure_dir(root)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = os.path.join(root, '%s_%s' % (prefix, stamp))
    if not os.path.exists(base):
        os.makedirs(base)
        return base
    idx = 1
    while True:
        candidate = os.path.join(root, '%s_%s_%02d' % (prefix, stamp, idx))
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        idx += 1


def write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with io.open(path, 'w', encoding='utf-8') as fp:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        if not isinstance(text, type(u'')):
            text = text.decode('utf-8')
        fp.write(text)


def wait_proc_exit(proc, wait_sec):
    deadline = time.time() + max(wait_sec, 0.0)
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return rc
        time.sleep(0.05)
    return None


def stop_process(proc, name, wait_sec):
    if proc is None:
        return None
    if proc.poll() is not None:
        return proc.returncode
    rospy.loginfo('Stopping %s with SIGINT ...', name)
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass
    rc = wait_proc_exit(proc, wait_sec)
    if rc is not None:
        return rc
    rospy.logwarn('%s did not exit after SIGINT, sending SIGTERM ...', name)
    try:
        proc.terminate()
    except Exception:
        pass
    rc = wait_proc_exit(proc, max(wait_sec, 1.0))
    if rc is not None:
        return rc
    rospy.logwarn('%s still alive, killing ...', name)
    try:
        proc.kill()
    except Exception:
        pass
    wait_proc_exit(proc, 3.0)
    return proc.poll()


def get_key_nonblocking():
    if select.select([sys.stdin], [], [], 0.02)[0]:
        return sys.stdin.read(1)
    return None


def quaternion_to_rpy(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return 0.0, 0.0, 0.0
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if sinp > 1.0:
        sinp = 1.0
    elif sinp < -1.0:
        sinp = -1.0
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class RobotStateMonitor(object):
    def __init__(self, robot_type):
        self.robot_type = robot_type
        self.pose_topic = '/%s_driver/out/tool_pose' % robot_type
        self.joint_state_topic = '/%s_driver/out/joint_state' % robot_type
        self.finger_topic = '/%s_driver/out/finger_position' % robot_type

        self.latest_pose = None
        self.latest_joint_state = None
        self.latest_finger = None

        rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(self.joint_state_topic, JointState, self._joint_cb, queue_size=1)
        if FingerPosition is not None:
            rospy.Subscriber(self.finger_topic, FingerPosition, self._finger_cb, queue_size=1)

    def _pose_cb(self, msg):
        self.latest_pose = msg

    def _joint_cb(self, msg):
        self.latest_joint_state = msg

    def _finger_cb(self, msg):
        self.latest_finger = msg

    def wait_until_ready(self, timeout_sec):
        t0 = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.latest_pose is not None and self.latest_joint_state is not None:
                return True
            if time.time() - t0 > timeout_sec:
                return False
            rate.sleep()
        return False

    @staticmethod
    def _pose_to_dict(msg):
        p = msg.pose.position
        q = msg.pose.orientation
        roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        return {
            'frame_id': msg.header.frame_id,
            'position': {'x': float(p.x), 'y': float(p.y), 'z': float(p.z)},
            'orientation_quat': {'x': float(q.x), 'y': float(q.y), 'z': float(q.z), 'w': float(q.w)},
            'orientation_rpy': {'roll': float(roll), 'pitch': float(pitch), 'yaw': float(yaw)},
        }

    @staticmethod
    def _extract_arm_joint_state(msg):
        arm_names = []
        arm_pos_rad = []
        for idx, name in enumerate(msg.name):
            if idx >= len(msg.position):
                continue
            if 'finger' in name.lower():
                continue
            arm_names.append(name)
            arm_pos_rad.append(float(msg.position[idx]))
        return {
            'names': arm_names,
            'position_rad': arm_pos_rad,
            'position_deg': [math.degrees(v) for v in arm_pos_rad],
        }

    @staticmethod
    def _finger_to_dict(msg):
        if msg is None:
            return None
        vals = {
            'finger1': float(getattr(msg, 'finger1', 0.0)),
            'finger2': float(getattr(msg, 'finger2', 0.0)),
            'finger3': float(getattr(msg, 'finger3', 0.0)),
        }
        vals['average'] = (vals['finger1'] + vals['finger2'] + vals['finger3']) / 3.0
        return vals

    def snapshot(self):
        if self.latest_pose is None or self.latest_joint_state is None:
            raise RuntimeError('Robot state is not ready yet.')
        return {
            'stamp': time.time(),
            'tool_pose': self._pose_to_dict(self.latest_pose),
            'joint_state': self._extract_arm_joint_state(self.latest_joint_state),
            'finger_state': self._finger_to_dict(self.latest_finger),
        }


class TeachSessionManager(object):
    def __init__(self, args):
        self.args = args
        self.robot_type = args.robot_type
        self.robot_prefix = args.robot_prefix
        self.base_link = args.base_link

        self.monitor = RobotStateMonitor(self.robot_type)
        self.teleop = FingerTeleopModeSwitch()

        self.home_srv_name = '/%s/in/home_arm' % self.robot_prefix
        self.recorder_proc = None
        self.current_session_dir = None
        self.initial_state = None
        self.is_recording = False

        self.joint_action_client = None
        self.pose_action_client = None
        self.gripper_client = None
        if HAS_ACTIONS:
            self.joint_action_client = actionlib.SimpleActionClient(
                '/%s/joints_action/joint_angles' % self.robot_prefix,
                ArmJointAnglesAction
            )
            self.pose_action_client = actionlib.SimpleActionClient(
                '/%s/pose_action/tool_pose' % self.robot_prefix,
                ArmPoseAction
            )
            self.gripper_client = actionlib.SimpleActionClient(
                '/%s/fingers_action/finger_positions' % self.robot_prefix,
                SetFingersPositionAction
            )

    def _build_recorder_cmd(self, session_dir):
        trajectory_dir = os.path.join(session_dir, 'trajectory')
        return [
            resolve_child_python(self.args.recorder_python),
            resolve_local_path(self.args.recorder_script),
            '_robot_type:={}'.format(self.robot_type),
            '_output_path:={}'.format(os.path.join(trajectory_dir, 'sequence.txt')),
            '_raw_action_path:={}'.format(os.path.join(trajectory_dir, 'sequence_raw.txt')),
            '_absolute_pose_path:={}'.format(os.path.join(trajectory_dir, 'sequence_pose.txt')),
            '_raw_csv_path:={}'.format(os.path.join(trajectory_dir, 'sequence_debug.csv')),
            '_stats_json_path:={}'.format(os.path.join(trajectory_dir, 'sequence_stats.json')),
            '_write_raw_action_file:=true',
            '_write_absolute_pose_file:=true',
            '_write_raw_csv:=true',
            '_write_initial_zero:={}'.format('true' if self.args.record_write_initial_zero else 'false'),
            '_player_step_duration:={}'.format(self.args.record_player_step_duration),
            '_sample_rate_hz:={}'.format(self.args.record_sample_rate),
            '_start_delay_sec:=0.0',
            '_record_duration_sec:=-1.0',
            '_gripper_mode:={}'.format(self.args.record_gripper_mode),
        ]

    def _call_home_arm(self):
        if not HAS_EMPTY:
            rospy.logwarn('std_srvs/Empty unavailable; cannot call home_arm.')
            return False
        try:
            rospy.wait_for_service(self.home_srv_name, timeout=3.0)
            proxy = rospy.ServiceProxy(self.home_srv_name, Empty)
            proxy()
            rospy.loginfo('home_arm called successfully: %s', self.home_srv_name)
            return True
        except Exception as exc:
            rospy.logwarn('home_arm failed: %s', exc)
            return False

    @staticmethod
    def _wait_action_server(client, name, timeout=3.0):
        if client is None:
            rospy.logwarn('%s client unavailable', name)
            return False
        ok = client.wait_for_server(rospy.Duration(timeout))
        if not ok:
            rospy.logwarn('%s action server unavailable', name)
        return ok

    def _restore_gripper(self, finger_state):
        if finger_state is None:
            rospy.loginfo('No initial finger state recorded, skip gripper restore.')
            return True
        if not HAS_ACTIONS or self.gripper_client is None:
            rospy.logwarn('Gripper action client unavailable, skip gripper restore.')
            return False
        if not self._wait_action_server(self.gripper_client, 'gripper'):
            return False
        try:
            goal = SetFingersPositionGoal()
            goal.fingers.finger1 = float(finger_state.get('finger1', 0.0))
            goal.fingers.finger2 = float(finger_state.get('finger2', 0.0))
            goal.fingers.finger3 = float(finger_state.get('finger3', 0.0))
            self.gripper_client.send_goal(goal)
            finished = self.gripper_client.wait_for_result(rospy.Duration(self.args.restore_gripper_timeout))
            if not finished:
                rospy.logwarn('Restore gripper timeout.')
                return False
            rospy.loginfo('Gripper restored.')
            return True
        except Exception as exc:
            rospy.logwarn('Restore gripper failed: %s', exc)
            return False

    def _restore_joint_angles(self, joint_state):
        if not HAS_ACTIONS or self.joint_action_client is None:
            rospy.logwarn('Joint action client unavailable.')
            return False
        if not self._wait_action_server(self.joint_action_client, 'joint_angles'):
            return False
        pos_deg = list(joint_state.get('position_deg', []))
        if not pos_deg:
            rospy.logwarn('No initial joint angles recorded.')
            return False
        try:
            goal = ArmJointAnglesGoal()
            angle_obj = goal.angles
            for idx, value_deg in enumerate(pos_deg, start=1):
                attr = 'joint{}'.format(idx)
                if hasattr(angle_obj, attr):
                    setattr(angle_obj, attr, float(value_deg))
            self.joint_action_client.send_goal(goal)
            finished = self.joint_action_client.wait_for_result(rospy.Duration(self.args.restore_joint_timeout))
            if not finished:
                rospy.logwarn('Restore joint angles timeout.')
                return False
            rospy.loginfo('Initial joint position restored.')
            return True
        except Exception as exc:
            rospy.logwarn('Restore joint angles failed: %s', exc)
            return False

    def _restore_cartesian_pose(self, pose_state):
        if not HAS_ACTIONS or self.pose_action_client is None:
            rospy.logwarn('Pose action client unavailable.')
            return False
        if not self._wait_action_server(self.pose_action_client, 'tool_pose'):
            return False
        try:
            goal = ArmPoseGoal()
            goal.pose.header = Header(frame_id=self.base_link)
            goal.pose.pose.position = Point(
                x=float(pose_state['position']['x']),
                y=float(pose_state['position']['y']),
                z=float(pose_state['position']['z'])
            )
            goal.pose.pose.orientation = Quaternion(
                x=float(pose_state['orientation_quat']['x']),
                y=float(pose_state['orientation_quat']['y']),
                z=float(pose_state['orientation_quat']['z']),
                w=float(pose_state['orientation_quat']['w'])
            )
            self.pose_action_client.send_goal(goal)
            finished = self.pose_action_client.wait_for_result(rospy.Duration(self.args.restore_pose_timeout))
            if not finished:
                rospy.logwarn('Restore cartesian pose timeout.')
                return False
            rospy.loginfo('Fallback cartesian pose restored.')
            return True
        except Exception as exc:
            rospy.logwarn('Restore cartesian pose failed: %s', exc)
            return False

    def _restore_initial_state(self):
        report = {
            'switch_to_trajectory_ok': False,
            'restore_joint_ok': False,
            'restore_gripper_ok': False,
            'restore_pose_fallback_ok': False,
            'home_called': False,
            'retry_joint_ok': False,
            'retry_gripper_ok': False,
            'retry_pose_fallback_ok': False,
        }
        if self.initial_state is None:
            return report

        report['switch_to_trajectory_ok'] = bool(self.teleop.switch_to_trajectory())
        time.sleep(0.3)
        report['restore_joint_ok'] = self._restore_joint_angles(self.initial_state['joint_state'])
        report['restore_gripper_ok'] = self._restore_gripper(self.initial_state.get('finger_state'))
        if report['restore_joint_ok'] and report['restore_gripper_ok']:
            return report

        if not report['restore_joint_ok']:
            report['restore_pose_fallback_ok'] = self._restore_cartesian_pose(self.initial_state['tool_pose'])
            if report['restore_pose_fallback_ok']:
                report['restore_gripper_ok'] = self._restore_gripper(self.initial_state.get('finger_state'))
                if report['restore_gripper_ok']:
                    return report

        report['home_called'] = self._call_home_arm()
        if report['home_called']:
            time.sleep(2.0)
            self.teleop.switch_to_trajectory()
            time.sleep(0.3)
            report['retry_joint_ok'] = self._restore_joint_angles(self.initial_state['joint_state'])
            report['retry_gripper_ok'] = self._restore_gripper(self.initial_state.get('finger_state'))
            if not report['retry_joint_ok']:
                report['retry_pose_fallback_ok'] = self._restore_cartesian_pose(self.initial_state['tool_pose'])
                if report['retry_pose_fallback_ok']:
                    report['retry_gripper_ok'] = self._restore_gripper(self.initial_state.get('finger_state'))
        return report

    def start_session(self):
        if self.is_recording:
            rospy.logwarn('A session is already running.')
            return
        if not self.monitor.wait_until_ready(self.args.wait_ready_timeout):
            rospy.logerr('Robot topics are not ready, cannot start.')
            return

        output_root = resolve_local_path(self.args.output_root)
        session_dir = make_unique_session_dir(output_root, self.args.session_prefix)
        state_dir = os.path.join(session_dir, 'state')
        self.initial_state = self.monitor.snapshot()
        write_json(os.path.join(state_dir, 'initial_state.json'), self.initial_state)

        cmd = self._build_recorder_cmd(session_dir)
        rospy.loginfo('Launching recorder: %s', ' '.join([str(x) for x in cmd]))
        self.recorder_proc = subprocess.Popen(cmd, cwd=THIS_DIR)
        time.sleep(max(self.args.startup_wait, 0.0))

        torque_ok = bool(self.teleop.switch_to_torque())
        if not torque_ok:
            rospy.logerr('Failed to switch to torque/gravity mode. Abort this session.')
            stop_process(self.recorder_proc, 'sequence_recorder', self.args.shutdown_wait)
            self.recorder_proc = None
            return

        manifest = {
            'session_dir': session_dir,
            'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'robot_type': self.robot_type,
            'robot_prefix': self.robot_prefix,
            'python': {
                'controller': sys.executable,
                'recorder': resolve_child_python(self.args.recorder_python),
            },
            'initial_state_file': os.path.join(state_dir, 'initial_state.json'),
            'trajectory_dir': os.path.join(session_dir, 'trajectory'),
            'trajectory_files': {
                'sequence': os.path.join(session_dir, 'trajectory', 'sequence.txt'),
                'sequence_raw': os.path.join(session_dir, 'trajectory', 'sequence_raw.txt'),
                'sequence_pose': os.path.join(session_dir, 'trajectory', 'sequence_pose.txt'),
                'sequence_debug_csv': os.path.join(session_dir, 'trajectory', 'sequence_debug.csv'),
                'sequence_stats': os.path.join(session_dir, 'trajectory', 'sequence_stats.json'),
            },
            'restore_strategy': 'joint_first_then_gripper_then_home_retry_pose_fallback',
        }
        write_json(os.path.join(session_dir, 'manifest.json'), manifest)

        self.current_session_dir = session_dir
        self.is_recording = True
        rospy.loginfo('Teach session started: %s', session_dir)

    def stop_session(self):
        if not self.is_recording:
            rospy.logwarn('No active session.')
            return
        state_dir = os.path.join(self.current_session_dir, 'state')
        stop_process(self.recorder_proc, 'sequence_recorder', self.args.shutdown_wait)
        self.recorder_proc = None

        restore_report = self._restore_initial_state()
        final_state = None
        try:
            if self.monitor.wait_until_ready(1.0):
                final_state = self.monitor.snapshot()
        except Exception as exc:
            rospy.logwarn('Failed to snapshot final state: %s', exc)

        write_json(os.path.join(state_dir, 'restore_report.json'), restore_report)
        if final_state is not None:
            write_json(os.path.join(state_dir, 'final_state.json'), final_state)

        summary = {
            'session_dir': self.current_session_dir,
            'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'initial_state_file': os.path.join(state_dir, 'initial_state.json'),
            'final_state_file': os.path.join(state_dir, 'final_state.json') if final_state is not None else None,
            'restore_report_file': os.path.join(state_dir, 'restore_report.json'),
            'trajectory_dir': os.path.join(self.current_session_dir, 'trajectory'),
            'trajectory_files': {
                'sequence': os.path.join(self.current_session_dir, 'trajectory', 'sequence.txt'),
                'sequence_raw': os.path.join(self.current_session_dir, 'trajectory', 'sequence_raw.txt'),
                'sequence_pose': os.path.join(self.current_session_dir, 'trajectory', 'sequence_pose.txt'),
                'sequence_debug_csv': os.path.join(self.current_session_dir, 'trajectory', 'sequence_debug.csv'),
                'sequence_stats': os.path.join(self.current_session_dir, 'trajectory', 'sequence_stats.json'),
            },
            'restore_report': restore_report,
        }
        write_json(os.path.join(self.current_session_dir, 'summary.json'), summary)

        self.is_recording = False
        self.current_session_dir = None
        self.initial_state = None
        rospy.loginfo('Teach session stopped and restore flow finished.')

    def do_open(self):
        self.teleop.do_open()

    def do_close(self):
        self.teleop.do_close()

    def do_home(self):
        self._call_home_arm()


def main():
    args = parse_args()

    rospy.init_node('teach_sequence_session', anonymous=True)
    if args.torque_service_name:
        rospy.set_param('~torque_service_name', args.torque_service_name)
    rospy.set_param('~open_position', args.open_position)
    rospy.set_param('~close_position', args.close_position)
    rospy.set_param('~switch_delay', args.switch_delay)
    rospy.set_param('~action_timeout', args.action_timeout)
    rospy.set_param('~robot_prefix', args.robot_prefix)

    manager = TeachSessionManager(args)

    print('\nTeach sequence session\n  s : start teach recording\n  e : end teach recording and restore initial state\n  o : open gripper\n  c : close gripper\n  h : call home_arm\n  q : quit\n')

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            key = get_key_nonblocking()
            if key is None:
                rate.sleep()
                continue
            key = key.lower()
            if key == 's':
                manager.start_session()
            elif key == 'e':
                manager.stop_session()
            elif key == 'o':
                manager.do_open()
            elif key == 'c':
                manager.do_close()
            elif key == 'h':
                manager.do_home()
            elif key == 'q':
                if manager.is_recording:
                    manager.stop_session()
                break
            rate.sleep()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':
    main()
