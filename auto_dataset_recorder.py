#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动化训练集录制脚本。

当前对齐点：
1. 默认调用 sequence_player_position_verified.py 进行回放。
2. 若未显式指定输入序列，则自动读取脚本目录下 teach_sessions 中最新一次示教结果。
3. 对 verified 播放优先选择 *_pose.txt / sequence_pose.txt 这类绝对位姿序列；找不到时再回退到 raw/normalized 序列。
4. 同步启动 sequence_recorder.py 与 RealSense 采集。
5. recorder 额外产出 sequence_pose.txt，并保存到 trajectory 目录。
6. 每轮数据写入独立 episode 时间戳文件夹，避免覆盖。
7. 采集结束后自动删除原始 RGB 图，只保留裁剪后的 rgb_224。

说明：
- 本脚本本身不直接依赖 rospy/tf，可由 python 或 python3 启动。
- 子脚本解释器会分别选择：
  - player / recorder: 在 Melodic 下优先 python2/python
  - camera: 优先 python3
"""

from __future__ import print_function

import argparse
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import glob


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PLAYER = os.path.join(THIS_DIR, 'sequence_player_position_verified.py')
DEFAULT_RECORDER = os.path.join(THIS_DIR, 'sequence_recorder.py')
DEFAULT_CAMERA = os.path.join(THIS_DIR, 'record_realsense_rgb_224.py')
DEFAULT_OUTPUT_ROOT = os.path.join(THIS_DIR, 'dataset_episodes')
DEFAULT_TEACH_ROOT = os.path.join(THIS_DIR, 'teach_sessions')


def parse_args():
    parser = argparse.ArgumentParser(description='Replay one sequence and record one dataset episode.')
    parser.add_argument('--raw-sequence', default='', help='可选：显式指定输入序列文件；若不指定，则自动读取 teach_sessions 最新序列')
    parser.add_argument('--teach-root', default=DEFAULT_TEACH_ROOT, help='teach_sessions 根目录；当 --raw-sequence 为空时使用')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT, help='episode 根目录；相对路径按脚本目录解析')
    parser.add_argument('--episode-prefix', default='episode', help='episode 文件夹前缀')

    parser.add_argument('--player-python', default='auto', help='player 使用的解释器')
    parser.add_argument('--recorder-python', default='auto', help='recorder 使用的解释器')
    parser.add_argument('--camera-python', default='auto', help='camera 使用的解释器')

    parser.add_argument('--player-script', default=DEFAULT_PLAYER, help='player 脚本路径')
    parser.add_argument('--recorder-script', default=DEFAULT_RECORDER, help='recorder 脚本路径')
    parser.add_argument('--camera-script', default=DEFAULT_CAMERA, help='camera 脚本路径')

    parser.add_argument('--robot-type', default='j2s6s300')
    parser.add_argument('--startup-wait', type=float, default=0.8)
    parser.add_argument('--shutdown-wait', type=float, default=3.0)
    parser.add_argument('--start-step', type=int, default=0)

    # position_verified player params
    parser.add_argument('--sequence-kind', default='auto', choices=['auto', 'absolute_pose', 'delta_action'])
    parser.add_argument('--action-mode', default='delta', choices=['delta', 'absolute'])
    parser.add_argument('--input-scale', type=float, default=1.0, help='兼容旧参数；会同时乘到 translation/rotation scale 上')
    parser.add_argument('--translation-scale', type=float, default=1.0)
    parser.add_argument('--rotation-scale', type=float, default=1.0)
    parser.add_argument('--use-orientation', action='store_true')
    parser.add_argument('--pause-between-steps', type=float, default=0.0)
    parser.add_argument('--step-action-timeout', type=float, default=8.0)
    parser.add_argument('--wait-pose-timeout', type=float, default=3.0)
    parser.add_argument('--driver-ready-sleep', type=float, default=1.0)

    parser.add_argument('--no-use-orientation', dest='use_orientation', action='store_false')
    parser.set_defaults(use_orientation=True)
    parser.add_argument('--goal-pos-tolerance', type=float, default=0.0005)
    parser.add_argument('--goal-ori-tolerance', type=float, default=0.02)
    parser.add_argument('--enable-gripper', dest='enable_gripper', action='store_true')
    parser.add_argument('--disable-gripper', dest='enable_gripper', action='store_false')
    parser.set_defaults(enable_gripper=True)

    parser.add_argument('--gripper-blocking', action='store_true')
    parser.add_argument('--verify-goal', dest='verify_goal', action='store_true')
    parser.add_argument('--disable-verify-goal', dest='verify_goal', action='store_false')
    parser.set_defaults(verify_goal=True)
    parser.add_argument('--verify-retries', type=int, default=1)
    parser.add_argument('--verify-settle-sec', type=float, default=0.15)

    # recorder params
    parser.add_argument('--record-sample-rate', type=float, default=10.0)
    parser.add_argument('--record-player-step-duration', type=float, default=0.1)
    parser.add_argument('--record-write-initial-zero', action='store_true')
    parser.add_argument('--record-gripper-mode', default='binary', choices=['binary', 'continuous'])

    # camera params
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--stream-fps', type=int, default=30)
    parser.add_argument('--save-hz', type=float, default=10.0)
    parser.add_argument('--output-size', type=int, default=224)
    parser.add_argument('--center-ratio', type=float, default=1.0)
    parser.add_argument('--ext', default='png', choices=['png', 'jpg', 'jpeg'])
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--preview', action='store_true')
    return parser.parse_args()


def resolve_local_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(THIS_DIR, path_str))


def ensure_dir(path_value):
    if path_value and not os.path.isdir(path_value):
        os.makedirs(path_value)


def _find_executable(name):
    for folder in os.environ.get('PATH', '').split(os.pathsep):
        if not folder:
            continue
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def resolve_child_python(role, requested):
    if requested and requested != 'auto':
        return requested
    distro = os.environ.get('ROS_DISTRO', '').strip().lower()
    if role in ('player', 'recorder'):
        if distro == 'melodic':
            return _find_executable('python2') or _find_executable('python') or 'python'
        return sys.executable
    if role == 'camera':
        return _find_executable('python3') or sys.executable
    return sys.executable


def make_unique_episode_dir(root, prefix):
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


def read_lines(path_value):
    with io.open(path_value, 'r', encoding='utf-8') as fp:
        return fp.readlines()


def parse_sequence_file(sequence_path):
    actions = []
    line_kind_counts = {'pose': 0, 'action': 0}
    with io.open(sequence_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if 'Pose=' in line:
                payload = line.split('Pose=', 1)[1].strip().strip('[]')
                line_kind_counts['pose'] += 1
            elif 'Action=' in line:
                payload = line.split('Action=', 1)[1].strip().strip('[]')
                line_kind_counts['action'] += 1
            else:
                continue
            if not payload:
                continue
            values = [float(x.strip()) for x in payload.split(',')]
            actions.append(values)
    detected_kind = 'unknown'
    if line_kind_counts['pose'] > 0 and line_kind_counts['action'] == 0:
        detected_kind = 'absolute_pose'
    elif line_kind_counts['action'] > 0 and line_kind_counts['pose'] == 0:
        detected_kind = 'delta_action'
    elif line_kind_counts['pose'] > 0 or line_kind_counts['action'] > 0:
        detected_kind = 'mixed'
    return actions, detected_kind, line_kind_counts


def series_stats(values):
    if not values:
        return {'count': 0, 'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'abs_max': 0.0}
    mean = sum(values) / float(len(values))
    var = sum((v - mean) ** 2 for v in values) / float(len(values))
    return {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'mean': mean,
        'std': math.sqrt(var),
        'abs_max': max(abs(min(values)), abs(max(values))),
    }


def analyze_input_sequence(sequence_path):
    actions, detected_kind, line_kind_counts = parse_sequence_file(sequence_path)
    if detected_kind == 'absolute_pose':
        dims = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    else:
        dims = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
    payload = {
        'source_file': os.path.abspath(sequence_path),
        'step_count': len(actions),
        'sequence_kind': detected_kind,
        'line_kind_counts': line_kind_counts,
        'has_gripper': any(len(a) >= 7 for a in actions),
        'dimensions': {},
    }
    for i, name in enumerate(dims):
        series = [a[i] for a in actions if len(a) > i]
        payload['dimensions'][name] = series_stats(series)
    return payload


def write_json(path_value, payload):
    parent = os.path.dirname(path_value)
    if parent:
        ensure_dir(parent)
    with io.open(path_value, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False, sort_keys=True)


def find_latest_teach_session(teach_root):
    teach_root = resolve_local_path(teach_root)
    if not os.path.isdir(teach_root):
        raise RuntimeError('teach_sessions 根目录不存在: %s' % teach_root)
    subdirs = []
    for name in os.listdir(teach_root):
        path_value = os.path.join(teach_root, name)
        if os.path.isdir(path_value):
            subdirs.append(path_value)
    if not subdirs:
        raise RuntimeError('teach_sessions 中没有找到任何示教会话目录: %s' % teach_root)
    subdirs.sort(key=lambda p: (os.path.getmtime(p), p), reverse=True)
    return subdirs[0]


def pick_preferred_sequence_from_session(session_dir):
    trajectory_dir = os.path.join(session_dir, 'trajectory')
    if not os.path.isdir(trajectory_dir):
        raise RuntimeError('最新 teach session 缺少 trajectory 目录: %s' % session_dir)

    candidates = []

    exact_pose = os.path.join(trajectory_dir, 'sequence_pose.txt')
    if os.path.isfile(exact_pose):
        candidates.append(exact_pose)

    for pattern in ['*_pose.txt', '*pose*.txt']:
        for path_value in sorted(glob.glob(os.path.join(trajectory_dir, pattern))):
            if os.path.isfile(path_value) and path_value not in candidates:
                candidates.append(path_value)

    for fallback_name in ['sequence_raw.txt', 'sequence.txt']:
        path_value = os.path.join(trajectory_dir, fallback_name)
        if os.path.isfile(path_value) and path_value not in candidates:
            candidates.append(path_value)

    if not candidates:
        raise RuntimeError('在最新 teach session 中没有找到可用序列文件: %s' % trajectory_dir)
    return candidates[0], trajectory_dir, candidates


def resolve_input_sequence(args):
    requested = (args.raw_sequence or '').strip()
    if requested:
        path_value = os.path.abspath(requested)
        if not os.path.isfile(path_value):
            raise RuntimeError('显式指定的序列文件不存在: %s' % path_value)
        return {
            'mode': 'manual',
            'sequence_file': path_value,
            'teach_session_dir': None,
            'trajectory_dir': os.path.dirname(path_value),
            'candidate_files': [path_value],
        }

    latest_session = find_latest_teach_session(args.teach_root)
    selected, trajectory_dir, candidates = pick_preferred_sequence_from_session(latest_session)
    return {
        'mode': 'latest_teach_session',
        'sequence_file': selected,
        'teach_session_dir': latest_session,
        'trajectory_dir': trajectory_dir,
        'candidate_files': candidates,
    }


def build_recorder_cmd(args, episode_dir):
    trajectory_dir = os.path.join(episode_dir, 'trajectory')
    return [
        resolve_child_python('recorder', args.recorder_python),
        resolve_local_path(args.recorder_script),
        '_robot_type:={}'.format(args.robot_type),
        '_output_path:={}'.format(os.path.join(trajectory_dir, 'sequence.txt')),
        '_raw_action_path:={}'.format(os.path.join(trajectory_dir, 'sequence_raw.txt')),
        '_absolute_pose_path:={}'.format(os.path.join(trajectory_dir, 'sequence_pose.txt')),
        '_raw_csv_path:={}'.format(os.path.join(trajectory_dir, 'sequence_debug.csv')),
        '_stats_json_path:={}'.format(os.path.join(trajectory_dir, 'sequence_stats.json')),
        '_write_raw_action_file:=true',
        '_write_absolute_pose_file:=true',
        '_write_raw_csv:=true',
        '_write_initial_zero:={}'.format('true' if args.record_write_initial_zero else 'false'),
        '_player_step_duration:={}'.format(args.record_player_step_duration),
        '_sample_rate_hz:={}'.format(args.record_sample_rate),
        '_start_delay_sec:=0.0',
        '_record_duration_sec:=-1.0',
        '_gripper_mode:={}'.format(args.record_gripper_mode),
    ]


def build_camera_cmd(args, episode_dir):
    camera_dir = os.path.join(episode_dir, 'camera')
    raw_dir = os.path.join(camera_dir, 'rgb_raw')
    fused_dir = os.path.join(camera_dir, 'rgb_224')
    cmd = [
        resolve_child_python('camera', args.camera_python),
        resolve_local_path(args.camera_script),
        '--raw-dir', raw_dir,
        '--fused-dir', fused_dir,
        '--width', str(args.width),
        '--height', str(args.height),
        '--stream-fps', str(args.stream_fps),
        '--save-hz', str(args.save_hz),
        '--output-size', str(args.output_size),
        '--center-ratio', str(args.center_ratio),
        '--ext', str(args.ext),
        '--jpeg-quality', str(args.jpeg_quality),
    ]
    if args.preview:
        cmd.append('--preview')
    return cmd


def build_player_cmd(args, sequence_file):
    translation_scale = args.input_scale * args.translation_scale
    rotation_scale = args.input_scale * args.rotation_scale
    cmd = [
        resolve_child_python('player', args.player_python),
        resolve_local_path(args.player_script),
        '_robot_type:={}'.format(args.robot_type),
        '_sequence_file:={}'.format(sequence_file),
        '_start_step:={}'.format(args.start_step),
        '_sequence_kind:={}'.format(args.sequence_kind),
        '_action_mode:={}'.format(args.action_mode),
        '_translation_scale:={}'.format(translation_scale),
        '_rotation_scale:={}'.format(rotation_scale),
        '_pause_between_steps:={}'.format(args.pause_between_steps),
        '_step_action_timeout:={}'.format(args.step_action_timeout),
        '_wait_pose_timeout:={}'.format(args.wait_pose_timeout),
        '_driver_ready_sleep:={}'.format(args.driver_ready_sleep),
        '_goal_pos_tolerance:={}'.format(args.goal_pos_tolerance),
        '_goal_ori_tolerance:={}'.format(args.goal_ori_tolerance),
        '_verify_goal:={}'.format('true' if args.verify_goal else 'false'),
        '_verify_retries:={}'.format(args.verify_retries),
        '_verify_settle_sec:={}'.format(args.verify_settle_sec),
        '_enable_gripper:={}'.format('true' if args.enable_gripper else 'false'),
        '_gripper_blocking:={}'.format('true' if args.gripper_blocking else 'false'),
    ]
    if args.use_orientation:
        cmd.append('_use_orientation:=true')
    else:
        cmd.append('_use_orientation:=false')
    return cmd


def spawn_process(cmd, name):
    print('[INFO] Launch {}: {}'.format(name, ' '.join(str(x) for x in cmd)))
    return subprocess.Popen(cmd, cwd=str(THIS_DIR))


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
    print('[INFO] Stopping {} with SIGINT ...'.format(name))
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass
    rc = wait_proc_exit(proc, wait_sec)
    if rc is not None:
        return rc
    print('[WARN] {} did not exit after SIGINT, sending SIGTERM ...'.format(name))
    try:
        proc.terminate()
    except Exception:
        pass
    rc = wait_proc_exit(proc, max(wait_sec, 1.0))
    if rc is not None:
        return rc
    print('[WARN] {} still alive, killing it ...'.format(name))
    try:
        proc.kill()
    except Exception:
        pass
    wait_proc_exit(proc, 3.0)
    return proc.poll()


def count_files(folder, suffixes):
    if not os.path.isdir(folder):
        return 0
    total = 0
    for name in os.listdir(folder):
        path_value = os.path.join(folder, name)
        if os.path.isfile(path_value) and os.path.splitext(name)[1].lower() in suffixes:
            total += 1
    return total


def remove_raw_images(camera_raw_dir):
    removed = 0
    if not os.path.isdir(camera_raw_dir):
        return removed
    for name in os.listdir(camera_raw_dir):
        path_value = os.path.join(camera_raw_dir, name)
        if os.path.isfile(path_value) and os.path.splitext(name)[1].lower() in ('.png', '.jpg', '.jpeg'):
            os.unlink(path_value)
            removed += 1
    try:
        os.rmdir(camera_raw_dir)
    except Exception:
        pass
    return removed


def main():
    args = parse_args()

    try:
        input_info = resolve_input_sequence(args)
    except Exception as exc:
        print('[ERROR] {}'.format(exc))
        return 1

    sequence_file = input_info['sequence_file']
    print('[INFO] Selected input sequence: {}'.format(sequence_file))
    if input_info['mode'] == 'latest_teach_session':
        print('[INFO] Latest teach session: {}'.format(input_info['teach_session_dir']))
        print('[INFO] Candidate files (priority order):')
        for path_value in input_info['candidate_files']:
            print('  - {}'.format(path_value))

    output_root = resolve_local_path(args.output_root)
    episode_dir = make_unique_episode_dir(output_root, args.episode_prefix)
    input_dir = os.path.join(episode_dir, 'input')
    ensure_dir(input_dir)

    copied_sequence = os.path.join(input_dir, os.path.basename(sequence_file))
    shutil.copy2(sequence_file, copied_sequence)

    input_stats = analyze_input_sequence(sequence_file)
    input_stats_path = os.path.join(input_dir, 'input_sequence_stats.json')
    write_json(input_stats_path, input_stats)

    manifest = {
        'robot_type': args.robot_type,
        'episode_dir': episode_dir,
        'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input': {
            'selection_mode': input_info['mode'],
            'selected_sequence_file': copied_sequence,
            'original_sequence_file': sequence_file,
            'teach_session_dir': input_info['teach_session_dir'],
            'teach_trajectory_dir': input_info['trajectory_dir'],
            'candidate_files': input_info['candidate_files'],
            'sequence_stats': input_stats_path,
            'sequence_kind': input_stats.get('sequence_kind', 'unknown'),
        },
        'python': {
            'controller': sys.executable,
            'player': resolve_child_python('player', args.player_python),
            'recorder': resolve_child_python('recorder', args.recorder_python),
            'camera': resolve_child_python('camera', args.camera_python),
        },
        'player_script': resolve_local_path(args.player_script),
        'recorder_script': resolve_local_path(args.recorder_script),
        'camera_script': resolve_local_path(args.camera_script),
    }
    write_json(os.path.join(episode_dir, 'manifest.json'), manifest)

    recorder_proc = None
    camera_proc = None
    player_proc = None
    t0 = time.time()

    try:
        recorder_proc = spawn_process(build_recorder_cmd(args, episode_dir), 'sequence_recorder')
        camera_proc = spawn_process(build_camera_cmd(args, episode_dir), 'realsense_recorder')
        time.sleep(max(args.startup_wait, 0.0))
        player_proc = spawn_process(build_player_cmd(args, sequence_file), 'sequence_player_position_verified')
        while True:
            rc = player_proc.poll()
            if rc is not None:
                print('[INFO] sequence_player_position_verified exited with return code: {}'.format(rc))
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\n[WARN] Interrupted by user.')
    finally:
        player_rc = stop_process(player_proc, 'sequence_player_position_verified', args.shutdown_wait)
        recorder_rc = stop_process(recorder_proc, 'sequence_recorder', args.shutdown_wait)
        camera_rc = stop_process(camera_proc, 'realsense_recorder', args.shutdown_wait)
        elapsed = time.time() - t0

        camera_raw_dir = os.path.join(episode_dir, 'camera', 'rgb_raw')
        camera_fused_dir = os.path.join(episode_dir, 'camera', 'rgb_224')
        raw_count_before_cleanup = count_files(camera_raw_dir, ('.png', '.jpg', '.jpeg'))
        removed_raw_count = remove_raw_images(camera_raw_dir)
        fused_count = count_files(camera_fused_dir, ('.png', '.jpg', '.jpeg'))

        summary = {
            'robot_type': args.robot_type,
            'elapsed_sec': elapsed,
            'episode_dir': episode_dir,
            'input_sequence': sequence_file,
            'input_sequence_kind': input_stats.get('sequence_kind', 'unknown'),
            'input_selection_mode': input_info['mode'],
            'player_return_code': player_rc,
            'recorder_return_code': recorder_rc,
            'camera_return_code': camera_rc,
            'counts': {
                'input_step_count': input_stats.get('step_count', 0),
                'camera_raw_image_count_before_cleanup': raw_count_before_cleanup,
                'camera_raw_image_count_kept': count_files(camera_raw_dir, ('.png', '.jpg', '.jpeg')),
                'camera_fused_image_count': fused_count,
            },
            'cleanup': {
                'raw_images_removed': True,
                'removed_raw_image_count': removed_raw_count,
                'camera_raw_dir_removed': (not os.path.exists(camera_raw_dir)),
            },
            'files': {
                'manifest': os.path.join(episode_dir, 'manifest.json'),
                'input_sequence_copy': copied_sequence,
                'input_sequence_stats': input_stats_path,
                'recorded_sequence': os.path.join(episode_dir, 'trajectory', 'sequence.txt'),
                'recorded_raw_sequence': os.path.join(episode_dir, 'trajectory', 'sequence_raw.txt'),
                'recorded_pose_sequence': os.path.join(episode_dir, 'trajectory', 'sequence_pose.txt'),
                'recorded_sequence_stats': os.path.join(episode_dir, 'trajectory', 'sequence_stats.json'),
                'recorded_debug_csv': os.path.join(episode_dir, 'trajectory', 'sequence_debug.csv'),
                'camera_metadata': os.path.join(episode_dir, 'camera', 'metadata.csv'),
                'camera_fused_dir': camera_fused_dir,
            },
        }
        write_json(os.path.join(episode_dir, 'summary.json'), summary)

        print('[INFO] Episode finished.')
        print('[INFO] Episode dir: {}'.format(episode_dir))
        print('[INFO] Summary: {}'.format(os.path.join(episode_dir, 'summary.json')))
        if player_rc not in (0, None):
            return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
