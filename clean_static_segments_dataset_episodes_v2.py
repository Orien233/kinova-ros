#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def to_float(x, default: float = 0.0) -> float:
    try:
        if x is None or x == '':
            return default
        return float(x)
    except Exception:
        return default


def norm3(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def list_images_sorted(img_dir: Path) -> List[Path]:
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def backup_file(src: Path, backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, backup_root / src.name)


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_action_vector_from_csv_row(row: Dict[str, str]) -> List[float]:
    dx = to_float(row.get('dx_phys', row.get('dx', row.get('delta_x', row.get('dx_norm', 0.0)))))
    dy = to_float(row.get('dy_phys', row.get('dy', row.get('delta_y', row.get('dy_norm', 0.0)))))
    dz = to_float(row.get('dz_phys', row.get('dz', row.get('delta_z', row.get('dz_norm', 0.0)))))

    drx = to_float(row.get('drx_phys', row.get('droll', row.get('delta_roll', row.get('drx_norm', 0.0)))))
    dry = to_float(row.get('dry_phys', row.get('dpitch', row.get('delta_pitch', row.get('dry_norm', 0.0)))))
    drz = to_float(row.get('drz_phys', row.get('dyaw', row.get('delta_yaw', row.get('drz_norm', 0.0)))))

    grip = to_float(row.get('gripper_state', row.get('gripper', row.get('gripper_cmd', 0.0))))
    return [dx, dy, dz, drx, dry, drz, grip]


def extract_gripper_value_from_csv_row(row: Dict[str, str]) -> float:
    return to_float(row.get('gripper_raw', row.get('gripper', row.get('gripper_cmd', 0.0))))


def detect_drop_indices(
    rows: List[Dict[str, str]],
    trans_thresh: float,
    rot_thresh: float,
    grip_stable_thresh: float,
    grip_event_thresh: float,
    min_static_len: int,
    keep_static_frames: int,
    fps: float,
    protect_after_grasp_seconds: float,
) -> Tuple[List[int], List[int]]:
    n = len(rows)
    actions = [extract_action_vector_from_csv_row(r) for r in rows]
    grippers = [extract_gripper_value_from_csv_row(r) for r in rows]

    static_mask = [False] * n
    for i, vec in enumerate(actions):
        trans_mag = norm3(vec[:3])
        rot_mag = norm3(vec[3:6])
        g_change = 0.0 if i == 0 else abs(grippers[i] - grippers[i - 1])
        static_mask[i] = (
            trans_mag <= trans_thresh and rot_mag <= rot_thresh and g_change <= grip_stable_thresh
        )

    protect_frames = max(0, int(round(protect_after_grasp_seconds * fps)))
    protected_mask = [False] * n
    grasp_event_indices: List[int] = []
    for i in range(1, n):
        if abs(grippers[i] - grippers[i - 1]) >= grip_event_thresh:
            grasp_event_indices.append(i)
    for event_idx in grasp_event_indices:
        end_idx = min(n - 1, event_idx + protect_frames)
        for j in range(event_idx, end_idx + 1):
            protected_mask[j] = True

    drop = set()
    i = 0
    while i < n:
        if not static_mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and static_mask[i + 1]:
            i += 1
        end = i
        seg_len = end - start + 1
        if seg_len >= min_static_len:
            for k in range(start + keep_static_frames, end + 1):
                if not protected_mask[k]:
                    drop.add(k)
        i += 1
    return sorted(drop), grasp_event_indices


def prune_by_indices(items: List, keep_indices: List[int]) -> List:
    keep_set = set(keep_indices)
    return [item for idx, item in enumerate(items) if idx in keep_set]


def try_prune_sidecar_csv(path: Path, keep_indices: List[int], expected_len: int, backup_root: Path, apply: bool) -> None:
    try:
        rows, fieldnames = read_csv_rows(path)
    except Exception as e:
        print(f'  [警告] 读取 CSV 失败 {path.name}: {e}')
        return
    if len(rows) != expected_len:
        print(f'  [跳过 sidecar CSV] {path.name}: 行数={len(rows)}，与主序列={expected_len} 不一致')
        return
    if not apply:
        print(f'  [dry-run] 将同步裁剪 CSV: {path}')
        return
    backup_file(path, backup_root)
    new_rows = prune_by_indices(rows, keep_indices)
    write_csv_rows(path, new_rows, fieldnames)
    print(f'  [CSV已裁剪] {path}')


def try_prune_sidecar_txt(path: Path, keep_indices: List[int], expected_len: int, backup_root: Path, apply: bool) -> None:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except Exception as e:
        print(f'  [警告] 读取 TXT 失败 {path.name}: {e}')
        return
    if len(lines) != expected_len:
        print(f'  [跳过 sidecar TXT] {path.name}: 行数={len(lines)}，与主序列={expected_len} 不一致')
        return
    if not apply:
        print(f'  [dry-run] 将同步裁剪 TXT: {path}')
        return
    backup_file(path, backup_root)
    new_lines = prune_by_indices(lines, keep_indices)
    path.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
    print(f'  [TXT已裁剪] {path}')


def is_episode_dir(ep_dir: Path, action_rel: str, image_rel: str) -> bool:
    return (ep_dir / action_rel).is_file() and (ep_dir / image_rel).is_dir()


def collect_episode_dirs(root: Path, action_rel: str, image_rel: str, episode_prefix: str, recursive: bool = True) -> List[Path]:
    """
    改进点：
    1. 先检查 root 自身是否就是 episode
    2. 再检查 root 的直接子目录
    3. 最后递归搜索任意深度下的 trajectory/sequence_debug.csv，再反推 episode 目录
    这样即使 dataset_episodes 下又套了一层目录，也能找到。
    """
    found: List[Path] = []
    seen = set()

    def add_candidate(ep: Path):
        rp = str(ep.resolve())
        if rp in seen:
            return
        if episode_prefix and episode_prefix not in ep.name:
            # 不强制必须以 episode_ 开头，但至少给一个弱约束；
            # 若用户不想限制可传 --episode-prefix ''
            pass
        if is_episode_dir(ep, action_rel, image_rel):
            found.append(ep)
            seen.add(rp)

    add_candidate(root)

    for p in sorted([x for x in root.iterdir() if x.is_dir()], key=lambda x: natural_key(x.name)):
        if episode_prefix and not p.name.startswith(episode_prefix):
            # 直接子目录层保留前缀过滤
            continue
        add_candidate(p)

    if recursive:
        action_parts = Path(action_rel).parts
        action_name = action_parts[-1]
        for f in root.rglob(action_name):
            if not f.is_file():
                continue
            rel_parts = f.relative_to(root).parts
            if len(rel_parts) < len(action_parts):
                continue
            if tuple(rel_parts[-len(action_parts):]) != action_parts:
                continue
            ep = f
            for _ in range(len(action_parts)):
                ep = ep.parent
            add_candidate(ep)

    return sorted(found, key=lambda p: natural_key(p.name))


def process_episode(
    ep_dir: Path,
    action_rel: str,
    image_rel: str,
    metadata_rel: Optional[str],
    raw_txt_rel: Optional[str],
    apply: bool,
    trans_thresh: float,
    rot_thresh: float,
    grip_stable_thresh: float,
    grip_event_thresh: float,
    min_static_len: int,
    keep_static_frames: int,
    fps: float,
    protect_after_grasp_seconds: float,
) -> None:
    action_file = ep_dir / action_rel
    image_dir = ep_dir / image_rel
    metadata_file = ep_dir / metadata_rel if metadata_rel else None
    raw_txt_file = ep_dir / raw_txt_rel if raw_txt_rel else None

    if not action_file.is_file():
        print(f'[跳过] {ep_dir}: 未找到动作文件 -> {action_file}')
        return
    if not image_dir.is_dir():
        print(f'[跳过] {ep_dir}: 未找到图片目录 -> {image_dir}')
        return

    images = list_images_sorted(image_dir)
    if not images:
        print(f'[跳过] {ep_dir}: 图片目录为空 -> {image_dir}')
        return

    try:
        rows, fieldnames = read_csv_rows(action_file)
    except Exception as e:
        print(f'[跳过] {ep_dir}: 动作 CSV 解析失败 -> {e}')
        return

    n = len(rows)
    if len(images) != n:
        print(f'[跳过] {ep_dir}: 图片数与 sequence_debug.csv 行数不一致，无法安全同步删除。 csv_rows={n}, images={len(images)}')
        return

    drop_indices, grasp_event_indices = detect_drop_indices(
        rows=rows,
        trans_thresh=trans_thresh,
        rot_thresh=rot_thresh,
        grip_stable_thresh=grip_stable_thresh,
        grip_event_thresh=grip_event_thresh,
        min_static_len=min_static_len,
        keep_static_frames=keep_static_frames,
        fps=fps,
        protect_after_grasp_seconds=protect_after_grasp_seconds,
    )

    if not drop_indices:
        print(f'[保留] {ep_dir.name}: 未发现需要删除的长静止段（检测到夹爪事件 {len(grasp_event_indices)} 次）')
        return

    keep_indices = [i for i in range(n) if i not in set(drop_indices)]
    removed_images = [images[i] for i in drop_indices]

    print(f'[检测到] {ep_dir.name}: 总帧数={n}, 删除={len(drop_indices)}, 保留={len(keep_indices)}, 夹爪事件={len(grasp_event_indices)}')

    if not apply:
        preview = drop_indices[:20]
        more = ' ...' if len(drop_indices) > 20 else ''
        print(f'  dry-run: 将删除下标 {preview}{more}')
        if metadata_file and metadata_file.is_file():
            print(f'  dry-run: 将同步裁剪 CSV -> {metadata_file}')
        if raw_txt_file and raw_txt_file.is_file():
            print(f'  dry-run: 将尝试同步裁剪 TXT -> {raw_txt_file}')
        return

    backup_root = ep_dir / '.backup_before_clean'
    backup_file(action_file, backup_root)
    new_rows = prune_by_indices(rows, keep_indices)
    write_csv_rows(action_file, new_rows, fieldnames)

    if metadata_file and metadata_file.is_file():
        try_prune_sidecar_csv(metadata_file, keep_indices, n, backup_root, apply=True)
    if raw_txt_file and raw_txt_file.is_file():
        try_prune_sidecar_txt(raw_txt_file, keep_indices, n, backup_root, apply=True)

    deleted = 0
    for p in removed_images:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f'  [警告] 删除图片失败 {p}: {e}')

    print(f'  [完成] 已写回 CSV，并删除图片 {deleted}/{len(removed_images)} 张')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='清洗 dataset_episodes 中长静止片段，并同步删除对应图片')
    parser.add_argument('--root', type=str, required=True, help='dataset_episodes 根目录，或单个 episode 目录')
    parser.add_argument('--apply', action='store_true', help='真正执行；默认只 dry-run')
    parser.add_argument('--debug-find', action='store_true', help='打印找到的候选 episode 目录，便于排查路径问题')

    parser.add_argument('--action-rel', type=str, default='trajectory/sequence_debug.csv', help='相对 episode 目录的动作 CSV 路径')
    parser.add_argument('--image-rel', type=str, default='camera/rgb_224', help='相对 episode 目录的图片目录路径')
    parser.add_argument('--metadata-rel', type=str, default='camera/metadata.CSV', help='相对 episode 目录的相机 metadata CSV 路径；不存在会自动跳过')
    parser.add_argument('--raw-txt-rel', type=str, default='trajectory/sequence_raw.txt', help='相对 episode 目录的原始 txt 路径；若每行对应一帧则同步裁剪')
    parser.add_argument('--episode-prefix', type=str, default='episode_', help='episode 目录前缀，默认 episode_；不想限制就传空字符串')

    parser.add_argument('--trans-thresh', type=float, default=0.0025, help='近静止判定：平移阈值（默认 0.0025 米）')
    parser.add_argument('--rot-thresh', type=float, default=0.03, help='近静止判定：旋转阈值（默认 0.03 rad）')
    parser.add_argument('--grip-stable-thresh', type=float, default=1e-6, help='近静止判定：夹爪变化阈值，小于此值视为夹爪未变化')
    parser.add_argument('--grip-event-thresh', type=float, default=0.05, help='判断夹爪发生明显变化的阈值')
    parser.add_argument('--min-static-len', type=int, default=8, help='连续静止长度达到该值才裁剪（默认 8 帧）')
    parser.add_argument('--keep-static-frames', type=int, default=2, help='普通长静止段保留前几帧（默认 2）')
    parser.add_argument('--fps', type=float, default=10.0, help='采样频率，用于把保护秒数换算成帧数（默认 10Hz）')
    parser.add_argument('--protect-after-grasp-seconds', type=float, default=5, help='夹爪发生明显变化后，保护多少秒内的帧不删除（默认 3 秒）')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f'目录不存在: {root}')

    episode_dirs = collect_episode_dirs(root, args.action_rel, args.image_rel, args.episode_prefix, recursive=True)
    if args.debug_find:
        print(f'[DEBUG] root = {root}')
        print(f'[DEBUG] action_rel = {args.action_rel}')
        print(f'[DEBUG] image_rel = {args.image_rel}')
        if episode_dirs:
            print('[DEBUG] 找到的候选 episode:')
            for ep in episode_dirs:
                print(f'  - {ep}')
        else:
            print('[DEBUG] 没有找到候选 episode。')
            print('[DEBUG] 你可以手动检查：')
            print(f'  find {root} -path "*/{args.action_rel}"')
            print(f'  find {root} -path "*/{args.image_rel}"')

    if not episode_dirs:
        print('未找到任何可处理的 episode 目录。')
        print(f'请检查是否存在：*/{args.action_rel} 和 */{args.image_rel}')
        return

    print(f'找到 {len(episode_dirs)} 个 episode 目录')
    print(f"{'执行模式: APPLY' if args.apply else '执行模式: DRY-RUN'}")
    print(f'参数: trans_thresh={args.trans_thresh}, rot_thresh={args.rot_thresh}, min_static_len={args.min_static_len}, keep_static_frames={args.keep_static_frames}, fps={args.fps}, protect_after_grasp_seconds={args.protect_after_grasp_seconds}')
    print(f'主 CSV: {args.action_rel}')
    print(f'图片目录: {args.image_rel}')
    print()

    for ep_dir in episode_dirs:
        process_episode(
            ep_dir=ep_dir,
            action_rel=args.action_rel,
            image_rel=args.image_rel,
            metadata_rel=args.metadata_rel,
            raw_txt_rel=args.raw_txt_rel,
            apply=args.apply,
            trans_thresh=args.trans_thresh,
            rot_thresh=args.rot_thresh,
            grip_stable_thresh=args.grip_stable_thresh,
            grip_event_thresh=args.grip_event_thresh,
            min_static_len=args.min_static_len,
            keep_static_frames=args.keep_static_frames,
            fps=args.fps,
            protect_after_grasp_seconds=args.protect_after_grasp_seconds,
        )
        print()

    print('全部处理完成。')


if __name__ == '__main__':
    main()
