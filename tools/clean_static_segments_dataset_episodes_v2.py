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
        print(f'  [WARN] Failed to read CSV {path.name}: {e}')
        return
    if len(rows) != expected_len:
        print(f'  [Skip sidecar CSV] {path.name}: rows={len(rows)}，vs main sequence={expected_len} mismatch')
        return
    if not apply:
        print(f'  [dry-run] Will prune CSV in sync: {path}')
        return
    backup_file(path, backup_root)
    new_rows = prune_by_indices(rows, keep_indices)
    write_csv_rows(path, new_rows, fieldnames)
    print(f'  [CSV pruned] {path}')


def try_prune_sidecar_txt(path: Path, keep_indices: List[int], expected_len: int, backup_root: Path, apply: bool) -> None:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
    except Exception as e:
        print(f'  [WARN] Failed to read TXT {path.name}: {e}')
        return
    if len(lines) != expected_len:
        print(f'  [Skip sidecar TXT] {path.name}: rows={len(lines)}，vs main sequence={expected_len} mismatch')
        return
    if not apply:
        print(f'  [dry-run] Will prune TXT in sync: {path}')
        return
    backup_file(path, backup_root)
    new_lines = prune_by_indices(lines, keep_indices)
    path.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
    print(f'  [TXT pruned] {path}')


def is_episode_dir(ep_dir: Path, action_rel: str, image_rel: str) -> bool:
    return (ep_dir / action_rel).is_file() and (ep_dir / image_rel).is_dir()


def collect_episode_dirs(root: Path, action_rel: str, image_rel: str, episode_prefix: str, recursive: bool = True) -> List[Path]:
    """
    Improvements:
    1. 1. Check whether root itself is an episode
    2. 2. Check direct child directories under root
    3. 3. Recursively search trajectory/sequence_debug.csv and infer episode dirs
    This still works when dataset_episodes has nested directories.
    """
    found: List[Path] = []
    seen = set()

    def add_candidate(ep: Path):
        rp = str(ep.resolve())
        if rp in seen:
            return
        if episode_prefix and episode_prefix not in ep.name:
            # No hard requirement to start with episode_, but keep weak filtering;
            # pass --episode-prefix "" to disable filtering
            pass
        if is_episode_dir(ep, action_rel, image_rel):
            found.append(ep)
            seen.add(rp)

    add_candidate(root)

    for p in sorted([x for x in root.iterdir() if x.is_dir()], key=lambda x: natural_key(x.name)):
        if episode_prefix and not p.name.startswith(episode_prefix):
            # Keep prefix filtering at direct child level
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
        print(f'[Skip] {ep_dir}: action file not found -> {action_file}')
        return
    if not image_dir.is_dir():
        print(f'[Skip] {ep_dir}: image directory not found -> {image_dir}')
        return

    images = list_images_sorted(image_dir)
    if not images:
        print(f'[Skip] {ep_dir}: image directory is empty -> {image_dir}')
        return

    try:
        rows, fieldnames = read_csv_rows(action_file)
    except Exception as e:
        print(f'[Skip] {ep_dir}: failed to parse action CSV -> {e}')
        return

    n = len(rows)
    if len(images) != n:
        print(f'[Skip] {ep_dir}: image count and sequence_debug.csv rows differ; cannot safely prune in sync. csv_rows={n}, images={len(images)}')
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
        print(f'[Keep] {ep_dir.name}: no long static segment to remove (detected gripper events: {len(grasp_event_indices)}  )')
        return

    keep_indices = [i for i in range(n) if i not in set(drop_indices)]
    removed_images = [images[i] for i in drop_indices]

    print(f'[Detected] {ep_dir.name}: total={n}, drop={len(drop_indices)}, keep={len(keep_indices)}, gripper_events={len(grasp_event_indices)}')

    if not apply:
        preview = drop_indices[:20]
        more = ' ...' if len(drop_indices) > 20 else ''
        print(f'  dry-run: indices to remove {preview}{more}')
        if metadata_file and metadata_file.is_file():
            print(f'  dry-run: will prune CSV -> {metadata_file}')
        if raw_txt_file and raw_txt_file.is_file():
            print(f'  dry-run: will attempt TXT prune -> {raw_txt_file}')
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
            print(f'  [WARN] Failed to delete image {p}: {e}')

    print(f'  [Done] CSV written back, images deleted {deleted}/{len(removed_images)} ')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clean long static segments in dataset_episodes and synchronously delete corresponding images')
    parser.add_argument('--root', type=str, required=True, help='dataset_episodes root, or a single episode directory')
    parser.add_argument('--apply', action='store_true', help='Apply changes; default is dry-run only')
    parser.add_argument('--debug-find', action='store_true', help='Print discovered episode candidates for path debugging')

    parser.add_argument('--action-rel', type=str, default='trajectory/sequence_debug.csv', help='Action CSV path relative to episode directory')
    parser.add_argument('--image-rel', type=str, default='camera/rgb_224', help='Image directory path relative to episode directory')
    parser.add_argument('--metadata-rel', type=str, default='camera/metadata.CSV', help='Camera metadata CSV path relative to episode directory; skipped if missing')
    parser.add_argument('--raw-txt-rel', type=str, default='trajectory/sequence_raw.txt', help='Raw txt path relative to episode directory; pruned in sync when line count matches frames')
    parser.add_argument('--episode-prefix', type=str, default='episode_', help='Episode prefix (default episode_); pass empty string to disable')

    parser.add_argument('--trans-thresh', type=float, default=0.0025, help='Near-static threshold: translation (default 0.0025 m)')
    parser.add_argument('--rot-thresh', type=float, default=0.03, help='Near-static threshold: rotation (default 0.03 rad)')
    parser.add_argument('--grip-stable-thresh', type=float, default=1e-6, help='Near-static threshold: gripper delta; below this value is considered unchanged')
    parser.add_argument('--grip-event-thresh', type=float, default=0.05, help='Threshold for significant gripper change events')
    parser.add_argument('--min-static-len', type=int, default=8, help='Prune only if static segment length reaches this value (default 8 frames)')
    parser.add_argument('--keep-static-frames', type=int, default=2, help='Keep first N frames in normal long static segments (default 2)')
    parser.add_argument('--fps', type=float, default=10.0, help='Sampling rate to convert protection seconds into frames (default 10Hz)')
    parser.add_argument('--protect-after-grasp-seconds', type=float, default=5, help='After significant gripper change, protect frames in this many seconds from deletion (default 3s)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f'Directory does not exist: {root}')

    episode_dirs = collect_episode_dirs(root, args.action_rel, args.image_rel, args.episode_prefix, recursive=True)
    if args.debug_find:
        print(f'[DEBUG] root = {root}')
        print(f'[DEBUG] action_rel = {args.action_rel}')
        print(f'[DEBUG] image_rel = {args.image_rel}')
        if episode_dirs:
            print('[DEBUG] Discovered episode candidates:')
            for ep in episode_dirs:
                print(f'  - {ep}')
        else:
            print('[DEBUG] No episode candidates found.')
            print('[DEBUG] You can manually check:')
            print(f'  find {root} -path "*/{args.action_rel}"')
            print(f'  find {root} -path "*/{args.image_rel}"')

    if not episode_dirs:
        print('No processable episode directory found.')
        print(f'Please check existence of:*/{args.action_rel}  and  */{args.image_rel}')
        return

    print(f'Found {len(episode_dirs)} episode directories')
    print(f"{'Mode: APPLY' if args.apply else 'Mode: DRY-RUN'}")
    print(f'Parameters: trans_thresh={args.trans_thresh}, rot_thresh={args.rot_thresh}, min_static_len={args.min_static_len}, keep_static_frames={args.keep_static_frames}, fps={args.fps}, protect_after_grasp_seconds={args.protect_after_grasp_seconds}')
    print(f'Main CSV: {args.action_rel}')
    print(f'Image dir: {args.image_rel}')
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

    print('All processing complete.')


if __name__ == '__main__':
    main()
