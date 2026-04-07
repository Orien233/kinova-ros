# Tools: Dataset Collection and Playback Workflow

This `tools/` directory contains utility scripts for teaching, replaying, recording, and cleaning Kinova dataset episodes.

## Scripts

- `teach_sequence_session.py`  
  Interactive teaching-session controller. Creates timestamped sessions under `tools/teach_sessions/`, restores robot state, and launches `sequence_recorder.py`.

- `sequence_recorder.py`  
  Records robot trajectory and writes synchronized outputs such as:
  - `sequence.txt` (normalized deltas)
  - `sequence_raw.txt` (physical deltas)
  - `sequence_pose.txt` (absolute poses)
  - `sequence_debug.csv`
  - `sequence_stats.json`

- `sequence_player_position_verified.py`  
  Position-based playback with goal verification/retry. Preferred player for high-fidelity replay.

- `sequence_player_position_cumulative.py`  
  Alternative position player that pre-accumulates deltas relative to an initial reference pose.

- `finger_telep_oc.py`  
  Reusable gripper/mode-switch helper for teleoperation flows.

- `record_realsense_rgb_224.py`  
  RealSense RGB capture pipeline with center crop and resize to `224x224`.

- `auto_dataset_recorder.py`  
  End-to-end orchestration for one episode: select sequence -> replay -> record trajectory -> record camera -> write episode folder.

- `clean_static_segments_dataset_episodes_v2.py`  
  Post-processing utility to remove long static segments and synchronize updates across images and sidecar files.

## Script Collaboration Graph

1. **Teaching stage**
   - `teach_sequence_session.py` -> starts `sequence_recorder.py`
   - `teach_sequence_session.py` -> uses `finger_telep_oc.py`
   - output saved in `tools/teach_sessions/<timestamp>/trajectory/`

2. **Automated dataset episode stage**
   - `auto_dataset_recorder.py` -> reads latest teach sequence (unless `--raw-sequence` is specified)
   - `auto_dataset_recorder.py` -> runs `sequence_player_position_verified.py` (default player)
   - `auto_dataset_recorder.py` -> runs `sequence_recorder.py`
   - `auto_dataset_recorder.py` -> runs `record_realsense_rgb_224.py`
   - output saved in `tools/dataset_episodes/<episode_timestamp>/`

3. **Cleanup stage**
   - `clean_static_segments_dataset_episodes_v2.py` -> processes `tools/dataset_episodes/` and synchronizes action/image pruning

## Path and Privacy Notes

- Default paths are now script-relative (resolved from `tools/`) instead of user-specific absolute paths.
- You can still override paths via CLI arguments when needed.

## Quick Examples

```bash
# 1) Run interactive teaching
python tools/teach_sequence_session.py

# 2) Record one automated dataset episode
python tools/auto_dataset_recorder.py

# 3) Dry-run cleanup of static segments
python3 tools/clean_static_segments_dataset_episodes_v2.py --root tools/dataset_episodes --debug-find

# 4) Apply cleanup changes
python3 tools/clean_static_segments_dataset_episodes_v2.py --root tools/dataset_episodes --apply
```
