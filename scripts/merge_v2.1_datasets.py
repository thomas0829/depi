#!/usr/bin/env python3
"""
Merge two LeRobot datasets
Dataset 1: pick_up_the_toy_car_and_put_it_into_the_narrow_box (41 episodes)
Dataset 2: pick_up_the_toy_car_and_put_it_into_the_narrow_box_ver2 (9 episodes)
Merged: 50 episodes
"""

import json
import shutil
from pathlib import Path
import pandas as pd
import os
import argparse

# Path settings
_HF_LEROBOT_HOME = os.environ.get("HF_LEROBOT_HOME")
if _HF_LEROBOT_HOME:
    # Derive base path from HF_LEROBOT_HOME for portability
    BASE_PATH = Path(_HF_LEROBOT_HOME) / "thomas0829"
else:
    # Fallback to original hard-coded default for backward compatibility
    BASE_PATH = Path.home() / ".cache/huggingface/lerobot/thomas0829"

DS1_PATH = BASE_PATH / "eval_car_box_depi_v1"
DS2_PATH = BASE_PATH / "eval_car_box_depi_v2"
OUTPUT_PATH = BASE_PATH / "eval_car_box_depi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two LeRobot datasets.")
    parser.add_argument(
        "--ds1",
        type=Path,
        default=DS1_PATH,
        help="Path to dataset 1 (default: %(default)s)",
    )
    parser.add_argument(
        "--ds2",
        type=Path,
        default=DS2_PATH,
        help="Path to dataset 2 (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output dataset path (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    global DS1_PATH, DS2_PATH, OUTPUT_PATH

    args = parse_args()
    DS1_PATH = args.ds1
    DS2_PATH = args.ds2
    OUTPUT_PATH = args.output
    print("=" * 60)
    print("Merge LeRobot Datasets")
    print("=" * 60)
    
    # Check source datasets
    if not DS1_PATH.exists():
        print(f"Error: Dataset 1 not found: {DS1_PATH}")
        return
    if not DS2_PATH.exists():
        print(f"Error: Dataset 2 not found: {DS2_PATH}")
        return
    
    # Read info.json
    with open(DS1_PATH / "meta/info.json") as f:
        info1 = json.load(f)
    with open(DS2_PATH / "meta/info.json") as f:
        info2 = json.load(f)
    
    ds1_episodes = info1["total_episodes"]
    ds2_episodes = info2["total_episodes"]
    ds1_frames = info1["total_frames"]
    ds2_frames = info2["total_frames"]
    
    print(f"\nDataset 1: {ds1_episodes} episodes, {ds1_frames} frames")
    print(f"Dataset 2: {ds2_episodes} episodes, {ds2_frames} frames")
    print(f"After merge: {ds1_episodes + ds2_episodes} episodes, {ds1_frames + ds2_frames} frames")
    
    # Create output directory
    if OUTPUT_PATH.exists():
        response = input(f"\nOutput directory exists: {OUTPUT_PATH}\nDelete and recreate? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
        shutil.rmtree(OUTPUT_PATH)
    
    OUTPUT_PATH.mkdir(parents=True)
    (OUTPUT_PATH / "data/chunk-000").mkdir(parents=True)
    (OUTPUT_PATH / "videos/chunk-000/observation.images.top").mkdir(parents=True)
    (OUTPUT_PATH / "videos/chunk-000/observation.images.wrist").mkdir(parents=True)
    (OUTPUT_PATH / "meta").mkdir(parents=True)
    
    print("\nMerging...")
    
    # 1. Copy dataset 1 parquet and videos (keep original numbering)
    print("\n[1/5] Copying dataset 1 parquet files...")
    for i in range(ds1_episodes):
        src = DS1_PATH / f"data/chunk-000/episode_{i:06d}.parquet"
        dst = OUTPUT_PATH / f"data/chunk-000/episode_{i:06d}.parquet"
        if src.exists():
            shutil.copy2(src, dst)
    
    print("[2/5] Copying dataset 1 video files...")
    for i in range(ds1_episodes):
        for cam in ["observation.images.top", "observation.images.wrist"]:
            src = DS1_PATH / f"videos/chunk-000/{cam}/episode_{i:06d}.mp4"
            dst = OUTPUT_PATH / f"videos/chunk-000/{cam}/episode_{i:06d}.mp4"
            if src.exists():
                shutil.copy2(src, dst)
    
    # 2. Copy dataset 2 parquet (need to update episode_index and index)
    print("[3/5] Copying and adjusting dataset 2 parquet files...")
    cumulative_index = ds1_frames  # Continue from dataset 1's last index
    
    for i in range(ds2_episodes):
        src = DS2_PATH / f"data/chunk-000/episode_{i:06d}.parquet"
        new_episode_idx = ds1_episodes + i
        dst = OUTPUT_PATH / f"data/chunk-000/episode_{new_episode_idx:06d}.parquet"
        
        if src.exists():
            df = pd.read_parquet(src)
            # Update episode_index
            df["episode_index"] = new_episode_idx
            # Update index (globally unique)
            num_frames = len(df)
            df["index"] = range(cumulative_index, cumulative_index + num_frames)
            cumulative_index += num_frames
            df.to_parquet(dst, index=False)
    
    print("[4/5] Copying dataset 2 video files (renumbered)...")
    for i in range(ds2_episodes):
        new_episode_idx = ds1_episodes + i
        for cam in ["observation.images.top", "observation.images.wrist"]:
            src = DS2_PATH / f"videos/chunk-000/{cam}/episode_{i:06d}.mp4"
            dst = OUTPUT_PATH / f"videos/chunk-000/{cam}/episode_{new_episode_idx:06d}.mp4"
            if src.exists():
                shutil.copy2(src, dst)
    
    # 3. Merge meta files
    print("[5/5] Merging meta files...")
    
    # info.json
    merged_info = info1.copy()
    merged_info["total_episodes"] = ds1_episodes + ds2_episodes
    merged_info["total_frames"] = ds1_frames + ds2_frames
    merged_info["total_videos"] = (ds1_episodes + ds2_episodes) * 2
    merged_info["splits"]["train"] = f"0:{ds1_episodes + ds2_episodes}"
    
    with open(OUTPUT_PATH / "meta/info.json", "w") as f:
        json.dump(merged_info, f, indent=4)
    
    # tasks.jsonl (should be the same, just copy one)
    shutil.copy2(DS1_PATH / "meta/tasks.jsonl", OUTPUT_PATH / "meta/tasks.jsonl")
    
    # episodes.jsonl (merge)
    with open(OUTPUT_PATH / "meta/episodes.jsonl", "w") as out_f:
        # Copy dataset 1 episodes
        with open(DS1_PATH / "meta/episodes.jsonl") as f:
            for line in f:
                out_f.write(line)
        
        # Copy dataset 2 episodes (adjust episode_index)
        with open(DS2_PATH / "meta/episodes.jsonl") as f:
            for line in f:
                ep = json.loads(line)
                ep["episode_index"] = ep["episode_index"] + ds1_episodes
                out_f.write(json.dumps(ep) + "\n")
    
    # episodes_stats.jsonl (if exists)
    stats_path1 = DS1_PATH / "meta/episodes_stats.jsonl"
    stats_path2 = DS2_PATH / "meta/episodes_stats.jsonl"
    if stats_path1.exists() and stats_path2.exists():
        with open(OUTPUT_PATH / "meta/episodes_stats.jsonl", "w") as out_f:
            with open(stats_path1) as f:
                for line in f:
                    out_f.write(line)
            with open(stats_path2) as f:
                for line in f:
                    stat = json.loads(line)
                    stat["episode_index"] = stat["episode_index"] + ds1_episodes
                    out_f.write(json.dumps(stat) + "\n")
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total episodes: {ds1_episodes + ds2_episodes}")
    print(f"Total frames: {ds1_frames + ds2_frames}")
    print("=" * 60)
    
    # Verify
    print("\nVerifying merged result...")
    parquet_count = len(list((OUTPUT_PATH / "data/chunk-000").glob("*.parquet")))
    top_video_count = len(list((OUTPUT_PATH / "videos/chunk-000/observation.images.top").glob("*.mp4")))
    wrist_video_count = len(list((OUTPUT_PATH / "videos/chunk-000/observation.images.wrist").glob("*.mp4")))
    
    print(f"  Parquet files: {parquet_count}")
    print(f"  Top camera videos: {top_video_count}")
    print(f"  Wrist camera videos: {wrist_video_count}")
    
    if parquet_count == ds1_episodes + ds2_episodes and top_video_count == ds1_episodes + ds2_episodes:
        print("\nVerification passed!")
    else:
        print("\nWarning: File count mismatch, please check")

if __name__ == "__main__":
    main()
