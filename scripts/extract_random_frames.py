"""Extract random frames from videos in the data folder.

This script:
1. Finds all episode directories in the data folder
2. For each episode, selects ONE video (front/top preferred, otherwise the only one)
3. Extracts a random frame between 1/2 and 2/3 of the video
4. Saves the frames to a new 'fig' folder

Usage:
    python extract_random_frames.py
"""

import random
from pathlib import Path

import cv2
from loguru import logger


def select_video_from_episode(episode_dir: Path) -> Path | None:
    """Select the best video from an episode directory.
    
    Priority:
    1. front_rgb.mp4 (YAM datasets)
    2. shoulder_view.mp4 (franka_data - top view)
    3. Any single .mp4 file (lerobot_annotations)
    4. First video containing 'front' or 'top' in name
    
    Returns:
        Path to the selected video, or None if no video found
    """
    videos = list(episode_dir.glob("*.mp4"))
    
    if not videos:
        return None
    
    # If only one video, use it
    if len(videos) == 1:
        return videos[0]
    
    # Priority order for video selection
    priority_names = ["front_rgb.mp4", "shoulder_view.mp4", "front.mp4", "top.mp4"]
    
    for name in priority_names:
        for video in videos:
            if video.name == name:
                return video
    
    # Check for videos containing 'front' or 'top'
    for video in videos:
        if "front" in video.name.lower() or "top" in video.name.lower():
            return video
    
    # Fallback: return first video
    return videos[0]


def find_episode_directories(data_dir: Path) -> list[Path]:
    """Find all episode directories that contain .mp4 files."""
    episodes = []
    
    # Find all directories that directly contain .mp4 files
    for mp4_file in data_dir.rglob("*.mp4"):
        episode_dir = mp4_file.parent
        if episode_dir not in episodes:
            # Skip .cache directories
            if ".cache" not in str(episode_dir):
                episodes.append(episode_dir)
    
    return episodes


def extract_random_frame(
    video_path: Path,
    output_dir: Path,
    episode_name: str,
    min_ratio: float = 0.5,
    max_ratio: float = 0.666,
) -> Path | None:
    """Extract a random frame from a video between min_ratio and max_ratio of total frames.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the extracted frame
        episode_name: Name for the output file
        min_ratio: Minimum position ratio (default: 0.5 = 1/2)
        max_ratio: Maximum position ratio (default: 0.666 = 2/3)

    Returns:
        Path to the saved frame, or None if extraction failed
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        logger.error(f"Cannot get frame count for: {video_path}")
        cap.release()
        return None

    # Calculate random frame position between 1/2 and 2/3
    min_frame = int(total_frames * min_ratio)
    max_frame = int(total_frames * max_ratio)
    if min_frame >= max_frame:
        max_frame = min_frame + 1
    target_frame = random.randint(min_frame, max_frame)

    # Seek to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error(f"Cannot read frame {target_frame} from: {video_path}")
        return None

    # Create output filename
    output_path = output_dir / f"{episode_name}.png"

    cv2.imwrite(str(output_path), frame)
    logger.info(f"Saved frame {target_frame}/{total_frames} -> {output_path.name}")

    return output_path


def main() -> None:
    """Main entry point."""
    # Get the project root (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    data_dir = project_root / "data"
    output_dir = project_root / "fig"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Find all episode directories
    episodes = find_episode_directories(data_dir)
    logger.info(f"Found {len(episodes)} episode directories")

    if not episodes:
        logger.warning("No episodes found!")
        return

    # Extract frames
    success_count = 0
    failed_count = 0

    for episode_dir in episodes:
        # Select video
        video = select_video_from_episode(episode_dir)
        if video is None:
            logger.warning(f"No video found in: {episode_dir}")
            failed_count += 1
            continue
        
        # Create episode name from path
        # e.g., data/lerobot_annotations/task/date -> lerobot_annotations_task_date
        try:
            relative_path = episode_dir.relative_to(data_dir)
            episode_name = str(relative_path).replace("/", "_")
        except ValueError:
            episode_name = episode_dir.name
        
        result = extract_random_frame(video, output_dir, episode_name)
        if result:
            success_count += 1
        else:
            failed_count += 1

    logger.info("=" * 60)
    logger.info(f"Extraction complete!")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
