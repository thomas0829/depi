#!/usr/bin/env python3
"""
Create a scattered photo collage effect for papers.
Photos appear randomly rotated and overlapping like they were dropped on a table.
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import argparse
from loguru import logger


def add_polaroid_frame(img: Image.Image, border_size: int = 15, bottom_extra: int = 40) -> Image.Image:
    """Add a polaroid-style white frame around the image."""
    # Create new image with white background
    new_width = img.width + border_size * 2
    new_height = img.height + border_size + bottom_extra
    framed = Image.new('RGB', (new_width, new_height), 'white')
    framed.paste(img, (border_size, border_size))
    return framed


def add_shadow(img: Image.Image, offset: tuple = (8, 8), blur_radius: int = 15, opacity: int = 100) -> Image.Image:
    """Add a drop shadow to the image."""
    # Create a larger canvas for shadow
    shadow_offset_x, shadow_offset_y = offset
    new_width = img.width + abs(shadow_offset_x) + blur_radius * 2
    new_height = img.height + abs(shadow_offset_y) + blur_radius * 2
    
    # Create shadow
    shadow = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
    shadow_layer = Image.new('RGBA', img.size, (0, 0, 0, opacity))
    
    # Position shadow
    shadow_x = blur_radius + max(0, shadow_offset_x)
    shadow_y = blur_radius + max(0, shadow_offset_y)
    shadow.paste(shadow_layer, (shadow_x, shadow_y))
    
    # Blur shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # Paste original image on top
    img_x = blur_radius + max(0, -shadow_offset_x)
    img_y = blur_radius + max(0, -shadow_offset_y)
    
    # Convert img to RGBA if needed
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    shadow.paste(img, (img_x, img_y), img if img.mode == 'RGBA' else None)
    
    return shadow


def create_scattered_collage(
    image_paths: list,
    output_path: str,
    canvas_size: tuple = (2400, 1800),
    photo_size: tuple = (300, 225),
    num_photos: int = 25,
    rotation_range: tuple = (-25, 25),
    add_frame: bool = True,
    background_color: tuple = (240, 240, 235),  # Light beige/paper color
    seed: int = None
):
    """
    Create a scattered photo collage.
    
    Args:
        image_paths: List of image file paths
        output_path: Output file path
        canvas_size: Size of the output canvas (width, height)
        photo_size: Size to resize each photo to (width, height)
        num_photos: Number of photos to include
        rotation_range: Range of random rotation angles (min, max)
        add_frame: Whether to add polaroid-style frames
        background_color: RGB tuple for background color
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample photos
    if len(image_paths) > num_photos:
        selected_paths = random.sample(image_paths, num_photos)
    else:
        selected_paths = image_paths
    
    logger.info(f"Creating collage with {len(selected_paths)} photos")
    
    # Create canvas
    canvas = Image.new('RGBA', canvas_size, (*background_color, 255))
    
    # Load and process each photo
    photos = []
    for path in selected_paths:
        try:
            img = Image.open(path).convert('RGB')
            
            # Resize maintaining aspect ratio
            img.thumbnail((photo_size[0], photo_size[1]), Image.Resampling.LANCZOS)
            
            # Add polaroid frame if requested
            if add_frame:
                img = add_polaroid_frame(img)
            
            # Add shadow
            img = add_shadow(img)
            
            photos.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    
    # Place photos randomly on canvas
    margin = 100
    for photo in photos:
        # Random rotation
        angle = random.uniform(rotation_range[0], rotation_range[1])
        rotated = photo.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        
        # Random position
        max_x = canvas_size[0] - rotated.width + margin
        max_y = canvas_size[1] - rotated.height + margin
        x = random.randint(-margin, max(0, max_x))
        y = random.randint(-margin, max(0, max_y))
        
        # Paste with alpha channel
        canvas.paste(rotated, (x, y), rotated)
    
    # Convert to RGB for saving
    final = Image.new('RGB', canvas_size, background_color)
    final.paste(canvas, mask=canvas.split()[3])
    
    # Save
    final.save(output_path, quality=95)
    logger.info(f"Saved collage to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create a scattered photo collage")
    parser.add_argument("--input-dir", type=str, default="fig", help="Directory containing images")
    parser.add_argument("--output", type=str, default="collage.png", help="Output file path")
    parser.add_argument("--num-photos", type=int, default=30, help="Number of photos to include")
    parser.add_argument("--canvas-width", type=int, default=2400, help="Canvas width")
    parser.add_argument("--canvas-height", type=int, default=1800, help="Canvas height")
    parser.add_argument("--photo-width", type=int, default=280, help="Photo width")
    parser.add_argument("--photo-height", type=int, default=210, help="Photo height")
    parser.add_argument("--no-frame", action="store_true", help="Don't add polaroid frames")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--filter", type=str, default=None, help="Filter images by keyword")
    
    args = parser.parse_args()
    
    # Find all images
    input_dir = Path(args.input_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_paths = [
        str(p) for p in input_dir.glob("*") 
        if p.suffix.lower() in image_extensions
    ]
    
    # Apply filter if specified
    if args.filter:
        image_paths = [p for p in image_paths if args.filter.lower() in p.lower()]
    
    logger.info(f"Found {len(image_paths)} images in {input_dir}")
    
    if not image_paths:
        logger.error("No images found!")
        return
    
    create_scattered_collage(
        image_paths=image_paths,
        output_path=args.output,
        canvas_size=(args.canvas_width, args.canvas_height),
        photo_size=(args.photo_width, args.photo_height),
        num_photos=args.num_photos,
        add_frame=not args.no_frame,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
