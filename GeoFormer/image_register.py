#!/usr/bin/env python3
"""
Register FIRE dataset images and save visualizations
"""

import argparse
import os
import cv2
import numpy as np
import sys

# Add current directory to path for imports
sys.path.insert(0, '.')

import eval_tool.immatch as immatch
from eval_tool.immatch.utils.model_helper import parse_model_config


def save_registered_images(im1, im2, H, save_dir, pair_name):
    """
    Save registered images with multiple visualization types
    
    Args:
        im1: First image (to be warped)
        im2: Second image (reference)
        H: Homography matrix (3x3)
        save_dir: Directory to save results
        pair_name: Name of the image pair
    
    Returns:
        bool: True if successful, False otherwise
    """
    if H is None:
        print(f"  WARNING: No homography for {pair_name}")
        return False
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dimensions from reference image
    h, w = im2.shape[:2]
    
    # Warp image 1 to align with image 2
    registered_im1 = cv2.warpPerspective(im1, H, (w, h))
    
    # Save registered image
    registered_path = os.path.join(save_dir, f'{pair_name}_registered.png')
    cv2.imwrite(registered_path, registered_im1)
    print(f"  ✓ Saved: {os.path.basename(registered_path)}")
    
    # Create overlay visualization (red-green)
    overlay_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Convert to grayscale if needed
    if len(registered_im1.shape) == 3:
        reg_gray = cv2.cvtColor(registered_im1, cv2.COLOR_BGR2GRAY)
    else:
        reg_gray = registered_im1
        
    if len(im2.shape) == 3:
        ref_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = im2
    
    # Red channel: registered image, Green channel: reference image
    overlay_img[:, :, 2] = reg_gray  # Red
    overlay_img[:, :, 1] = ref_gray  # Green
    # Overlap appears as yellow
    
    overlay_path = os.path.join(save_dir, f'{pair_name}_overlay.png')
    cv2.imwrite(overlay_path, overlay_img)
    
    # Create checkerboard pattern
    checker_size = 50
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                mask[i:i+checker_size, j:j+checker_size] = 1
    
    checker_img = np.zeros((h, w, 3), dtype=np.uint8)
    if len(im2.shape) == 3:
        checker_img[mask == 1] = im2[mask == 1]
        checker_img[mask == 0] = registered_im1[mask == 0]
    else:
        im2_color = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
        reg_color = cv2.cvtColor(registered_im1, cv2.COLOR_GRAY2BGR)
        checker_img[mask == 1] = im2_color[mask == 1]
        checker_img[mask == 0] = reg_color[mask == 0]
    
    checker_path = os.path.join(save_dir, f'{pair_name}_checkerboard.png')
    cv2.imwrite(checker_path, checker_img)
    
    # Create side-by-side comparison
    if len(im2.shape) == 2:
        im2_display = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
    else:
        im2_display = im2.copy()
    
    if len(registered_im1.shape) == 2:
        reg_display = cv2.cvtColor(registered_im1, cv2.COLOR_GRAY2BGR)
    else:
        reg_display = registered_im1.copy()
    
    # Resize if too large
    max_width = 1920
    if w * 2 > max_width:
        scale = max_width / (w * 2)
        new_w = int(w * scale)
        new_h = int(h * scale)
        im2_display = cv2.resize(im2_display, (new_w, new_h))
        reg_display = cv2.resize(reg_display, (new_w, new_h))
    
    sidebyside = np.hstack([reg_display, im2_display])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sidebyside, 'Registered', (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(sidebyside, 'Reference', (reg_display.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
    
    sidebyside_path = os.path.join(save_dir, f'{pair_name}_comparison.png')
    cv2.imwrite(sidebyside_path, sidebyside)
    
    return True


def register_fire_images(
    root_dir='.',
    config='geoformer',
    h_solver='cv',
    ransac_thres=15,
    match_thres=None,
    odir='outputs/fire_registered',
    max_pairs=None,
):
    """
    Register FIRE dataset images using GeoFormer or other matching methods
    """
    
    print("=" * 80)
    print("FIRE Dataset Image Registration")
    print("=" * 80)
    
    # Setup paths
    data_root = os.path.join(root_dir, 'data/datasets/FIRE')
    gt_dir = os.path.join(data_root, 'Ground Truth')
    im_dir = os.path.join(data_root, 'Images')
    
    # Check if dataset exists
    if not os.path.exists(data_root):
        print(f"\nERROR: FIRE dataset not found at: {data_root}")
        print("Expected structure:")
        print("  data/datasets/FIRE/")
        print("  ├── Ground Truth/")
        print("  └── Images/")
        return
    
    if not os.path.exists(gt_dir) or not os.path.exists(im_dir):
        print(f"\nERROR: Missing subdirectories in FIRE dataset")
        return
    
    # Get image pairs
    match_pairs = sorted([x for x in os.listdir(gt_dir) if x.endswith('.txt')])
    
    if max_pairs:
        match_pairs = match_pairs[:max_pairs]
        print(f"\nProcessing first {max_pairs} image pairs (out of {len(match_pairs)} total)")
    else:
        print(f"\nFound {len(match_pairs)} image pairs")
    
    # Parse model config
    print(f"\nLoading model configuration: {config}")
    try:
        args = parse_model_config(config, 'fire', root_dir)
        class_name = args['class']
        print(f"Model class: {class_name}")
    except Exception as e:
        print(f"\nERROR loading config: {e}")
        return
    
    # Set matching threshold
    if match_thres is not None:
        args['match_threshold'] = match_thres
        print(f"Matching threshold: {match_thres}")
    else:
        print(f"Matching threshold: {args.get('match_threshold', 'default')}")
    
    # Create output directory
    output_dir = os.path.join(root_dir, odir, class_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize model
    print(f"\nInitializing {class_name}...")
    try:
        model = immatch.__dict__[class_name](args)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        return
    
    # Process each pair
    print(f"\n{'='*80}")
    print("Processing image pairs...")
    print(f"{'='*80}\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, pair_file in enumerate(match_pairs, 1):
        pair_name = pair_file.replace('.txt', '')
        print(f"[{idx}/{len(match_pairs)}] {pair_name}")
        
        # Parse image names from filename
        # Format: control_points_XXX_Y_Z.txt where XXX is image ID, Y and Z are view numbers
        parts = pair_name.split('_')
        
        # Extract the base image ID and view numbers
        file_name = pair_file.replace('.txt', '')
        gt_file = os.path.join(gt_dir, pair_file)
        
        refer = file_name.split('_')[2] + '_' + file_name.split('_')[3]
        query = file_name.split('_')[2] + '_' + file_name.split('_')[4]
        
        im1_path = os.path.join(im_dir, query + '.jpg')
        im2_path = os.path.join(im_dir, refer + '.jpg')
        
        print(f"  Query:  {query}.jpg")
        print(f"  Refer:  {refer}.jpg")
        
        # Check if images exist
        if not os.path.exists(im1_path):
            print(f"  ERROR: Image not found: {im1_path}")
            fail_count += 1
            continue
        
        if not os.path.exists(im2_path):
            print(f"  ERROR: Image not found: {im2_path}")
            fail_count += 1
            continue
        
        # Load images
        im1 = cv2.imread(im1_path)
        im2 = cv2.imread(im2_path)
        
        if im1 is None or im2 is None:
            print(f"  ERROR: Failed to read images")
            fail_count += 1
            continue
        
        print(f"  Shapes: {im1.shape} -> {im2.shape}")
        
        # Perform matching
        try:
            print(f"  Matching...")
            match_result = model.match_pairs(im1_path, im2_path)
            
            # Extract matches
            if len(match_result) >= 3:
                matches = match_result[0]
                print(f"  Matches: {len(matches)}")
            else:
                print(f"  ERROR: Unexpected match result format")
                fail_count += 1
                continue
            
            # Estimate homography
            if len(matches) >= 4:
                print(f"  Estimating homography ({h_solver})...")
                
                if h_solver == 'cv':
                    H, inliers = cv2.findHomography(
                        matches[:, :2], matches[:, 2:4], 
                        cv2.RANSAC, ransac_thres
                    )
                elif h_solver == 'degensac':
                    import pydegensac
                    H, inliers = pydegensac.findHomography(
                        matches[:, :2], matches[:, 2:4], 
                        ransac_thres
                    )
                else:
                    print(f"  ERROR: Unknown solver: {h_solver}")
                    fail_count += 1
                    continue
                
                if H is not None:
                    inlier_count = np.sum(inliers) if inliers is not None else 0
                    print(f"  Inliers: {inlier_count}/{len(matches)}")
                    
                    # Save registered images
                    if save_registered_images(im1, im2, H, output_dir, pair_name):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    print(f"  FAILED: Could not estimate homography")
                    fail_count += 1
            else:
                print(f"  FAILED: Not enough matches (need >= 4)")
                fail_count += 1
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total pairs:     {len(match_pairs)}")
    print(f"Successful:      {success_count} ({100*success_count/len(match_pairs):.1f}%)")
    print(f"Failed:          {fail_count} ({100*fail_count/len(match_pairs):.1f}%)")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files for each pair:")
    print(f"  - *_registered.png     : Warped query image")
    print(f"  - *_overlay.png        : Red-green overlay (yellow = good alignment)")
    print(f"  - *_checkerboard.png   : Checkerboard pattern")
    print(f"  - *_comparison.png     : Side-by-side comparison")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Register FIRE dataset images using image matching methods',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID (default: 0)')
    parser.add_argument('--root_dir', type=str, default='.',
                        help='Root directory of project (default: current directory)')
    parser.add_argument('--config', type=str, default='geoformer',
                        help='Method configuration name (default: geoformer)')
    parser.add_argument('--h_solver', type=str, default='cv',
                        choices=['cv', 'degensac'],
                        help='Homography solver (default: cv)')
    parser.add_argument('--ransac_thres', type=float, default=15.0,
                        help='RANSAC threshold (default: 15.0)')
    parser.add_argument('--match_thres', type=float, default=None,
                        help='Matching threshold (default: use config value)')
    parser.add_argument('--odir', type=str, default='outputs/fire_registered',
                        help='Output directory (default: outputs/fire_registered)')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Maximum number of pairs to process (default: all)')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Run registration
    register_fire_images(
        root_dir=args.root_dir,
        config=args.config,
        h_solver=args.h_solver,
        ransac_thres=args.ransac_thres,
        match_thres=args.match_thres,
        odir=args.odir,
        max_pairs=args.max_pairs,
    )