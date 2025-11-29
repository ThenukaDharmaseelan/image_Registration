import os
import numpy as np
import glob
import time
import pydegensac
import cv2
import torch
from tqdm import tqdm


def compute_auc(s_error, p_error, a_error):
    assert (len(s_error) == 71)  # Easy pairs
    assert (len(p_error) == 49)  # Hard pairs. Note file control_points_P37_1_2.txt is ignored
    assert (len(a_error) == 14)  # Moderate pairs

    s_error = np.array(s_error)
    p_error = np.array(p_error)
    a_error = np.array(a_error)

    limit = 25
    gs_error = np.zeros(limit + 1)
    gp_error = np.zeros(limit + 1)
    ga_error = np.zeros(limit + 1)

    accum_s = 0
    accum_p = 0
    accum_a = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_s = accum_s + gs_error[i]
        accum_p = accum_p + gp_error[i]
        accum_a = accum_a + ga_error[i]

    auc_s = accum_s / (limit * 100)
    auc_p = accum_p / (limit * 100)
    auc_a = accum_a / (limit * 100)
    mAUC = (auc_s + auc_p + auc_a) / 3.0
    return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}


def cal_reproj_dists(p1s, p2s, homography):
    '''Compute the reprojection errors using the GT homography'''

    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist


def mle_homography_estimation(src_pts, dst_pts, ransac_thres=2.0, max_iters=2000, confidence=0.999):
    """
    MLE-based homography estimation with RANSAC
    
    Args:
        src_pts: Source points (N x 2)
        dst_pts: Destination points (N x 2)
        ransac_thres: RANSAC threshold for inlier determination
        max_iters: Maximum RANSAC iterations
        confidence: Desired confidence level
    
    Returns:
        H: Homography matrix (3 x 3)
        inliers: Boolean mask of inliers
        stats: Dictionary with MLE statistics
    """
    if len(src_pts) < 4:
        return None, None, {}
    
    best_H = None
    best_inliers = None
    best_score = -1
    n_points = len(src_pts)
    iterations_run = 0
    
    # RANSAC loop
    for iteration in range(max_iters):
        iterations_run += 1
        # Randomly sample 4 points
        indices = np.random.choice(n_points, 4, replace=False)
        sample_src = src_pts[indices]
        sample_dst = dst_pts[indices]
        
        # Compute homography from 4 points using DLT
        try:
            H = cv2.getPerspectiveTransform(
                sample_src.astype(np.float32), 
                sample_dst.astype(np.float32)
            )
        except:
            continue
        
        # Compute reprojection errors
        src_homogeneous = np.concatenate([src_pts, np.ones((n_points, 1))], axis=1)
        dst_proj = (H @ src_homogeneous.T).T
        dst_proj = dst_proj[:, :2] / dst_proj[:, 2:3]
        
        errors = np.linalg.norm(dst_pts - dst_proj, axis=1)
        
        # Identify inliers
        inliers = errors < ransac_thres
        n_inliers = np.sum(inliers)
        
        if n_inliers > best_score:
            best_score = n_inliers
            best_inliers = inliers
            best_H = H
            
            # Early termination if we have enough inliers
            inlier_ratio = n_inliers / n_points
            if inlier_ratio > 0.9:
                break
    
    # Statistics
    stats = {
        'iterations': iterations_run,
        'initial_inliers': best_score if best_H is not None else 0,
        'refined': False
    }
    
    # Refine homography using all inliers with MLE (Levenberg-Marquardt)
    if best_H is not None and best_score >= 4:
        inlier_src = src_pts[best_inliers]
        inlier_dst = dst_pts[best_inliers]
        
        try:
            # Refine using all inliers
            H_refined = cv2.findHomography(
                inlier_src, inlier_dst, 
                method=0  # Use least-squares refinement (MLE)
            )[0]
            
            if H_refined is not None:
                best_H = H_refined
                stats['refined'] = True
                
                # Recompute inliers after refinement
                src_homogeneous = np.concatenate([src_pts, np.ones((n_points, 1))], axis=1)
                dst_proj = (H_refined @ src_homogeneous.T).T
                dst_proj = dst_proj[:, :2] / dst_proj[:, 2:3]
                errors = np.linalg.norm(dst_pts - dst_proj, axis=1)
                best_inliers = errors < ransac_thres
                stats['refined_inliers'] = np.sum(best_inliers)
        except:
            pass
    
    return best_H, best_inliers, stats


def eval_summary_homography(dists_ss, dists_sp, dists_sa):
    dists_ss, dists_sp, dists_sa = map(lambda dist: np.array(dist), [dists_ss, dists_sp, dists_sa])
    # Compute aucs
    auc = compute_auc(dists_ss, dists_sp, dists_sa)

    # Calculate total AUC (average of s, p, a)
    total_auc = (auc["s"] + auc["p"] + auc["a"]) / 3.0

    # Generate summary - single line format
    summary = f'Hest AUC: s={auc["s"]:.4f} p={auc["p"]:.4f} a={auc["a"]:.4f} total={total_auc:.4f} m={auc["mAUC"]:.4f}'
    print(summary)
    return auc["mAUC"], total_auc, {'s': auc["s"], 'p': auc["p"], 'a': auc["a"]}


def scale_homography(sw, sh):
    return np.array([[sw,  0, 0],
                     [ 0, sh, 0],
                     [ 0,  0, 1]])


def eval_fire(
        matcher,
        match_pairs,
        im_dir,
        gt_dir,
        method='',
        task='homography',
        scale_H=False,
        h_solver='degensac',
        ransac_thres=2,
        lprint_=print,
        debug=False,
       
):
    np.set_printoptions(precision=4)

    assert task == 'homography'
    lprint_(f'\n>>>>Eval hpatches: task={task} method={method} scale_H={scale_H} rthres={ransac_thres}')
    # Homography

    inlier_ratio = []
    h_failed = 0
    dists_ss = []
    dists_sp = []
    dists_sa = []
    image_num = 0
    failed = 0
    inaccurate = 0
    first_ransac_num = 0
    first_match_num = 0
    first_match = 0

    match_failed = 0
    n_matches = []
    match_time = []
    
    # NEW: Per-image MLE tracking
    per_image_mle = {}
    
    # MLE statistics
    mle_stats = {
        'total_iterations': 0,
        'total_refined': 0,
        'total_pairs': 0,
        'avg_initial_inliers': [],
        'avg_refined_inliers': [],
        # Per-category statistics
        's': {'iterations': 0, 'refined': 0, 'pairs': 0, 'initial_inliers': [], 'refined_inliers': [], 'distances': []},
        'p': {'iterations': 0, 'refined': 0, 'pairs': 0, 'initial_inliers': [], 'refined_inliers': [], 'distances': []},
        'a': {'iterations': 0, 'refined': 0, 'pairs': 0, 'initial_inliers': [], 'refined_inliers': [], 'distances': []},
        'all_distances': []
    }
    
    start_time = time.time()
    for pair_idx, pair_file in tqdm(enumerate(match_pairs), total=len(match_pairs), smoothing=.5):
        if debug and pair_idx > 10:
            break
        file_name = pair_file.replace('.txt', '')
        gt_file = os.path.join(gt_dir, pair_file)

        refer = file_name.split('_')[2] + '_' + file_name.split('_')[3]
        query = file_name.split('_')[2] + '_' + file_name.split('_')[4]
        im1_path = os.path.join(im_dir, query + '.jpg')
        im2_path = os.path.join(im_dir, refer + '.jpg')
        # Eval on composed pairs within seq
        image_num += 1
        category = file_name.split('_')[2][0]
        # Predict matches
        try:
            t0 = time.time()
            match_res = matcher(im1_path, im2_path)
            if len(match_res) > 5:
                first_match_num += match_res[-2]
                first_ransac_num += match_res[-1]
                first_match += 1
            match_time.append(time.time() - t0)
            matches, p1s, p2s = match_res[0:3]
        except Exception as e:
            print(str(e))
            p1s = p2s = matches = []
            match_failed += 1
        n_matches.append(len(matches))

        if 'homography' in task:
            try:
                # Choose homography solver
                if 'cv' in h_solver:
                    H_pred, inliers = cv2.findHomography(matches[:, :2], matches[:, 2:4], cv2.RANSAC, ransac_thres)
                elif 'mle' in h_solver:
                    # Use MLE-based homography estimation
                    H_pred, inliers, stats = mle_homography_estimation(
                        matches[:, :2], matches[:, 2:4], 
                        ransac_thres=ransac_thres
                    )
                    # Collect MLE statistics
                    if stats:
                        mle_stats['total_iterations'] += stats.get('iterations', 0)
                        mle_stats['total_refined'] += int(stats.get('refined', False))
                        mle_stats['total_pairs'] += 1
                        mle_stats['avg_initial_inliers'].append(stats.get('initial_inliers', 0))
                        if 'refined_inliers' in stats:
                            mle_stats['avg_refined_inliers'].append(stats['refined_inliers'])
                        
                        # Collect per-category statistics
                        cat_key = category.lower()
                        if cat_key in mle_stats:
                            mle_stats[cat_key]['iterations'] += stats.get('iterations', 0)
                            mle_stats[cat_key]['refined'] += int(stats.get('refined', False))
                            mle_stats[cat_key]['pairs'] += 1
                            mle_stats[cat_key]['initial_inliers'].append(stats.get('initial_inliers', 0))
                            if 'refined_inliers' in stats:
                                mle_stats[cat_key]['refined_inliers'].append(stats['refined_inliers'])
                else:  # degensac
                    H_pred, inliers = pydegensac.findHomography(matches[:, :2], matches[:, 2:4], ransac_thres)
                
                if scale_H:
                    scale = match_res[4]

                    # Scale gt homoragphies
                    H_scale_im1 = scale_homography(1/scale[0], 1/scale[1])
                    H_scale_im2 = scale_homography(1/scale[2], 1/scale[3])
                    H_pred = np.linalg.inv(H_scale_im2) @ H_pred @ H_scale_im1

            except:
                H_pred = None

            if H_pred is None:
                avg_dist = big_num = 1e6
                irat = 0
                h_failed += 1
                failed += 1
                inliers = []
            else:
                points_gd = np.loadtxt(gt_file)
                raw = np.zeros([len(points_gd), 2])
                dst = np.zeros([len(points_gd), 2])
                raw[:, 0] = points_gd[:, 2]
                raw[:, 1] = points_gd[:, 3]
                dst[:, 0] = points_gd[:, 0]
                dst[:, 1] = points_gd[:, 1]
                
                dst_pred = cv2.perspectiveTransform(raw.reshape(-1, 1, 2), H_pred).squeeze()
                dis = (dst - dst_pred) ** 2
                dis = np.sqrt(dis[:, 0] + dis[:, 1])
                avg_dist = dis.mean()
                irat = np.mean(inliers)
                mae = dis.max()
                mee = np.median(dis)
                if mae > 50 or mee > 20:
                    inaccurate += 1
                
                # Store distance for MLE statistics
                if 'mle' in h_solver:
                    cat_key = category.lower()
                    if cat_key in mle_stats:
                        mle_stats[cat_key]['distances'].append(avg_dist)
                    mle_stats['all_distances'].append(avg_dist)
            
            # NEW: Store per-image MLE distance (avg_dist is the reprojection error)
            per_image_mle[str(pair_idx)] = float(avg_dist)

            inlier_ratio.append(irat)

            if category == 'S':
                dists_ss.append(avg_dist)
            if category == 'P':
                dists_sp.append(avg_dist)
            if category == 'A':
                dists_sa.append(avg_dist)

    lprint_(
        f'>>Finished, pairs={len(match_time)} match_failed={match_failed} matches={np.mean(n_matches):.1f} match_time={np.mean(match_time):.2f}s')
    print('----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------')
    lprint_('==== Homography Estimation ====')
    lprint_(
        f'Hest solver={h_solver} est_failed={h_failed} ransac_thres={ransac_thres} inlier_rate={np.mean(inlier_ratio):.2f}')
    
    # Print MLE statistics if MLE solver was used
    mle_dict = {}
    if 'mle' in h_solver and mle_stats['total_pairs'] > 0:
        # Calculate mean reprojection distance for each category
        mle_s = np.mean(mle_stats['s']['distances']) if mle_stats['s']['distances'] else 0
        mle_p = np.mean(mle_stats['p']['distances']) if mle_stats['p']['distances'] else 0
        mle_a = np.mean(mle_stats['a']['distances']) if mle_stats['a']['distances'] else 0
        mle_m = np.mean(mle_stats['all_distances']) if mle_stats['all_distances'] else 0
        
        # Calculate total MLE (average of s, p, a)
        total_mle = (mle_s + mle_p + mle_a) / 3.0
        
        # Store MLE results
        mle_dict = {'s': mle_s, 'p': mle_p, 'a': mle_a, 'total': total_mle, 'm': mle_m}
    
    # Always returns tuple now
    mauc, total_auc, auc_dict = eval_summary_homography(dists_ss, dists_sp, dists_sa)
    
    # Print MLE after AUC
    if mle_dict:
        lprint_(f'MLE dist: s={mle_dict["s"]:.4f} p={mle_dict["p"]:.4f} a={mle_dict["a"]:.4f} total={mle_dict["total"]:.4f} m={mle_dict["m"]:.4f}')

    print('-' * 40)
    print(f"Failed:{'%.2f' % (100 * failed / image_num)}%, Inaccurate:{'%.2f' % (100 * inaccurate / image_num)}%, "
          f"Acceptable:{'%.2f' % (100 * (image_num - inaccurate - failed) / image_num)}%")

    print('-' * 40)

    # Always return dictionary with per-image MLE
    result = {
        'mauc': mauc, 
        'total_auc': total_auc, 
        'auc': auc_dict,
        'per_image_mle': per_image_mle  # NEW: Add per-image MLE to results
    }
    if mle_dict:
        result['mle'] = mle_dict
    
    return result


import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, '.')

import eval_tool.immatch as immatch
from eval_tool.immatch.utils.data_io import lprint
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
    ransac_thres=3,
    match_thres=None,
    odir='outputs/fire_registered',
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
    print(f"\nFound {len(match_pairs)} image pairs")
    
    # Parse model config
    print(f"\nLoading model configuration: {config}")
    try:
        args = parse_model_config(config, 'fire', root_dir)
        class_name = args['class']
        print(f"Model class: {class_name}")
    except FileNotFoundError as e:
        print(f"\nERROR: Config file not found")
        print(f"Looking for: ./eval_configs/{config}.yml")
        print("\nAvailable configs should be in: ./eval_configs/")
        print("\nTry one of these common configs:")
        print("  --config geoformer")
        print("  --config loftr")
        print("  --config superglue")
        return
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
        
        # Read ground truth file to get image names
        gt_file = os.path.join(gt_dir, pair_file)
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        # Parse image names from filename
        # Format: control_points_P{num}_{category}_{query}_{refer}.txt
        # e.g., control_points_P1_S_S001_L1_S001_R1.txt
        parts = pair_name.split('_')
        if len(parts) >= 6:
            category = parts[2]  # S, P, or A
            refer = parts[3] + '_' + parts[4]  # e.g., S001_L1
            query = parts[3] + '_' + parts[5]  # e.g., S001_R1
        else:
            print(f"  WARNING: Unexpected filename format, skipping")
            fail_count += 1
            continue
        
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
    parser.add_argument('--ransac_thres', type=float, default=3.0,
                        help='RANSAC threshold (default: 3.0)')
    parser.add_argument('--match_thres', type=float, default=None,
                        help='Matching threshold (default: use config value)')
    parser.add_argument('--odir', type=str, default='outputs/fire_registered',
                        help='Output directory (default: outputs/fire_registered)')
    
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
    )

#!/usr/bin/env python3
"""
Script to uncomment the eval_fire function in fire_helper.py
Run this from your GeoFormer root directory:
    python /tmp/fix_fire_helper.py
"""

def uncomment_file(filepath):
    """Uncomment all lines in a Python file"""
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print(f"Current directory: {os.getcwd()}")
        print("\nMake sure you run this from the GeoFormer root directory!")
        return False
    
    print(f"Reading: {filepath}")
    
    # Read the file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Uncomment all lines
    uncommented_lines = []
    changes = 0
    
    for i, line in enumerate(lines):
        if line.startswith('# ') and not line.startswith('# #'):
            # Remove '# ' prefix
            uncommented_lines.append(line[2:])
            changes += 1
        elif line.startswith('#') and len(line) > 1 and line[1] not in ['#', ' ']:
            # Remove single '#' prefix (but keep shebang and double ##)
            uncommented_lines.append(line[1:])
            changes += 1
        else:
            # Keep line as is
            uncommented_lines.append(line)
    
    if changes == 0:
        print("No changes needed - file is already uncommented!")
        return True
    
    # Create backup
    backup_path = filepath + '.backup'
    print(f"Creating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        f.writelines(lines)
    
    # Write uncommented version
    print(f"Writing uncommented version: {filepath}")
    with open(filepath, 'w') as f:
        f.writelines(uncommented_lines)
    
    print(f"✓ Successfully uncommented {changes} lines!")
    print(f"✓ Backup saved to: {backup_path}")
    return True

if __name__ == '__main__':
    # Target file
    target = 'eval_tool/immatch/utils/fire_helper.py'
    
    print("="*80)
    print("Fire Helper Uncomment Script")
    print("="*80)
    print(f"Current directory: {os.getcwd()}")
    print(f"Target file: {target}")
    print()
    
    success = uncomment_file(target)
    
    if success:
        print("\n" + "="*80)
        print("SUCCESS! Now you can run eval_FIRE.py")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("FAILED - Please check the error messages above")
        print("="*80)
        sys.exit(1)