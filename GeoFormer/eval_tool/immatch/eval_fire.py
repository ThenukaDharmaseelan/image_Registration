# import argparse
# from argparse import Namespace
# import os

# import eval_tool.immatch as immatch
# from eval_tool.immatch.utils.data_io import lprint
# import eval_tool.immatch.utils.fire_helper as helper
# from eval_tool.immatch.utils.model_helper import parse_model_config

# def eval_fire(
#     root_dir,
#     config_list,
#     task='homography',
#     h_solver='degensac',
#     ransac_thres=2,
#     match_thres=None,
#     odir='outputs/fire',
#     save_npy=False,
#     print_out=False,
#     debug=False,
# ):
#     # Init paths
#     data_root = os.path.join(root_dir, 'data/datasets/FIRE')
#     cache_dir = os.path.join(root_dir, odir, 'cache')
#     result_dir = os.path.join(root_dir, odir, 'results', task)

#     gt_dir = os.path.join(data_root, 'Ground Truth')
#     im_dir = os.path.join(data_root, 'Images')
#     match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')]
#     if not os.path.exists(cache_dir):
#         os.makedirs(cache_dir)    
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir)    
        
    
#     # Iterate over methods
#     for config_name in config_list:
#         # Load model
#         args = parse_model_config(config_name, 'fire', root_dir)
#         class_name = args['class']
        
#         # One log file per method
#         log_file = os.path.join(result_dir, f'{class_name}.txt')        
#         log = open(log_file, 'a')
#         lprint_ = lambda ms: lprint(ms, log)

#         # Iterate over matching thresholds
#         thresholds = match_thres if match_thres else [args['match_threshold']] 
#         lprint_(f'\n>>>> Method={class_name} Default loftr_config: {args} '
#                 f'Thres: {thresholds}')        
        
#         for thres in thresholds:
#             args['match_threshold'] = thres   # Set to target thresholds
            
#             # Init model
#             model = immatch.__dict__[class_name](args)
#             matcher = lambda im1, im2: model.match_pairs(im1, im2)
            
#             # Init result save path (for matching results)            
#             result_npy = None            
#             if save_npy:
#                 result_tag = model.name
#                 if args['imsize'] > 0:
#                     result_tag += f".im{args['imsize']}"
#                 if thres > 0:
#                     result_tag += f'.m{thres}'
#                 result_npy = os.path.join(cache_dir, f'{result_tag}.npy')
            
#             lprint_(f'Matching thres: {thres}  Save to: {result_npy}')
            
#             # Eval on the specified task(s)
#             helper.eval_fire(
#                 matcher,
#                 match_pairs,
#                 im_dir,
#                 gt_dir,
#                 model.name,
#                 task=task,
#                 scale_H=getattr(model, 'no_match_upscale', False),
#                 h_solver=h_solver,
#                 ransac_thres=ransac_thres,
#                 lprint_=lprint_,
#                 print_out=print_out,
#                 save_npy=result_npy,
#                 debug=debug,
#             )
#         log.close()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Benchmark HPatches')
#     parser.add_argument('--gpu', '-gpu', type=str, default='0')
#     parser.add_argument('--root_dir', type=str, default='.')
#     parser.add_argument('--odir', type=str, default='outputs/fire')
#     parser.add_argument('--loftr_config', type=str, nargs='*', default=['my'])
#     parser.add_argument('--match_thres', type=float, nargs='*', default=None)
#     parser.add_argument(
#         '--task', type=str, default='homography',
#         choices=['matching', 'homography', 'both']
#     )
#     parser.add_argument(
#         '--h_solver', type=str, default='cv',
#         choices=['degensac', 'cv']
#     )
#     parser.add_argument('--ransac_thres', type=float, default=3)
#     parser.add_argument('--save_npy', action='store_true')
#     parser.add_argument('--print_out', action='store_true', default=True)

#     args = parser.parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     eval_fire(
#         args.root_dir, args.config, args.task,
#         h_solver=args.h_solver,
#         ransac_thres=args.ransac_thres,
#         match_thres=args.match_thres,
#         odir=args.odir,
#         save_npy=args.save_npy,
#         print_out=args.print_out
#     )

import argparse
from argparse import Namespace
import os
import cv2
import numpy as np

import eval_tool.immatch as immatch
from eval_tool.immatch.utils.data_io import lprint
import eval_tool.immatch.utils.fire_helper as helper
from eval_tool.immatch.utils.model_helper import parse_model_config

def save_registered_images(im1, im2, H, save_dir, pair_name, overlay=True):
    """
    Save registered images and optionally create an overlay visualization
    
    Args:
        im1: First image (to be warped)
        im2: Second image (reference)
        H: Homography matrix
        save_dir: Directory to save results
        pair_name: Name of the image pair
        overlay: Whether to create an overlay visualization
    """
    if H is None:
        return
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dimensions from reference image
    h, w = im2.shape[:2]
    
    # Warp image 1 to align with image 2
    registered_im1 = cv2.warpPerspective(im1, H, (w, h))
    
    # Save registered image
    registered_path = os.path.join(save_dir, f'{pair_name}_registered.png')
    cv2.imwrite(registered_path, registered_im1)
    
    if overlay:
        # Create overlay visualization (registered image in red, reference in green)
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
        
        # Red channel: registered image
        overlay_img[:, :, 2] = reg_gray
        # Green channel: reference image
        overlay_img[:, :, 1] = ref_gray
        # Overlap appears as yellow
        
        overlay_path = os.path.join(save_dir, f'{pair_name}_overlay.png')
        cv2.imwrite(overlay_path, overlay_img)
        
        # Also save a checkerboard pattern for better visualization
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
    
    return registered_path

def eval_fire_with_registration(
    root_dir,
    config_list,
    task='homography',
    h_solver='degensac',
    ransac_thres=2,
    match_thres=None,
    odir='outputs/fire',
    save_npy=False,
    save_registered=True,
    print_out=False,
    debug=False,
):
    # Init paths
    data_root = os.path.join(root_dir, 'data/datasets/FIRE')
    cache_dir = os.path.join(root_dir, odir, 'cache')
    result_dir = os.path.join(root_dir, odir, 'results', task)
    registered_dir = os.path.join(root_dir, odir, 'registered_images')

    gt_dir = os.path.join(data_root, 'Ground Truth')
    im_dir = os.path.join(data_root, 'Images')
    match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')]
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if save_registered and not os.path.exists(registered_dir):
        os.makedirs(registered_dir)
    
    # Iterate over methods
    for config_name in config_list:
        # Load model
        args = parse_model_config(config_name, 'fire', root_dir)
        class_name = args['class']
        
        # Create method-specific registered images directory
        method_registered_dir = os.path.join(registered_dir, class_name)
        if save_registered:
            os.makedirs(method_registered_dir, exist_ok=True)
        
        # One log file per method
        log_file = os.path.join(result_dir, f'{class_name}.txt')        
        log = open(log_file, 'a')
        lprint_ = lambda ms: lprint(ms, log)

        # Iterate over matching thresholds
        thresholds = match_thres if match_thres else [args['match_threshold']] 
        lprint_(f'\n>>>> Method={class_name} Default config: {args} '
                f'Thres: {thresholds}')        
        
        for thres in thresholds:
            args['match_threshold'] = thres   # Set to target thresholds
            
            # Init model
            model = immatch.__dict__[class_name](args)
            matcher = lambda im1, im2: model.match_pairs(im1, im2)
            
            # Init result save path (for matching results)            
            result_npy = None            
            if save_npy:
                result_tag = model.name
                if args['imsize'] > 0:
                    result_tag += f".im{args['imsize']}"
                if thres > 0:
                    result_tag += f'.m{thres}'
                result_npy = os.path.join(cache_dir, f'{result_tag}.npy')
            
            lprint_(f'Matching thres: {thres}  Save to: {result_npy}')
            if save_registered:
                lprint_(f'Registered images will be saved to: {method_registered_dir}')
            
            # Process each pair if we want to save registered images
            if save_registered:
                for pair_file in match_pairs:
                    pair_name = pair_file.replace('.txt', '')
                    lprint_(f'Processing pair: {pair_name}')
                    
                    # Read ground truth
                    gt_file = os.path.join(gt_dir, pair_file)
                    with open(gt_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Parse image names
                    im1_name = lines[0].strip()
                    im2_name = lines[1].strip()
                    
                    # Load images
                    im1_path = os.path.join(im_dir, im1_name)
                    im2_path = os.path.join(im_dir, im2_name)
                    
                    im1 = cv2.imread(im1_path)
                    im2 = cv2.imread(im2_path)
                    
                    if im1 is None or im2 is None:
                        lprint_(f'Warning: Could not load images for {pair_name}')
                        continue
                    
                    # Perform matching
                    try:
                        match_result = matcher(im1, im2)
                        
                        # Extract keypoints and matches
                        if hasattr(match_result, 'mkpts0'):
                            mkpts0 = match_result.mkpts0
                            mkpts1 = match_result.mkpts1
                        else:
                            mkpts0 = match_result['mkpts0']
                            mkpts1 = match_result['mkpts1']
                        
                        # Estimate homography
                        if len(mkpts0) >= 4:
                            if h_solver == 'degensac':
                                try:
                                    import pydegensac
                                    H, inliers = pydegensac.findHomography(
                                        mkpts0, mkpts1, ransac_thres
                                    )
                                except:
                                    H, inliers = cv2.findHomography(
                                        mkpts0, mkpts1, cv2.RANSAC, ransac_thres
                                    )
                            else:
                                H, inliers = cv2.findHomography(
                                    mkpts0, mkpts1, cv2.RANSAC, ransac_thres
                                )
                            
                            if H is not None:
                                # Save registered images
                                save_registered_images(
                                    im1, im2, H, method_registered_dir, pair_name
                                )
                                lprint_(f'Saved registered images for {pair_name}')
                            else:
                                lprint_(f'Failed to estimate homography for {pair_name}')
                        else:
                            lprint_(f'Not enough matches for {pair_name}: {len(mkpts0)}')
                    
                    except Exception as e:
                        lprint_(f'Error processing {pair_name}: {str(e)}')
                        if debug:
                            import traceback
                            lprint_(traceback.format_exc())
            
            # Run standard evaluation
            helper.eval_fire(
                matcher,
                match_pairs,
                im_dir,
                gt_dir,
                model.name,
                task=task,
                scale_H=getattr(model, 'no_match_upscale', False),
                h_solver=h_solver,
                ransac_thres=ransac_thres,
                lprint_=lprint_,
                print_out=print_out,
                save_npy=result_npy,
                debug=debug,
            )
        log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark FIRE with Image Registration')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--odir', type=str, default='outputs/fire')
    parser.add_argument('--config', type=str, nargs='*', default=['my'])  # Fixed from loftr_config
    parser.add_argument('--match_thres', type=float, nargs='*', default=None)
    parser.add_argument(
        '--task', type=str, default='homography',
        choices=['matching', 'homography', 'both']
    )
    parser.add_argument(
        '--h_solver', type=str, default='cv',
        choices=['degensac', 'cv']
    )
    parser.add_argument('--ransac_thres', type=float, default=3)
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--save_registered', action='store_true', default=True)
    parser.add_argument('--print_out', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    eval_fire_with_registration(
        args.root_dir, 
        args.config,  # Fixed bug
        args.task,
        h_solver=args.h_solver,
        ransac_thres=args.ransac_thres,
        match_thres=args.match_thres,
        odir=args.odir,
        save_npy=args.save_npy,
        save_registered=args.save_registered,
        print_out=args.print_out,
        debug=args.debug
    )
