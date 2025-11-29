# # # ''' Evaluates EyeLiner Pairwise Registration Pipeline on an Image Dataset '''

# # # # =================
# # # # Install libraries
# # # # =================

# # # import argparse
# # # import os, sys
# # # from tqdm import tqdm
# # # from utils import none_or_str, compute_dice
# # # from PIL import Image
# # # import torch
# # # from torchvision.transforms import ToPILImage
# # # from data import ImageDataset
# # # from eyeliner import EyeLinerP
# # # from visualize import create_flicker, create_checkerboard, create_diff_map
# # # from matplotlib import pyplot as plt

# # # def parse_args():
# # #     parser = argparse.ArgumentParser()
# # #     # data args
# # #     parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
# # #     parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
# # #     parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
# # #     parser.add_argument('-fv', '--fixed-vessel', default='fixed_vessel_seg', type=none_or_str, help='Fixed vessel column')
# # #     parser.add_argument('-mv', '--moving-vessel', default='moving_vessel_seg', type=none_or_str, help='Moving vessel column')
# # #     parser.add_argument('-fd', '--fixed-disk', default='fixed_disk', type=none_or_str, help='Fixed disk column')
# # #     parser.add_argument('-md', '--moving-disk', default='moving_disk', type=none_or_str, help='Moving disk column')
# # #     parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
# # #     parser.add_argument('-r', '--registration', default='registration_path', type=none_or_str, help='Registration column')
# # #     parser.add_argument('-l', '--lmbda', default=None, type=float, help='Lambda value for computing the TPS quantitative result')
# # #     parser.add_argument('--detected_keypoints', default='keypoints', type=none_or_str, help='Model Detected Keypoints column')
# # #     parser.add_argument('--manual_keypoints', default='keypoints', type=none_or_str, help='Manually annotated Keypoints column')
# # #     parser.add_argument('--fire_eval', action='store_true', help='Run FIRE evaluation')

# # #     # misc
# # #     parser.add_argument('--device', default='cpu', help='Device to run program on')
# # #     parser.add_argument('--save', default='trained_models/', help='Location to save results')
# # #     args = parser.parse_args()
# # #     return args

# # # def main(args):

# # #     device = torch.device(args.device)

# # #     # load dataset
# # #     dataset = ImageDataset(
# # #         path=args.data, 
# # #         input_col=args.fixed, 
# # #         output_col=args.moving,
# # #         input_vessel_col=args.fixed_vessel,
# # #         output_vessel_col=args.moving_vessel,
# # #         input_od_col=args.fixed_disk,
# # #         output_od_col=args.moving_disk,
# # #         input_dim=(args.size, args.size), 
# # #         cmode='rgb',
# # #         input='vessel',
# # #         detected_keypoints_col=args.detected_keypoints,
# # #         manual_keypoints_col=args.manual_keypoints,
# # #         registration_col=args.registration
# # #     )

# # #     # make directory to store registrations
# # #     reg_images_save_folder = os.path.join(args.save, 'registration_images')
# # #     seg_overlaps_save_folder = os.path.join(args.save, 'seg_overlaps')
# # #     checkerboard_after_save_folder = os.path.join(args.save, 'ckbd_images_after')
# # #     checkerboard_before_save_folder = os.path.join(args.save, 'ckbd_images_before')
# # #     flicker_save_folder = os.path.join(args.save, 'flicker_images')
# # #     difference_map_save_folder = os.path.join(args.save, 'diff_map_images')
# # #     os.makedirs(reg_images_save_folder, exist_ok=True)
# # #     os.makedirs(checkerboard_after_save_folder, exist_ok=True)
# # #     os.makedirs(checkerboard_before_save_folder, exist_ok=True)
# # #     os.makedirs(flicker_save_folder, exist_ok=True)
# # #     os.makedirs(difference_map_save_folder, exist_ok=True)
# # #     os.makedirs(seg_overlaps_save_folder, exist_ok=True)

# # #     images_filenames = []
# # #     ckbd_filenames_before = []
# # #     ckbd_filenames_after = []
# # #     segmentation_overlaps = []
# # #     flicker_filenames = []
# # #     difference_maps_filenames = []
# # #     dice_before = []
# # #     dice_after = []
# # #     if args.manual_keypoints is not None:
# # #         error_before = []
# # #         error_after = []

# # #     for i in tqdm(range(len(dataset))):
        
# # #         # load images
# # #         batch_data = dataset[i]
# # #         fixed_image = batch_data['fixed_image'] # (3, 256, 256)
# # #         moving_image = batch_data['moving_image'] # (3, 256, 256)
# # #         fixed_vessel = batch_data['fixed_input'] # (3, 256, 256)
# # #         fixed_vessel = (fixed_vessel > 0.5).float()
# # #         moving_vessel = batch_data['moving_input'] # (3, 256, 256
# # #         moving_vessel = (moving_vessel > 0.5).float()
# # #         theta = batch_data['theta'] # (3, 3) or ((N, 2), (256, 256, 2))

# # #         # if image pair could not be registered
# # #         if theta is None:
# # #             images_filenames.append(None)
# # #             ckbd_filenames_after.append(None)
# # #             ckbd_filenames_before.append(None)
# # #             flicker_filenames.append(None)
# # #             difference_maps_filenames.append(None)
# # #             dice_before.append(None)
# # #             dice_after.append(None)
# # #             if args.manual_keypoints is not None:
# # #                 error_before.append(None)
# # #                 error_after.append(None)
# # #             continue

# # #         # register moving image
# # #         try:
# # #             reg_image = EyeLinerP.apply_transform(theta, moving_image) # (3, 256, 256)
# # #         except:
# # #             reg_image = EyeLinerP.apply_transform(theta[1], moving_image)

# # #         # register moving segmentation
# # #         try:
# # #             reg_vessel = EyeLinerP.apply_transform(theta, moving_vessel) # (3, 256, 256)
# # #         except:
# # #             reg_vessel = EyeLinerP.apply_transform(theta[1], moving_vessel)
# # #         reg_vessel = (reg_vessel > 0.5).float()

# # #         # create mask
# # #         reg_mask = torch.ones_like(moving_image)
# # #         try:
# # #             reg_mask = EyeLinerP.apply_transform(theta, reg_mask) # (3, 256, 256)
# # #         except:
# # #             reg_mask = EyeLinerP.apply_transform(theta[1], reg_mask)

# # #         # apply mask to images
# # #         fixed_image = fixed_image * reg_mask
# # #         moving_image = moving_image * reg_mask
# # #         reg_image = reg_image * reg_mask

# # #         # apply mask to segmentations
# # #         fixed_vessel = fixed_vessel #* reg_mask
# # #         moving_vessel = moving_vessel #* reg_mask
# # #         reg_vessel = reg_vessel * reg_mask

# # #         # save registration image
# # #         filename = os.path.join(reg_images_save_folder, f'reg_{i}.png')
# # #         ToPILImage()(reg_image).save(filename)
# # #         images_filenames.append(filename)

# # #         # qualitative evaluation

# # #         # segmentation overlap
# # #         seg1 = ToPILImage()(fixed_vessel)
# # #         seg2 = ToPILImage()(reg_vessel)
# # #         seg_overlap = Image.blend(seg1, seg2, alpha=0.5)
# # #         filename = os.path.join(seg_overlaps_save_folder, f'seg_overlap_{i}.png')
# # #         seg_overlap.save(filename)
# # #         segmentation_overlaps.append(filename)

# # #         # checkerboards - before and after registration
# # #         ckbd = create_checkerboard(fixed_image, reg_image, patch_size=32)
# # #         filename = os.path.join(checkerboard_after_save_folder, f'ckbd_{i}.png')
# # #         ToPILImage()(ckbd).save(filename)
# # #         ckbd_filenames_after.append(filename)

# # #         ckbd = create_checkerboard(fixed_image, moving_image, patch_size=32)
# # #         filename = os.path.join(checkerboard_before_save_folder, f'ckbd_{i}.png')
# # #         ToPILImage()(ckbd).save(filename)
# # #         ckbd_filenames_before.append(filename)

# # #         # flicker animation
# # #         filename = os.path.join(flicker_save_folder, f'flicker_{i}.gif')
# # #         create_flicker(fixed_image, reg_image, output_path=filename)
# # #         flicker_filenames.append(filename)

# # #         # subtraction maps
# # #         filename = os.path.join(difference_map_save_folder, f'diff_map_{i}.png')
# # #         create_diff_map(fixed_image, reg_image, filename)
# # #         difference_maps_filenames.append(filename)

# # #         # compute dice between segmentation maps
# # #         seg_dice_before = compute_dice(fixed_vessel, moving_vessel)
# # #         seg_dice_after = compute_dice(fixed_vessel, reg_vessel)
# # #         dice_before.append(seg_dice_before)
# # #         dice_after.append(seg_dice_after)

# # #         # quantitative evaluation
# # #         if args.manual_keypoints is not None:

# # #             fixed_kp_manual = batch_data['fixed_keypoints_manual']
# # #             moving_kp_manual = batch_data['moving_keypoints_manual']
# # #             fixed_kp_detected = batch_data['fixed_keypoints_detected']
# # #             moving_kp_detected = batch_data['moving_keypoints_detected']

# # #             # apply theta to keypoints
# # #             try:
# # #                 reg_kp = EyeLinerP.apply_transform_points(theta, moving_keypoints=moving_kp_manual)
# # #             except:                
# # #                 reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=moving_kp_manual, ctrl_keypoints=moving_kp_detected, tgt_keypoints=fixed_kp_detected, lmbda=args.lmbda)
# # #                 # reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=fixed_kp_manual, ctrl_keypoints=fixed_kp_detected)

# # #             if args.fire_eval:
# # #                 fixed_kp_manual = 2912. * fixed_kp_manual / 256.
# # #                 moving_kp_manual = 2912. * moving_kp_manual / 256.
# # #                 reg_kp = 2912. * reg_kp / 256.

# # #             # compute mean distance between fixed and registered keypoints
# # #             md_before = torch.sqrt(torch.sum((fixed_kp_manual - moving_kp_manual)**2, dim=-1)).mean().item()
# # #             md_after = torch.sqrt(torch.sum((fixed_kp_manual - reg_kp)**2, dim=-1)).mean().item()
# # #             error_before.append(md_before)
# # #             error_after.append(md_after)

# # #     dataset.data['registration_path'] = images_filenames
# # #     dataset.data['checkerboard_before'] = ckbd_filenames_before
# # #     dataset.data['checkerboard_after'] = ckbd_filenames_after
# # #     dataset.data['flicker'] = flicker_filenames
# # #     dataset.data['seg_overlap'] = segmentation_overlaps
# # #     dataset.data['difference_map'] = difference_maps_filenames
# # #     dataset.data['DICE_before'] = dice_before
# # #     dataset.data['DICE_after'] = dice_after
# # #     # add columns to dataframe
# # #     if args.manual_keypoints is not None:
# # #         dataset.data['MD_before'] = error_before
# # #         dataset.data['MD_after'] = error_after

# # #     # save results
# # #     csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
# # #     dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)

# # #     return

# # # if __name__ == '__main__':
# # #     args = parse_args()
# # #     main(args)
    


# # ''' Evaluates EyeLiner Pairwise Registration Pipeline on an Image Dataset '''

# # # =================
# # # Install libraries
# # # =================

# # import argparse
# # import os, sys
# # from tqdm import tqdm
# # from utils import none_or_str, compute_dice
# # from PIL import Image
# # import torch
# # from torchvision.transforms import ToPILImage
# # from data import ImageDataset
# # from eyeliner import EyeLinerP
# # from visualize import create_flicker, create_checkerboard, create_diff_map
# # from matplotlib import pyplot as plt

# # def parse_args():
# #     parser = argparse.ArgumentParser()
# #     # data args
# #     parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
# #     parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
# #     parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
# #     parser.add_argument('-fv', '--fixed-vessel', default='fixed_vessel_seg', type=none_or_str, help='Fixed vessel column')
# #     parser.add_argument('-mv', '--moving-vessel', default='moving_vessel_seg', type=none_or_str, help='Moving vessel column')
# #     parser.add_argument('-fd', '--fixed-disk', default='fixed_disk', type=none_or_str, help='Fixed disk column')
# #     parser.add_argument('-md', '--moving-disk', default='moving_disk', type=none_or_str, help='Moving disk column')
# #     parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
# #     parser.add_argument('-r', '--registration', default='registration_path', type=none_or_str, help='Registration column')
# #     parser.add_argument('-l', '--lmbda', default=None, type=float, help='Lambda value for computing the TPS quantitative result')
# #     parser.add_argument('--detected_keypoints', default='keypoints', type=none_or_str, help='Model Detected Keypoints column')
# #     parser.add_argument('--manual_keypoints', default='keypoints', type=none_or_str, help='Manually annotated Keypoints column')
# #     parser.add_argument('--fire_eval', action='store_true', help='Run FIRE evaluation')

# #     # misc
# #     parser.add_argument('--device', default='cpu', help='Device to run program on')
# #     parser.add_argument('--save', default='trained_models/', help='Location to save results')
# #     args = parser.parse_args()
# #     return args

# # def main(args):

# #     device = torch.device(args.device)

# #     # load dataset
# #     dataset = ImageDataset(
# #         path=args.data, 
# #         input_col=args.fixed, 
# #         output_col=args.moving,
# #         input_vessel_col=args.fixed_vessel,
# #         output_vessel_col=args.moving_vessel,
# #         input_od_col=args.fixed_disk,
# #         output_od_col=args.moving_disk,
# #         input_dim=(args.size, args.size), 
# #         cmode='rgb',
# #         input='vessel',
# #         detected_keypoints_col=args.detected_keypoints,
# #         manual_keypoints_col=args.manual_keypoints,
# #         registration_col=args.registration
# #     )

# #     # make directory to store registrations
# #     reg_images_save_folder = os.path.join(args.save, 'registration_images')
# #     seg_overlaps_save_folder = os.path.join(args.save, 'seg_overlaps')
# #     checkerboard_after_save_folder = os.path.join(args.save, 'ckbd_images_after')
# #     checkerboard_before_save_folder = os.path.join(args.save, 'ckbd_images_before')
# #     flicker_save_folder = os.path.join(args.save, 'flicker_images')
# #     difference_map_save_folder = os.path.join(args.save, 'diff_map_images')
# #     os.makedirs(reg_images_save_folder, exist_ok=True)
# #     os.makedirs(checkerboard_after_save_folder, exist_ok=True)
# #     os.makedirs(checkerboard_before_save_folder, exist_ok=True)
# #     os.makedirs(flicker_save_folder, exist_ok=True)
# #     os.makedirs(difference_map_save_folder, exist_ok=True)
# #     os.makedirs(seg_overlaps_save_folder, exist_ok=True)

# #     images_filenames = []
# #     ckbd_filenames_before = []
# #     ckbd_filenames_after = []
# #     segmentation_overlaps = []
# #     flicker_filenames = []
# #     difference_maps_filenames = []
# #     dice_before = []
# #     dice_after = []
# #     if args.manual_keypoints is not None:
# #         error_before = []
# #         error_after = []

# #     for i in tqdm(range(len(dataset))):
        
# #         # load images
# #         batch_data = dataset[i]
# #         fixed_image = batch_data['fixed_image'] # (3, 256, 256)
# #         moving_image = batch_data['moving_image'] # (3, 256, 256)
# #         fixed_vessel = batch_data['fixed_input'] # (3, 256, 256)
# #         fixed_vessel = (fixed_vessel > 0.5).float()
# #         moving_vessel = batch_data['moving_input'] # (3, 256, 256
# #         moving_vessel = (moving_vessel > 0.5).float()
# #         theta = batch_data['theta'] # (3, 3) or ((N, 2), (256, 256, 2))

# #         # if image pair could not be registered
# #         if theta is None:
# #             images_filenames.append(None)
# #             ckbd_filenames_after.append(None)
# #             ckbd_filenames_before.append(None)
# #             flicker_filenames.append(None)
# #             difference_maps_filenames.append(None)
# #             dice_before.append(None)
# #             dice_after.append(None)
# #             if args.manual_keypoints is not None:
# #                 error_before.append(None)
# #                 error_after.append(None)
# #             continue

# #         # register moving image
# #         try:
# #             reg_image = EyeLinerP.apply_transform(theta, moving_image) # (3, 256, 256)
# #         except:
# #             reg_image = EyeLinerP.apply_transform(theta[1], moving_image)

# #         # register moving segmentation
# #         try:
# #             reg_vessel = EyeLinerP.apply_transform(theta, moving_vessel) # (3, 256, 256)
# #         except:
# #             reg_vessel = EyeLinerP.apply_transform(theta[1], moving_vessel)
# #         reg_vessel = (reg_vessel > 0.5).float()

# #         # create mask
# #         reg_mask = torch.ones_like(moving_image)
# #         try:
# #             reg_mask = EyeLinerP.apply_transform(theta, reg_mask) # (3, 256, 256)
# #         except:
# #             reg_mask = EyeLinerP.apply_transform(theta[1], reg_mask)

# #         # apply mask to images
# #         fixed_image = fixed_image * reg_mask
# #         moving_image = moving_image * reg_mask
# #         reg_image = reg_image * reg_mask

# #         # apply mask to segmentations
# #         fixed_vessel = fixed_vessel #* reg_mask
# #         moving_vessel = moving_vessel #* reg_mask
# #         reg_vessel = reg_vessel * reg_mask

# #         # save registration image
# #         filename = os.path.join(reg_images_save_folder, f'reg_{i}.png')
# #         ToPILImage()(reg_image).save(filename)
# #         images_filenames.append(filename)

# #         # qualitative evaluation

# #         # segmentation overlap
# #         seg1 = ToPILImage()(fixed_vessel)
# #         seg2 = ToPILImage()(reg_vessel)
# #         seg_overlap = Image.blend(seg1, seg2, alpha=0.5)
# #         filename = os.path.join(seg_overlaps_save_folder, f'seg_overlap_{i}.png')
# #         seg_overlap.save(filename)
# #         segmentation_overlaps.append(filename)

# #         # checkerboards - before and after registration
# #         ckbd = create_checkerboard(fixed_image, reg_image, patch_size=32)
# #         filename = os.path.join(checkerboard_after_save_folder, f'ckbd_{i}.png')
# #         ToPILImage()(ckbd).save(filename)
# #         ckbd_filenames_after.append(filename)

# #         ckbd = create_checkerboard(fixed_image, moving_image, patch_size=32)
# #         filename = os.path.join(checkerboard_before_save_folder, f'ckbd_{i}.png')
# #         ToPILImage()(ckbd).save(filename)
# #         ckbd_filenames_before.append(filename)

# #         # flicker animation
# #         filename = os.path.join(flicker_save_folder, f'flicker_{i}.gif')
# #         create_flicker(fixed_image, reg_image, output_path=filename)
# #         flicker_filenames.append(filename)

# #         # subtraction maps
# #         filename = os.path.join(difference_map_save_folder, f'diff_map_{i}.png')
# #         create_diff_map(fixed_image, reg_image, filename)
# #         difference_maps_filenames.append(filename)

# #         # compute dice between segmentation maps
# #         seg_dice_before = compute_dice(fixed_vessel, moving_vessel)
# #         seg_dice_after = compute_dice(fixed_vessel, reg_vessel)
# #         dice_before.append(seg_dice_before)
# #         dice_after.append(seg_dice_after)

# #         # quantitative evaluation
# #         if args.manual_keypoints is not None:

# #             fixed_kp_manual = batch_data['fixed_keypoints_manual']
# #             moving_kp_manual = batch_data['moving_keypoints_manual']
# #             fixed_kp_detected = batch_data['fixed_keypoints_detected']
# #             moving_kp_detected = batch_data['moving_keypoints_detected']

# #             # apply theta to keypoints
# #             try:
# #                 reg_kp = EyeLinerP.apply_transform_points(theta, moving_keypoints=moving_kp_manual)
# #             except:                
# #                 reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=moving_kp_manual, ctrl_keypoints=moving_kp_detected, tgt_keypoints=fixed_kp_detected, lmbda=args.lmbda)
# #                 # reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=fixed_kp_manual, ctrl_keypoints=fixed_kp_detected)

# #             if args.fire_eval:
# #                 fixed_kp_manual = 2912. * fixed_kp_manual / 256.
# #                 moving_kp_manual = 2912. * moving_kp_manual / 256.
# #                 reg_kp = 2912. * reg_kp / 256.

# #             # compute mean distance between fixed and registered keypoints
# #             md_before = torch.sqrt(torch.sum((fixed_kp_manual - moving_kp_manual)**2, dim=-1)).mean().item()
# #             md_after = torch.sqrt(torch.sum((fixed_kp_manual - reg_kp)**2, dim=-1)).mean().item()
# #             error_before.append(md_before)
# #             error_after.append(md_after)

# #     # ========================================
# #     # SAVE RESULTS TO DATAFRAME
# #     # ========================================
# #     dataset.data['registration_path'] = images_filenames
# #     dataset.data['checkerboard_before'] = ckbd_filenames_before
# #     dataset.data['checkerboard_after'] = ckbd_filenames_after
# #     dataset.data['flicker'] = flicker_filenames
# #     dataset.data['seg_overlap'] = segmentation_overlaps
# #     dataset.data['difference_map'] = difference_maps_filenames
# #     dataset.data['DICE_before'] = dice_before
# #     dataset.data['DICE_after'] = dice_after
    
# #     # add columns to dataframe
# #     if args.manual_keypoints is not None:
# #         dataset.data['MD_before'] = error_before
# #         dataset.data['MD_after'] = error_after

# #     # save results
# #     csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
# #     dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)

# #     # ========================================
# #     # PRINT SUMMARY STATISTICS
# #     # ========================================
    
# #     print("\n" + "="*60)
# #     print("REGISTRATION RESULTS SUMMARY")
# #     print("="*60)
# #     print(f"Dataset: {os.path.basename(args.data)}")
# #     print(f"Save location: {args.save}")
# #     print(f"Total image pairs: {len(dataset)}")
    
# #     # Print MD statistics
# #     if args.manual_keypoints is not None:
# #         # Filter out None values
# #         valid_md_before = [x for x in error_before if x is not None]
# #         valid_md_after = [x for x in error_after if x is not None]
        
# #         if valid_md_before:
# #             print(f"\n{'─'*60}")
# #             print(f"MEAN DISTANCE (MD) RESULTS")
# #             print(f"{'─'*60}")
# #             print(f"Lambda (TPS): {args.lmbda}")
# #             print(f"Successful registrations: {len(valid_md_after)}/{len(dataset)}")
# #             print(f"Failed registrations: {len(dataset) - len(valid_md_after)}")
            
# #             print(f"\nBefore Registration:")
# #             print(f"  Mean MD:   {sum(valid_md_before)/len(valid_md_before):8.2f} pixels")
# #             print(f"  Median MD: {sorted(valid_md_before)[len(valid_md_before)//2]:8.2f} pixels")
# #             print(f"  Min MD:    {min(valid_md_before):8.2f} pixels")
# #             print(f"  Max MD:    {max(valid_md_before):8.2f} pixels")
            
# #             print(f"\nAfter Registration:")
# #             mean_after = sum(valid_md_after)/len(valid_md_after)
# #             median_after = sorted(valid_md_after)[len(valid_md_after)//2]
# #             print(f"  Mean MD:   {mean_after:8.2f} pixels")
# #             print(f"  Median MD: {median_after:8.2f} pixels")
# #             print(f"  Min MD:    {min(valid_md_after):8.2f} pixels")
# #             print(f"  Max MD:    {max(valid_md_after):8.2f} pixels")
            
# #             improvement = sum(valid_md_before)/len(valid_md_before) - mean_after
# #             print(f"\n  ✓ Improvement: {improvement:8.2f} pixels")
            
# #             # Success rates
# #             success_5 = sum(1 for x in valid_md_after if x < 5) / len(valid_md_after) * 100
# #             success_10 = sum(1 for x in valid_md_after if x < 10) / len(valid_md_after) * 100
# #             print(f"  ✓ Success Rate (MD < 5px):  {success_5:5.1f}%")
# #             print(f"  ✓ Success Rate (MD < 10px): {success_10:5.1f}%")

# #     # Print DICE scores
# #     valid_dice_before = [x for x in dice_before if x is not None]
# #     valid_dice_after = [x for x in dice_after if x is not None]

# #     if valid_dice_before:
# #         print(f"\n{'─'*60}")
# #         print(f"DICE SCORES (Vessel Segmentation Overlap)")
# #         print(f"{'─'*60}")
# #         avg_dice_before = sum(valid_dice_before)/len(valid_dice_before)
# #         avg_dice_after = sum(valid_dice_after)/len(valid_dice_after)
# #         print(f"Before Registration: {avg_dice_before:.4f}")
# #         print(f"After Registration:  {avg_dice_after:.4f}")
# #         print(f"Improvement:         {(avg_dice_after - avg_dice_before):+.4f}")
    
# #     print("="*60)
# #     print(f"✓ Results saved to: {os.path.join(args.save, csv_save)}")
# #     print("="*60 + "\n")

# #     return

# # if __name__ == '__main__':
# #     args = parse_args()
# #     main(args)



# # import argparse
# # import os, sys
# # from tqdm import tqdm
# # from utils import none_or_str, compute_dice
# # from PIL import Image
# # import torch
# # from torchvision.transforms import ToPILImage
# # from data import ImageDataset
# # from eyeliner import EyeLinerP
# # from visualize import create_flicker, create_checkerboard, create_diff_map
# # from matplotlib import pyplot as plt

# # def parse_args():
# #     parser = argparse.ArgumentParser()
# #     # data args
# #     parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
# #     parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
# #     parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
# #     parser.add_argument('-fv', '--fixed-vessel', default='fixed_vessel_seg', type=none_or_str, help='Fixed vessel column')
# #     parser.add_argument('-mv', '--moving-vessel', default='moving_vessel_seg', type=none_or_str, help='Moving vessel column')
# #     parser.add_argument('-fd', '--fixed-disk', default='fixed_disk', type=none_or_str, help='Fixed disk column')
# #     parser.add_argument('-md', '--moving-disk', default='moving_disk', type=none_or_str, help='Moving disk column')
# #     parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
# #     parser.add_argument('-r', '--registration', default='registration_path', type=none_or_str, help='Registration column')
# #     parser.add_argument('-l', '--lmbda', default=None, type=float, help='Lambda value for computing the TPS quantitative result')
# #     parser.add_argument('--detected_keypoints', default='keypoints', type=none_or_str, help='Model Detected Keypoints column')
# #     parser.add_argument('--manual_keypoints', default='keypoints', type=none_or_str, help='Manually annotated Keypoints column')
# #     parser.add_argument('--fire_eval', action='store_true', help='Run FIRE evaluation')

# #     # misc
# #     parser.add_argument('--device', default='cpu', help='Device to run program on')
# #     parser.add_argument('--save', default='trained_models/', help='Location to save results')
# #     args = parser.parse_args()
# #     return args

# # def get_class_from_filename(filepath):
# #     """
# #     Extract class (A, P, or S) from image filename
# #     Example: 'data/retina_datasets/FIRE/images/A01_1.jpg' -> 'A'
# #     """
# #     filename = os.path.basename(filepath)
# #     filename_no_ext = os.path.splitext(filename)[0]
# #     first_letter = filename_no_ext[0].upper()
    
# #     if first_letter in ['A', 'P', 'S']:
# #         return first_letter
# #     else:
# #         return None

# # def main(args):

# #     device = torch.device(args.device)

# #     # load dataset
# #     dataset = ImageDataset(
# #         path=args.data, 
# #         input_col=args.fixed, 
# #         output_col=args.moving,
# #         input_vessel_col=args.fixed_vessel,
# #         output_vessel_col=args.moving_vessel,
# #         input_od_col=args.fixed_disk,
# #         output_od_col=args.moving_disk,
# #         input_dim=(args.size, args.size), 
# #         cmode='rgb',
# #         input='vessel',
# #         detected_keypoints_col=args.detected_keypoints,
# #         manual_keypoints_col=args.manual_keypoints,
# #         registration_col=args.registration
# #     )

# #     # make directory to store registrations
# #     reg_images_save_folder = os.path.join(args.save, 'registration_images')
# #     seg_overlaps_save_folder = os.path.join(args.save, 'seg_overlaps')
# #     checkerboard_after_save_folder = os.path.join(args.save, 'ckbd_images_after')
# #     checkerboard_before_save_folder = os.path.join(args.save, 'ckbd_images_before')
# #     flicker_save_folder = os.path.join(args.save, 'flicker_images')
# #     difference_map_save_folder = os.path.join(args.save, 'diff_map_images')
# #     os.makedirs(reg_images_save_folder, exist_ok=True)
# #     os.makedirs(checkerboard_after_save_folder, exist_ok=True)
# #     os.makedirs(checkerboard_before_save_folder, exist_ok=True)
# #     os.makedirs(flicker_save_folder, exist_ok=True)
# #     os.makedirs(difference_map_save_folder, exist_ok=True)
# #     os.makedirs(seg_overlaps_save_folder, exist_ok=True)

# #     images_filenames = []
# #     ckbd_filenames_before = []
# #     ckbd_filenames_after = []
# #     segmentation_overlaps = []
# #     flicker_filenames = []
# #     difference_maps_filenames = []
# #     dice_before = []
# #     dice_after = []
# #     image_classes = []
    
# #     if args.manual_keypoints is not None:
# #         error_before = []
# #         error_after = []

# #     # Separate results by class
# #     class_stats = {
# #         'A': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': []},
# #         'P': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': []},
# #         'S': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': []}
# #     }

# #     for i in tqdm(range(len(dataset))):
        
# #         # load images
# #         batch_data = dataset[i]
        
# #         # Get fixed image path to determine class
# #         fixed_image_path = dataset.data.iloc[i][args.fixed]
# #         img_class = get_class_from_filename(fixed_image_path)
# #         image_classes.append(img_class)
        
# #         fixed_image = batch_data['fixed_image']
# #         moving_image = batch_data['moving_image']
# #         fixed_vessel = batch_data['fixed_input']
# #         fixed_vessel = (fixed_vessel > 0.5).float()
# #         moving_vessel = batch_data['moving_input']
# #         moving_vessel = (moving_vessel > 0.5).float()
# #         theta = batch_data['theta']

# #         # if image pair could not be registered
# #         if theta is None:
# #             images_filenames.append(None)
# #             ckbd_filenames_after.append(None)
# #             ckbd_filenames_before.append(None)
# #             flicker_filenames.append(None)
# #             difference_maps_filenames.append(None)
# #             dice_before.append(None)
# #             dice_after.append(None)
# #             if args.manual_keypoints is not None:
# #                 error_before.append(None)
# #                 error_after.append(None)
# #             continue

# #         # register moving image
# #         try:
# #             reg_image = EyeLinerP.apply_transform(theta, moving_image)
# #         except:
# #             reg_image = EyeLinerP.apply_transform(theta[1], moving_image)

# #         # register moving segmentation
# #         try:
# #             reg_vessel = EyeLinerP.apply_transform(theta, moving_vessel)
# #         except:
# #             reg_vessel = EyeLinerP.apply_transform(theta[1], moving_vessel)
# #         reg_vessel = (reg_vessel > 0.5).float()

# #         # create mask
# #         reg_mask = torch.ones_like(moving_image)
# #         try:
# #             reg_mask = EyeLinerP.apply_transform(theta, reg_mask)
# #         except:
# #             reg_mask = EyeLinerP.apply_transform(theta[1], reg_mask)

# #         # apply mask to images
# #         fixed_image = fixed_image * reg_mask
# #         moving_image = moving_image * reg_mask
# #         reg_image = reg_image * reg_mask

# #         # apply mask to segmentations
# #         fixed_vessel = fixed_vessel
# #         moving_vessel = moving_vessel
# #         reg_vessel = reg_vessel * reg_mask

# #         # save registration image
# #         filename = os.path.join(reg_images_save_folder, f'reg_{i}.png')
# #         ToPILImage()(reg_image).save(filename)
# #         images_filenames.append(filename)

# #         # segmentation overlap
# #         seg1 = ToPILImage()(fixed_vessel)
# #         seg2 = ToPILImage()(reg_vessel)
# #         seg_overlap = Image.blend(seg1, seg2, alpha=0.5)
# #         filename = os.path.join(seg_overlaps_save_folder, f'seg_overlap_{i}.png')
# #         seg_overlap.save(filename)
# #         segmentation_overlaps.append(filename)

# #         # checkerboards
# #         ckbd = create_checkerboard(fixed_image, reg_image, patch_size=32)
# #         filename = os.path.join(checkerboard_after_save_folder, f'ckbd_{i}.png')
# #         ToPILImage()(ckbd).save(filename)
# #         ckbd_filenames_after.append(filename)

# #         ckbd = create_checkerboard(fixed_image, moving_image, patch_size=32)
# #         filename = os.path.join(checkerboard_before_save_folder, f'ckbd_{i}.png')
# #         ToPILImage()(ckbd).save(filename)
# #         ckbd_filenames_before.append(filename)

# #         # flicker animation
# #         filename = os.path.join(flicker_save_folder, f'flicker_{i}.gif')
# #         create_flicker(fixed_image, reg_image, output_path=filename)
# #         flicker_filenames.append(filename)

# #         # subtraction maps
# #         filename = os.path.join(difference_map_save_folder, f'diff_map_{i}.png')
# #         create_diff_map(fixed_image, reg_image, filename)
# #         difference_maps_filenames.append(filename)

# #         # compute dice between segmentation maps
# #         seg_dice_before = compute_dice(fixed_vessel, moving_vessel)
# #         seg_dice_after = compute_dice(fixed_vessel, reg_vessel)
# #         dice_before.append(seg_dice_before)
# #         dice_after.append(seg_dice_after)
        
# #         # Store class-specific DICE scores
# #         if img_class in class_stats:
# #             class_stats[img_class]['dice_before'].append(seg_dice_before)
# #             class_stats[img_class]['dice_after'].append(seg_dice_after)

# #         # quantitative evaluation
# #         if args.manual_keypoints is not None:

# #             fixed_kp_manual = batch_data['fixed_keypoints_manual']
# #             moving_kp_manual = batch_data['moving_keypoints_manual']
# #             fixed_kp_detected = batch_data['fixed_keypoints_detected']
# #             moving_kp_detected = batch_data['moving_keypoints_detected']

# #             # apply theta to keypoints
# #             try:
# #                 reg_kp = EyeLinerP.apply_transform_points(theta, moving_keypoints=moving_kp_manual)
# #             except:                
# #                 reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=moving_kp_manual, ctrl_keypoints=moving_kp_detected, tgt_keypoints=fixed_kp_detected, lmbda=args.lmbda)

# #             if args.fire_eval:
# #                 fixed_kp_manual = 2912. * fixed_kp_manual / 256.
# #                 moving_kp_manual = 2912. * moving_kp_manual / 256.
# #                 reg_kp = 2912. * reg_kp / 256.

# #             # compute mean distance between fixed and registered keypoints
# #             md_before = torch.sqrt(torch.sum((fixed_kp_manual - moving_kp_manual)**2, dim=-1)).mean().item()
# #             md_after = torch.sqrt(torch.sum((fixed_kp_manual - reg_kp)**2, dim=-1)).mean().item()
# #             error_before.append(md_before)
# #             error_after.append(md_after)
            
# #             # Store class-specific errors
# #             if img_class in class_stats:
# #                 class_stats[img_class]['error_before'].append(md_before)
# #                 class_stats[img_class]['error_after'].append(md_after)

# #     # ========================================
# #     # SAVE RESULTS TO DATAFRAME
# #     # ========================================
# #     dataset.data['registration_path'] = images_filenames
# #     dataset.data['checkerboard_before'] = ckbd_filenames_before
# #     dataset.data['checkerboard_after'] = ckbd_filenames_after
# #     dataset.data['flicker'] = flicker_filenames
# #     dataset.data['seg_overlap'] = segmentation_overlaps
# #     dataset.data['difference_map'] = difference_maps_filenames
# #     dataset.data['DICE_before'] = dice_before
# #     dataset.data['DICE_after'] = dice_after
# #     dataset.data['image_class'] = image_classes
    
# #     if args.manual_keypoints is not None:
# #         dataset.data['MD_before'] = error_before
# #         dataset.data['MD_after'] = error_after

# #     # save results
# #     csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
# #     dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)

# #     # ========================================
# #     # PRINT SUMMARY STATISTICS
# #     # ========================================
    
# #     print("\n" + "="*60)
# #     print("REGISTRATION RESULTS SUMMARY")
# #     print("="*60)
# #     print(f"Dataset: {os.path.basename(args.data)}")
# #     print(f"Save location: {args.save}")
# #     print(f"Total image pairs: {len(dataset)}")
    
# #     # Print class distribution
# #     print(f"\n{'─'*60}")
# #     print(f"CLASS DISTRIBUTION")
# #     print(f"{'─'*60}")
# #     for cls in ['A', 'P', 'S']:
# #         count = len(class_stats[cls]['dice_before'])
# #         print(f"Class {cls}: {count} images")
    
# #     # ========================================
# #     # TOTAL (ALL CLASSES COMBINED) STATISTICS
# #     # ========================================
# #     if args.manual_keypoints is not None:
# #         valid_md_before = [x for x in error_before if x is not None]
# #         valid_md_after = [x for x in error_after if x is not None]
        
# #         if valid_md_before:
# #             print(f"\n{'='*60}")
# #             print(f"TOTAL - ALL CLASSES COMBINED")
# #             print(f"{'='*60}")
# #             print(f"Lambda (TPS): {args.lmbda}")
# #             print(f"Successful registrations: {len(valid_md_after)}/{len(dataset)}")
# #             print(f"Failed registrations: {len(dataset) - len(valid_md_after)}")
            
# #             total_mean_before = sum(valid_md_before)/len(valid_md_before)
# #             total_mean_after = sum(valid_md_after)/len(valid_md_after)
# #             total_sum_before = sum(valid_md_before)
# #             total_sum_after = sum(valid_md_after)
            
# #             print(f"\nBefore Registration:")
# #             print(f"  Mean MD:   {total_mean_before:8.2f} pixels")
# #             print(f"  Sum MD:    {total_sum_before:8.2f} pixels")
# #             print(f"  Median MD: {sorted(valid_md_before)[len(valid_md_before)//2]:8.2f} pixels")
# #             print(f"  Min MD:    {min(valid_md_before):8.2f} pixels")
# #             print(f"  Max MD:    {max(valid_md_before):8.2f} pixels")
            
# #             print(f"\nAfter Registration:")
# #             print(f"  Mean MD:   {total_mean_after:8.2f} pixels")
# #             print(f"  Sum MD:    {total_sum_after:8.2f} pixels")
# #             print(f"  Median MD: {sorted(valid_md_after)[len(valid_md_after)//2]:8.2f} pixels")
# #             print(f"  Min MD:    {min(valid_md_after):8.2f} pixels")
# #             print(f"  Max MD:    {max(valid_md_after):8.2f} pixels")
            
# #             improvement = total_mean_before - total_mean_after
# #             print(f"\n  ✓ Improvement: {improvement:8.2f} pixels")
            
# #             success_5 = sum(1 for x in valid_md_after if x < 5) / len(valid_md_after) * 100
# #             success_10 = sum(1 for x in valid_md_after if x < 10) / len(valid_md_after) * 100
# #             print(f"  ✓ Success Rate (MD < 5px):  {success_5:5.1f}%")
# #             print(f"  ✓ Success Rate (MD < 10px): {success_10:5.1f}%")
            
# #             # ========================================
# #             # CLASS-SPECIFIC STATISTICS
# #             # ========================================
# #             print(f"\n{'='*60}")
# #             print(f"CLASS-SPECIFIC MD RESULTS")
# #             print(f"{'='*60}")
            
# #             sum_of_class_sums_before = 0
# #             sum_of_class_sums_after = 0
            
# #             for cls in ['A', 'P', 'S']:
# #                 cls_error_before = class_stats[cls]['error_before']
# #                 cls_error_after = class_stats[cls]['error_after']
                
# #                 if len(cls_error_before) > 0:
# #                     class_mean_before = sum(cls_error_before) / len(cls_error_before)
# #                     class_mean_after = sum(cls_error_after) / len(cls_error_after)
# #                     class_sum_before = sum(cls_error_before)
# #                     class_sum_after = sum(cls_error_after)
# #                     class_count = len(cls_error_before)
                    
# #                     sum_of_class_sums_before += class_sum_before
# #                     sum_of_class_sums_after += class_sum_after
                    
# #                     print(f"\nClass {cls} ({class_count} images):")
# #                     print(f"  Before: Mean = {class_mean_before:8.2f} pixels, Sum = {class_sum_before:8.2f} pixels")
# #                     print(f"  After:  Mean = {class_mean_after:8.2f} pixels, Sum = {class_sum_after:8.2f} pixels")
# #                     improvement = class_mean_before - class_mean_after
# #                     print(f"  ✓ Improvement: {improvement:8.2f} pixels")
            
# #             # ========================================
# #             # VERIFICATION
# #             # ========================================
# #             print(f"\n{'='*60}")
# #             print(f"VERIFICATION")
# #             print(f"{'='*60}")
            
# #             # Calculate what the total mean should be from class data
# #             calculated_total_mean_before = sum_of_class_sums_before / len(valid_md_before)
# #             calculated_total_mean_after = sum_of_class_sums_after / len(valid_md_after)
            
# #             print(f"Before Registration:")
# #             print(f"  Sum of all class sums:      {sum_of_class_sums_before:8.2f} pixels")
# #             print(f"  Total sum (direct):         {total_sum_before:8.2f} pixels")
# #             print(f"  Sums match: {'✓ YES' if abs(sum_of_class_sums_before - total_sum_before) < 0.01 else '✗ NO'}")
# #             print(f"\n  Total mean (direct):        {total_mean_before:8.2f} pixels")
# #             print(f"  Calculated from class sums: {calculated_total_mean_before:8.2f} pixels")
# #             print(f"  Formula: ({sum_of_class_sums_before:.2f}) / {len(valid_md_before)} = {calculated_total_mean_before:.2f}")
# #             print(f"  Means match: {'✓ YES' if abs(calculated_total_mean_before - total_mean_before) < 0.01 else '✗ NO'}")
            
# #             print(f"\nAfter Registration:")
# #             print(f"  Sum of all class sums:      {sum_of_class_sums_after:8.2f} pixels")
# #             print(f"  Total sum (direct):         {total_sum_after:8.2f} pixels")
# #             print(f"  Sums match: {'✓ YES' if abs(sum_of_class_sums_after - total_sum_after) < 0.01 else '✗ NO'}")
# #             print(f"\n  Total mean (direct):        {total_mean_after:8.2f} pixels")
# #             print(f"  Calculated from class sums: {calculated_total_mean_after:8.2f} pixels")
# #             print(f"  Formula: ({sum_of_class_sums_after:.2f}) / {len(valid_md_after)} = {calculated_total_mean_after:.2f}")
# #             print(f"  Means match: {'✓ YES' if abs(calculated_total_mean_after - total_mean_after) < 0.01 else '✗ NO'}")
            
# #             print(f"\n{'─'*60}")
# #             print(f"WEIGHTED AVERAGE VERIFICATION")
# #             print(f"{'─'*60}")
            
# #             # Calculate weighted average of class means
# #             weighted_mean_before = 0
# #             weighted_mean_after = 0
# #             total_count = 0
            
# #             for cls in ['A', 'P', 'S']:
# #                 if len(class_stats[cls]['error_after']) > 0:
# #                     class_mean_before = sum(class_stats[cls]['error_before']) / len(class_stats[cls]['error_before'])
# #                     class_mean_after = sum(class_stats[cls]['error_after']) / len(class_stats[cls]['error_after'])
# #                     class_count = len(class_stats[cls]['error_after'])
                    
# #                     weighted_mean_before += class_mean_before * class_count
# #                     weighted_mean_after += class_mean_after * class_count
# #                     total_count += class_count
            
# #             weighted_mean_before = weighted_mean_before / total_count
# #             weighted_mean_after = weighted_mean_after / total_count
            
# #             print(f"After Registration (HIGH PRECISION):")
# #             print(f"  Total mean (direct):             {total_mean_after:10.6f} pixels")
# #             print(f"  Weighted average of class means: {weighted_mean_after:10.6f} pixels")
# #             print(f"  Difference:                      {abs(total_mean_after - weighted_mean_after):10.6f} pixels")
# #             print(f"  Match: {'✓ YES' if abs(weighted_mean_after - total_mean_after) < 0.001 else '✗ NO'}")
            
# #             print(f"\nRounded to 2 decimals:")
# #             print(f"  Total mean:                      {total_mean_after:10.2f} pixels")
# #             print(f"  Weighted average:                {weighted_mean_after:10.2f} pixels")
# #             print(f"  ✓ These should be identical (any difference is rounding)")

# #     # DICE scores
# #     valid_dice_before = [x for x in dice_before if x is not None]
# #     valid_dice_after = [x for x in dice_after if x is not None]

# #     if valid_dice_before:
# #         print(f"\n{'='*60}")
# #         print(f"TOTAL DICE SCORES (Vessel Segmentation Overlap)")
# #         print(f"{'='*60}")
# #         total_dice_before = sum(valid_dice_before)/len(valid_dice_before)
# #         total_dice_after = sum(valid_dice_after)/len(valid_dice_after)
# #         print(f"Before Registration: {total_dice_before:.4f}")
# #         print(f"After Registration:  {total_dice_after:.4f}")
# #         print(f"Improvement:         {(total_dice_after - total_dice_before):+.4f}")
        
# #         # Class-specific DICE
# #         print(f"\n{'='*60}")
# #         print(f"CLASS-SPECIFIC DICE SCORES")
# #         print(f"{'='*60}")
# #         for cls in ['A', 'P', 'S']:
# #             cls_dice_before = class_stats[cls]['dice_before']
# #             cls_dice_after = class_stats[cls]['dice_after']
            
# #             if len(cls_dice_before) > 0:
# #                 class_dice_before = sum(cls_dice_before)/len(cls_dice_before)
# #                 class_dice_after = sum(cls_dice_after)/len(cls_dice_after)
# #                 print(f"\nClass {cls} ({len(cls_dice_before)} images):")
# #                 print(f"  Before: {class_dice_before:.4f}")
# #                 print(f"  After:  {class_dice_after:.4f}")
# #                 improvement = class_dice_after - class_dice_before
# #                 print(f"  Improvement: {improvement:+.4f}")
    
# #     print("="*60)
# #     print(f"✓ Results saved to: {os.path.join(args.save, csv_save)}")
# #     print("="*60 + "\n")

# #     return

# # if __name__ == '__main__':
# #     args = parse_args()
# #     main(args)


# import argparse
# import os, sys
# import time
# import json
# from tqdm import tqdm
# from utils import none_or_str, compute_dice
# from PIL import Image
# import torch
# from torchvision.transforms import ToPILImage
# from data import ImageDataset
# from eyeliner import EyeLinerP
# from visualize import create_flicker, create_checkerboard, create_diff_map
# from matplotlib import pyplot as plt

# def parse_args():
#     parser = argparse.ArgumentParser()
#     # data args
#     parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
#     parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
#     parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
#     parser.add_argument('-fv', '--fixed-vessel', default='fixed_vessel_seg', type=none_or_str, help='Fixed vessel column')
#     parser.add_argument('-mv', '--moving-vessel', default='moving_vessel_seg', type=none_or_str, help='Moving vessel column')
#     parser.add_argument('-fd', '--fixed-disk', default='fixed_disk', type=none_or_str, help='Fixed disk column')
#     parser.add_argument('-md', '--moving-disk', default='moving_disk', type=none_or_str, help='Moving disk column')
#     parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
#     parser.add_argument('-r', '--registration', default='registration_path', type=none_or_str, help='Registration column')
#     parser.add_argument('-l', '--lmbda', default=None, type=float, help='Lambda value for computing the TPS quantitative result')
#     parser.add_argument('--detected_keypoints', default='keypoints', type=none_or_str, help='Model Detected Keypoints column')
#     parser.add_argument('--manual_keypoints', default='keypoints', type=none_or_str, help='Manually annotated Keypoints column')
#     parser.add_argument('--fire_eval', action='store_true', help='Run FIRE evaluation')

#     # misc
#     parser.add_argument('--device', default='cpu', help='Device to run program on')
#     parser.add_argument('--save', default='trained_models/', help='Location to save results')
#     parser.add_argument('--save_runtime', action='store_true', help='Save detailed runtime statistics to JSON file')
#     args = parser.parse_args()
#     return args

# def get_class_from_filename(filepath):
#     """
#     Extract class (A, P, or S) from image filename
#     Example: 'data/retina_datasets/FIRE/images/A01_1.jpg' -> 'A'
#     """
#     filename = os.path.basename(filepath)
#     filename_no_ext = os.path.splitext(filename)[0]
#     first_letter = filename_no_ext[0].upper()
    
#     if first_letter in ['A', 'P', 'S']:
#         return first_letter
#     else:
#         return None

# def main(args):
#     # Start total execution timer
#     total_start_time = time.time()
    
#     # Runtime tracking dictionaries
#     per_image_times = []
#     per_image_breakdown = []
#     stage_times = {
#         'setup': 0,
#         'data_loading': 0,
#         'registration': 0,
#         'visualization': 0,
#         'metrics': 0,
#         'saving': 0
#     }
    
#     # Setup stage timing
#     setup_start = time.time()
    
#     device = torch.device(args.device)

#     # load dataset
#     dataset = ImageDataset(
#         path=args.data, 
#         input_col=args.fixed, 
#         output_col=args.moving,
#         input_vessel_col=args.fixed_vessel,
#         output_vessel_col=args.moving_vessel,
#         input_od_col=args.fixed_disk,
#         output_od_col=args.moving_disk,
#         input_dim=(args.size, args.size), 
#         cmode='rgb',
#         input='vessel',
#         detected_keypoints_col=args.detected_keypoints,
#         manual_keypoints_col=args.manual_keypoints,
#         registration_col=args.registration
#     )

#     # make directory to store registrations
#     reg_images_save_folder = os.path.join(args.save, 'registration_images')
#     seg_overlaps_save_folder = os.path.join(args.save, 'seg_overlaps')
#     checkerboard_after_save_folder = os.path.join(args.save, 'ckbd_images_after')
#     checkerboard_before_save_folder = os.path.join(args.save, 'ckbd_images_before')
#     flicker_save_folder = os.path.join(args.save, 'flicker_images')
#     difference_map_save_folder = os.path.join(args.save, 'diff_map_images')
#     os.makedirs(reg_images_save_folder, exist_ok=True)
#     os.makedirs(checkerboard_after_save_folder, exist_ok=True)
#     os.makedirs(checkerboard_before_save_folder, exist_ok=True)
#     os.makedirs(flicker_save_folder, exist_ok=True)
#     os.makedirs(difference_map_save_folder, exist_ok=True)
#     os.makedirs(seg_overlaps_save_folder, exist_ok=True)

#     images_filenames = []
#     ckbd_filenames_before = []
#     ckbd_filenames_after = []
#     segmentation_overlaps = []
#     flicker_filenames = []
#     difference_maps_filenames = []
#     dice_before = []
#     dice_after = []
#     image_classes = []
    
#     if args.manual_keypoints is not None:
#         error_before = []
#         error_after = []

#     # Separate results by class
#     class_stats = {
#         'A': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': []},
#         'P': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': []},
#         'S': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': []}
#     }
    
#     stage_times['setup'] = time.time() - setup_start

#     # Main processing loop with progress bar
#     for i in tqdm(range(len(dataset)), desc="Processing images"):
        
#         # Track time for this image
#         image_start_time = time.time()
#         breakdown = {
#             'data_loading': 0,
#             'registration': 0,
#             'visualization': 0,
#             'metrics': 0,
#             'saving': 0
#         }
        
#         # DATA LOADING
#         data_load_start = time.time()
#         batch_data = dataset[i]
        
#         # Get fixed image path to determine class
#         fixed_image_path = dataset.data.iloc[i][args.fixed]
#         img_class = get_class_from_filename(fixed_image_path)
#         image_classes.append(img_class)
        
#         fixed_image = batch_data['fixed_image']
#         moving_image = batch_data['moving_image']
#         fixed_vessel = batch_data['fixed_input']
#         fixed_vessel = (fixed_vessel > 0.5).float()
#         moving_vessel = batch_data['moving_input']
#         moving_vessel = (moving_vessel > 0.5).float()
#         theta = batch_data['theta']
        
#         breakdown['data_loading'] = time.time() - data_load_start
#         stage_times['data_loading'] += breakdown['data_loading']

#         # if image pair could not be registered
#         if theta is None:
#             images_filenames.append(None)
#             ckbd_filenames_after.append(None)
#             ckbd_filenames_before.append(None)
#             flicker_filenames.append(None)
#             difference_maps_filenames.append(None)
#             dice_before.append(None)
#             dice_after.append(None)
#             if args.manual_keypoints is not None:
#                 error_before.append(None)
#                 error_after.append(None)
            
#             # Record time even for failed registrations
#             total_image_time = time.time() - image_start_time
#             per_image_times.append(total_image_time)
#             breakdown['total'] = total_image_time
#             per_image_breakdown.append(breakdown)
#             continue

#         # REGISTRATION
#         reg_start = time.time()
#         try:
#             reg_image = EyeLinerP.apply_transform(theta, moving_image)
#         except:
#             reg_image = EyeLinerP.apply_transform(theta[1], moving_image)

#         # register moving segmentation
#         try:
#             reg_vessel = EyeLinerP.apply_transform(theta, moving_vessel)
#         except:
#             reg_vessel = EyeLinerP.apply_transform(theta[1], moving_vessel)
#         reg_vessel = (reg_vessel > 0.5).float()

#         # create mask
#         reg_mask = torch.ones_like(moving_image)
#         try:
#             reg_mask = EyeLinerP.apply_transform(theta, reg_mask)
#         except:
#             reg_mask = EyeLinerP.apply_transform(theta[1], reg_mask)

#         # apply mask to images
#         fixed_image = fixed_image * reg_mask
#         moving_image = moving_image * reg_mask
#         reg_image = reg_image * reg_mask

#         # apply mask to segmentations
#         fixed_vessel = fixed_vessel
#         moving_vessel = moving_vessel
#         reg_vessel = reg_vessel * reg_mask
        
#         breakdown['registration'] = time.time() - reg_start
#         stage_times['registration'] += breakdown['registration']

#         # SAVING REGISTRATION IMAGE
#         save_start = time.time()
#         filename = os.path.join(reg_images_save_folder, f'reg_{i}.png')
#         ToPILImage()(reg_image).save(filename)
#         images_filenames.append(filename)
#         breakdown['saving'] = time.time() - save_start
#         stage_times['saving'] += breakdown['saving']

#         # VISUALIZATION
#         viz_start = time.time()
        
#         # segmentation overlap
#         seg1 = ToPILImage()(fixed_vessel)
#         seg2 = ToPILImage()(reg_vessel)
#         seg_overlap = Image.blend(seg1, seg2, alpha=0.5)
#         filename = os.path.join(seg_overlaps_save_folder, f'seg_overlap_{i}.png')
#         seg_overlap.save(filename)
#         segmentation_overlaps.append(filename)

#         # checkerboards
#         ckbd = create_checkerboard(fixed_image, reg_image, patch_size=32)
#         filename = os.path.join(checkerboard_after_save_folder, f'ckbd_{i}.png')
#         ToPILImage()(ckbd).save(filename)
#         ckbd_filenames_after.append(filename)

#         ckbd = create_checkerboard(fixed_image, moving_image, patch_size=32)
#         filename = os.path.join(checkerboard_before_save_folder, f'ckbd_{i}.png')
#         ToPILImage()(ckbd).save(filename)
#         ckbd_filenames_before.append(filename)

#         # flicker animation
#         filename = os.path.join(flicker_save_folder, f'flicker_{i}.gif')
#         create_flicker(fixed_image, reg_image, output_path=filename)
#         flicker_filenames.append(filename)

#         # subtraction maps
#         filename = os.path.join(difference_map_save_folder, f'diff_map_{i}.png')
#         create_diff_map(fixed_image, reg_image, filename)
#         difference_maps_filenames.append(filename)
        
#         breakdown['visualization'] = time.time() - viz_start
#         stage_times['visualization'] += breakdown['visualization']

#         # METRICS COMPUTATION
#         metrics_start = time.time()
        
#         # compute dice between segmentation maps
#         seg_dice_before = compute_dice(fixed_vessel, moving_vessel)
#         seg_dice_after = compute_dice(fixed_vessel, reg_vessel)
#         dice_before.append(seg_dice_before)
#         dice_after.append(seg_dice_after)
        
#         # Store class-specific DICE scores
#         if img_class in class_stats:
#             class_stats[img_class]['dice_before'].append(seg_dice_before)
#             class_stats[img_class]['dice_after'].append(seg_dice_after)

#         # quantitative evaluation
#         if args.manual_keypoints is not None:

#             fixed_kp_manual = batch_data['fixed_keypoints_manual']
#             moving_kp_manual = batch_data['moving_keypoints_manual']
#             fixed_kp_detected = batch_data['fixed_keypoints_detected']
#             moving_kp_detected = batch_data['moving_keypoints_detected']

#             # apply theta to keypoints
#             try:
#                 reg_kp = EyeLinerP.apply_transform_points(theta, moving_keypoints=moving_kp_manual)
#             except:                
#                 reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=moving_kp_manual, ctrl_keypoints=moving_kp_detected, tgt_keypoints=fixed_kp_detected, lmbda=args.lmbda)

#             if args.fire_eval:
#                 fixed_kp_manual = 2912. * fixed_kp_manual / 256.
#                 moving_kp_manual = 2912. * moving_kp_manual / 256.
#                 reg_kp = 2912. * reg_kp / 256.

#             # compute mean distance between fixed and registered keypoints
#             md_before = torch.sqrt(torch.sum((fixed_kp_manual - moving_kp_manual)**2, dim=-1)).mean().item()
#             md_after = torch.sqrt(torch.sum((fixed_kp_manual - reg_kp)**2, dim=-1)).mean().item()
#             error_before.append(md_before)
#             error_after.append(md_after)
            
#             # Store class-specific errors
#             if img_class in class_stats:
#                 class_stats[img_class]['error_before'].append(md_before)
#                 class_stats[img_class]['error_after'].append(md_after)
        
#         breakdown['metrics'] = time.time() - metrics_start
#         stage_times['metrics'] += breakdown['metrics']
        
#         # Record total time for this image
#         total_image_time = time.time() - image_start_time
#         per_image_times.append(total_image_time)
#         breakdown['total'] = total_image_time
#         per_image_breakdown.append(breakdown)

#     # ========================================
#     # SAVE RESULTS TO DATAFRAME
#     # ========================================
#     save_results_start = time.time()
    
#     dataset.data['registration_path'] = images_filenames
#     dataset.data['checkerboard_before'] = ckbd_filenames_before
#     dataset.data['checkerboard_after'] = ckbd_filenames_after
#     dataset.data['flicker'] = flicker_filenames
#     dataset.data['seg_overlap'] = segmentation_overlaps
#     dataset.data['difference_map'] = difference_maps_filenames
#     dataset.data['DICE_before'] = dice_before
#     dataset.data['DICE_after'] = dice_after
#     dataset.data['image_class'] = image_classes
#     dataset.data['processing_time'] = per_image_times
    
#     if args.manual_keypoints is not None:
#         dataset.data['MD_before'] = error_before
#         dataset.data['MD_after'] = error_after

#     # save results
#     csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
#     dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)
    
#     stage_times['saving'] += time.time() - save_results_start

#     # ========================================
#     # PRINT SUMMARY STATISTICS
#     # ========================================
    
#     print("\n" + "="*60)
#     print("REGISTRATION RESULTS SUMMARY")
#     print("="*60)
#     print(f"Dataset: {os.path.basename(args.data)}")
#     print(f"Save location: {args.save}")
#     print(f"Total image pairs: {len(dataset)}")
    
#     # Print class distribution
#     print(f"\n{'─'*60}")
#     print(f"CLASS DISTRIBUTION")
#     print(f"{'─'*60}")
#     for cls in ['A', 'P', 'S']:
#         count = len(class_stats[cls]['dice_before'])
#         print(f"Class {cls}: {count} images")
    
#     # ========================================
#     # TOTAL (ALL CLASSES COMBINED) STATISTICS
#     # ========================================
#     if args.manual_keypoints is not None:
#         valid_md_before = [x for x in error_before if x is not None]
#         valid_md_after = [x for x in error_after if x is not None]
        
#         if valid_md_before:
#             print(f"\n{'='*60}")
#             print(f"TOTAL - ALL CLASSES COMBINED")
#             print(f"{'='*60}")
#             print(f"Lambda (TPS): {args.lmbda}")
#             print(f"Successful registrations: {len(valid_md_after)}/{len(dataset)}")
#             print(f"Failed registrations: {len(dataset) - len(valid_md_after)}")
            
#             total_mean_before = sum(valid_md_before)/len(valid_md_before)
#             total_mean_after = sum(valid_md_after)/len(valid_md_after)
#             total_sum_before = sum(valid_md_before)
#             total_sum_after = sum(valid_md_after)
            
#             print(f"\nBefore Registration:")
#             print(f"  Mean MD:   {total_mean_before:8.2f} pixels")
#             print(f"  Sum MD:    {total_sum_before:8.2f} pixels")
#             print(f"  Median MD: {sorted(valid_md_before)[len(valid_md_before)//2]:8.2f} pixels")
#             print(f"  Min MD:    {min(valid_md_before):8.2f} pixels")
#             print(f"  Max MD:    {max(valid_md_before):8.2f} pixels")
            
#             print(f"\nAfter Registration:")
#             print(f"  Mean MD:   {total_mean_after:8.2f} pixels")
#             print(f"  Sum MD:    {total_sum_after:8.2f} pixels")
#             print(f"  Median MD: {sorted(valid_md_after)[len(valid_md_after)//2]:8.2f} pixels")
#             print(f"  Min MD:    {min(valid_md_after):8.2f} pixels")
#             print(f"  Max MD:    {max(valid_md_after):8.2f} pixels")
            
#             improvement = total_mean_before - total_mean_after
#             print(f"\n  ✓ Improvement: {improvement:8.2f} pixels")
            
#             success_5 = sum(1 for x in valid_md_after if x < 5) / len(valid_md_after) * 100
#             success_10 = sum(1 for x in valid_md_after if x < 10) / len(valid_md_after) * 100
#             print(f"  ✓ Success Rate (MD < 5px):  {success_5:5.1f}%")
#             print(f"  ✓ Success Rate (MD < 10px): {success_10:5.1f}%")
            
#             # ========================================
#             # CLASS-SPECIFIC STATISTICS
#             # ========================================
#             print(f"\n{'='*60}")
#             print(f"CLASS-SPECIFIC MD RESULTS")
#             print(f"{'='*60}")
            
#             sum_of_class_sums_before = 0
#             sum_of_class_sums_after = 0
            
#             for cls in ['A', 'P', 'S']:
#                 cls_error_before = class_stats[cls]['error_before']
#                 cls_error_after = class_stats[cls]['error_after']
                
#                 if len(cls_error_before) > 0:
#                     class_mean_before = sum(cls_error_before) / len(cls_error_before)
#                     class_mean_after = sum(cls_error_after) / len(cls_error_after)
#                     class_sum_before = sum(cls_error_before)
#                     class_sum_after = sum(cls_error_after)
#                     class_count = len(cls_error_before)
                    
#                     sum_of_class_sums_before += class_sum_before
#                     sum_of_class_sums_after += class_sum_after
                    
#                     print(f"\nClass {cls} ({class_count} images):")
#                     print(f"  Before: Mean = {class_mean_before:8.2f} pixels, Sum = {class_sum_before:8.2f} pixels")
#                     print(f"  After:  Mean = {class_mean_after:8.2f} pixels, Sum = {class_sum_after:8.2f} pixels")
#                     improvement = class_mean_before - class_mean_after
#                     print(f"  ✓ Improvement: {improvement:8.2f} pixels")
            
#             # ========================================
#             # VERIFICATION
#             # ========================================
#             print(f"\n{'='*60}")
#             print(f"VERIFICATION")
#             print(f"{'='*60}")
            
#             # Calculate what the total mean should be from class data
#             calculated_total_mean_before = sum_of_class_sums_before / len(valid_md_before)
#             calculated_total_mean_after = sum_of_class_sums_after / len(valid_md_after)
            
#             print(f"Before Registration:")
#             print(f"  Sum of all class sums:      {sum_of_class_sums_before:8.2f} pixels")
#             print(f"  Total sum (direct):         {total_sum_before:8.2f} pixels")
#             print(f"  Sums match: {'✓ YES' if abs(sum_of_class_sums_before - total_sum_before) < 0.01 else '✗ NO'}")
#             print(f"\n  Total mean (direct):        {total_mean_before:8.2f} pixels")
#             print(f"  Calculated from class sums: {calculated_total_mean_before:8.2f} pixels")
#             print(f"  Formula: ({sum_of_class_sums_before:.2f}) / {len(valid_md_before)} = {calculated_total_mean_before:.2f}")
#             print(f"  Means match: {'✓ YES' if abs(calculated_total_mean_before - total_mean_before) < 0.01 else '✗ NO'}")
            
#             print(f"\nAfter Registration:")
#             print(f"  Sum of all class sums:      {sum_of_class_sums_after:8.2f} pixels")
#             print(f"  Total sum (direct):         {total_sum_after:8.2f} pixels")
#             print(f"  Sums match: {'✓ YES' if abs(sum_of_class_sums_after - total_sum_after) < 0.01 else '✗ NO'}")
#             print(f"\n  Total mean (direct):        {total_mean_after:8.2f} pixels")
#             print(f"  Calculated from class sums: {calculated_total_mean_after:8.2f} pixels")
#             print(f"  Formula: ({sum_of_class_sums_after:.2f}) / {len(valid_md_after)} = {calculated_total_mean_after:.2f}")
#             print(f"  Means match: {'✓ YES' if abs(calculated_total_mean_after - total_mean_after) < 0.01 else '✗ NO'}")
            
#             print(f"\n{'─'*60}")
#             print(f"WEIGHTED AVERAGE VERIFICATION")
#             print(f"{'─'*60}")
            
#             # Calculate weighted average of class means
#             weighted_mean_before = 0
#             weighted_mean_after = 0
#             total_count = 0
            
#             for cls in ['A', 'P', 'S']:
#                 if len(class_stats[cls]['error_after']) > 0:
#                     class_mean_before = sum(class_stats[cls]['error_before']) / len(class_stats[cls]['error_before'])
#                     class_mean_after = sum(class_stats[cls]['error_after']) / len(class_stats[cls]['error_after'])
#                     class_count = len(class_stats[cls]['error_after'])
                    
#                     weighted_mean_before += class_mean_before * class_count
#                     weighted_mean_after += class_mean_after * class_count
#                     total_count += class_count
            
#             weighted_mean_before = weighted_mean_before / total_count
#             weighted_mean_after = weighted_mean_after / total_count
            
#             print(f"After Registration (HIGH PRECISION):")
#             print(f"  Total mean (direct):             {total_mean_after:10.6f} pixels")
#             print(f"  Weighted average of class means: {weighted_mean_after:10.6f} pixels")
#             print(f"  Difference:                      {abs(total_mean_after - weighted_mean_after):10.6f} pixels")
#             print(f"  Match: {'✓ YES' if abs(weighted_mean_after - total_mean_after) < 0.001 else '✗ NO'}")
            
#             print(f"\nRounded to 2 decimals:")
#             print(f"  Total mean:                      {total_mean_after:10.2f} pixels")
#             print(f"  Weighted average:                {weighted_mean_after:10.2f} pixels")
#             print(f"  ✓ These should be identical (any difference is rounding)")

#     # DICE scores
#     valid_dice_before = [x for x in dice_before if x is not None]
#     valid_dice_after = [x for x in dice_after if x is not None]

#     if valid_dice_before:
#         print(f"\n{'='*60}")
#         print(f"TOTAL DICE SCORES (Vessel Segmentation Overlap)")
#         print(f"{'='*60}")
#         total_dice_before = sum(valid_dice_before)/len(valid_dice_before)
#         total_dice_after = sum(valid_dice_after)/len(valid_dice_after)
#         print(f"Before Registration: {total_dice_before:.4f}")
#         print(f"After Registration:  {total_dice_after:.4f}")
#         print(f"Improvement:         {(total_dice_after - total_dice_before):+.4f}")
        
#         # Class-specific DICE
#         print(f"\n{'='*60}")
#         print(f"CLASS-SPECIFIC DICE SCORES")
#         print(f"{'='*60}")
#         for cls in ['A', 'P', 'S']:
#             cls_dice_before = class_stats[cls]['dice_before']
#             cls_dice_after = class_stats[cls]['dice_after']
            
#             if len(cls_dice_before) > 0:
#                 class_dice_before = sum(cls_dice_before)/len(cls_dice_before)
#                 class_dice_after = sum(cls_dice_after)/len(cls_dice_after)
#                 print(f"\nClass {cls} ({len(cls_dice_before)} images):")
#                 print(f"  Before: {class_dice_before:.4f}")
#                 print(f"  After:  {class_dice_after:.4f}")
#                 improvement = class_dice_after - class_dice_before
#                 print(f"  Improvement: {improvement:+.4f}")
    
#     print("="*60)
#     print(f"✓ Results saved to: {os.path.join(args.save, csv_save)}")
#     print("="*60 + "\n")
    
#     # ========================================
#     # RUNTIME STATISTICS
#     # ========================================
#     total_runtime = time.time() - total_start_time
    
#     print("\n" + "="*60)
#     print("RUNTIME STATISTICS")
#     print("="*60)
    
#     # Per-image statistics
#     valid_times = [t for t in per_image_times if t is not None]
#     if valid_times:
#         avg_time = sum(valid_times) / len(valid_times)
#         min_time = min(valid_times)
#         max_time = max(valid_times)
#         median_time = sorted(valid_times)[len(valid_times)//2]
        
#         print(f"\nPer-Image Processing Time:")
#         print(f"  Total images processed: {len(valid_times)}")
#         print(f"  Average time per image: {avg_time:.4f}s")
#         print(f"  Median time per image:  {median_time:.4f}s")
#         print(f"  Min time per image:     {min_time:.4f}s")
#         print(f"  Max time per image:     {max_time:.4f}s")
#         print(f"  Total processing time:  {sum(valid_times):.2f}s ({sum(valid_times)/60:.2f} min)")
#         print(f"  Throughput:             {len(valid_times)/sum(valid_times):.2f} images/sec")
    
#     # Stage breakdown
#     print(f"\nStage-wise Time Breakdown:")
#     print(f"  Setup:           {stage_times['setup']:8.2f}s ({stage_times['setup']/total_runtime*100:5.1f}%)")
#     print(f"  Data Loading:    {stage_times['data_loading']:8.2f}s ({stage_times['data_loading']/total_runtime*100:5.1f}%)")
#     print(f"  Registration:    {stage_times['registration']:8.2f}s ({stage_times['registration']/total_runtime*100:5.1f}%)")
#     print(f"  Visualization:   {stage_times['visualization']:8.2f}s ({stage_times['visualization']/total_runtime*100:5.1f}%)")
#     print(f"  Metrics:         {stage_times['metrics']:8.2f}s ({stage_times['metrics']/total_runtime*100:5.1f}%)")
#     print(f"  Saving:          {stage_times['saving']:8.2f}s ({stage_times['saving']/total_runtime*100:5.1f}%)")
    
#     print(f"\n{'='*60}")
#     print(f"TOTAL EXECUTION TIME: {total_runtime:.2f}s ({total_runtime/60:.2f} min)")
#     print(f"{'='*60}\n")
    
#     # ========================================
#     # SAVE RUNTIME STATISTICS TO JSON
#     # ========================================
#     if args.save_runtime:
#         runtime_stats = {
#             'total_execution_time_seconds': total_runtime,
#             'total_execution_time_minutes': total_runtime / 60,
#             'total_images': len(dataset),
#             'successful_registrations': len(valid_times) if valid_times else 0,
#             'failed_registrations': len(dataset) - (len(valid_times) if valid_times else 0),
#             'per_image_stats': {
#                 'average_time_seconds': avg_time if valid_times else None,
#                 'median_time_seconds': median_time if valid_times else None,
#                 'min_time_seconds': min_time if valid_times else None,
#                 'max_time_seconds': max_time if valid_times else None,
#                 'throughput_images_per_second': len(valid_times)/sum(valid_times) if valid_times else None
#             },
#             'stage_breakdown_seconds': stage_times,
#             'stage_breakdown_percentage': {
#                 stage: (time_val/total_runtime*100) for stage, time_val in stage_times.items()
#             },
#             'individual_image_times': per_image_times,
#             'detailed_breakdown_per_image': per_image_breakdown
#         }
        
#         runtime_file = os.path.join(args.save, 'runtime_statistics.json')
#         with open(runtime_file, 'w') as f:
#             json.dump(runtime_stats, f, indent=2)
        
#         print(f"✓ Runtime statistics saved to: {runtime_file}\n")

#     return

# if __name__ == '__main__':
#     args = parse_args()
#     # main(args)



# import argparse
# import os, sys
# import time
# import json
# from tqdm import tqdm
# from utils import none_or_str, compute_dice
# from PIL import Image
# import torch
# from torchvision.transforms import ToPILImage
# from data import ImageDataset
# from eyeliner import EyeLinerP
# from visualize import create_flicker, create_checkerboard, create_diff_map
# from matplotlib import pyplot as plt
# import numpy as np
# from sklearn.metrics import mutual_info_score

# def compute_mutual_information(img1, img2, bins=256):
#     """
#     Compute Mutual Information between two images.
#     Higher MI indicates better alignment.
    
#     Args:
#         img1: First image tensor (C, H, W)
#         img2: Second image tensor (C, H, W)
#         bins: Number of histogram bins
    
#     Returns:
#         Mutual information score (float)
#     """
#     # Convert to numpy and flatten
#     img1_np = img1.cpu().numpy().flatten()
#     img2_np = img2.cpu().numpy().flatten()
    
#     # Discretize to histogram bins
#     img1_binned = np.digitize(img1_np, bins=np.linspace(0, 1, bins))
#     img2_binned = np.digitize(img2_np, bins=np.linspace(0, 1, bins))
    
#     # Compute mutual information
#     mi = mutual_info_score(img1_binned, img2_binned)
    
#     return mi

# def compute_normalized_mutual_information(img1, img2, bins=256):
#     """
#     Compute Normalized Mutual Information between two images.
#     NMI is in range [0, 1] where 1 indicates perfect alignment.
    
#     Args:
#         img1: First image tensor (C, H, W)
#         img2: Second image tensor (C, H, W)
#         bins: Number of histogram bins
    
#     Returns:
#         Normalized mutual information score (float)
#     """
#     # Convert to numpy and flatten
#     img1_np = img1.cpu().numpy().flatten()
#     img2_np = img2.cpu().numpy().flatten()
    
#     # Discretize to histogram bins
#     img1_binned = np.digitize(img1_np, bins=np.linspace(0, 1, bins))
#     img2_binned = np.digitize(img2_np, bins=np.linspace(0, 1, bins))
    
#     # Compute MI
#     mi = mutual_info_score(img1_binned, img2_binned)
    
#     # Compute entropies for normalization
#     h1 = compute_entropy(img1_binned, bins)
#     h2 = compute_entropy(img2_binned, bins)
    
#     # Normalized MI
#     nmi = 2 * mi / (h1 + h2) if (h1 + h2) > 0 else 0
    
#     return nmi

# def compute_entropy(data, bins):
#     """Compute entropy of discretized data"""
#     hist, _ = np.histogram(data, bins=bins)
#     hist = hist / hist.sum()
#     hist = hist[hist > 0]  # Remove zeros
#     return -np.sum(hist * np.log2(hist))

# def parse_args():
#     parser = argparse.ArgumentParser()
#     # data args
#     parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
#     parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
#     parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
#     parser.add_argument('-fv', '--fixed-vessel', default='fixed_vessel_seg', type=none_or_str, help='Fixed vessel column')
#     parser.add_argument('-mv', '--moving-vessel', default='moving_vessel_seg', type=none_or_str, help='Moving vessel column')
#     parser.add_argument('-fd', '--fixed-disk', default='fixed_disk', type=none_or_str, help='Fixed disk column')
#     parser.add_argument('-md', '--moving-disk', default='moving_disk', type=none_or_str, help='Moving disk column')
#     parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
#     parser.add_argument('-r', '--registration', default='registration_path', type=none_or_str, help='Registration column')
#     parser.add_argument('-l', '--lmbda', default=None, type=float, help='Lambda value for computing the TPS quantitative result')
#     parser.add_argument('--detected_keypoints', default='keypoints', type=none_or_str, help='Model Detected Keypoints column')
#     parser.add_argument('--manual_keypoints', default='keypoints', type=none_or_str, help='Manually annotated Keypoints column')
#     parser.add_argument('--fire_eval', action='store_true', help='Run FIRE evaluation')
#     parser.add_argument('--mle_bins', type=int, default=256, help='Number of bins for MI/NMI computation')

#     # misc
#     parser.add_argument('--device', default='cpu', help='Device to run program on')
#     parser.add_argument('--save', default='trained_models/', help='Location to save results')
#     parser.add_argument('--save_runtime', action='store_true', help='Save detailed runtime statistics to JSON file')
#     args = parser.parse_args()
#     return args

# def get_class_from_filename(filepath):
#     """
#     Extract class (A, P, or S) from image filename
#     Example: 'data/retina_datasets/FIRE/images/A01_1.jpg' -> 'A'
#     """
#     filename = os.path.basename(filepath)
#     filename_no_ext = os.path.splitext(filename)[0]
#     first_letter = filename_no_ext[0].upper()
    
#     if first_letter in ['A', 'P', 'S']:
#         return first_letter
#     else:
#         return None

# def main(args):
#     # Start total execution timer
#     total_start_time = time.time()
    
#     # Runtime tracking dictionaries
#     per_image_times = []
#     per_image_breakdown = []
#     stage_times = {
#         'setup': 0,
#         'data_loading': 0,
#         'registration': 0,
#         'visualization': 0,
#         'metrics': 0,
#         'saving': 0
#     }
    
#     # Setup stage timing
#     setup_start = time.time()
    
#     device = torch.device(args.device)

#     # load dataset
#     dataset = ImageDataset(
#         path=args.data, 
#         input_col=args.fixed, 
#         output_col=args.moving,
#         input_vessel_col=args.fixed_vessel,
#         output_vessel_col=args.moving_vessel,
#         input_od_col=args.fixed_disk,
#         output_od_col=args.moving_disk,
#         input_dim=(args.size, args.size), 
#         cmode='rgb',
#         input='vessel',
#         detected_keypoints_col=args.detected_keypoints,
#         manual_keypoints_col=args.manual_keypoints,
#         registration_col=args.registration
#     )

#     # make directory to store registrations
#     reg_images_save_folder = os.path.join(args.save, 'registration_images')
#     seg_overlaps_save_folder = os.path.join(args.save, 'seg_overlaps')
#     checkerboard_after_save_folder = os.path.join(args.save, 'ckbd_images_after')
#     checkerboard_before_save_folder = os.path.join(args.save, 'ckbd_images_before')
#     flicker_save_folder = os.path.join(args.save, 'flicker_images')
#     difference_map_save_folder = os.path.join(args.save, 'diff_map_images')
#     os.makedirs(reg_images_save_folder, exist_ok=True)
#     os.makedirs(checkerboard_after_save_folder, exist_ok=True)
#     os.makedirs(checkerboard_before_save_folder, exist_ok=True)
#     os.makedirs(flicker_save_folder, exist_ok=True)
#     os.makedirs(difference_map_save_folder, exist_ok=True)
#     os.makedirs(seg_overlaps_save_folder, exist_ok=True)

#     images_filenames = []
#     ckbd_filenames_before = []
#     ckbd_filenames_after = []
#     segmentation_overlaps = []
#     flicker_filenames = []
#     difference_maps_filenames = []
#     dice_before = []
#     dice_after = []
#     image_classes = []
    
#     # NEW: MLE score lists
#     mi_before = []
#     mi_after = []
#     nmi_before = []
#     nmi_after = []
    
#     if args.manual_keypoints is not None:
#         error_before = []
#         error_after = []

#     # Separate results by class
#     class_stats = {
#         'A': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': [], 
#               'mi_before': [], 'mi_after': [], 'nmi_before': [], 'nmi_after': []},
#         'P': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': [],
#               'mi_before': [], 'mi_after': [], 'nmi_before': [], 'nmi_after': []},
#         'S': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': [],
#               'mi_before': [], 'mi_after': [], 'nmi_before': [], 'nmi_after': []}
#     }
    
#     stage_times['setup'] = time.time() - setup_start

#     # Main processing loop with progress bar
#     for i in tqdm(range(len(dataset)), desc="Processing images"):
        
#         # Track time for this image
#         image_start_time = time.time()
#         breakdown = {
#             'data_loading': 0,
#             'registration': 0,
#             'visualization': 0,
#             'metrics': 0,
#             'saving': 0
#         }
        
#         # DATA LOADING
#         data_load_start = time.time()
#         batch_data = dataset[i]
        
#         # Get fixed image path to determine class
#         fixed_image_path = dataset.data.iloc[i][args.fixed]
#         img_class = get_class_from_filename(fixed_image_path)
#         image_classes.append(img_class)
        
#         fixed_image = batch_data['fixed_image']
#         moving_image = batch_data['moving_image']
#         fixed_vessel = batch_data['fixed_input']
#         fixed_vessel = (fixed_vessel > 0.5).float()
#         moving_vessel = batch_data['moving_input']
#         moving_vessel = (moving_vessel > 0.5).float()
#         theta = batch_data['theta']
        
#         breakdown['data_loading'] = time.time() - data_load_start
#         stage_times['data_loading'] += breakdown['data_loading']

#         # if image pair could not be registered
#         if theta is None:
#             images_filenames.append(None)
#             ckbd_filenames_after.append(None)
#             ckbd_filenames_before.append(None)
#             flicker_filenames.append(None)
#             difference_maps_filenames.append(None)
#             dice_before.append(None)
#             dice_after.append(None)
#             mi_before.append(None)
#             mi_after.append(None)
#             nmi_before.append(None)
#             nmi_after.append(None)
#             if args.manual_keypoints is not None:
#                 error_before.append(None)
#                 error_after.append(None)
            
#             # Record time even for failed registrations
#             total_image_time = time.time() - image_start_time
#             per_image_times.append(total_image_time)
#             breakdown['total'] = total_image_time
#             per_image_breakdown.append(breakdown)
#             continue

#         # REGISTRATION
#         reg_start = time.time()
#         try:
#             reg_image = EyeLinerP.apply_transform(theta, moving_image)
#         except:
#             reg_image = EyeLinerP.apply_transform(theta[1], moving_image)

#         # register moving segmentation
#         try:
#             reg_vessel = EyeLinerP.apply_transform(theta, moving_vessel)
#         except:
#             reg_vessel = EyeLinerP.apply_transform(theta[1], moving_vessel)
#         reg_vessel = (reg_vessel > 0.5).float()

#         # create mask
#         reg_mask = torch.ones_like(moving_image)
#         try:
#             reg_mask = EyeLinerP.apply_transform(theta, reg_mask)
#         except:
#             reg_mask = EyeLinerP.apply_transform(theta[1], reg_mask)

#         # apply mask to images
#         fixed_image = fixed_image * reg_mask
#         moving_image = moving_image * reg_mask
#         reg_image = reg_image * reg_mask

#         # apply mask to segmentations
#         fixed_vessel = fixed_vessel
#         moving_vessel = moving_vessel
#         reg_vessel = reg_vessel * reg_mask
        
#         breakdown['registration'] = time.time() - reg_start
#         stage_times['registration'] += breakdown['registration']

#         # SAVING REGISTRATION IMAGE
#         save_start = time.time()
#         filename = os.path.join(reg_images_save_folder, f'reg_{i}.png')
#         ToPILImage()(reg_image).save(filename)
#         images_filenames.append(filename)
#         breakdown['saving'] = time.time() - save_start
#         stage_times['saving'] += breakdown['saving']

#         # VISUALIZATION
#         viz_start = time.time()
        
#         # segmentation overlap
#         seg1 = ToPILImage()(fixed_vessel)
#         seg2 = ToPILImage()(reg_vessel)
#         seg_overlap = Image.blend(seg1, seg2, alpha=0.5)
#         filename = os.path.join(seg_overlaps_save_folder, f'seg_overlap_{i}.png')
#         seg_overlap.save(filename)
#         segmentation_overlaps.append(filename)

#         # checkerboards
#         ckbd = create_checkerboard(fixed_image, reg_image, patch_size=32)
#         filename = os.path.join(checkerboard_after_save_folder, f'ckbd_{i}.png')
#         ToPILImage()(ckbd).save(filename)
#         ckbd_filenames_after.append(filename)

#         ckbd = create_checkerboard(fixed_image, moving_image, patch_size=32)
#         filename = os.path.join(checkerboard_before_save_folder, f'ckbd_{i}.png')
#         ToPILImage()(ckbd).save(filename)
#         ckbd_filenames_before.append(filename)

#         # flicker animation
#         filename = os.path.join(flicker_save_folder, f'flicker_{i}.gif')
#         create_flicker(fixed_image, reg_image, output_path=filename)
#         flicker_filenames.append(filename)

#         # subtraction maps
#         filename = os.path.join(difference_map_save_folder, f'diff_map_{i}.png')
#         create_diff_map(fixed_image, reg_image, filename)
#         difference_maps_filenames.append(filename)
        
#         breakdown['visualization'] = time.time() - viz_start
#         stage_times['visualization'] += breakdown['visualization']

#         # METRICS COMPUTATION
#         metrics_start = time.time()
        
#         # compute dice between segmentation maps
#         seg_dice_before = compute_dice(fixed_vessel, moving_vessel)
#         seg_dice_after = compute_dice(fixed_vessel, reg_vessel)
#         dice_before.append(seg_dice_before)
#         dice_after.append(seg_dice_after)
        
#         # NEW: Compute MLE scores (Mutual Information)
#         img_mi_before = compute_mutual_information(fixed_image, moving_image, bins=args.mle_bins)
#         img_mi_after = compute_mutual_information(fixed_image, reg_image, bins=args.mle_bins)
#         mi_before.append(img_mi_before)
#         mi_after.append(img_mi_after)
        
#         # NEW: Compute Normalized Mutual Information
#         img_nmi_before = compute_normalized_mutual_information(fixed_image, moving_image, bins=args.mle_bins)
#         img_nmi_after = compute_normalized_mutual_information(fixed_image, reg_image, bins=args.mle_bins)
#         nmi_before.append(img_nmi_before)
#         nmi_after.append(img_nmi_after)
        
#         # Store class-specific scores
#         if img_class in class_stats:
#             class_stats[img_class]['dice_before'].append(seg_dice_before)
#             class_stats[img_class]['dice_after'].append(seg_dice_after)
#             class_stats[img_class]['mi_before'].append(img_mi_before)
#             class_stats[img_class]['mi_after'].append(img_mi_after)
#             class_stats[img_class]['nmi_before'].append(img_nmi_before)
#             class_stats[img_class]['nmi_after'].append(img_nmi_after)

#         # quantitative evaluation
#         if args.manual_keypoints is not None:

#             fixed_kp_manual = batch_data['fixed_keypoints_manual']
#             moving_kp_manual = batch_data['moving_keypoints_manual']
#             fixed_kp_detected = batch_data['fixed_keypoints_detected']
#             moving_kp_detected = batch_data['moving_keypoints_detected']

#             # apply theta to keypoints
#             try:
#                 reg_kp = EyeLinerP.apply_transform_points(theta, moving_keypoints=moving_kp_manual)
#             except:                
#                 reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=moving_kp_manual, ctrl_keypoints=moving_kp_detected, tgt_keypoints=fixed_kp_detected, lmbda=args.lmbda)

#             if args.fire_eval:
#                 fixed_kp_manual = 2912. * fixed_kp_manual / 256.
#                 moving_kp_manual = 2912. * moving_kp_manual / 256.
#                 reg_kp = 2912. * reg_kp / 256.

#             # compute mean distance between fixed and registered keypoints
#             md_before = torch.sqrt(torch.sum((fixed_kp_manual - moving_kp_manual)**2, dim=-1)).mean().item()
#             md_after = torch.sqrt(torch.sum((fixed_kp_manual - reg_kp)**2, dim=-1)).mean().item()
#             error_before.append(md_before)
#             error_after.append(md_after)
            
#             # Store class-specific errors
#             if img_class in class_stats:
#                 class_stats[img_class]['error_before'].append(md_before)
#                 class_stats[img_class]['error_after'].append(md_after)
        
#         breakdown['metrics'] = time.time() - metrics_start
#         stage_times['metrics'] += breakdown['metrics']
        
#         # Record total time for this image
#         total_image_time = time.time() - image_start_time
#         per_image_times.append(total_image_time)
#         breakdown['total'] = total_image_time
#         per_image_breakdown.append(breakdown)

#     # ========================================
#     # SAVE RESULTS TO DATAFRAME
#     # ========================================
#     save_results_start = time.time()
    
#     dataset.data['registration_path'] = images_filenames
#     dataset.data['checkerboard_before'] = ckbd_filenames_before
#     dataset.data['checkerboard_after'] = ckbd_filenames_after
#     dataset.data['flicker'] = flicker_filenames
#     dataset.data['seg_overlap'] = segmentation_overlaps
#     dataset.data['difference_map'] = difference_maps_filenames
#     dataset.data['DICE_before'] = dice_before
#     dataset.data['DICE_after'] = dice_after
#     dataset.data['MI_before'] = mi_before
#     dataset.data['MI_after'] = mi_after
#     dataset.data['NMI_before'] = nmi_before
#     dataset.data['NMI_after'] = nmi_after
#     dataset.data['image_class'] = image_classes
#     dataset.data['processing_time'] = per_image_times
    
#     if args.manual_keypoints is not None:
#         dataset.data['MD_before'] = error_before
#         dataset.data['MD_after'] = error_after

#     # save results
#     csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
#     dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)
    
#     stage_times['saving'] += time.time() - save_results_start

#     # ========================================
#     # PRINT SUMMARY STATISTICS
#     # ========================================
    
#     print("\n" + "="*60)
#     print("REGISTRATION RESULTS SUMMARY")
#     print("="*60)
#     print(f"Dataset: {os.path.basename(args.data)}")
#     print(f"Save location: {args.save}")
#     print(f"Total image pairs: {len(dataset)}")
    
#     # Print class distribution
#     print(f"\n{'─'*60}")
#     print(f"CLASS DISTRIBUTION")
#     print(f"{'─'*60}")
#     for cls in ['A', 'P', 'S']:
#         count = len(class_stats[cls]['dice_before'])
#         print(f"Class {cls}: {count} images")
    
#     # ========================================
#     # TOTAL (ALL CLASSES COMBINED) STATISTICS
#     # ========================================
#     if args.manual_keypoints is not None:
#         valid_md_before = [x for x in error_before if x is not None]
#         valid_md_after = [x for x in error_after if x is not None]
        
#         if valid_md_before:
#             print(f"\n{'='*60}")
#             print(f"TOTAL - ALL CLASSES COMBINED")
#             print(f"{'='*60}")
#             print(f"Lambda (TPS): {args.lmbda}")
#             print(f"Successful registrations: {len(valid_md_after)}/{len(dataset)}")
#             print(f"Failed registrations: {len(dataset) - len(valid_md_after)}")
            
#             total_mean_before = sum(valid_md_before)/len(valid_md_before)
#             total_mean_after = sum(valid_md_after)/len(valid_md_after)
#             total_sum_before = sum(valid_md_before)
#             total_sum_after = sum(valid_md_after)
            
#             print(f"\nBefore Registration:")
#             print(f"  Mean MD:   {total_mean_before:8.2f} pixels")
#             print(f"  Sum MD:    {total_sum_before:8.2f} pixels")
#             print(f"  Median MD: {sorted(valid_md_before)[len(valid_md_before)//2]:8.2f} pixels")
#             print(f"  Min MD:    {min(valid_md_before):8.2f} pixels")
#             print(f"  Max MD:    {max(valid_md_before):8.2f} pixels")
            
#             print(f"\nAfter Registration:")
#             print(f"  Mean MD:   {total_mean_after:8.2f} pixels")
#             print(f"  Sum MD:    {total_sum_after:8.2f} pixels")
#             print(f"  Median MD: {sorted(valid_md_after)[len(valid_md_after)//2]:8.2f} pixels")
#             print(f"  Min MD:    {min(valid_md_after):8.2f} pixels")
#             print(f"  Max MD:    {max(valid_md_after):8.2f} pixels")
            
#             improvement = total_mean_before - total_mean_after
#             print(f"\n  ✓ Improvement: {improvement:8.2f} pixels")
            
#             success_5 = sum(1 for x in valid_md_after if x < 5) / len(valid_md_after) * 100
#             success_10 = sum(1 for x in valid_md_after if x < 10) / len(valid_md_after) * 100
#             print(f"  ✓ Success Rate (MD < 5px):  {success_5:5.1f}%")
#             print(f"  ✓ Success Rate (MD < 10px): {success_10:5.1f}%")
            
#             # ========================================
#             # CLASS-SPECIFIC STATISTICS
#             # ========================================
#             print(f"\n{'='*60}")
#             print(f"CLASS-SPECIFIC MD RESULTS")
#             print(f"{'='*60}")
            
#             sum_of_class_sums_before = 0
#             sum_of_class_sums_after = 0
            
#             for cls in ['A', 'P', 'S']:
#                 cls_error_before = class_stats[cls]['error_before']
#                 cls_error_after = class_stats[cls]['error_after']
                
#                 if len(cls_error_before) > 0:
#                     class_mean_before = sum(cls_error_before) / len(cls_error_before)
#                     class_mean_after = sum(cls_error_after) / len(cls_error_after)
#                     class_sum_before = sum(cls_error_before)
#                     class_sum_after = sum(cls_error_after)
#                     class_count = len(cls_error_before)
                    
#                     sum_of_class_sums_before += class_sum_before
#                     sum_of_class_sums_after += class_sum_after
                    
#                     print(f"\nClass {cls} ({class_count} images):")
#                     print(f"  Before: Mean = {class_mean_before:8.2f} pixels, Sum = {class_sum_before:8.2f} pixels")
#                     print(f"  After:  Mean = {class_mean_after:8.2f} pixels, Sum = {class_sum_after:8.2f} pixels")
#                     improvement = class_mean_before - class_mean_after
#                     print(f"  ✓ Improvement: {improvement:8.2f} pixels")
            
#             # ========================================
#             # VERIFICATION
#             # ========================================
#             print(f"\n{'='*60}")
#             print(f"VERIFICATION")
#             print(f"{'='*60}")
            
#             # Calculate what the total mean should be from class data
#             calculated_total_mean_before = sum_of_class_sums_before / len(valid_md_before)
#             calculated_total_mean_after = sum_of_class_sums_after / len(valid_md_after)
            
#             print(f"Before Registration:")
#             print(f"  Sum of all class sums:      {sum_of_class_sums_before:8.2f} pixels")
#             print(f"  Total sum (direct):         {total_sum_before:8.2f} pixels")
#             print(f"  Sums match: {'✓ YES' if abs(sum_of_class_sums_before - total_sum_before) < 0.01 else '✗ NO'}")
#             print(f"\n  Total mean (direct):        {total_mean_before:8.2f} pixels")
#             print(f"  Calculated from class sums: {calculated_total_mean_before:8.2f} pixels")
#             print(f"  Formula: ({sum_of_class_sums_before:.2f}) / {len(valid_md_before)} = {calculated_total_mean_before:.2f}")
#             print(f"  Means match: {'✓ YES' if abs(calculated_total_mean_before - total_mean_before) < 0.01 else '✗ NO'}")
            
#             print(f"\nAfter Registration:")
#             print(f"  Sum of all class sums:      {sum_of_class_sums_after:8.2f} pixels")
#             print(f"  Total sum (direct):         {total_sum_after:8.2f} pixels")
#             print(f"  Sums match: {'✓ YES' if abs(sum_of_class_sums_after - total_sum_after) < 0.01 else '✗ NO'}")
#             print(f"\n  Total mean (direct):        {total_mean_after:8.2f} pixels")
#             print(f"  Calculated from class sums: {calculated_total_mean_after:8.2f} pixels")
#             print(f"  Formula: ({sum_of_class_sums_after:.2f}) / {len(valid_md_after)} = {calculated_total_mean_after:.2f}")
#             print(f"  Means match: {'✓ YES' if abs(calculated_total_mean_after - total_mean_after) < 0.01 else '✗ NO'}")
            
#             print(f"\n{'─'*60}")
#             print(f"WEIGHTED AVERAGE VERIFICATION")
#             print(f"{'─'*60}")
            
#             # Calculate weighted average of class means
#             weighted_mean_before = 0
#             weighted_mean_after = 0
#             total_count = 0
            
#             for cls in ['A', 'P', 'S']:
#                 if len(class_stats[cls]['error_after']) > 0:
#                     class_mean_before = sum(class_stats[cls]['error_before']) / len(class_stats[cls]['error_before'])
#                     class_mean_after = sum(class_stats[cls]['error_after']) / len(class_stats[cls]['error_after'])
#                     class_count = len(class_stats[cls]['error_after'])
                    
#                     weighted_mean_before += class_mean_before * class_count
#                     weighted_mean_after += class_mean_after * class_count
#                     total_count += class_count
            
#             weighted_mean_before = weighted_mean_before / total_count
#             weighted_mean_after = weighted_mean_after / total_count
            
#             print(f"After Registration (HIGH PRECISION):")
#             print(f"  Total mean (direct):             {total_mean_after:10.6f} pixels")
#             print(f"  Weighted average of class means: {weighted_mean_after:10.6f} pixels")
#             print(f"  Difference:                      {abs(total_mean_after - weighted_mean_after):10.6f} pixels")
#             print(f"  Match: {'✓ YES' if abs(weighted_mean_after - total_mean_after) < 0.001 else '✗ NO'}")
            
#             print(f"\nRounded to 2 decimals:")
#             print(f"  Total mean:                      {total_mean_after:10.2f} pixels")
#             print(f"  Weighted average:                {weighted_mean_after:10.2f} pixels")
#             print(f"  ✓ These should be identical (any difference is rounding)")

#     # DICE scores
#     valid_dice_before = [x for x in dice_before if x is not None]
#     valid_dice_after = [x for x in dice_after if x is not None]

#     if valid_dice_before:
#         print(f"\n{'='*60}")
#         print(f"TOTAL DICE SCORES (Vessel Segmentation Overlap)")
#         print(f"{'='*60}")
#         total_dice_before = sum(valid_dice_before)/len(valid_dice_before)
#         total_dice_after = sum(valid_dice_after)/len(valid_dice_after)
#         print(f"Before Registration: {total_dice_before:.4f}")
#         print(f"After Registration:  {total_dice_after:.4f}")
#         print(f"Improvement:         {(total_dice_after - total_dice_before):+.4f}")
        
#         # Class-specific DICE
#         print(f"\n{'='*60}")
#         print(f"CLASS-SPECIFIC DICE SCORES")
#         print(f"{'='*60}")
#         for cls in ['A', 'P', 'S']:
#             cls_dice_before = class_stats[cls]['dice_before']
#             cls_dice_after = class_stats[cls]['dice_after']
            
#             if len(cls_dice_before) > 0:
#                 class_dice_before = sum(cls_dice_before)/len(cls_dice_before)
#                 class_dice_after = sum(cls_dice_after)/len(cls_dice_after)
#                 print(f"\nClass {cls} ({len(cls_dice_before)} images):")
#                 print(f"  Before: {class_dice_before:.4f}")
#                 print(f"  After:  {class_dice_after:.4f}")
#                 improvement = class_dice_after - class_dice_before
#                 print(f"  Improvement: {improvement:+.4f}")
    
#     # ========================================
#     # NEW: MLE SCORES (MUTUAL INFORMATION)
#     # ========================================
#     valid_mi_before = [x for x in mi_before if x is not None]
#     valid_mi_after = [x for x in mi_after if x is not None]
#     valid_nmi_before = [x for x in nmi_before if x is not None]
#     valid_nmi_after = [x for x in nmi_after if x is not None]
    
#     if valid_mi_before:
#         print(f"\n{'='*60}")
#         print(f"MUTUAL INFORMATION (MI) SCORES")
#         print(f"{'='*60}")
#         print(f"Note: Higher MI indicates better alignment")
#         total_mi_before = sum(valid_mi_before)/len(valid_mi_before)
#         total_mi_after = sum(valid_mi_after)/len(valid_mi_after)
#         print(f"Before Registration: {total_mi_before:.4f}")
#         print(f"After Registration:  {total_mi_after:.4f}")
#         print(f"Improvement:         {(total_mi_after - total_mi_before):+.4f}")
        
#         # Class-specific MI
#         print(f"\n{'='*60}")
#         print(f"CLASS-SPECIFIC MI SCORES")
#         print(f"{'='*60}")
#         for cls in ['A', 'P', 'S']:
#             cls_mi_before = class_stats[cls]['mi_before']
#             cls_mi_after = class_stats[cls]['mi_after']
            
#             if len(cls_mi_before) > 0:
#                 class_mi_before = sum(cls_mi_before)/len(cls_mi_before)
#                 class_mi_after = sum(cls_mi_after)/len(cls_mi_after)
#                 print(f"\nClass {cls} ({len(cls_mi_before)} images):")
#                 print(f"  Before: {class_mi_before:.4f}")
#                 print(f"  After:  {class_mi_after:.4f}")
#                 improvement = class_mi_after - class_mi_before
#                 print(f"  Improvement: {improvement:+.4f}")
    
#     if valid_nmi_before:
#         print(f"\n{'='*60}")
#         print(f"NORMALIZED MUTUAL INFORMATION (NMI) SCORES")
#         print(f"{'='*60}")
#         print(f"Note: NMI range [0, 1], higher is better")
#         total_nmi_before = sum(valid_nmi_before)/len(valid_nmi_before)
#         total_nmi_after = sum(valid_nmi_after)/len(valid_nmi_after)
#         print(f"Before Registration: {total_nmi_before:.4f}")
#         print(f"After Registration:  {total_nmi_after:.4f}")
#         print(f"Improvement:         {(total_nmi_after - total_nmi_before):+.4f}")
        
#         # Class-specific NMI
#         print(f"\n{'='*60}")
#         print(f"CLASS-SPECIFIC NMI SCORES")
#         print(f"{'='*60}")
#         for cls in ['A', 'P', 'S']:
#             cls_nmi_before = class_stats[cls]['nmi_before']
#             cls_nmi_after = class_stats[cls]['nmi_after']
            
#             if len(cls_nmi_before) > 0:
#                 class_nmi_before = sum(cls_nmi_before)/len(cls_nmi_before)
#                 class_nmi_after = sum(cls_nmi_after)/len(cls_nmi_after)
#                 print(f"\nClass {cls} ({len(cls_nmi_before)} images):")
#                 print(f"  Before: {class_nmi_before:.4f}")
#                 print(f"  After:  {class_nmi_after:.4f}")
#                 improvement = class_nmi_after - class_nmi_before
#                 print(f"  Improvement: {improvement:+.4f}")
    
#     print("="*60)
#     print(f"✓ Results saved to: {os.path.join(args.save, csv_save)}")
#     print("="*60 + "\n")
    
#     # ========================================
#     # RUNTIME STATISTICS
#     # ========================================
#     total_runtime = time.time() - total_start_time
    
#     print("\n" + "="*60)
#     print("RUNTIME STATISTICS")
#     print("="*60)
    
#     # Per-image statistics
#     valid_times = [t for t in per_image_times if t is not None]
#     if valid_times:
#         avg_time = sum(valid_times) / len(valid_times)
#         min_time = min(valid_times)
#         max_time = max(valid_times)
#         median_time = sorted(valid_times)[len(valid_times)//2]
        
#         print(f"\nPer-Image Processing Time:")
#         print(f"  Total images processed: {len(valid_times)}")
#         print(f"  Average time per image: {avg_time:.4f}s")
#         print(f"  Median time per image:  {median_time:.4f}s")
#         print(f"  Min time per image:     {min_time:.4f}s")
#         print(f"  Max time per image:     {max_time:.4f}s")
#         print(f"  Total processing time:  {sum(valid_times):.2f}s ({sum(valid_times)/60:.2f} min)")
#         print(f"  Throughput:             {len(valid_times)/sum(valid_times):.2f} images/sec")
    
#     # Stage breakdown
#     print(f"\nStage-wise Time Breakdown:")
#     print(f"  Setup:           {stage_times['setup']:8.2f}s ({stage_times['setup']/total_runtime*100:5.1f}%)")
#     print(f"  Data Loading:    {stage_times['data_loading']:8.2f}s ({stage_times['data_loading']/total_runtime*100:5.1f}%)")
#     print(f"  Registration:    {stage_times['registration']:8.2f}s ({stage_times['registration']/total_runtime*100:5.1f}%)")
#     print(f"  Visualization:   {stage_times['visualization']:8.2f}s ({stage_times['visualization']/total_runtime*100:5.1f}%)")
#     print(f"  Metrics:         {stage_times['metrics']:8.2f}s ({stage_times['metrics']/total_runtime*100:5.1f}%)")
#     print(f"  Saving:          {stage_times['saving']:8.2f}s ({stage_times['saving']/total_runtime*100:5.1f}%)")
    
#     print(f"\n{'='*60}")
#     print(f"TOTAL EXECUTION TIME: {total_runtime:.2f}s ({total_runtime/60:.2f} min)")
#     print(f"{'='*60}\n")
    
#     # ========================================
#     # SAVE RUNTIME STATISTICS TO JSON
#     # ========================================
#     if args.save_runtime:
#         runtime_stats = {
#             'total_execution_time_seconds': total_runtime,
#             'total_execution_time_minutes': total_runtime / 60,
#             'total_images': len(dataset),
#             'successful_registrations': len(valid_times) if valid_times else 0,
#             'failed_registrations': len(dataset) - (len(valid_times) if valid_times else 0),
#             'per_image_stats': {
#                 'average_time_seconds': avg_time if valid_times else None,
#                 'median_time_seconds': median_time if valid_times else None,
#                 'min_time_seconds': min_time if valid_times else None,
#                 'max_time_seconds': max_time if valid_times else None,
#                 'throughput_images_per_second': len(valid_times)/sum(valid_times) if valid_times else None
#             },
#             'stage_breakdown_seconds': stage_times,
#             'stage_breakdown_percentage': {
#                 stage: (time_val/total_runtime*100) for stage, time_val in stage_times.items()
#             },
#             'individual_image_times': per_image_times,
#             'detailed_breakdown_per_image': per_image_breakdown
#         }
        
#         runtime_file = os.path.join(args.save, 'runtime_statistics.json')
#         with open(runtime_file, 'w') as f:
#             json.dump(runtime_stats, f, indent=2)
        
#         print(f"✓ Runtime statistics saved to: {runtime_file}\n")

#     return

# if __name__ == '__main__':
#     args = parse_args()
#     main(args)

import argparse
import os, sys
import time
import json
from tqdm import tqdm
from utils import none_or_str, compute_dice
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
from data import ImageDataset
from eyeliner import EyeLinerP
from visualize import create_flicker, create_checkerboard, create_diff_map
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mutual_info_score


def compute_mutual_information(img1, img2, bins=256):
    """
    Compute Mutual Information between two images.
    Higher MI indicates better alignment.
    
    Args:
        img1: First image tensor (C, H, W)
        img2: Second image tensor (C, H, W)
        bins: Number of histogram bins
    
    Returns:
        Mutual information score (float)
    """
    # Convert to numpy and flatten
    img1_np = img1.cpu().numpy().flatten()
    img2_np = img2.cpu().numpy().flatten()
    
    # Discretize to histogram bins
    img1_binned = np.digitize(img1_np, bins=np.linspace(0, 1, bins))
    img2_binned = np.digitize(img2_np, bins=np.linspace(0, 1, bins))
    
    # Compute mutual information
    mi = mutual_info_score(img1_binned, img2_binned)
    
    return mi

def compute_normalized_mutual_information(img1, img2, bins=256):
    """
    Compute Normalized Mutual Information between two images.
    NMI is in range [0, 1] where 1 indicates perfect alignment.
    
    Args:
        img1: First image tensor (C, H, W)
        img2: Second image tensor (C, H, W)
        bins: Number of histogram bins
    
    Returns:
        Normalized mutual information score (float)
    """
    # Convert to numpy and flatten
    img1_np = img1.cpu().numpy().flatten()
    img2_np = img2.cpu().numpy().flatten()
    
    # Discretize to histogram bins
    img1_binned = np.digitize(img1_np, bins=np.linspace(0, 1, bins))
    img2_binned = np.digitize(img2_np, bins=np.linspace(0, 1, bins))
    
    # Compute MI
    mi = mutual_info_score(img1_binned, img2_binned)
    
    # Compute entropies for normalization
    h1 = compute_entropy(img1_binned, bins)
    h2 = compute_entropy(img2_binned, bins)
    
    # Normalized MI
    nmi = 2 * mi / (h1 + h2) if (h1 + h2) > 0 else 0
    
    return nmi

def compute_entropy(data, bins):
    """Compute entropy of discretized data"""
    hist, _ = np.histogram(data, bins=bins)
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros
    return -np.sum(hist * np.log2(hist))

def compute_pairwise_statistics(values):
    """
    Compute mean and standard deviation across image pairs.
    
    Formula:
        mean = (1/n) * Σx_i
        std = sqrt( (1/(n-1)) * Σ(x_i - mean)² )
    
    Args:
        values: List of values (one per image pair)
    
    Returns:
        dict with mean, std, n, min, max, median
    """
    if len(values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'n': 0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    values_array = np.array(values)
    n = len(values_array)
    
    # Calculate mean
    mean = np.mean(values_array)
    
    # Calculate sample standard deviation (ddof=1 for n-1 denominator)
    std = np.std(values_array, ddof=1) if n > 1 else 0.0
    
    return {
        'mean': float(mean),
        'std': float(std),
        'n': int(n),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array))
    }

def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
    parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
    parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
    parser.add_argument('-fv', '--fixed-vessel', default='fixed_vessel_seg', type=none_or_str, help='Fixed vessel column')
    parser.add_argument('-mv', '--moving-vessel', default='moving_vessel_seg', type=none_or_str, help='Moving vessel column')
    parser.add_argument('-fd', '--fixed-disk', default='fixed_disk', type=none_or_str, help='Fixed disk column')
    parser.add_argument('-md', '--moving-disk', default='moving_disk', type=none_or_str, help='Moving disk column')
    parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
    parser.add_argument('-r', '--registration', default='registration_path', type=none_or_str, help='Registration column')
    parser.add_argument('-l', '--lmbda', default=None, type=float, help='Lambda value for computing the TPS quantitative result')
    parser.add_argument('--detected_keypoints', default='keypoints', type=none_or_str, help='Model Detected Keypoints column')
    parser.add_argument('--manual_keypoints', default='keypoints', type=none_or_str, help='Manually annotated Keypoints column')
    parser.add_argument('--fire_eval', action='store_true', help='Run FIRE evaluation')
    parser.add_argument('--mle_bins', type=int, default=256, help='Number of bins for MI/NMI computation')

    # misc
    parser.add_argument('--device', default='cpu', help='Device to run program on')
    parser.add_argument('--save', default='trained_models/', help='Location to save results')
    parser.add_argument('--save_runtime', action='store_true', help='Save detailed runtime statistics to JSON file')
    parser.add_argument('--save_pairwise_stats', action='store_true', help='Save pair-wise statistics to JSON file')
    parser.add_argument('--print_per_image', action='store_true', help='Print MLE for each image pair')
    args = parser.parse_args()
    return args

def get_class_from_filename(filepath):
    """
    Extract class (A, P, or S) from image filename
    Example: 'data/retina_datasets/FIRE/images/A01_1.jpg' -> 'A'
    """
    filename = os.path.basename(filepath)
    filename_no_ext = os.path.splitext(filename)[0]
    first_letter = filename_no_ext[0].upper()
    
    if first_letter in ['A', 'P', 'S']:
        return first_letter
    else:
        return None

def main(args):
    # Start total execution timer
    total_start_time = time.time()
    
    # Runtime tracking dictionaries
    per_image_times = []
    per_image_breakdown = []
    stage_times = {
        'setup': 0,
        'data_loading': 0,
        'registration': 0,
        'visualization': 0,
        'metrics': 0,
        'saving': 0
    }
    
    # Setup stage timing
    setup_start = time.time()
    
    device = torch.device(args.device)

    # load dataset
    dataset = ImageDataset(
        path=args.data, 
        input_col=args.fixed, 
        output_col=args.moving,
        input_vessel_col=args.fixed_vessel,
        output_vessel_col=args.moving_vessel,
        input_od_col=args.fixed_disk,
        output_od_col=args.moving_disk,
        input_dim=(args.size, args.size), 
        cmode='rgb',
        input='vessel',
        detected_keypoints_col=args.detected_keypoints,
        manual_keypoints_col=args.manual_keypoints,
        registration_col=args.registration
    )

    # make directory to store registrations
    reg_images_save_folder = os.path.join(args.save, 'registration_images')
    seg_overlaps_save_folder = os.path.join(args.save, 'seg_overlaps')
    checkerboard_after_save_folder = os.path.join(args.save, 'ckbd_images_after')
    checkerboard_before_save_folder = os.path.join(args.save, 'ckbd_images_before')
    flicker_save_folder = os.path.join(args.save, 'flicker_images')
    difference_map_save_folder = os.path.join(args.save, 'diff_map_images')
    os.makedirs(reg_images_save_folder, exist_ok=True)
    os.makedirs(checkerboard_after_save_folder, exist_ok=True)
    os.makedirs(checkerboard_before_save_folder, exist_ok=True)
    os.makedirs(flicker_save_folder, exist_ok=True)
    os.makedirs(difference_map_save_folder, exist_ok=True)
    os.makedirs(seg_overlaps_save_folder, exist_ok=True)

    images_filenames = []
    ckbd_filenames_before = []
    ckbd_filenames_after = []
    segmentation_overlaps = []
    flicker_filenames = []
    difference_maps_filenames = []
    dice_before = []
    dice_after = []
    image_classes = []
    
    # NEW: MLE score lists
    mi_before = []
    mi_after = []
    nmi_before = []
    nmi_after = []
    
    if args.manual_keypoints is not None:
        error_before = []
        error_after = []

    # Separate results by class
    class_stats = {
        'A': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': [], 
              'mi_before': [], 'mi_after': [], 'nmi_before': [], 'nmi_after': []},
        'P': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': [],
              'mi_before': [], 'mi_after': [], 'nmi_before': [], 'nmi_after': []},
        'S': {'error_before': [], 'error_after': [], 'dice_before': [], 'dice_after': [],
              'mi_before': [], 'mi_after': [], 'nmi_before': [], 'nmi_after': []}
    }
    
    stage_times['setup'] = time.time() - setup_start

    # Print header for per-image output if requested
    if args.print_per_image:
        print("\n" + "="*100)
        print("PER-IMAGE MLE VALUES")
        print("="*100)
        print(f"{'Idx':<5} {'Fixed Image':<40} {'Class':<7} {'MD Before':<12} {'MD After':<12} {'MI Before':<12} {'MI After':<12} {'NMI Before':<12} {'NMI After':<12}")
        print("-"*100)

    # Main processing loop with progress bar
    for i in tqdm(range(len(dataset)), desc="Processing images", disable=args.print_per_image):
        
        # Track time for this image
        image_start_time = time.time()
        breakdown = {
            'data_loading': 0,
            'registration': 0,
            'visualization': 0,
            'metrics': 0,
            'saving': 0
        }
        
        # DATA LOADING
        data_load_start = time.time()
        batch_data = dataset[i]
        
        # Get fixed image path to determine class
        fixed_image_path = dataset.data.iloc[i][args.fixed]
        img_class = get_class_from_filename(fixed_image_path)
        image_classes.append(img_class)
        fixed_image_name = os.path.basename(fixed_image_path)
        
        fixed_image = batch_data['fixed_image']
        moving_image = batch_data['moving_image']
        fixed_vessel = batch_data['fixed_input']
        fixed_vessel = (fixed_vessel > 0.5).float()
        moving_vessel = batch_data['moving_input']
        moving_vessel = (moving_vessel > 0.5).float()
        theta = batch_data['theta']
        
        breakdown['data_loading'] = time.time() - data_load_start
        stage_times['data_loading'] += breakdown['data_loading']

        # if image pair could not be registered
        if theta is None:
            images_filenames.append(None)
            ckbd_filenames_after.append(None)
            ckbd_filenames_before.append(None)
            flicker_filenames.append(None)
            difference_maps_filenames.append(None)
            dice_before.append(None)
            dice_after.append(None)
            mi_before.append(None)
            mi_after.append(None)
            nmi_before.append(None)
            nmi_after.append(None)
            if args.manual_keypoints is not None:
                error_before.append(None)
                error_after.append(None)
            
            if args.print_per_image:
                print(f"{i:<5} {fixed_image_name:<40} {img_class if img_class else 'N/A':<7} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<12}")
            
            # Record time even for failed registrations
            total_image_time = time.time() - image_start_time
            per_image_times.append(total_image_time)
            breakdown['total'] = total_image_time
            per_image_breakdown.append(breakdown)
            continue

        # REGISTRATION
        reg_start = time.time()
        try:
            reg_image = EyeLinerP.apply_transform(theta, moving_image)
        except:
            reg_image = EyeLinerP.apply_transform(theta[1], moving_image)

        # register moving segmentation
        try:
            reg_vessel = EyeLinerP.apply_transform(theta, moving_vessel)
        except:
            reg_vessel = EyeLinerP.apply_transform(theta[1], moving_vessel)
        reg_vessel = (reg_vessel > 0.5).float()

        # create mask
        reg_mask = torch.ones_like(moving_image)
        try:
            reg_mask = EyeLinerP.apply_transform(theta, reg_mask)
        except:
            reg_mask = EyeLinerP.apply_transform(theta[1], reg_mask)

        # apply mask to images
        fixed_image = fixed_image * reg_mask
        moving_image = moving_image * reg_mask
        reg_image = reg_image * reg_mask

        # apply mask to segmentations
        fixed_vessel = fixed_vessel
        moving_vessel = moving_vessel
        reg_vessel = reg_vessel * reg_mask
        
        breakdown['registration'] = time.time() - reg_start
        stage_times['registration'] += breakdown['registration']

        # SAVING REGISTRATION IMAGE
        save_start = time.time()
        filename = os.path.join(reg_images_save_folder, f'reg_{i}.png')
        ToPILImage()(reg_image).save(filename)
        images_filenames.append(filename)
        breakdown['saving'] = time.time() - save_start
        stage_times['saving'] += breakdown['saving']

        # VISUALIZATION
        viz_start = time.time()
        
        # segmentation overlap
        seg1 = ToPILImage()(fixed_vessel)
        seg2 = ToPILImage()(reg_vessel)
        seg_overlap = Image.blend(seg1, seg2, alpha=0.5)
        filename = os.path.join(seg_overlaps_save_folder, f'seg_overlap_{i}.png')
        seg_overlap.save(filename)
        segmentation_overlaps.append(filename)

        # checkerboards
        ckbd = create_checkerboard(fixed_image, reg_image, patch_size=32)
        filename = os.path.join(checkerboard_after_save_folder, f'ckbd_{i}.png')
        ToPILImage()(ckbd).save(filename)
        ckbd_filenames_after.append(filename)

        ckbd = create_checkerboard(fixed_image, moving_image, patch_size=32)
        filename = os.path.join(checkerboard_before_save_folder, f'ckbd_{i}.png')
        ToPILImage()(ckbd).save(filename)
        ckbd_filenames_before.append(filename)

        # flicker animation
        filename = os.path.join(flicker_save_folder, f'flicker_{i}.gif')
        create_flicker(fixed_image, reg_image, output_path=filename)
        flicker_filenames.append(filename)

        # subtraction maps
        filename = os.path.join(difference_map_save_folder, f'diff_map_{i}.png')
        create_diff_map(fixed_image, reg_image, filename)
        difference_maps_filenames.append(filename)
        
        breakdown['visualization'] = time.time() - viz_start
        stage_times['visualization'] += breakdown['visualization']

        # METRICS COMPUTATION
        metrics_start = time.time()
        
        # compute dice between segmentation maps
        seg_dice_before = compute_dice(fixed_vessel, moving_vessel)
        seg_dice_after = compute_dice(fixed_vessel, reg_vessel)
        dice_before.append(seg_dice_before)
        dice_after.append(seg_dice_after)
        
        # NEW: Compute MLE scores (Mutual Information)
        img_mi_before = compute_mutual_information(fixed_image, moving_image, bins=args.mle_bins)
        img_mi_after = compute_mutual_information(fixed_image, reg_image, bins=args.mle_bins)
        mi_before.append(img_mi_before)
        mi_after.append(img_mi_after)
        
        # NEW: Compute Normalized Mutual Information
        img_nmi_before = compute_normalized_mutual_information(fixed_image, moving_image, bins=args.mle_bins)
        img_nmi_after = compute_normalized_mutual_information(fixed_image, reg_image, bins=args.mle_bins)
        nmi_before.append(img_nmi_before)
        nmi_after.append(img_nmi_after)
        
        # Store class-specific scores
        if img_class in class_stats:
            class_stats[img_class]['dice_before'].append(seg_dice_before)
            class_stats[img_class]['dice_after'].append(seg_dice_after)
            class_stats[img_class]['mi_before'].append(img_mi_before)
            class_stats[img_class]['mi_after'].append(img_mi_after)
            class_stats[img_class]['nmi_before'].append(img_nmi_before)
            class_stats[img_class]['nmi_after'].append(img_nmi_after)

        # Initialize MD values for printing
        md_before_val = None
        md_after_val = None
        
        # quantitative evaluation
        if args.manual_keypoints is not None:

            fixed_kp_manual = batch_data['fixed_keypoints_manual']
            moving_kp_manual = batch_data['moving_keypoints_manual']
            fixed_kp_detected = batch_data['fixed_keypoints_detected']
            moving_kp_detected = batch_data['moving_keypoints_detected']

            # apply theta to keypoints
            try:
                reg_kp = EyeLinerP.apply_transform_points(theta, moving_keypoints=moving_kp_manual)
            except:                
                reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=moving_kp_manual, ctrl_keypoints=moving_kp_detected, tgt_keypoints=fixed_kp_detected, lmbda=args.lmbda)

            if args.fire_eval:
                fixed_kp_manual = 2912. * fixed_kp_manual / 256.
                moving_kp_manual = 2912. * moving_kp_manual / 256.
                reg_kp = 2912. * reg_kp / 256.

            # compute mean distance between fixed and registered keypoints
            md_before = torch.sqrt(torch.sum((fixed_kp_manual - moving_kp_manual)**2, dim=-1)).mean().item()
            md_after = torch.sqrt(torch.sum((fixed_kp_manual - reg_kp)**2, dim=-1)).mean().item()
            error_before.append(md_before)
            error_after.append(md_after)
            
            md_before_val = md_before
            md_after_val = md_after
            
            # Store class-specific errors
            if img_class in class_stats:
                class_stats[img_class]['error_before'].append(md_before)
                class_stats[img_class]['error_after'].append(md_after)
        
        breakdown['metrics'] = time.time() - metrics_start
        stage_times['metrics'] += breakdown['metrics']
        
        # PRINT PER-IMAGE MLE VALUES
        md_before_str = f"{md_before_val:>11.4f}" if md_before_val is not None else "N/A".rjust(11)
        md_after_str = f"{md_after_val:>11.4f}" if md_after_val is not None else "N/A".rjust(11)
        class_str = img_class if img_class else 'N/A'

        # Print the row
        
        # Print the row with descriptive column headers
        print(f"{i:<5} {fixed_image_name:<40} {class_str:<7} "
              f"MD_Before: {md_before_str} MD_After: {md_after_str} "
           )
        sys.stdout.flush()
 

        # Print per-image MLE values if requested
        if args.print_per_image:
            md_before_str = f"{md_before_val:>11.4f}" if md_before_val is not None else "N/A".rjust(11)
            md_after_str = f"{md_after_val:>11.4f}" if md_after_val is not None else "N/A".rjust(11)
            print(f"{i:<5} {fixed_image_name:<40} {img_class if img_class else 'N/A':<7} "
                  f"{md_before_str} {md_after_str} "
                  f"{img_mi_before:>11.4f} {img_mi_after:>11.4f} "
                  f"{img_nmi_before:>11.4f} {img_nmi_after:>11.4f}")
        
        # Record total time for this image
        total_image_time = time.time() - image_start_time
        per_image_times.append(total_image_time)
        breakdown['total'] = total_image_time
        per_image_breakdown.append(breakdown)

    if args.print_per_image:
        print("="*100 + "\n")

    # ========================================
    # SAVE RESULTS TO DATAFRAME
    # ========================================
    save_results_start = time.time()
    
    dataset.data['registration_path'] = images_filenames
    dataset.data['checkerboard_before'] = ckbd_filenames_before
    dataset.data['checkerboard_after'] = ckbd_filenames_after
    dataset.data['flicker'] = flicker_filenames
    dataset.data['seg_overlap'] = segmentation_overlaps
    dataset.data['difference_map'] = difference_maps_filenames
    dataset.data['DICE_before'] = dice_before
    dataset.data['DICE_after'] = dice_after
    dataset.data['MI_before'] = mi_before
    dataset.data['MI_after'] = mi_after
    dataset.data['NMI_before'] = nmi_before
    dataset.data['NMI_after'] = nmi_after
    dataset.data['image_class'] = image_classes
    dataset.data['processing_time'] = per_image_times
    
    if args.manual_keypoints is not None:
        dataset.data['MD_before'] = error_before
        dataset.data['MD_after'] = error_after

    # save results
    csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
    dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)
    
    stage_times['saving'] += time.time() - save_results_start

    # ========================================
    # COMPUTE PAIR-WISE STATISTICS
    # ========================================
    
    print("\n" + "="*80)
    print("COMPUTING PAIR-WISE STATISTICS")
    print("="*80)
    print("Note: Standard deviation calculated across image pairs using formula:")
    print("  std = sqrt( (1/(n-1)) * Σ(x_i - mean)² )")
    print("="*80)
    
    # Overall statistics
    overall_stats = {}
    class_pairwise_stats = {}
    
    # Metrics to compute statistics for
    metric_configs = [
        ('error_before', error_before if args.manual_keypoints else []),
        ('error_after', error_after if args.manual_keypoints else []),
        ('dice_before', dice_before),
        ('dice_after', dice_after),
        ('mi_before', mi_before),
        ('mi_after', mi_after),
        ('nmi_before', nmi_before),
        ('nmi_after', nmi_after)
    ]
    
    for metric_name, metric_values in metric_configs:
        # Compute overall statistics
        valid_values = [x for x in metric_values if x is not None]
        overall_stats[metric_name] = compute_pairwise_statistics(valid_values)
        
        # Compute class-specific statistics
        for cls in ['A', 'P', 'S']:
            cls_values = class_stats[cls].get(metric_name, [])
            if cls not in class_pairwise_stats:
                class_pairwise_stats[cls] = {}
            class_pairwise_stats[cls][metric_name] = compute_pairwise_statistics(cls_values)

    # ========================================
    # PRINT SUMMARY WITH PAIR-WISE STATISTICS
    # ========================================
    
    print("\n" + "="*80)
    print("REGISTRATION RESULTS SUMMARY (PAIR-WISE STATISTICS)")
    print("="*80)
    print(f"Dataset: {os.path.basename(args.data)}")
    print(f"Save location: {args.save}")
    print(f"Total image pairs: {len(dataset)}")
    
    # Print class distribution
    print(f"\n{'─'*80}")
    print(f"CLASS DISTRIBUTION")
    print(f"{'─'*80}")
    for cls in ['A', 'P', 'S']:
        count = len(class_stats[cls]['dice_before'])
        print(f"Class {cls}: {count} images")
    
    # ========================================
    # MD STATISTICS WITH PAIR-WISE STD
    # ========================================
    if args.manual_keypoints is not None and overall_stats['error_after']['n'] > 0:
        print(f"\n{'='*80}")
        print(f"MEAN DISTANCE (MD) - PAIR-WISE STATISTICS")
        print(f"{'='*80}")
        print(f"Format: Mean ± Std (n=count)")
        print(f"Note: Std represents variation across different image pairs")
        
        # Overall
        print(f"\nOverall:")
        before_stats = overall_stats['error_before']
        after_stats = overall_stats['error_after']
        print(f"  Before: {before_stats['mean']:7.2f} ± {before_stats['std']:6.2f} (n={before_stats['n']}) pixels")
        print(f"  After:  {after_stats['mean']:7.2f} ± {after_stats['std']:6.2f} (n={after_stats['n']}) pixels")
        improvement = before_stats['mean'] - after_stats['mean']
        print(f"  Improvement: {improvement:7.2f} pixels")
        
        # Class-specific
        print(f"\nClass-Specific:")
        for cls in ['S', 'A', 'P']:
            cls_before = class_pairwise_stats[cls]['error_before']
            cls_after = class_pairwise_stats[cls]['error_after']
            
            if cls_after['n'] > 0:
                print(f"\n  Class {cls}:")
                print(f"    Before: {cls_before['mean']:7.2f} ± {cls_before['std']:6.2f} (n={cls_before['n']}) pixels")
                print(f"    After:  {cls_after['mean']:7.2f} ± {cls_after['std']:6.2f} (n={cls_after['n']}) pixels")
                improvement = cls_before['mean'] - cls_after['mean']
                print(f"    Improvement: {improvement:7.2f} pixels")
    
    # ========================================
    # DICE STATISTICS WITH PAIR-WISE STD
    # ========================================
    if overall_stats['dice_after']['n'] > 0:
        print(f"\n{'='*80}")
        print(f"DICE SCORES - PAIR-WISE STATISTICS")
        print(f"{'='*80}")
        print(f"Format: Mean ± Std (n=count)")
        
        # Overall
        print(f"\nOverall:")
        before_stats = overall_stats['dice_before']
        after_stats = overall_stats['dice_after']
        print(f"  Before: {before_stats['mean']:.4f} ± {before_stats['std']:.4f} (n={before_stats['n']})")
        print(f"  After:  {after_stats['mean']:.4f} ± {after_stats['std']:.4f} (n={after_stats['n']})")
        improvement = after_stats['mean'] - before_stats['mean']
        print(f"  Improvement: {improvement:+.4f}")
        
        # Class-specific
        print(f"\nClass-Specific:")
        for cls in ['S', 'A', 'P']:
            cls_before = class_pairwise_stats[cls]['dice_before']
            cls_after = class_pairwise_stats[cls]['dice_after']
            
            if cls_after['n'] > 0:
                print(f"\n  Class {cls}:")
                print(f"    Before: {cls_before['mean']:.4f} ± {cls_before['std']:.4f} (n={cls_before['n']})")
                print(f"    After:  {cls_after['mean']:.4f} ± {cls_after['std']:.4f} (n={cls_after['n']})")
                improvement = cls_after['mean'] - cls_before['mean']
                print(f"    Improvement: {improvement:+.4f}")
    
    # ========================================
    # MI STATISTICS WITH PAIR-WISE STD
    # ========================================
    if overall_stats['mi_after']['n'] > 0:
        print(f"\n{'='*80}")
        print(f"MUTUAL INFORMATION (MI) - PAIR-WISE STATISTICS")
        print(f"{'='*80}")
        print(f"Format: Mean ± Std (n=count)")
        print(f"Note: Higher MI indicates better alignment")
        
        # Overall
        print(f"\nOverall:")
        before_stats = overall_stats['mi_before']
        after_stats = overall_stats['mi_after']
        print(f"  Before: {before_stats['mean']:.4f} ± {before_stats['std']:.4f} (n={before_stats['n']})")
        print(f"  After:  {after_stats['mean']:.4f} ± {after_stats['std']:.4f} (n={after_stats['n']})")
        improvement = after_stats['mean'] - before_stats['mean']
        print(f"  Improvement: {improvement:+.4f}")
        
        # Class-specific
        print(f"\nClass-Specific:")
        for cls in ['S', 'A', 'P']:
            cls_before = class_pairwise_stats[cls]['mi_before']
            cls_after = class_pairwise_stats[cls]['mi_after']
            
            if cls_after['n'] > 0:
                print(f"\n  Class {cls}:")
                print(f"    Before: {cls_before['mean']:.4f} ± {cls_before['std']:.4f} (n={cls_before['n']})")
                print(f"    After:  {cls_after['mean']:.4f} ± {cls_after['std']:.4f} (n={cls_after['n']})")
                improvement = cls_after['mean'] - cls_before['mean']
                print(f"    Improvement: {improvement:+.4f}")
    
    # ========================================
    # NMI STATISTICS WITH PAIR-WISE STD
    # ========================================
    if overall_stats['nmi_after']['n'] > 0:
        print(f"\n{'='*80}")
        print(f"NORMALIZED MUTUAL INFORMATION (NMI) - PAIR-WISE STATISTICS")
        print(f"{'='*80}")
        print(f"Format: Mean ± Std (n=count)")
        print(f"Note: NMI range [0, 1], higher is better")
        
        # Overall
        print(f"\nOverall:")
        before_stats = overall_stats['nmi_before']
        after_stats = overall_stats['nmi_after']
        print(f"  Before: {before_stats['mean']:.4f} ± {before_stats['std']:.4f} (n={before_stats['n']})")
        print(f"  After:  {after_stats['mean']:.4f} ± {after_stats['std']:.4f} (n={after_stats['n']})")
        improvement = after_stats['mean'] - before_stats['mean']
        print(f"  Improvement: {improvement:+.4f}")
        
        # Class-specific
        print(f"\nClass-Specific:")
        for cls in ['S', 'A', 'P']:
            cls_before = class_pairwise_stats[cls]['nmi_before']
            cls_after = class_pairwise_stats[cls]['nmi_after']
            
            if cls_after['n'] > 0:
                print(f"\n  Class {cls}:")
                print(f"    Before: {cls_before['mean']:.4f} ± {cls_before['std']:.4f} (n={cls_before['n']})")
                print(f"    After:  {cls_after['mean']:.4f} ± {cls_after['std']:.4f} (n={cls_after['n']})")
                improvement = cls_after['mean'] - cls_before['mean']
                print(f"    Improvement: {improvement:+.4f}")
    
    # ========================================
    # LATEX TABLE FORMAT
    # ========================================
    print(f"\n{'='*80}")
    print(f"LATEX TABLE FORMAT (Mean ± Std)")
    print(f"{'='*80}")
    print("Copy and paste the lines below into your LaTeX document:\n")
    
    if args.manual_keypoints is not None and overall_stats['error_after']['n'] > 0:
        print("% Mean Distance (MD) [pixels]")
        o = overall_stats['error_after']
        s = class_pairwise_stats['S']['error_after']
        a = class_pairwise_stats['A']['error_after']
        p = class_pairwise_stats['P']['error_after']
        print(f"MD & {o['mean']:.2f}\\pm{o['std']:.2f} & "
              f"{s['mean']:.2f}\\pm{s['std']:.2f} & "
              f"{a['mean']:.2f}\\pm{a['std']:.2f} & "
              f"{p['mean']:.2f}\\pm{p['std']:.2f} \\\\")
    
    print("\n% DICE Score")
    o = overall_stats['dice_after']
    s = class_pairwise_stats['S']['dice_after']
    a = class_pairwise_stats['A']['dice_after']
    p = class_pairwise_stats['P']['dice_after']
    print(f"DICE & {o['mean']:.4f}\\pm{o['std']:.4f} & "
          f"{s['mean']:.4f}\\pm{s['std']:.4f} & "
          f"{a['mean']:.4f}\\pm{a['std']:.4f} & "
          f"{p['mean']:.4f}\\pm{p['std']:.4f} \\\\")
    
    if overall_stats['mi_after']['n'] > 0:
        print("\n% Mutual Information (MI)")
        o = overall_stats['mi_after']
        s = class_pairwise_stats['S']['mi_after']
        a = class_pairwise_stats['A']['mi_after']
        p = class_pairwise_stats['P']['mi_after']
        print(f"MI & {o['mean']:.4f}\\pm{o['std']:.4f} & "
              f"{s['mean']:.4f}\\pm{s['std']:.4f} & "
              f"{a['mean']:.4f}\\pm{a['std']:.4f} & "
              f"{p['mean']:.4f}\\pm{p['std']:.4f} \\\\")
    
    if overall_stats['nmi_after']['n'] > 0:
        print("\n% Normalized Mutual Information (NMI)")
        o = overall_stats['nmi_after']
        s = class_pairwise_stats['S']['nmi_after']
        a = class_pairwise_stats['A']['nmi_after']
        p = class_pairwise_stats['P']['nmi_after']
        print(f"NMI & {o['mean']:.4f}\\pm{o['std']:.4f} & "
              f"{s['mean']:.4f}\\pm{s['std']:.4f} & "
              f"{a['mean']:.4f}\\pm{a['std']:.4f} & "
              f"{p['mean']:.4f}\\pm{p['std']:.4f} \\\\")
    
    print("\n" + "="*80)
    print(f"✓ Results saved to: {os.path.join(args.save, csv_save)}")
    print("="*80 + "\n")
    
    # ========================================
    # RUNTIME STATISTICS
    # ========================================
    total_runtime = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("RUNTIME STATISTICS")
    print("="*60)
    
    # Per-image statistics
    valid_times = [t for t in per_image_times if t is not None]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        min_time = min(valid_times)
        max_time = max(valid_times)
        median_time = sorted(valid_times)[len(valid_times)//2]
        
        print(f"\nPer-Image Processing Time:")
        print(f"  Total images processed: {len(valid_times)}")
        print(f"  Average time per image: {avg_time:.4f}s")
        print(f"  Median time per image:  {median_time:.4f}s")
        print(f"  Min time per image:     {min_time:.4f}s")
        print(f"  Max time per image:     {max_time:.4f}s")
        print(f"  Total processing time:  {sum(valid_times):.2f}s ({sum(valid_times)/60:.2f} min)")
        print(f"  Throughput:             {len(valid_times)/sum(valid_times):.2f} images/sec")
    
    # Stage breakdown
    print(f"\nStage-wise Time Breakdown:")
    print(f"  Setup:           {stage_times['setup']:8.2f}s ({stage_times['setup']/total_runtime*100:5.1f}%)")
    print(f"  Data Loading:    {stage_times['data_loading']:8.2f}s ({stage_times['data_loading']/total_runtime*100:5.1f}%)")
    print(f"  Registration:    {stage_times['registration']:8.2f}s ({stage_times['registration']/total_runtime*100:5.1f}%)")
    print(f"  Visualization:   {stage_times['visualization']:8.2f}s ({stage_times['visualization']/total_runtime*100:5.1f}%)")
    print(f"  Metrics:         {stage_times['metrics']:8.2f}s ({stage_times['metrics']/total_runtime*100:5.1f}%)")
    print(f"  Saving:          {stage_times['saving']:8.2f}s ({stage_times['saving']/total_runtime*100:5.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"TOTAL EXECUTION TIME: {total_runtime:.2f}s ({total_runtime/60:.2f} min)")
    print(f"{'='*60}\n")
    
    # ========================================
    # SAVE STATISTICS TO JSON
    # ========================================
    if args.save_runtime or args.save_pairwise_stats:
        output_stats = {
            'total_execution_time_seconds': total_runtime,
            'total_execution_time_minutes': total_runtime / 60,
            'total_images': len(dataset),
            'successful_registrations': len(valid_times) if valid_times else 0,
            'failed_registrations': len(dataset) - (len(valid_times) if valid_times else 0),
        }
        
        if args.save_runtime:
            output_stats['per_image_stats'] = {
                'average_time_seconds': avg_time if valid_times else None,
                'median_time_seconds': median_time if valid_times else None,
                'min_time_seconds': min_time if valid_times else None,
                'max_time_seconds': max_time if valid_times else None,
                'throughput_images_per_second': len(valid_times)/sum(valid_times) if valid_times else None
            }
            output_stats['stage_breakdown_seconds'] = stage_times
            output_stats['stage_breakdown_percentage'] = {
                stage: (time_val/total_runtime*100) for stage, time_val in stage_times.items()
            }
            output_stats['individual_image_times'] = per_image_times
            output_stats['detailed_breakdown_per_image'] = per_image_breakdown
        
        if args.save_pairwise_stats:
            output_stats['pairwise_statistics'] = {
                'overall': overall_stats,
                'class_specific': class_pairwise_stats
            }
        
        stats_file = os.path.join(args.save, 'statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(output_stats, f, indent=2)
        
        print(f"✓ Statistics saved to: {stats_file}\n")

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)