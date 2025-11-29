# # # import argparse
# # # from argparse import Namespace
# # # import os


# # # import eval_tool.immatch as immatch
# # # from eval_tool.immatch.utils.data_io import lprint
# # # import eval_tool.immatch.utils.fire_helper as helper
# # # from eval_tool.immatch.utils.model_helper import parse_model_config

# # # def eval_fire(
# # #         root_dir,
# # #         config_list,
# # #         task='homography',
# # #         h_solver='degensac',
# # #         ransac_thres=2,
# # #         match_thres=None,
# # #         odir='outputs/fire',
# # #         save_npy=False,
# # #         print_out=False,
# # #         debug=False,
# # # ):
# # #     # Init paths
# # #     data_root = os.path.join(root_dir, 'data/datasets/FIRE')
# # #     cache_dir = os.path.join(root_dir, odir, 'cache')
# # #     result_dir = os.path.join(root_dir, odir, 'results', task)
# # #     mauc = 0
# # #     total_auc = 0
# # #     auc_results = {}
# # #     mle_results = {}
# # #     gt_dir = os.path.join(data_root, 'Ground Truth')
# # #     im_dir = os.path.join(data_root, 'Images')
# # #     match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')]
# # #     if not os.path.exists(cache_dir):
# # #         os.makedirs(cache_dir)
# # #     if not os.path.exists(result_dir):
# # #         os.makedirs(result_dir)

# # #     # Iterate over methods
# # #     for config_name in config_list:
# # #         # Load model
# # #         args = parse_model_config(config_name, 'fire', root_dir)
# # #         class_name = args['class']

# # #         # One log file per method
# # #         log_file = os.path.join(result_dir, f'{class_name}.txt')
# # #         log = open(log_file, 'a')
# # #         lprint_ = lambda ms: lprint(ms, log)
# # #         ransac_thres = args['ransac_thres']
# # #         # Iterate over matching thresholds
# # #         thresholds = match_thres if match_thres else [args['match_threshold']]
# # #         lprint_(f'\n>>>> Method={class_name} Default config: {args} '
# # #                 f'Thres: {thresholds}')

# # #         for thres in thresholds:
# # #             args['match_threshold'] = thres  # Set to target thresholds

# # #             # Init model
# # #             model = immatch.__dict__[class_name](args)

# # #             matcher = lambda im1, im2: model.match_pairs(im1, im2)

# # #             # Init result save path (for matching results)
# # #             result_npy = None
# # #             if save_npy:
# # #                 result_tag = model.name
# # #                 if args['imsize'] > 0:
# # #                     result_tag += f".im{args['imsize']}"
# # #                 if thres > 0:
# # #                     result_tag += f'.m{thres}'
# # #                 result_npy = os.path.join(cache_dir, f'{result_tag}.npy')

# # #             lprint_(f'Matching thres: {thres}  Save to: {result_npy}')

# # #             # Eval on the specified task(s)
# # #             result = helper.eval_fire(
# # #                 matcher,
# # #                 match_pairs,
# # #                 im_dir,
# # #                 gt_dir,
# # #                 model.name,
# # #                 task=task,
# # #                 scale_H=getattr(model, 'no_match_upscale', False),
# # #                 h_solver=h_solver,
# # #                 ransac_thres=ransac_thres,
# # #                 lprint_=lprint_,
# # #                 debug=debug,
# # #             )
# # #             mauc = result['mauc']
# # #             total_auc = result['total_auc']
# # #             auc_results = result.get('auc', {})
# # #             mle_results = result.get('mle', {})
# # #         log.close()
    
# # #     return mauc, total_auc, auc_results, mle_results


# # # if __name__ == '__main__':
# # #     parser = argparse.ArgumentParser(description='Benchmark FIRE')
# # #     parser.add_argument('--gpu', '-gpu', type=str, default='0')
# # #     parser.add_argument('--root_dir', type=str, default='.')
# # #     parser.add_argument('--odir', type=str, default='outputs/fire')
# # #     parser.add_argument('--config', type=str, nargs='*', default=['geoformer'])
# # #     parser.add_argument('--match_thres', type=float, nargs='*', default=None)
# # #     parser.add_argument(
# # #         '--task', type=str, default='homography',
# # #         choices=['matching', 'homography', 'both']
# # #     )
# # #     parser.add_argument(
# # #         '--h_solver', type=str, default='cv',
# # #         choices=['degensac', 'cv', 'mle']  # Added 'mle' option
# # #     )
# # #     parser.add_argument('--ransac_thres', type=float, default=15)
# # #     parser.add_argument('--save_npy', action='store_true')
# # #     parser.add_argument('--print_out', action='store_true', default=True)

# # #     args = parser.parse_args()
# # #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
# # #     print('='*80)
# # #     print(f'FIRE Evaluation Configuration:')
# # #     print(f'  - Homography Solver: {args.h_solver.upper()}')
# # #     print(f'  - RANSAC Threshold: {args.ransac_thres}')
# # #     print(f'  - Config: {args.config}')
# # #     print(f'  - Task: {args.task}')
# # #     print('='*80)
    
# # #     mauc, total_auc, auc_results, mle_results = eval_fire(
# # #         args.root_dir, args.config, args.task,
# # #         h_solver=args.h_solver,
# # #         ransac_thres=args.ransac_thres,
# # #         match_thres=args.match_thres,
# # #         odir=args.odir,
# # #         save_npy=args.save_npy,
# # #         print_out=args.print_out
# # #     )
    
# # #     print('='*80)
# # #     print(f'Final Results:')
# # #     print(f'  - Solver: {args.h_solver.upper()}')
# # #     if auc_results:
# # #         print(f'  - Hest AUC: s={auc_results["s"]:.4f} p={auc_results["p"]:.4f} a={auc_results["a"]:.4f} total={total_auc:.4f} m={mauc:.4f}')
# # #     else:
# # #         print(f'  - Hest AUC: total={total_auc:.4f} m={mauc:.4f}')
# # #     if mle_results:
# # #         print(f'  - MLE dist: s={mle_results["s"]:.4f} p={mle_results["p"]:.4f} a={mle_results["a"]:.4f} total={mle_results["total"]:.4f} m={mle_results["m"]:.4f}')
# # #     print('='*80)


# # import argparse
# # from argparse import Namespace
# # import os
# # import time
# # from collections import defaultdict


# # import eval_tool.immatch as immatch
# # from eval_tool.immatch.utils.data_io import lprint
# # import eval_tool.immatch.utils.fire_helper as helper
# # from eval_tool.immatch.utils.model_helper import parse_model_config

# # def eval_fire(
# #         root_dir,
# #         config_list,
# #         task='homography',
# #         h_solver='degensac',
# #         ransac_thres=2,
# #         match_thres=None,
# #         odir='outputs/fire',
# #         save_npy=False,
# #         print_out=False,
# #         debug=False,
# # ):
# #     # Start overall timing
# #     total_start_time = time.time()
    
# #     # Init paths
# #     data_root = os.path.join(root_dir, 'data/datasets/FIRE')
# #     cache_dir = os.path.join(root_dir, odir, 'cache')
# #     result_dir = os.path.join(root_dir, odir, 'results', task)
# #     mauc = 0
# #     total_auc = 0
# #     auc_results = {}
# #     mle_results = {}
# #     gt_dir = os.path.join(data_root, 'Ground Truth')
# #     im_dir = os.path.join(data_root, 'Images')
# #     match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')]
# #     if not os.path.exists(cache_dir):
# #         os.makedirs(cache_dir)
# #     if not os.path.exists(result_dir):
# #         os.makedirs(result_dir)

# #     # Runtime tracking dictionaries
# #     all_runtime_stats = {}
    
# #     # Iterate over methods
# #     for config_name in config_list:
# #         method_start_time = time.time()
        
# #         # Load model
# #         args = parse_model_config(config_name, 'fire', root_dir)
# #         class_name = args['class']

# #         # One log file per method
# #         log_file = os.path.join(result_dir, f'{class_name}.txt')
# #         log = open(log_file, 'a')
# #         lprint_ = lambda ms: lprint(ms, log)
# #         ransac_thres = args['ransac_thres']
        
# #         # Iterate over matching thresholds
# #         thresholds = match_thres if match_thres else [args['match_threshold']]
# #         lprint_(f'\n>>>> Method={class_name} Default config: {args} '
# #                 f'Thres: {thresholds}')

# #         for thres in thresholds:
# #             threshold_start_time = time.time()
            
# #             args['match_threshold'] = thres  # Set to target thresholds

# #             # Init model
# #             model = immatch.__dict__[class_name](args)

# #             # Wrapper to track per-image runtime
# #             image_runtimes = []
            
# #             def timed_matcher(im1, im2):
# #                 img_start = time.time()
# #                 result = model.match_pairs(im1, im2)
# #                 img_time = time.time() - img_start
# #                 image_runtimes.append(img_time)
# #                 return result

# #             matcher = timed_matcher

# #             # Init result save path (for matching results)
# #             result_npy = None
# #             if save_npy:
# #                 result_tag = model.name
# #                 if args['imsize'] > 0:
# #                     result_tag += f".im{args['imsize']}"
# #                 if thres > 0:
# #                     result_tag += f'.m{thres}'
# #                 result_npy = os.path.join(cache_dir, f'{result_tag}.npy')

# #             lprint_(f'Matching thres: {thres}  Save to: {result_npy}')

# #             # Eval on the specified task(s)
# #             result = helper.eval_fire(
# #                 matcher,
# #                 match_pairs,
# #                 im_dir,
# #                 gt_dir,
# #                 model.name,
# #                 task=task,
# #                 scale_H=getattr(model, 'no_match_upscale', False),
# #                 h_solver=h_solver,
# #                 ransac_thres=ransac_thres,
# #                 lprint_=lprint_,
# #                 debug=debug,
# #             )
            
# #             threshold_time = time.time() - threshold_start_time
            
# #             # Calculate runtime statistics
# #             if image_runtimes:
# #                 avg_img_time = sum(image_runtimes) / len(image_runtimes)
# #                 min_img_time = min(image_runtimes)
# #                 max_img_time = max(image_runtimes)
# #                 total_img_time = sum(image_runtimes)
                
# #                 runtime_stats = {
# #                     'total_images': len(image_runtimes),
# #                     'avg_per_image': avg_img_time,
# #                     'min_per_image': min_img_time,
# #                     'max_per_image': max_img_time,
# #                     'total_matching_time': total_img_time,
# #                     'threshold_total_time': threshold_time,
# #                     'overhead_time': threshold_time - total_img_time,
# #                     'all_image_times': image_runtimes
# #                 }
                
# #                 # Store stats
# #                 stats_key = f"{class_name}_thres{thres}"
# #                 all_runtime_stats[stats_key] = runtime_stats
                
# #                 # Log runtime information
# #                 lprint_(f'\n{"="*60}')
# #                 lprint_(f'Runtime Statistics for threshold {thres}:')
# #                 lprint_(f'  Total images processed: {len(image_runtimes)}')
# #                 lprint_(f'  Average time per image: {avg_img_time:.4f}s')
# #                 lprint_(f'  Min time per image: {min_img_time:.4f}s')
# #                 lprint_(f'  Max time per image: {max_img_time:.4f}s')
# #                 lprint_(f'  Total matching time: {total_img_time:.4f}s')
# #                 lprint_(f'  Total threshold time: {threshold_time:.4f}s')
# #                 lprint_(f'  Overhead (non-matching): {threshold_time - total_img_time:.4f}s')
# #                 lprint_(f'{"="*60}\n')
                
# #                 # Print to console if requested
# #                 if print_out:
# #                     print(f'\nRuntime Statistics for {class_name} (threshold={thres}):')
# #                     print(f'  Total images: {len(image_runtimes)}')
# #                     print(f'  Avg per image: {avg_img_time:.4f}s')
# #                     print(f'  Total time: {threshold_time:.4f}s')
            
# #             mauc = result['mauc']
# #             total_auc = result['total_auc']
# #             auc_results = result.get('auc', {})
# #             mle_results = result.get('mle', {})
        
# #         method_time = time.time() - method_start_time
# #         lprint_(f'\n{"="*60}')
# #         lprint_(f'Total runtime for method {class_name}: {method_time:.4f}s ({method_time/60:.2f} min)')
# #         lprint_(f'{"="*60}\n')
        
# #         log.close()
    
# #     # Calculate total runtime
# #     total_runtime = time.time() - total_start_time
    
# #     # Return results with runtime stats
# #     return mauc, total_auc, auc_results, mle_results, all_runtime_stats, total_runtime


# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description='Benchmark FIRE')
# #     parser.add_argument('--gpu', '-gpu', type=str, default='0')
# #     parser.add_argument('--root_dir', type=str, default='.')
# #     parser.add_argument('--odir', type=str, default='outputs/fire')
# #     parser.add_argument('--config', type=str, nargs='*', default=['geoformer'])
# #     parser.add_argument('--match_thres', type=float, nargs='*', default=None)
# #     parser.add_argument(
# #         '--task', type=str, default='homography',
# #         choices=['matching', 'homography', 'both']
# #     )
# #     parser.add_argument(
# #         '--h_solver', type=str, default='cv',
# #         choices=['degensac', 'cv', 'mle']  # Added 'mle' option
# #     )
# #     parser.add_argument('--ransac_thres', type=float, default=15)
# #     parser.add_argument('--save_npy', action='store_true')
# #     parser.add_argument('--print_out', action='store_true', default=True)
# #     parser.add_argument('--save_runtime', action='store_true', 
# #                        help='Save detailed runtime statistics to file')

# #     args = parser.parse_args()
# #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
# #     print('='*80)
# #     print(f'FIRE Evaluation Configuration:')
# #     print(f'  - Homography Solver: {args.h_solver.upper()}')
# #     print(f'  - RANSAC Threshold: {args.ransac_thres}')
# #     print(f'  - Config: {args.config}')
# #     print(f'  - Task: {args.task}')
# #     print('='*80)
    
# #     # Run evaluation with timing
# #     overall_start = time.time()
    
# #     mauc, total_auc, auc_results, mle_results, runtime_stats, total_runtime = eval_fire(
# #         args.root_dir, args.config, args.task,
# #         h_solver=args.h_solver,
# #         ransac_thres=args.ransac_thres,
# #         match_thres=args.match_thres,
# #         odir=args.odir,
# #         save_npy=args.save_npy,
# #         print_out=args.print_out
# #     )
    
# #     print('='*80)
# #     print(f'Final Results:')
# #     print(f'  - Solver: {args.h_solver.upper()}')
# #     if auc_results:
# #         print(f'  - Hest AUC: s={auc_results["s"]:.4f} p={auc_results["p"]:.4f} a={auc_results["a"]:.4f} total={total_auc:.4f} m={mauc:.4f}')
# #     else:
# #         print(f'  - Hest AUC: total={total_auc:.4f} m={mauc:.4f}')
# #     if mle_results:
# #         print(f'  - MLE dist: s={mle_results["s"]:.4f} p={mle_results["p"]:.4f} a={mle_results["a"]:.4f} total={mle_results["total"]:.4f} m={mle_results["m"]:.4f}')
# #     print('='*80)
    
# #     # Print runtime summary
# #     print('\n' + '='*80)
# #     print('RUNTIME SUMMARY:')
# #     print('='*80)
    
# #     for config_key, stats in runtime_stats.items():
# #         print(f'\nConfiguration: {config_key}')
# #         print(f'  Total images processed: {stats["total_images"]}')
# #         print(f'  Average time per image: {stats["avg_per_image"]:.4f}s')
# #         print(f'  Min time per image: {stats["min_per_image"]:.4f}s')
# #         print(f'  Max time per image: {stats["max_per_image"]:.4f}s')
# #         print(f'  Total matching time: {stats["total_matching_time"]:.2f}s ({stats["total_matching_time"]/60:.2f} min)')
# #         print(f'  Throughput: {stats["total_images"]/stats["total_matching_time"]:.2f} images/sec')
    
# #     print(f'\n{"="*80}')
# #     print(f'TOTAL EXECUTION TIME: {total_runtime:.2f}s ({total_runtime/60:.2f} min)')
# #     print(f'{"="*80}\n')
    
# #     # Optionally save runtime statistics to file
# #     if args.save_runtime:
# #         import json
# #         runtime_file = os.path.join(args.root_dir, args.odir, 'runtime_statistics.json')
        
# #         # Prepare data for JSON (convert non-serializable types)
# #         json_stats = {}
# #         for key, stats in runtime_stats.items():
# #             json_stats[key] = {
# #                 'total_images': stats['total_images'],
# #                 'avg_per_image': stats['avg_per_image'],
# #                 'min_per_image': stats['min_per_image'],
# #                 'max_per_image': stats['max_per_image'],
# #                 'total_matching_time': stats['total_matching_time'],
# #                 'threshold_total_time': stats['threshold_total_time'],
# #                 'overhead_time': stats['overhead_time'],
# #                 'throughput_imgs_per_sec': stats['total_images'] / stats['total_matching_time'],
# #                 'all_image_times': stats['all_image_times']
# #             }
        
# #         json_stats['total_execution_time'] = total_runtime
        
# #         with open(runtime_file, 'w') as f:
# #             json.dump(json_stats, f, indent=2)
        
# #         print(f'Runtime statistics saved to: {runtime_file}\n')


# import argparse
# from argparse import Namespace
# import os
# import time
# from collections import defaultdict


# import eval_tool.immatch as immatch
# from eval_tool.immatch.utils.data_io import lprint
# import eval_tool.immatch.utils.fire_helper as helper
# from eval_tool.immatch.utils.model_helper import parse_model_config

# def eval_fire(
#         root_dir,
#         config_list,
#         task='homography',
#         h_solver='degensac',
#         ransac_thres=2,
#         match_thres=None,
#         odir='outputs/fire',
#         save_npy=False,
#         print_out=False,
#         debug=False,
# ):
#     # Start overall timing
#     total_start_time = time.time()
    
#     # Init paths
#     data_root = os.path.join(root_dir, 'data/datasets/FIRE')
#     cache_dir = os.path.join(root_dir, odir, 'cache')
#     result_dir = os.path.join(root_dir, odir, 'results', task)
#     mauc = 0
#     total_auc = 0
#     auc_results = {}
#     mle_results = {}
#     gt_dir = os.path.join(data_root, 'Ground Truth')
#     im_dir = os.path.join(data_root, 'Images')
#     match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')]
#     if not os.path.exists(cache_dir):
#         os.makedirs(cache_dir)
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir)

#     # Runtime tracking dictionaries
#     all_runtime_stats = {}
#     all_mle_per_image = {}  # NEW: Store per-image MLE results
    
#     # Iterate over methods
#     for config_name in config_list:
#         method_start_time = time.time()
        
#         # Load model
#         args = parse_model_config(config_name, 'fire', root_dir)
#         class_name = args['class']

#         # One log file per method
#         log_file = os.path.join(result_dir, f'{class_name}.txt')
#         log = open(log_file, 'a')
#         lprint_ = lambda ms: lprint(ms, log)
#         ransac_thres = args['ransac_thres']
        
#         # Iterate over matching thresholds
#         thresholds = match_thres if match_thres else [args['match_threshold']]
#         lprint_(f'\n>>>> Method={class_name} Default config: {args} '
#                 f'Thres: {thresholds}')

#         for thres in thresholds:
#             threshold_start_time = time.time()
            
#             args['match_threshold'] = thres  # Set to target thresholds

#             # Init model
#             model = immatch.__dict__[class_name](args)

#             # Wrapper to track per-image runtime and image pairs
#             image_runtimes = []
#             image_pairs = []  # NEW: Track image pair names
            
#             def timed_matcher(im1, im2):
#                 img_start = time.time()
#                 result = model.match_pairs(im1, im2)
#                 img_time = time.time() - img_start
#                 image_runtimes.append(img_time)
#                 # Store image pair identifiers
#                 pair_name = f"{os.path.basename(im1)}__{os.path.basename(im2)}"
#                 image_pairs.append(pair_name)
#                 return result

#             matcher = timed_matcher

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
#             result = helper.eval_fire(
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
#                 debug=debug
                
#             )
            
#             threshold_time = time.time() - threshold_start_time
            
#             # Extract per-image MLE results if available
#             per_image_mle = result.get('per_image_mle', {})
            
#             # Calculate runtime statistics
#             if image_runtimes:
#                 avg_img_time = sum(image_runtimes) / len(image_runtimes)
#                 min_img_time = min(image_runtimes)
#                 max_img_time = max(image_runtimes)
#                 total_img_time = sum(image_runtimes)
                
#                 runtime_stats = {
#                     'total_images': len(image_runtimes),
#                     'avg_per_image': avg_img_time,
#                     'min_per_image': min_img_time,
#                     'max_per_image': max_img_time,
#                     'total_matching_time': total_img_time,
#                     'threshold_total_time': threshold_time,
#                     'overhead_time': threshold_time - total_img_time,
#                     'all_image_times': image_runtimes,
#                     'image_pairs': image_pairs  # NEW: Include image pair names
#                 }
                
#                 # Store stats
#                 stats_key = f"{class_name}_thres{thres}"
#                 all_runtime_stats[stats_key] = runtime_stats
                
#                 # Store per-image MLE results
#                 if per_image_mle:
#                     all_mle_per_image[stats_key] = per_image_mle
                    
#                     # Log per-image MLE statistics
#                     lprint_(f'\n{"="*60}')
#                     lprint_(f'Per-Image MLE Results for threshold {thres}:')
#                     lprint_(f'{"="*60}')
                    
#                     # Create detailed table
#                     lprint_(f'{"Image Pair":<50} {"MLE Distance":>15}')
#                     lprint_(f'{"-"*65}')
                    
#                     for pair_id, mle_dist in per_image_mle.items():
#                         # Try to get friendly name
#                         pair_name = pair_id
#                         if pair_id.isdigit() and int(pair_id) < len(image_pairs):
#                             pair_name = image_pairs[int(pair_id)]
#                         lprint_(f'{pair_name:<50} {mle_dist:>15.4f}')
                    
#                     # Calculate statistics
#                     mle_values = list(per_image_mle.values())
#                     avg_mle = sum(mle_values) / len(mle_values)
#                     min_mle = min(mle_values)
#                     max_mle = max(mle_values)
                    
#                     lprint_(f'{"-"*65}')
#                     lprint_(f'MLE Statistics:')
#                     lprint_(f'  Average MLE: {avg_mle:.4f}')
#                     lprint_(f'  Min MLE: {min_mle:.4f}')
#                     lprint_(f'  Max MLE: {max_mle:.4f}')
#                     lprint_(f'{"="*60}\n')
                    
#                     # Print to console if requested
#                     if print_out:
#                         print(f'\nPer-Image MLE for {class_name} (threshold={thres}):')
#                         print(f'  Average: {avg_mle:.4f}')
#                         print(f'  Min: {min_mle:.4f}')
#                         print(f'  Max: {max_mle:.4f}')
#                         print(f'  Total images: {len(mle_values)}')
                
#                 # Log runtime information
#                 lprint_(f'\n{"="*60}')
#                 lprint_(f'Runtime Statistics for threshold {thres}:')
#                 lprint_(f'  Total images processed: {len(image_runtimes)}')
#                 lprint_(f'  Average time per image: {avg_img_time:.4f}s')
#                 lprint_(f'  Min time per image: {min_img_time:.4f}s')
#                 lprint_(f'  Max time per image: {max_img_time:.4f}s')
#                 lprint_(f'  Total matching time: {total_img_time:.4f}s')
#                 lprint_(f'  Total threshold time: {threshold_time:.4f}s')
#                 lprint_(f'  Overhead (non-matching): {threshold_time - total_img_time:.4f}s')
#                 lprint_(f'{"="*60}\n')
                
#                 # Print to console if requested
#                 if print_out:
#                     print(f'\nRuntime Statistics for {class_name} (threshold={thres}):')
#                     print(f'  Total images: {len(image_runtimes)}')
#                     print(f'  Avg per image: {avg_img_time:.4f}s')
#                     print(f'  Total time: {threshold_time:.4f}s')
            
#             mauc = result['mauc']
#             total_auc = result['total_auc']
#             auc_results = result.get('auc', {})
#             mle_results = result.get('mle', {})
        
#         method_time = time.time() - method_start_time
#         lprint_(f'\n{"="*60}')
#         lprint_(f'Total runtime for method {class_name}: {method_time:.4f}s ({method_time/60:.2f} min)')
#         lprint_(f'{"="*60}\n')
        
#         log.close()
    
#     # Calculate total runtime
#     total_runtime = time.time() - total_start_time
    
#     # Return results with runtime stats and per-image MLE
#     return mauc, total_auc, auc_results, mle_results, all_runtime_stats, all_mle_per_image, total_runtime


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Benchmark FIRE')
#     parser.add_argument('--gpu', '-gpu', type=str, default='0')
#     parser.add_argument('--root_dir', type=str, default='.')
#     parser.add_argument('--odir', type=str, default='outputs/fire')
#     parser.add_argument('--config', type=str, nargs='*', default=['geoformer'])
#     parser.add_argument('--match_thres', type=float, nargs='*', default=None)
#     parser.add_argument(
#         '--task', type=str, default='homography',
#         choices=['matching', 'homography', 'both']
#     )
#     parser.add_argument(
#         '--h_solver', type=str, default='cv',
#         choices=['degensac', 'cv', 'mle']  # Added 'mle' option
#     )
#     parser.add_argument('--ransac_thres', type=float, default=15)
#     parser.add_argument('--save_npy', action='store_true')
#     parser.add_argument('--print_out', action='store_true', default=True)
#     parser.add_argument('--save_runtime', action='store_true', 
#                        help='Save detailed runtime statistics to file')
#     parser.add_argument('--save_mle_per_image', action='store_true',
#                        help='Save per-image MLE results to separate file')

#     args = parser.parse_args()
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
#     print('='*80)
#     print(f'FIRE Evaluation Configuration:')
#     print(f'  - Homography Solver: {args.h_solver.upper()}')
#     print(f'  - RANSAC Threshold: {args.ransac_thres}')
#     print(f'  - Config: {args.config}')
#     print(f'  - Task: {args.task}')
#     print('='*80)
    
#     # Run evaluation with timing
#     overall_start = time.time()
    
#     mauc, total_auc, auc_results, mle_results, runtime_stats, mle_per_image, total_runtime = eval_fire(
#         args.root_dir, args.config, args.task,
#         h_solver=args.h_solver,
#         ransac_thres=args.ransac_thres,
#         match_thres=args.match_thres,
#         odir=args.odir,
#         save_npy=args.save_npy,
#         print_out=args.print_out
#     )
    
#     print('='*80)
#     print(f'Final Results:')
#     print(f'  - Solver: {args.h_solver.upper()}')
#     if auc_results:
#         print(f'  - Hest AUC: s={auc_results["s"]:.4f} p={auc_results["p"]:.4f} a={auc_results["a"]:.4f} total={total_auc:.4f} m={mauc:.4f}')
#     else:
#         print(f'  - Hest AUC: total={total_auc:.4f} m={mauc:.4f}')
#     if mle_results:
#         print(f'  - MLE dist: s={mle_results["s"]:.4f} p={mle_results["p"]:.4f} a={mle_results["a"]:.4f} total={mle_results["total"]:.4f} m={mle_results["m"]:.4f}')
#     print('='*80)
    
#     # Print runtime summary
#     print('\n' + '='*80)
#     print('RUNTIME SUMMARY:')
#     print('='*80)
    
#     for config_key, stats in runtime_stats.items():
#         print(f'\nConfiguration: {config_key}')
#         print(f'  Total images processed: {stats["total_images"]}')
#         print(f'  Average time per image: {stats["avg_per_image"]:.4f}s')
#         print(f'  Min time per image: {stats["min_per_image"]:.4f}s')
#         print(f'  Max time per image: {stats["max_per_image"]:.4f}s')
#         print(f'  Total matching time: {stats["total_matching_time"]:.2f}s ({stats["total_matching_time"]/60:.2f} min)')
#         print(f'  Throughput: {stats["total_images"]/stats["total_matching_time"]:.2f} images/sec')
    
#     # Print per-image MLE summary
#     if mle_per_image:
#         print('\n' + '='*80)
#         print('PER-IMAGE MLE SUMMARY:')
#         print('='*80)
        
#         for config_key, mle_dict in mle_per_image.items():
#             if mle_dict:
#                 mle_values = list(mle_dict.values())
#                 print(f'\nConfiguration: {config_key}')
#                 print(f'  Total images: {len(mle_values)}')
#                 print(f'  Average MLE: {sum(mle_values)/len(mle_values):.4f}')
#                 print(f'  Min MLE: {min(mle_values):.4f}')
#                 print(f'  Max MLE: {max(mle_values):.4f}')
    
#     print(f'\n{"="*80}')
#     print(f'TOTAL EXECUTION TIME: {total_runtime:.2f}s ({total_runtime/60:.2f} min)')
#     print(f'{"="*80}\n')
    
#     # Optionally save runtime statistics to file
#     if args.save_runtime:
#         import json
#         runtime_file = os.path.join(args.root_dir, args.odir, 'runtime_statistics.json')
        
#         # Prepare data for JSON (convert non-serializable types)
#         json_stats = {}
#         for key, stats in runtime_stats.items():
#             json_stats[key] = {
#                 'total_images': stats['total_images'],
#                 'avg_per_image': stats['avg_per_image'],
#                 'min_per_image': stats['min_per_image'],
#                 'max_per_image': stats['max_per_image'],
#                 'total_matching_time': stats['total_matching_time'],
#                 'threshold_total_time': stats['threshold_total_time'],
#                 'overhead_time': stats['overhead_time'],
#                 'throughput_imgs_per_sec': stats['total_images'] / stats['total_matching_time'],
#                 'all_image_times': stats['all_image_times'],
#                 'image_pairs': stats.get('image_pairs', [])
#             }
        
#         json_stats['total_execution_time'] = total_runtime
        
#         with open(runtime_file, 'w') as f:
#             json.dump(json_stats, f, indent=2)
        
#         print(f'Runtime statistics saved to: {runtime_file}')
    
#     # Optionally save per-image MLE results
#     if args.save_mle_per_image and mle_per_image:
#         import json
#         import csv
        
#         # Save as JSON
#         mle_json_file = os.path.join(args.root_dir, args.odir, 'mle_per_image.json')
#         with open(mle_json_file, 'w') as f:
#             json.dump(mle_per_image, f, indent=2)
#         print(f'Per-image MLE results (JSON) saved to: {mle_json_file}')
        
#         # Also save as CSV for easier analysis
#         mle_csv_file = os.path.join(args.root_dir, args.odir, 'mle_per_image.csv')
#         with open(mle_csv_file, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['Configuration', 'Image_Pair_ID', 'Image_Pair_Name', 'MLE_Distance', 'Runtime_Seconds'])
            
#             for config_key in mle_per_image.keys():
#                 mle_dict = mle_per_image[config_key]
#                 runtime_dict = runtime_stats.get(config_key, {})
#                 image_pairs = runtime_dict.get('image_pairs', [])
#                 image_times = runtime_dict.get('all_image_times', [])
                
#                 for pair_id, mle_dist in mle_dict.items():
#                     # Get friendly name if available
#                     pair_name = pair_id
#                     runtime = ''
#                     if pair_id.isdigit():
#                         idx = int(pair_id)
#                         if idx < len(image_pairs):
#                             pair_name = image_pairs[idx]
#                         if idx < len(image_times):
#                             runtime = f'{image_times[idx]:.4f}'
                    
#                     writer.writerow([config_key, pair_id, pair_name, f'{mle_dist:.6f}', runtime])
        
#         print(f'Per-image MLE results (CSV) saved to: {mle_csv_file}\n')

import argparse
from argparse import Namespace
import os
import time
from collections import defaultdict
import numpy as np

import eval_tool.immatch as immatch
from eval_tool.immatch.utils.data_io import lprint
import eval_tool.immatch.utils.fire_helper as helper
from eval_tool.immatch.utils.model_helper import parse_model_config

def determine_category_from_gt_file(gt_file):
    """
    Determine category from FIRE ground truth filename
    Format: control_points_S10_1_2.txt, control_points_P26_1_2.txt, control_points_A02_1_2.txt
    """
    basename = os.path.basename(gt_file).replace('.txt', '')
    
    # Remove 'control_points_' prefix if present
    if basename.startswith('control_points_'):
        basename = basename.replace('control_points_', '')
    
    # Now check the first character
    first_char = basename[0].upper() if len(basename) > 0 else ''
    
    if first_char == 'S':
        return 'S'
    elif first_char == 'A':
        return 'A'
    elif first_char == 'P':
        return 'P'
    
    return 'Unknown'

def calculate_pairwise_statistics(mle_dict, match_pairs, gt_dir, lprint_=None):
    """
    Calculate mean and standard deviation across image pairs
    
    Parameters:
    -----------
    mle_dict : dict
        Dictionary mapping pair_id to MLE distance
    match_pairs : list
        List of ground truth filenames
    gt_dir : str
        Ground truth directory path
    lprint_ : function
        Logging function
        
    Returns:
    --------
    tuple: (statistics dict, pair_categories dict, detailed_pairs list)
    """
    
    if lprint_ is None:
        lprint_ = print
    
    # Map each pair to its category and collect detailed info
    pair_categories = {}
    detailed_pairs = []
    
    lprint_(f'\n{"="*100}')
    lprint_(f'DETAILED MLE VALUES FOR EACH IMAGE PAIR')
    lprint_(f'{"="*100}')
    lprint_(f'{"Pair ID":<10} {"GT File":<35} {"Category":<10} {"MLE Distance":<15}')
    lprint_(f'{"-"*100}')
    
    for pair_id, mle_val in sorted(mle_dict.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        gt_file = ''
        category = 'Unknown'
        
        if pair_id.isdigit():
            idx = int(pair_id)
            if idx < len(match_pairs):
                gt_file = match_pairs[idx]
                category = determine_category_from_gt_file(gt_file)
                pair_categories[pair_id] = category
        
        detailed_pairs.append({
            'pair_id': pair_id,
            'gt_file': gt_file,
            'category': category,
            'mle': mle_val
        })
        
        lprint_(f'{pair_id:<10} {gt_file:<35} {category:<10} {mle_val:<15.6f}')
    
    lprint_(f'{"="*100}\n')
    
    # Organize MLE values by category
    mle_by_category = {
        'S': [],
        'A': [],
        'P': [],
        'Overall': []
    }
    
    for pair_id, mle_val in mle_dict.items():
        category = pair_categories.get(pair_id, 'Unknown')
        if category in ['S', 'A', 'P']:
            mle_by_category[category].append(mle_val)
        mle_by_category['Overall'].append(mle_val)
    
    # Print category summary
    lprint_(f'\n{"="*100}')
    lprint_(f'CATEGORY DISTRIBUTION')
    lprint_(f'{"="*100}')
    for cat in ['S', 'A', 'P', 'Overall']:
        count = len(mle_by_category[cat])
        lprint_(f'Class {cat}: {count} pairs')
    lprint_(f'{"="*100}\n')
    
    # Calculate statistics for each category
    statistics = {}
    
    lprint_(f'\n{"="*100}')
    lprint_(f'CALCULATING PAIR-WISE STATISTICS FOR EACH CATEGORY')
    lprint_(f'{"="*100}\n')
    
    for cat_name in ['Overall', 'S', 'A', 'P']:
        mle_values = mle_by_category[cat_name]
        
        if len(mle_values) > 0:
            mle_array = np.array(mle_values)
            n = len(mle_array)
            
            # Step 1: Calculate mean
            mean = np.mean(mle_array)
            
            # Step 2: Calculate deviations
            deviations = mle_array - mean
            
            # Step 3: Square deviations
            squared_deviations = deviations ** 2
            
            # Step 4: Sum of squared deviations
            sum_squared_dev = np.sum(squared_deviations)
            
            # Step 5: Variance (divide by n-1)
            if n > 1:
                variance = sum_squared_dev / (n - 1)
                std = np.sqrt(variance)
            else:
                variance = 0.0
                std = 0.0
            
            statistics[cat_name] = {
                'mean': mean,
                'std': std,
                'variance': variance,
                'n': n,
                'min': np.min(mle_array),
                'max': np.max(mle_array),
                'median': np.median(mle_array),
                'sum': np.sum(mle_array),
                'sum_squared_dev': sum_squared_dev
            }
            
            # Print detailed calculation
            lprint_(f'--- Class {cat_name} ---')
            lprint_(f'Number of pairs (n): {n}')
            lprint_(f'\nStep 1: Sum of all MLE values')
            lprint_(f'  Σx_i = {np.sum(mle_array):.6f}')
            lprint_(f'\nStep 2: Calculate Mean')
            lprint_(f'  Mean = Σx_i / n = {np.sum(mle_array):.6f} / {n} = {mean:.6f}')
            lprint_(f'\nStep 3: Calculate Deviations (x_i - mean)')
            if n <= 10:
                for i, (val, dev) in enumerate(zip(mle_array, deviations)):
                    lprint_(f'  Pair {i}: {val:.6f} - {mean:.6f} = {dev:.6f}')
            else:
                lprint_(f'  First 5 deviations:')
                for i in range(5):
                    lprint_(f'  Pair {i}: {mle_array[i]:.6f} - {mean:.6f} = {deviations[i]:.6f}')
                lprint_(f'  ... ({n-5} more pairs)')
            
            lprint_(f'\nStep 4: Square the Deviations')
            if n <= 10:
                for i, sq_dev in enumerate(squared_deviations):
                    lprint_(f'  (deviation_{i})² = {sq_dev:.6f}')
            else:
                lprint_(f'  First 5 squared deviations:')
                for i in range(5):
                    lprint_(f'  (deviation_{i})² = {squared_deviations[i]:.6f}')
                lprint_(f'  ... ({n-5} more)')
            
            lprint_(f'\nStep 5: Sum of Squared Deviations')
            lprint_(f'  Σ(x_i - mean)² = {sum_squared_dev:.6f}')
            
            lprint_(f'\nStep 6: Calculate Variance')
            lprint_(f'  Variance = Σ(x_i - mean)² / (n-1)')
            lprint_(f'  Variance = {sum_squared_dev:.6f} / {n-1} = {variance:.6f}')
            
            lprint_(f'\nStep 7: Calculate Standard Deviation')
            lprint_(f'  Std = √(Variance) = √({variance:.6f}) = {std:.6f}')
            
            lprint_(f'\n  FINAL RESULTS:')
            lprint_(f'  Mean ± Std = {mean:.6f} ± {std:.6f}')
            lprint_(f'  Min: {np.min(mle_array):.6f}')
            lprint_(f'  Max: {np.max(mle_array):.6f}')
            lprint_(f'  Median: {np.median(mle_array):.6f}')
            lprint_(f'\n')
            
        else:
            statistics[cat_name] = {
                'mean': 0.0,
                'std': 0.0,
                'variance': 0.0,
                'n': 0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'sum': 0.0,
                'sum_squared_dev': 0.0
            }
            lprint_(f'--- Class {cat_name} ---')
            lprint_(f'No pairs in this category\n')
    
    return statistics, pair_categories, detailed_pairs

def eval_fire(
        root_dir,
        config_list,
        task='homography',
        h_solver='degensac',
        ransac_thres=2,
        match_thres=None,
        odir='outputs/fire',
        save_npy=False,
        print_out=False,
        debug=False,
):
    # Start overall timing
    total_start_time = time.time()
    
    # Init paths
    data_root = os.path.join(root_dir, 'data/datasets/FIRE')
    cache_dir = os.path.join(root_dir, odir, 'cache')
    result_dir = os.path.join(root_dir, odir, 'results', task)
    mauc = 0
    total_auc = 0
    auc_results = {}
    mle_results = {}
    gt_dir = os.path.join(data_root, 'Ground Truth')
    im_dir = os.path.join(data_root, 'Images')
    match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')]
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Runtime tracking dictionaries
    all_runtime_stats = {}
    all_mle_per_image = {}
    all_pairwise_stats = {}
    all_detailed_pairs = {}
    
    # Iterate over methods
    for config_name in config_list:
        method_start_time = time.time()
        
        # Load model
        args = parse_model_config(config_name, 'fire', root_dir)
        class_name = args['class']

        # One log file per method
        log_file = os.path.join(result_dir, f'{class_name}.txt')
        log = open(log_file, 'a')
        lprint_ = lambda ms: lprint(ms, log)
        ransac_thres = args['ransac_thres']
        
        # Iterate over matching thresholds
        thresholds = match_thres if match_thres else [args['match_threshold']]
        lprint_(f'\n>>>> Method={class_name} Default config: {args} '
                f'Thres: {thresholds}')

        for thres in thresholds:
            threshold_start_time = time.time()
            
            args['match_threshold'] = thres

            # Init model
            model = immatch.__dict__[class_name](args)

            # Wrapper to track per-image runtime and image pairs
            image_runtimes = []
            image_pairs = []
            
            def timed_matcher(im1, im2):
                img_start = time.time()
                result = model.match_pairs(im1, im2)
                img_time = time.time() - img_start
                image_runtimes.append(img_time)
                pair_name = f"{os.path.basename(im1)}__{os.path.basename(im2)}"
                image_pairs.append(pair_name)
                return result

            matcher = timed_matcher

            # Init result save path
            result_npy = None
            if save_npy:
                result_tag = model.name
                if args['imsize'] > 0:
                    result_tag += f".im{args['imsize']}"
                if thres > 0:
                    result_tag += f'.m{thres}'
                result_npy = os.path.join(cache_dir, f'{result_tag}.npy')

            lprint_(f'Matching thres: {thres}  Save to: {result_npy}')

            # Eval on the specified task(s)
            result = helper.eval_fire(
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
                debug=debug
            )
            
            threshold_time = time.time() - threshold_start_time
            
            # Extract per-image MLE results
            per_image_mle = result.get('per_image_mle', {})
            
            # Calculate pair-wise statistics with detailed output
            if per_image_mle:
                pairwise_stats, pair_categories, detailed_pairs = calculate_pairwise_statistics(
                    per_image_mle, match_pairs, gt_dir, lprint_
                )
                
                stats_key = f"{class_name}_thres{thres}"
                all_pairwise_stats[stats_key] = pairwise_stats
                all_detailed_pairs[stats_key] = detailed_pairs
                
                # Summary table
                lprint_(f'\n{"="*100}')
                lprint_(f'SUMMARY TABLE: PAIR-WISE MLE STATISTICS')
                lprint_(f'{"="*100}')
                lprint_(f'{"Category":<15} {"N":>6} {"Mean":>12} {"Std":>12} {"Min":>12} {"Max":>12} {"Median":>12}')
                lprint_(f'{"-"*100}')
                
                for cat in ['Overall', 'S', 'A', 'P']:
                    if cat in pairwise_stats:
                        stats = pairwise_stats[cat]
                        lprint_(f'{cat:<15} {stats["n"]:>6} {stats["mean"]:>12.6f} '
                               f'{stats["std"]:>12.6f} {stats["min"]:>12.6f} '
                               f'{stats["max"]:>12.6f} {stats["median"]:>12.6f}')
                
                lprint_(f'{"="*100}\n')
                
                # LaTeX table format
                if pairwise_stats['S']['n'] > 0 and pairwise_stats['A']['n'] > 0 and pairwise_stats['P']['n'] > 0:
                    lprint_(f'\n{"="*100}')
                    lprint_(f'LATEX TABLE FORMAT:')
                    lprint_(f'{"="*100}')
                    lprint_(f'{class_name}  & '
                           f'{pairwise_stats["Overall"]["mean"]:.2f}\\pm{pairwise_stats["Overall"]["std"]:.2f} & '
                           f'{pairwise_stats["S"]["mean"]:.2f}\\pm{pairwise_stats["S"]["std"]:.2f} & '
                           f'{pairwise_stats["A"]["mean"]:.2f}\\pm{pairwise_stats["A"]["std"]:.2f} & '
                           f'{pairwise_stats["P"]["mean"]:.2f}\\pm{pairwise_stats["P"]["std"]:.2f} \\\\')
                    lprint_(f'{"="*100}\n')
                else:
                    lprint_(f'\nWARNING: Some categories are empty!')
                    lprint_(f'  S: {pairwise_stats["S"]["n"]} pairs')
                    lprint_(f'  A: {pairwise_stats["A"]["n"]} pairs')
                    lprint_(f'  P: {pairwise_stats["P"]["n"]} pairs\n')
                
                # Console output
                if print_out:
                    print(f'\n{"="*80}')
                    print(f'Pair-wise MLE Statistics for {class_name} (threshold={thres}):')
                    print(f'{"="*80}')
                    print(f'{"Category":<15} {"N":>6} {"Mean":>12} {"Std":>12}')
                    print(f'{"-"*60}')
                    for cat in ['Overall', 'S', 'A', 'P']:
                        if cat in pairwise_stats:
                            stats = pairwise_stats[cat]
                            print(f'{cat:<15} {stats["n"]:>6} {stats["mean"]:>12.6f} {stats["std"]:>12.6f}')
                    print(f'{"="*80}\n')
            
            # Calculate runtime statistics
            if image_runtimes:
                avg_img_time = sum(image_runtimes) / len(image_runtimes)
                min_img_time = min(image_runtimes)
                max_img_time = max(image_runtimes)
                total_img_time = sum(image_runtimes)
                
                runtime_stats = {
                    'total_images': len(image_runtimes),
                    'avg_per_image': avg_img_time,
                    'min_per_image': min_img_time,
                    'max_per_image': max_img_time,
                    'total_matching_time': total_img_time,
                    'threshold_total_time': threshold_time,
                    'overhead_time': threshold_time - total_img_time,
                    'all_image_times': image_runtimes,
                    'image_pairs': image_pairs
                }
                
                stats_key = f"{class_name}_thres{thres}"
                all_runtime_stats[stats_key] = runtime_stats
                all_mle_per_image[stats_key] = per_image_mle
                
                # Log runtime information
                lprint_(f'\n{"="*60}')
                lprint_(f'Runtime Statistics for threshold {thres}:')
                lprint_(f'  Total images processed: {len(image_runtimes)}')
                lprint_(f'  Average time per image: {avg_img_time:.4f}s')
                lprint_(f'  Min time per image: {min_img_time:.4f}s')
                lprint_(f'  Max time per image: {max_img_time:.4f}s')
                lprint_(f'  Total matching time: {total_img_time:.4f}s')
                lprint_(f'  Total threshold time: {threshold_time:.4f}s')
                lprint_(f'  Overhead (non-matching): {threshold_time - total_img_time:.4f}s')
                lprint_(f'{"="*60}\n')
            
            mauc = result['mauc']
            total_auc = result['total_auc']
            auc_results = result.get('auc', {})
            mle_results = result.get('mle', {})
        
        method_time = time.time() - method_start_time
        lprint_(f'\n{"="*60}')
        lprint_(f'Total runtime for method {class_name}: {method_time:.4f}s ({method_time/60:.2f} min)')
        lprint_(f'{"="*60}\n')
        
        log.close()
    
    # Calculate total runtime
    total_runtime = time.time() - total_start_time
    
    return mauc, total_auc, auc_results, mle_results, all_runtime_stats, all_mle_per_image, all_pairwise_stats, all_detailed_pairs, total_runtime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark FIRE with detailed MLE analysis')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--odir', type=str, default='outputs/fire')
    parser.add_argument('--config', type=str, nargs='*', default=['geoformer'])
    parser.add_argument('--match_thres', type=float, nargs='*', default=None)
    parser.add_argument(
        '--task', type=str, default='homography',
        choices=['matching', 'homography', 'both']
    )
    parser.add_argument(
        '--h_solver', type=str, default='cv',
        choices=['degensac', 'cv', 'mle']
    )
    parser.add_argument('--ransac_thres', type=float, default=15)
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--print_out', action='store_true', default=True)
    parser.add_argument('--save_runtime', action='store_true')
    parser.add_argument('--save_mle_per_image', action='store_true')
    parser.add_argument('--save_pairwise_stats', action='store_true')
    parser.add_argument('--save_detailed_pairs', action='store_true')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    print('='*80)
    print(f'FIRE Evaluation Configuration:')
    print(f'  - Homography Solver: {args.h_solver.upper()}')
    print(f'  - RANSAC Threshold: {args.ransac_thres}')
    print(f'  - Config: {args.config}')
    print(f'  - Task: {args.task}')
    print('='*80)
    
    # Run evaluation
    mauc, total_auc, auc_results, mle_results, runtime_stats, mle_per_image, pairwise_stats, detailed_pairs, total_runtime = eval_fire(
        args.root_dir, args.config, args.task,
        h_solver=args.h_solver,
        ransac_thres=args.ransac_thres,
        match_thres=args.match_thres,
        odir=args.odir,
        save_npy=args.save_npy,
        print_out=args.print_out
    )
    
    print('='*80)
    print(f'Final Results:')
    print(f'  - Solver: {args.h_solver.upper()}')
    if auc_results:
        print(f'  - Hest AUC: s={auc_results["s"]:.4f} p={auc_results["p"]:.4f} a={auc_results["a"]:.4f} total={total_auc:.4f} m={mauc:.4f}')
    if mle_results:
        print(f'  - MLE dist: s={mle_results["s"]:.4f} p={mle_results["p"]:.4f} a={mle_results["a"]:.4f} total={mle_results["total"]:.4f} m={mle_results["m"]:.4f}')
    print('='*80)
    
    # Print pair-wise statistics summary
    if pairwise_stats:
        print('\n' + '='*80)
        print('PAIR-WISE MLE STATISTICS SUMMARY:')
        print('='*80)
        
        for config_key, stats_dict in pairwise_stats.items():
            print(f'\nConfiguration: {config_key}')
            print(f'{"Category":<15} {"N":>6} {"Mean±Std":>24}')
            print(f'{"-"*60}')
            for cat in ['Overall', 'S', 'A', 'P']:
                if cat in stats_dict:
                    stats = stats_dict[cat]
                    print(f'{cat:<15} {stats["n"]:>6}   {stats["mean"]:>10.6f} ± {stats["std"]:<10.6f}')
    
    print(f'\n{"="*80}')
    print(f'TOTAL EXECUTION TIME: {total_runtime:.2f}s ({total_runtime/60:.2f} min)')
    print(f'{"="*80}\n')
    
    # Save files
    if args.save_detailed_pairs and detailed_pairs:
        import csv
        
        for config_key, pairs_list in detailed_pairs.items():
            csv_file = os.path.join(args.root_dir, args.odir, f'{config_key}_detailed_pairs.csv')
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Pair_ID', 'GT_File', 'Category', 'MLE_Distance'])
                
                for pair_info in pairs_list:
                    writer.writerow([
                        pair_info['pair_id'],
                        pair_info['gt_file'],
                        pair_info['category'],
                        f"{pair_info['mle']:.6f}"
                    ])
            
            print(f'Detailed pairs saved to: {csv_file}')
    
    if args.save_pairwise_stats and pairwise_stats:
        import json
        import csv
        
        stats_json_file = os.path.join(args.root_dir, args.odir, 'pairwise_statistics.json')
        with open(stats_json_file, 'w') as f:
            json.dump(pairwise_stats, f, indent=2)
        print(f'Pair-wise statistics (JSON) saved to: {stats_json_file}')
        
        stats_csv_file = os.path.join(args.root_dir, args.odir, 'pairwise_statistics.csv')
        with open(stats_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Configuration', 'Category', 'N', 'Mean', 'Std', 'Variance', 'Min', 'Max', 'Median'])
            
            for config_key, stats_dict in pairwise_stats.items():
                for cat in ['Overall', 'S', 'A', 'P']:
                    if cat in stats_dict:
                        stats = stats_dict[cat]
                        writer.writerow([
                            config_key, cat, stats['n'], 
                            f'{stats["mean"]:.6f}', f'{stats["std"]:.6f}',
                            f'{stats["variance"]:.6f}',
                            f'{stats["min"]:.6f}', f'{stats["max"]:.6f}',
                            f'{stats["median"]:.6f}'
                        ])
        
        print(f'Pair-wise statistics (CSV) saved to: {stats_csv_file}')
        
        latex_file = os.path.join(args.root_dir, args.odir, 'pairwise_latex_table.txt')
        with open(latex_file, 'w') as f:
            f.write('% Pair-wise MLE Statistics for LaTeX\n')
            f.write('% Mean ± Standard Deviation (calculated across image pairs)\n\n')
            
            for config_key, stats_dict in pairwise_stats.items():
                if stats_dict['S']['n'] > 0 and stats_dict['A']['n'] > 0 and stats_dict['P']['n'] > 0:
                    method_name = config_key.split('_thres')[0]
                    overall = stats_dict['Overall']
                    s = stats_dict['S']
                    a = stats_dict['A']
                    p = stats_dict['P']
                    
                    f.write(f'{method_name}  & '
                           f'{overall["mean"]:.2f}\\pm{overall["std"]:.2f} (n={overall["n"]}) & '
                           f'{s["mean"]:.2f}\\pm{s["std"]:.2f} (n={s["n"]}) & '
                           f'{a["mean"]:.2f}\\pm{a["std"]:.2f} (n={a["n"]}) & '
                           f'{p["mean"]:.2f}\\pm{p["std"]:.2f} (n={p["n"]}) \\\\\n')
        
        print(f'LaTeX table saved to: {latex_file}\n')