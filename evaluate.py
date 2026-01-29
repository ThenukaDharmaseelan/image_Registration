import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import normalized_mutual_info_score
from skimage.morphology import skeletonize, binary_opening, disk


# 1. parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Single image mode (evaluate one set of images)
    parser.add_argument('--fixed_img', type=str, default=None, help="Path to single fixed color image")
    parser.add_argument('--fixed_map', type=str, default=None, help="Path to single fixed vessel map")
    parser.add_argument('--reg_eye', type=str, default=None, help="Path to single EyeLiner registered image")
    parser.add_argument('--reg_bin', type=str, default=None, help="Path to single Binary Process registered map")
    
    # Batch mode (original folder-based)
    parser.add_argument("--fixed_imgs", type=str, default='data/Images', help="Path to fixed color images folder (batch mode)")
    parser.add_argument('--fixed_maps', type=str, default='data/vessel_segmentations_automorph', help="Path to fixed vessel maps folder (batch mode)")
    parser.add_argument('--reg_eyelnr', type=str, default='data/registration_approaches/Eyeliner', help="Path to EyeLiner registered images folder (batch mode)")
    parser.add_argument('--reg_bin_folder', type=str, default='data/registration_approaches/binary_process', help="Path to Binary Process registered maps folder (batch mode)")
    
    parser.add_argument('--output', type=str, default='evaluation_results.csv', help="Output CSV filename")
    return parser.parse_args()


def is_single_mode(args):
    """Check if running in single image mode."""
    return args.fixed_img is not None


# 2. Image processing
def load_image(path, as_gray=False):
    """Image loader handling extension mismatches"""
    path = Path(path)
    if not path.exists():
        found = False
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            alt_path = path.with_suffix(ext)
            if alt_path.exists():
                path = alt_path
                found = True
                break
        if not found:
            return None
            
    img = cv2.imread(str(path))
    if img is None: return None

    if as_gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def get_fov_mask(img_gray):
    """Generates mask to exclude black corners (Field of View)."""
    mask = img_gray > 10
    return binary_opening(mask, disk(3))


def ensure_binary_map(img):
    """
    Returns a boolean vessel map.
    - If input is Color (EyeLiner): Auto-extracts vessels from Green channel.
    - If input is Binary: Thresholds directly.
    """
    if img is None: return None
    
    # colour image (eyeliner) extract green channel
    if img.ndim == 3 and not np.array_equal(img[:,:,0], img[:,:,1]):
        green_channel = img[:,:,1]
        inv_green = cv2.bitwise_not(green_channel)
        mask = get_fov_mask(green_channel)
        
        # adaptive thresholding for vessel extraction
        thresh = cv2.adaptiveThreshold(inv_green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
        return (thresh > 0) & mask
    
    # binary image/ grayscale map
    if img.ndim == 3: img = img[:, :, 0]
    return img > 127


# 3. Metrics
def intensity_metrics(fixed, registered):
    """Computes NMI, Wasserstein, and NCC on FOV pixels."""
    if fixed is None or registered is None: 
        return np.nan, np.nan, np.nan

    if fixed.shape != registered.shape:
        registered = cv2.resize(registered, (fixed.shape[1], fixed.shape[0]))
        
    fov_mask = get_fov_mask(fixed)
    
    # NMI Quanized
    f_bins = (fixed[fov_mask] // 4).astype(int)
    r_bins = (registered[fov_mask] // 4).astype(int)
    nmi = normalized_mutual_info_score(f_bins, r_bins) # using scikit-learn implementation
    
    # Wasserstein Distance
    f_flat = fixed[fov_mask].flatten()
    r_flat = registered[fov_mask].flatten()
    if len(f_flat) > 10000: # subsample for efficiency if large
        sample_indices = np.random.choice(len(f_flat), size=10000, replace=False)
        f_flat = f_flat[sample_indices]
        r_flat = r_flat[sample_indices]
    wsd = wasserstein_distance(f_flat, r_flat) # using scipy implementation
    
    # Normalized Cross-Correlation (NCC)
    if np.std(f_flat) == 0 or np.std(r_flat) == 0:
        ncc = 0.0
    else:
        f_norm = (f_flat - np.mean(f_flat))
        r_norm = (r_flat - np.mean(r_flat))
        ncc = np.mean(f_norm * r_norm) / (np.std(f_flat) * np.std(r_flat)) # NCC = (I(x,y) - mean_I) * (J(x,y) - mean_J) / (std_I * std_J)
        
    return nmi, wsd, ncc


def geometric_metrics(fixed_mask, reg_mask):
    """Computes Hausdorff Distance (HD95) and Centerline Dice."""
    if fixed_mask is None or reg_mask is None:
        return np.nan, np.nan, np.nan
    
    if fixed_mask.shape != reg_mask.shape:
        reg_mask = cv2.resize(reg_mask.astype(np.uint8), (fixed_mask.shape[1], fixed_mask.shape[0])).astype(bool)
        
    # Hausdorff Distance
    u = np.argwhere(fixed_mask) # coords of fixed vessels
    v = np.argwhere(reg_mask)   # coords of registered vessels
    
    if len(u) == 0 or len(v) == 0:
        hd100, hd95 = np.nan, np.nan
    else:
        # Use KDTree for fast nearest-neighbor lookup
        tree_u = cKDTree(u)
        tree_v = cKDTree(v)
        
        # For every point in U, find distance to nearest V
        d_u_to_v, _ = tree_v.query(u, k=1)
        # For every point in V, find distance to nearest U
        d_v_to_u, _ = tree_u.query(v, k=1)
        
        # HD100 (Max)
        hd100 = max(np.max(d_u_to_v), np.max(d_v_to_u))
        
        # HD95 (95th Percentile)
        hd95 = max(np.percentile(d_u_to_v, 95), np.percentile(d_v_to_u, 95)) # page 10 of Taha and Hansbury "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool" mentions q is selected to factor out outliers. we can use 95th percentile instead of max 100.
    
    # Centerline Dice
    fixed_skeleton = skeletonize(fixed_mask)
    reg_skeleton = skeletonize(reg_mask)
    
    len_fixed_skeleton = np.sum(fixed_skeleton)
    len_reg_skeleton = np.sum(reg_skeleton)
    
    if len_fixed_skeleton == 0 or len_reg_skeleton == 0:
        cldice = 0.0
    else:
        # calculate topological precision (how much of the registered skeleton is inside the fixed skeleton)
        match_reg_skel = np.sum(reg_skeleton * fixed_mask)
        topology_precision = match_reg_skel / len_reg_skeleton
        
        # calculate topological sensitivity (how much of the fixed skeleton is captured by the registered skeleton)
        match_fixed_skel = np.sum(fixed_skeleton * reg_mask)
        topology_sensitivity = match_fixed_skel / len_fixed_skeleton
        
        # harmonic mean (clDice)
        cldice = (2 * topology_precision * topology_sensitivity) / (topology_precision + topology_sensitivity + 1e-8)
        
    return hd95, hd100, cldice


          
def evaluate_single(args):
    """Evaluate a single set of 4 images."""
    img_id = Path(args.fixed_img).stem
    
    img_fixed_col = load_image(args.fixed_img, as_gray=True)
    img_fixed_map = load_image(args.fixed_map)
    img_eye_reg = load_image(args.reg_eye) if args.reg_eye else None
    img_bin_reg = load_image(args.reg_bin) if args.reg_bin else None
    
    # Check required files
    if img_fixed_col is None:
        print(f"Error: Could not load fixed image: {args.fixed_img}")
        return
    if img_fixed_map is None:
        print(f"Error: Could not load fixed map: {args.fixed_map}")
        return
    
    row = {'ID': img_id}
    
    # A. Intensity Metrics
    if img_eye_reg is not None:
        eye_gray = cv2.cvtColor(img_eye_reg, cv2.COLOR_BGR2GRAY) if img_eye_reg.ndim == 3 else img_eye_reg
        nmi, wsd, ncc = intensity_metrics(img_fixed_col, eye_gray)
    else:
        nmi, wsd, ncc = np.nan, np.nan, np.nan
    
    row.update({'Eye_NMI': nmi, 'Eye_WSD': wsd, 'Eye_NCC': ncc})
    
    # B. Geometric Metrics
    map_fix = ensure_binary_map(img_fixed_map)
    map_eye = ensure_binary_map(img_eye_reg)
    map_bin = ensure_binary_map(img_bin_reg)
    
    eye_hd95, eye_hd100, eye_cld = geometric_metrics(map_fix, map_eye)
    bin_hd95, bin_hd100, bin_cld = geometric_metrics(map_fix, map_bin)
    
    row.update({
        'Eye_HD100': eye_hd100, 'Eye_HD95': eye_hd95, 'Eye_clDice': eye_cld,
        'Bin_HD100': bin_hd100, 'Bin_HD95': bin_hd95, 'Bin_clDice': bin_cld
    })
    
    # Print results
    header = f"{'ID':<15} | {'Eye_NMI':<8} {'Eye_WSD':<8} | {'Eye_HD95':<8} {'Bin_HD95':<8} | {'Eye_HD100':<9} {'Bin_HD100':<9} | {'Eye_clD':<8} {'Bin_clD':<8}"
    print(header)
    print("-" * 130)
    print(f"{img_id:<15} | {nmi:<8.4f} {wsd:<8.2f} | {eye_hd95:<8.2f} {bin_hd95:<8.2f} | {eye_hd100:<9.2f} {bin_hd100:<9.2f} | {eye_cld:<8.4f} {bin_cld:<8.4f}")
    
    # Save
    df = pd.DataFrame([row])
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


def evaluate_batch(args):
    """Evaluate all images in folders (original batch mode)."""
    # Setup Paths
    p_fixed = Path(args.fixed_imgs)
    p_maps = Path(args.fixed_maps)
    p_eye = Path(args.reg_eyelnr)
    p_bin = Path(args.reg_bin_folder)
    
    # Verify directories
    for p in [p_fixed, p_maps, p_eye, p_bin]:
        if not p.exists():
            print(f"Error: Directory not found: {p}")
            return

    # Get list of fixed images
    fixed_files = sorted([f for f in p_fixed.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    print(f"Starting evaluation on {len(fixed_files)} images...")
    header = f"{'ID':<15} | {'Eye_NMI':<8} {'Eye_WSD':<8} | {'Eye_HD95':<8} {'Bin_HD95':<8} | {'Eye_HD100':<9} {'Bin_HD100':<9} | {'Eye_clD':<8} {'Bin_clD':<8}"
    print(header)
    print("-" * 130)
    
    results = []

    for idx, f_path in enumerate(fixed_files):
        img_id = f_path.stem
        
        def find_match(folder, id_str):
            matches = list(folder.glob(f"*{id_str}*"))
            return load_image(matches[0]) if matches else None

        img_fixed_col = load_image(f_path, as_gray=True)
        img_fixed_map = find_match(p_maps, img_id)
        img_eye_reg = find_match(p_eye, img_id)
        img_bin_reg = find_match(p_bin, img_id)
        
        row = {'ID': img_id}
        
        # A. Intensity Metrics
        if img_eye_reg is not None:
            eye_gray = cv2.cvtColor(img_eye_reg, cv2.COLOR_BGR2GRAY) if img_eye_reg.ndim == 3 else img_eye_reg
            nmi, wsd, ncc = intensity_metrics(img_fixed_col, eye_gray)
        else:
            nmi, wsd, ncc = np.nan, np.nan, np.nan
        
        row.update({'Eye_NMI': nmi, 'Eye_WSD': wsd, 'Eye_NCC': ncc})
        
        # B. Geometric Metrics
        map_fix = ensure_binary_map(img_fixed_map)
        map_eye = ensure_binary_map(img_eye_reg)
        map_bin = ensure_binary_map(img_bin_reg)
        
        eye_hd95, eye_hd100, eye_cld = geometric_metrics(map_fix, map_eye)
        bin_hd95, bin_hd100, bin_cld = geometric_metrics(map_fix, map_bin)
        
        row.update({
            'Eye_HD100': eye_hd100, 'Eye_HD95': eye_hd95, 'Eye_clDice': eye_cld,
            'Bin_HD100': bin_hd100, 'Bin_HD95': bin_hd95, 'Bin_clDice': bin_cld
        })
        
        results.append(row)
        
        print(f"{img_id:<15} | {nmi:<8.4f} {wsd:<8.2f} | {eye_hd95:<8.2f} {bin_hd95:<8.2f} | {eye_hd100:<9.2f} {bin_hd100:<9.2f} | {eye_cld:<8.4f} {bin_cld:<8.4f}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nEvaluation complete. Results saved to {args.output}")


def main():
    args = parse_args()
    
    if is_single_mode(args):
        evaluate_single(args)
    else:
        evaluate_batch(args)

if __name__ == "__main__":
    main()