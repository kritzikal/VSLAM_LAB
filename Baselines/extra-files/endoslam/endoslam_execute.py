#!/usr/bin/env python3
"""
VSLAM-LAB EndoSLAM Execution Script
Wrapper for Endo-SfMLearner pose estimation
"""

import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Add EndoSLAM to path
endoslam_path = os.path.join(os.getcwd(), 'Baselines', 'EndoSLAM', 'EndoSfMLearner')
sys.path.append(endoslam_path)

from models import PoseExpNet


def load_tensor_image(filename, img_height, img_width):
    """Load and preprocess image for PoseNet"""
    img = Image.open(filename).convert('RGB')
    img = img.resize((img_width, img_height), Image.LANCZOS)
    img = np.array(img).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = torch.from_numpy(img).unsqueeze(0) / 255.0
    return img


def pose_vec_to_mat(vec):
    """Convert 6DoF pose vector to 4x4 transformation matrix

    Args:
        vec: [tx, ty, tz, rx, ry, rz] pose vector (Euler angles)
    Returns:
        4x4 transformation matrix
    """
    translation = vec[:3]
    rotation = vec[3:]

    # Convert Euler angles to rotation matrix
    rot = R.from_euler('xyz', rotation)
    rot_mat = rot.as_matrix()

    # Build 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = translation

    return transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VSLAM-LAB EndoSLAM Execution")

    # Standard VSLAM-LAB arguments
    parser.add_argument("--sequence_path", type=str, required=True, help="Path to image sequence")
    parser.add_argument("--calibration_yaml", type=str, help="Path to calibration file")
    parser.add_argument("--rgb_csv", type=str, required=True, help="Path to RGB image list")
    parser.add_argument("--exp_folder", type=str, required=True, help="Path to save results")
    parser.add_argument("--exp_it", type=str, required=True, help="Experiment iteration")
    parser.add_argument("--settings_yaml", type=str, help="Path to settings YAML")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose output")
    parser.add_argument("--mode", type=str, default="mono", help="Mode: mono")

    # EndoSLAM-specific arguments
    parser.add_argument("--pretrained_posenet", type=str, required=True, help="Path to PoseNet weights")
    parser.add_argument("--pretrained_dispnet", type=str, help="Path to DispNet weights (optional)")
    parser.add_argument("--img_height", type=int, default=256, help="Image height")
    parser.add_argument("--img_width", type=int, default=832, help="Image width")
    parser.add_argument("--rotation_mode", type=str, default='euler', choices=['euler', 'quat'])

    args, unknown = parser.parse_known_args()

    verbose = bool(args.verbose)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n[EndoSLAM] Starting pose estimation...")
        print(f"[EndoSLAM] Device: {device}")
        print(f"[EndoSLAM] PoseNet weights: {args.pretrained_posenet}")
        print(f"[EndoSLAM] Sequence path: {args.sequence_path}")

    # Load image list
    image_list = []
    timestamps = []
    with open(args.rgb_csv, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                timestamp = parts[0]
                img_path = parts[1]
                image_list.append(os.path.join(args.sequence_path, img_path))
                timestamps.append(timestamp)

    if verbose:
        print(f"[EndoSLAM] Loaded {len(image_list)} images")

    # Load PoseNet model
    weights = torch.load(args.pretrained_posenet, map_location=device)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval()

    if verbose:
        print(f"[EndoSLAM] PoseNet loaded (sequence length: {seq_length})")

    # Initialize trajectory
    poses = [np.eye(4)]  # Start with identity

    # Process image pairs
    with torch.no_grad():
        for i in range(len(image_list) - 1):
            # Load target and reference images
            tgt_img = load_tensor_image(image_list[i + 1], args.img_height, args.img_width).to(device)
            ref_img = load_tensor_image(image_list[i], args.img_height, args.img_width).to(device)

            # Stack reference images (PoseNet expects specific input format)
            ref_imgs = ref_img.unsqueeze(1)  # Add sequence dimension

            # Predict pose
            _, pose_pred = pose_net(tgt_img, ref_imgs)
            pose_vec = pose_pred[0, 0].cpu().numpy()  # Get first prediction

            # Convert to transformation matrix
            rel_pose = pose_vec_to_mat(pose_vec)

            # Accumulate pose (chain transformations)
            abs_pose = poses[-1] @ rel_pose
            poses.append(abs_pose)

            if verbose and (i + 1) % 50 == 0:
                print(f"[EndoSLAM] Processed {i + 1}/{len(image_list) - 1} frames")

    # Save trajectory in TUM format
    trajectory_file = os.path.join(args.exp_folder, args.exp_it.zfill(5) + '_KeyFrameTrajectory.txt')
    os.makedirs(args.exp_folder, exist_ok=True)

    with open(trajectory_file, 'w') as f:
        for i, pose in enumerate(poses):
            # Extract translation
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]

            # Extract rotation and convert to quaternion
            rot_mat = pose[:3, :3]
            rotation = R.from_matrix(rot_mat)
            qx, qy, qz, qw = rotation.as_quat()

            # Write in TUM format: timestamp tx ty tz qx qy qz qw
            ts = timestamps[i] if i < len(timestamps) else str(i)
            f.write(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

    if verbose:
        print(f"[EndoSLAM] Trajectory saved to: {trajectory_file}")
        print(f"[EndoSLAM] Total poses: {len(poses)}")
        print("[EndoSLAM] Done!")
