from pathlib import Path
import tomllib as toml
import numpy as np

from vid2pose_utils import (sam3_tracker,
                            write_mask_overlay,
                            estimate_keypoints,
                            write_keypoints_to_npz,
                            write_keypoint_overlay,
                            load_calibration,
                            triangulate_poses_in_video,
                            write_results_to_npz,
                            write_npz_to_trc,
                            interpolate_nan_keypoints,
                            )

ROOT = Path(__file__).resolve().parent

def load_config(cfg_fpath: Path):
    with open(cfg_fpath, "rb") as f:
        cfg = toml.load(f)
    paths = {k:Path(v) for k, v in cfg["paths"].items()}
    seeds = {k: [tuple(p) for p in v] for k, v in cfg["seeds"].items()}
    return paths, seeds

def main():
    # load paths and seed points
    paths, seeds = load_config(ROOT / "config.toml")

    # per video tracking and pose inference loop
    keypoint_npzs = {}
    calibration_data = {}
    for cam in list(seeds.keys()):
        video_path = paths["input_path"] / f"{cam}.mp4"
        calib_path = paths["input_path"] / "cal" / f"{cam}_calibration.json"
        masked_video_path = paths["output_path"] / f"{cam}_masked.mp4"
        keypoint_npz_path = paths["output_path"] / f"{cam}_keypoints.npz"
        keypoint_video_path = paths["output_path"] / f"{cam}_keypoints.mp4"
        
        # load the calibration data
        calib = load_calibration(calib_path)
        calibration_data[cam] = calib

        # run the inference pipe if the data doesn't already exist
        if not Path(keypoint_npz_path).exists():
            print(f"Processing video from camera {cam}")
            masks = sam3_tracker(video_path, 
                                coords=seeds[cam])
            print(f"Generated masks for {len(masks)} frames in video {video_path.name}.")
            
            write_mask_overlay(video_path,
                                masks,
                                masked_video_path,
                                alpha=0.5,
                                color=(0, 0, 255))

            results = estimate_keypoints(video_path,
                                        masks,
                                        )
            
            # cache the 2D keypoints for 3D reconstruction
            write_keypoints_to_npz(keypoint_npz_path, results)
            keypoint_npzs[cam] = keypoint_npz_path
            write_keypoint_overlay(video_path,
                                    results,
                                    keypoint_video_path)
            
            # optional: cache the full 3DB output
            # TODO: pass the 3D keypoints for monocular usage 
            #write_results_to_npz(ROOT / "output" / f"{cam}_results.npz", results)

        # load existing keypoints if they do exist
        else:
            print(f"Keypoints for camera {cam} already exist at {keypoint_npz_path}")
            keypoint_npzs[cam] = keypoint_npz_path

    # multiview reconstruction
    kp = {c: np.load(p)['keypoints'] for c, p in keypoint_npzs.items()}
    X3d = triangulate_poses_in_video(keypoints=kp, 
                                     calibration_data=calibration_data, 
                                     method="all", 
                                     threshold_px=50.0, # only used if method="consensus"
                                     )
    
    # Interpolate NaN keypoints if they exist (can happen with consensus reconstruction)
    if np.isnan(X3d).any():
        print("interpolating NaN keypoints before writing to file")
        X3d = interpolate_nan_keypoints(X3d)
        
    # save the triangulated 3D keypoints
    np.savez(paths["output_path"] / "X3d.npz", X3d=X3d)
    write_npz_to_trc(paths["output_path"] / "X3d.npz", data_rate=30)
    
if __name__ == "__main__":
    main()
    