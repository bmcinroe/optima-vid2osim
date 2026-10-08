import json
import cv2
import torch
import numpy as np
from itertools import combinations
from pathlib import Path
import os

def load_calibration(calib_fpath):
    with open(calib_fpath, encoding="utf-8") as f:
        calib = json.load(f)
        
    return {
        int(frame): {
            "K": np.array(matrices["intrinsic_matrix"], dtype=np.float32),
            "E": np.array(matrices["extrinsic_matrix"], dtype=np.float32)
        }
        for frame, matrices in calib.items()
    }

def sam3_tracker(video_path, coords: tuple = None, obj_id: int = 1, tracker=None, max_frames=None):
    video_path = str(video_path)
    
    if tracker is None:
        from sam3.model_builder import build_sam3_video_model
        model = build_sam3_video_model()
        tracker = model.tracker
        tracker.backbone = model.detector.backbone
        print("Initialized SAM3 tracker.")
    
    state = tracker.init_state(video_path=video_path)
    H, W = state["video_height"], state["video_width"]
    
    return _track_subject(state, tracker, coords, obj_id=obj_id, max_frames=max_frames)

def _track_subject(state, tracker, coords: list, obj_id: int = 1, max_frames=None):
    H, W = state["video_height"], state["video_width"]
    point_list = []
    for coord in coords:
        point_list.append([coord[0] / W, coord[1] / H])
    points = torch.tensor(point_list, dtype=torch.float32)
    labels = torch.tensor([1]*points.shape[0], dtype=torch.int64)

    _, obj_ids, _, seed_masks = tracker.add_new_points_or_box(
        inference_state = state,
        frame_idx = 0,
        obj_id = obj_id,
        points = points,
        labels = labels,
        clear_old_points = True,
        rel_coordinates = True,
    )
    if seed_masks[list(obj_ids).index(obj_id)].sum() == 0:
        raise RuntimeError(f"seed coordinate {coords} produced an empty mask")

    masks = {}
    for frame_idx, obj_ids, _, video_res_masks, _ in tracker.propagate_in_video(
        state,
        start_frame_idx = 0,
        max_frame_num_to_track = max_frames,
        reverse = False,
        propagate_preflight = True,
    ):
        ids = list(obj_ids)
        if obj_id not in ids:
            masks[frame_idx] = None
            continue
        m = (video_res_masks[ids.index(obj_id), 0] > 0.0).cpu().numpy()
        cleaned = _clean_mask(m)
        if cleaned is not None and m.sum() - cleaned.sum() > 0.05 * m.sum():
            print(f"frame {frame_idx}: dropped {m.sum() - cleaned.sum()} pixels")
        masks[frame_idx] = cleaned
    return masks

def _clean_mask(mask, min_area = 0):
    if mask is None:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n<= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas))
    if areas[k] < min_area:
        return None
    return labels == (k + 1)

def load_estimator(device="cuda"):
    from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
    from sam_3d_body_tools.build_detector import HumanDetector
    from sam_3d_body_tools.build_sam import HumanSegmentor
    from sam_3d_body_tools.build_fov_estimator import FOVEstimator
    
    detector_path = os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = os.environ.get("SAM3D_FOV_PATH", "")
    
    human_detector = HumanDetector(name="vitdet", device=device, path=detector_path)
    human_segmentor = HumanSegmentor(name="sam3", device=device, path=segmentor_path)
    fov_estimator = FOVEstimator(name="moge2", device=device, path=fov_path)
    
    model, model_cfg = load_sam_3d_body_hf("facebook/sam-3d-body-dinov3")
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model, 
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator
    )
    return estimator

def estimate_keypoints(vid_path, masks, cam_int=None, box_scale=1.2):
    estimator = load_estimator(device="cuda")
    
    cap = cv2.VideoCapture(str(vid_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    results = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        m = masks.get(idx)
        if m is None:
            results.append(None)
            idx += 1
            continue

        ys, xs = np.nonzero(m)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        bw = (xs.max() - xs.min()) * box_scale
        bh = (ys.max() - ys.min()) * box_scale
        bbox = np.array([[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]], dtype=np.float32)

        # disable autocast to avoid contamination with sam3
        with torch.autocast(device_type="cuda", enabled=False):
            outputs = estimator.process_one_image(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                bboxes=bbox,
                masks = m[None].astype(np.uint8),
                cam_int = cam_int,
                inference_type = "body",
            )
        results.append(outputs[0])
        idx += 1
    cap.release()
    return results

def _as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def write_keypoints_to_npz(npz_path, results, keypoint_id: str):
    n_frames = len(results)
    kp_shape = next(
        _as_numpy(r[keypoint_id]).squeeze().shape
        for r in results if r is not None
    )
    n_kpts, dim = kp_shape

    arr = np.full((n_frames, n_kpts, dim), np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        if r is not None:
            arr[i] = _as_numpy(r[keypoint_id]).squeeze()
    np.savez(npz_path, keypoints=arr)

def load_keypoints_from_npz(npz_paths: list):
    arrays = [np.load(p)["keypoints"][:, :, :2] for p in npz_paths]
    return np.stack(arrays, axis=0)

def write_results_to_npz(npz_path, results):
    n_frames = len(results)
    first_valid = next((r for r in results if r is not None), None)
    if first_valid is None:
        raise ValueError("results contains no non-None frames")

    # union of keys across all valid frames (in case some frames lack keys)
    keys = []
    for r in results:
        if r is not None:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)

    arrays = {}
    skipped = {}
    for k in keys:
        # reference shape/dtype from the first frame that has this key
        ref = next(
            (_as_numpy(r[k]).squeeze() for r in results
             if r is not None and k in r),
            None,
        )
        if ref is None or ref.dtype.kind not in "fiub":
            skipped[k] = "non-numeric"
            continue

        shape = ref.shape
        out = np.full((n_frames, *shape), np.nan, dtype=np.float32)

        ok = True
        for i, r in enumerate(results):
            if r is None or k not in r:
                continue
            v = _as_numpy(r[k]).squeeze()
            if v.shape != shape:
                skipped[k] = f"ragged shape: {shape} vs {v.shape} at frame {i}"
                ok = False
                break
            out[i] = v

        if ok:
            arrays[k] = out

    if skipped:
        print(f"skipped keys: {skipped}")
    print(f"saved keys: {[f'{k}{v.shape}' for k, v in arrays.items()]}")

    np.savez(npz_path, **arrays)
    return npz_path

def write_npz_to_trc(npz_path, data_rate=None, output_path=None, time_key=None):
    npz_path = Path(npz_path)
    output = Path(output_path) if output_path is not None else npz_path.with_suffix(".trc")
    with np.load(npz_path, allow_pickle=False) as archive:
        label = archive.files[0]
        xyz = np.asarray(archive[label], dtype=float)
        if time_key is None:
            time_key = next((key for key in ("time", "times", "timestamps", "t")
                             if key in archive), None)
        times = None if time_key is None else np.asarray(archive[time_key], dtype=float).ravel()

    if xyz.ndim != 3 or xyz.shape[1:] != (70, 3) or not len(xyz):
        raise ValueError("X3d must have shape (frames, 70, 3).")
    if not np.isfinite(xyz).all():
        raise ValueError("X3d must contain finite coordinates.")
    nframes = len(xyz)
    if times is not None:
        if times.shape != (nframes,) or not np.isfinite(times).all():
            raise ValueError("Provide one finite timestamp per frame.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps must be strictly increasing.")
        rate = (nframes - 1) / (times[-1] - times[0]) if nframes > 1 else data_rate
    else:
        rate = data_rate
    if rate is None or not np.isfinite(rate) or rate <= 0:
        raise ValueError("Provide a positive data_rate when timestamps cannot determine it.")
    if times is None:
        times = np.arange(nframes) / rate

    # correspondence between MHR indicies and opensim markers
    markers = ("RSJC LSJC REJC LEJC RHJC LHJC RKJC LKJC RAJC LAJC "
               "RLEL RMEL LLEL LMEL RACR LACR RCAL LCAL RTOE LTOE RMT5 LMT5 "
               "RWJC LWJC RTHUMB RPINKY LTHUMB LPINKY NECK NOSE REYE LEYE REAR LEAR").split()
    indices = [6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 64, 66, 63, 65,
               68, 67, 20, 17, 18, 15, 19, 16, 41, 62, 24, 40, 45, 61,
               69, 0, 2, 1, 4, 3]
    coordinates = xyz[:, indices][:, :, [0, 2, 1]] * np.array([1000, 1000, -1000])
    # boilerplate trc header and formatting
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{rate:.17g}\t{rate:.17g}\t{nframes}\t{len(markers)}\tmm\t{rate:.17g}\t1\t{nframes}",
        "Frame#\tTime\t" + "\t\t\t".join(markers) + "\t\t\t",
        "\t\t" + "\t".join(f"{axis}{i}" for i in range(1, len(markers) + 1)
                            for axis in "XYZ") + "\t",
        "",
    ]
    rows = np.column_stack((np.arange(1, nframes + 1), times,
                            coordinates.reshape(nframes, -1)))
    np.savetxt(output, rows, delimiter="\t", comments="", header="\n".join(header),
               fmt=["%d", "%.17g"] + ["%.5f"] * (3 * len(markers)))
    return output