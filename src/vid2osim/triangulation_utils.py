import numpy as np
from itertools import combinations

def _prepare_dlt(keypoints, calibration_data):
    cams = list(keypoints)
    if len(cams) < 2:
        raise ValueError("At least two cameras are required.")
    uv = np.stack([keypoints[c] for c in cams], axis=2).astype(float)
    projections = []
    for cam in cams:
        cal = calibration_data[cam]
        K = np.array([cal[t]["K"] for t in range(len(uv))], dtype=float)
        E = np.array([cal[t]["E"] for t in range(len(uv))], dtype=float)
        projections.append(K @ E)
    P = np.stack(projections, axis=1)  # (N, cameras, 3, 4)
    A = np.stack([
        uv[..., 0, None] * P[:, None, :, 2] - P[:, None, :, 0],
        uv[..., 1, None] * P[:, None, :, 2] - P[:, None, :, 1],
    ], axis=-2)  # (N, J, cameras, 2, 4)
    return cams, uv, P, A

def _solve_dlt(rows):
    A = rows.reshape(rows.shape[:-3] + (-1, 4))
    return np.linalg.svd(A, full_matrices=False)[2][..., -1, :]

def triangulate_poses_in_video(keypoints, calibration_data, method="all",
                               threshold_px=5.0):
    if method == "all":
        return triangulate_dlt(keypoints, calibration_data)
    if method == "consensus":
        return triangulate_consensus(
            keypoints, calibration_data, threshold_px=threshold_px
        )
    raise ValueError("method must be 'all' or 'consensus'.")

def triangulate_dlt(keypoints, calibration_data):
    _, _, _, A = _prepare_dlt(keypoints, calibration_data)
    Xh = _solve_dlt(A)
    return Xh[..., :3] / Xh[..., 3:4]


def triangulate_consensus(keypoints, calibration_data, threshold_px=5.0,
                          return_inliers=False):
    # Return (N, J, 3), NaN without a strict majority of camera inliers.
    # With return_inliers=True, also return {cam: (N, J) boolean mask}.
    cams, uv, P, A = _prepare_dlt(keypoints, calibration_data)
    if len(cams) < 4:
        raise ValueError("At least four cameras are required for consensus.")
    xyz = np.full(uv.shape[:2] + (3,), np.nan)
    kept = np.zeros(uv.shape[:-1], dtype=bool)
    pairs = list(combinations(range(len(cams)), 2))
    majority = len(cams) // 2 + 1

    for frame, joint in np.ndindex(uv.shape[:2]):
        rows, cameras, points = A[frame, joint], P[frame], uv[frame, joint]
        best, best_score = None, (0, -np.inf)
        for pair in pairs:
            Xh = _solve_dlt(rows[list(pair)])
            projected = cameras @ Xh
            with np.errstate(divide="ignore", invalid="ignore"):
                errors = np.linalg.norm(
                    projected[:, :2] / projected[:, 2:3] - points, axis=1)
            inliers = (errors <= threshold_px) & (projected[:, 2] * Xh[3] > 0)
            score = (int(inliers.sum()), -errors[inliers].sum())
            if score[0] >= majority and score > best_score:
                best, best_score = inliers, score

        if best is not None:
            Xh = _solve_dlt(rows[best])
            if abs(Xh[3]) > 1e-12:
                xyz[frame, joint] = Xh[:3] / Xh[3]
                kept[frame, joint] = best

    if return_inliers:
        return xyz, {c: kept[..., i] for i, c in enumerate(cams)}
    return xyz

def interpolate_nan_keypoints(xyz):
    result = np.array(xyz, dtype=float, copy=True)
    frames = np.arange(len(result))

    for joint, axis in np.ndindex(result.shape[1:]):
        values = result[:, joint, axis]
        valid = ~np.isnan(values)
        if valid.any():
            values[~valid] = np.interp(
                frames[~valid], frames[valid], values[valid]
            )

    return result