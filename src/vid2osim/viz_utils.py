import cv2
import torch
import numpy as np

def write_mask_overlay(video_path, masks, out_path, alpha=0.5, color=(0, 0, 255)):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tint = np.array(color, dtype=np.float32)

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            m = masks.get(idx)
            if m is not None:
                sel = m.astype(bool)
                frame[sel] = (frame[sel] * (1 - alpha) + tint * alpha).astype(np.uint8)
            else:
                cv2.putText(frame, "no mask", (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path

def write_keypoint_overlay(
    video_path,
    results,
    out_path,
    radius=4,
    thickness=2,
    draw_skeleton=True,
    temporal_opacity=True,
    temporal_sigma=0.02,
    min_alpha=0.10,
    max_alpha=1.0,
):
    from sam_3d_body.metadata.mhr70 import pose_info

    def as_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def draw_with_alpha(frame, draw_fn, alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))

        overlay = frame.copy()
        draw_fn(overlay)

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1.0 - alpha,
            0.0,
            dst=frame,
        )

    if temporal_opacity:
        temporal_confidence, temporal_residuals = (
            _temporal_keypoint_confidence(
                results,
                sigma=temporal_sigma,
            )
        )
    else:
        temporal_confidence = [None] * len(results)
        temporal_residuals = [None] * len(results)

    name_to_idx = {
        info["name"]: idx
        for idx, info in pose_info["keypoint_info"].items()
    }

    skeleton_edges = []

    if draw_skeleton:
        for edge in pose_info["skeleton_info"].values():

            name_a, name_b = edge["link"]

            if (
                name_a in name_to_idx
                and name_b in name_to_idx
            ):
                skeleton_edges.append(
                    (
                        name_to_idx[name_a],
                        name_to_idx[name_b],
                    )
                )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open output video: {out_path}"
        )

    idx = 0

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                break

            result = (
                results[idx]
                if idx < len(results)
                else None
            )

            if result is None:

                cv2.putText(
                    frame,
                    "no pose",
                    (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

                writer.write(frame)
                idx += 1
                continue

            keypoints = (
                as_numpy(
                    result["pred_keypoints_2d"]
                )
                .squeeze()
            )

            xy = keypoints[:, :2]

            valid = (
                np.isfinite(xy).all(axis=1)
                & (xy[:, 0] >= 0)
                & (xy[:, 0] < W)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < H)
            )

            if (
                temporal_opacity
                and temporal_confidence[idx] is not None
            ):
                confidence = temporal_confidence[idx]
            else:
                confidence = np.ones(
                    len(xy),
                    dtype=np.float32,
                )

            alpha = (
                min_alpha
                + confidence * (max_alpha - min_alpha)
            )

            if draw_skeleton:

                for a, b in skeleton_edges:

                    if a >= len(xy) or b >= len(xy):
                        continue

                    if not (valid[a] and valid[b]):
                        continue

                    pa = tuple(
                        np.round(xy[a]).astype(int)
                    )

                    pb = tuple(
                        np.round(xy[b]).astype(int)
                    )

                    edge_alpha = min(
                        alpha[a],
                        alpha[b],
                    )

                    def draw_edge(
                        image,
                        pa=pa,
                        pb=pb,
                    ):
                        cv2.line(
                            image,
                            pa,
                            pb,
                            (0, 255, 255),
                            thickness,
                            cv2.LINE_AA,
                        )

                    draw_with_alpha(
                        frame,
                        draw_edge,
                        edge_alpha,
                    )

            for k, point in enumerate(xy):

                if not valid[k]:
                    continue

                p = tuple(
                    np.round(point).astype(int)
                )

                def draw_point(
                    image,
                    p=p,
                ):
                    cv2.circle(
                        image,
                        p,
                        radius,
                        (0, 0, 255),
                        -1,
                        cv2.LINE_AA,
                    )

                draw_with_alpha(
                    frame,
                    draw_point,
                    alpha[k],
                )

            writer.write(frame)
            idx += 1

    finally:
        cap.release()
        writer.release()

    return out_path

def _temporal_keypoint_confidence(
    results,
    sigma=0.02,
    fallback_confidence=0.5,
):
    """
    temporary confidence heuristic based on temporal consistency of 2D keypoints
    TODO: replace in future update
    """
    def as_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def get_keypoints(result):
        if result is None:
            return None

        kpts = as_numpy(result["pred_keypoints_2d"]).squeeze()

        if kpts.ndim != 2 or kpts.shape[1] < 2:
            raise ValueError(
                f"Unexpected pred_keypoints_2d shape: {kpts.shape}"
            )

        return kpts[:, :2].astype(np.float32)

    def get_bbox_diagonal(result):
        bbox = as_numpy(result["bbox"]).squeeze()

        if bbox.shape != (4,):
            raise ValueError(
                f"Unexpected bbox shape: {bbox.shape}"
            )

        x1, y1, x2, y2 = bbox
        return max(
            np.hypot(x2 - x1, y2 - y1),
            1.0,
        )

    keypoints = [get_keypoints(r) for r in results]

    confidences = []
    residuals = []

    for kpts in keypoints:
        if kpts is None:
            confidences.append(None)
            residuals.append(None)
        else:
            K = len(kpts)

            confidences.append(
                np.full(
                    K,
                    fallback_confidence,
                    dtype=np.float32,
                )
            )

            residuals.append(
                np.full(
                    K,
                    np.nan,
                    dtype=np.float32,
                )
            )

    for t in range(1, len(results) - 1):

        prev_kpts = keypoints[t - 1]
        curr_kpts = keypoints[t]
        next_kpts = keypoints[t + 1]

        if (
            prev_kpts is None
            or curr_kpts is None
            or next_kpts is None
        ):
            continue

        if not (
            prev_kpts.shape
            == curr_kpts.shape
            == next_kpts.shape
        ):
            continue

        finite = (
            np.isfinite(prev_kpts).all(axis=1)
            & np.isfinite(curr_kpts).all(axis=1)
            & np.isfinite(next_kpts).all(axis=1)
        )

        predicted = 0.5 * (prev_kpts + next_kpts)

        residual_px = np.linalg.norm(
            curr_kpts - predicted,
            axis=1,
        )

        bbox_diag = get_bbox_diagonal(results[t])

        residual_normalized = residual_px / bbox_diag

        confidence = np.exp(
            -0.5 * (residual_normalized / sigma) ** 2
        )

        residuals[t][finite] = residual_px[finite]
        confidences[t][finite] = confidence[finite]

    return confidences, residuals