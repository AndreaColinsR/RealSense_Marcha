import cv2
import numpy as np


def detect_grid_corners(image_path, roi_top_frac=0.60, tophat_kernel=15,
                         min_component_area=150, min_line_length=35,
                         max_line_gap=20, hough_threshold=25,
                         horizontal_angle_thresh=15, angle_merge_deg=8,
                         dist_merge_px=15, min_merged_line_length=80,
                         roi_polygon=None):
    """
    Detect corners formed by 2 (roughly) horizontal boundary lines and
    several near-vertical/diagonal divider lines painted on a floor
    (e.g. a marked walking lane / gait mat), as seen at an angle.

    Strategy: rather than intersecting every detected line with every
    other line (which is noisy when there's clutter), this only looks
    at the near-vertical divider lines and uses each one's own two
    endpoints as the two corners it forms with the (unmodeled) top and
    bottom boundary lines. This sidesteps having to explicitly and
    reliably detect the horizontal lines themselves, which is fragile
    when there are other horizontal-ish distractors in the scene
    (e.g. a mat seam or the edge of a rug).

    Returns
    -------
    top_row, bottom_row : lists of (x, y) corners, sorted left-to-right
    debug : dict with intermediate 'mask' and 'vis' images, plus
            'merged_lines' (the detected divider lines) for tuning
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    y0 = int(h * roi_top_frac)
    roi = gray[y0:h, :]

    # --- isolate thin bright lines, robust to uneven illumination ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_kernel, tophat_kernel))
    tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if roi_polygon is not None:
        poly_mask = np.zeros_like(roi, dtype=np.uint8)
        cv2.fillPoly(poly_mask, [np.array(roi_polygon, dtype=np.int32) - [0, y0]], 255)
        mask[poly_mask > 0] = 0

    # discard small blobs (carpet texture, reflections, noise)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n_comp):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            clean[labels == i] = 255
    mask = clean

    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_d = cv2.dilate(mask, k2, iterations=1)
    mask_c = cv2.morphologyEx(mask_d, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    edges = cv2.Canny(mask_c, 50, 150)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold,
                                 minLineLength=min_line_length, maxLineGap=max_line_gap)

    debug = {"mask": mask_c, "vis": img.copy(), "merged_lines": []}
    if raw_lines is None:
        return [], [], debug

    # keep only steep (non-horizontal) segments -> these are the dividers
    steep = []
    for l in raw_lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) > horizontal_angle_thresh:
            steep.append((x1, y1, x2, y2))

    merged = _merge_lines(steep, angle_merge_deg, dist_merge_px)
    merged = [l for l in merged
              if np.hypot(l[2] - l[0], l[3] - l[1]) >= min_merged_line_length]
    debug["merged_lines"] = [(x1, y1 + y0, x2, y2 + y0) for (x1, y1, x2, y2) in merged]

    # each divider's own two endpoints ARE its two corners
    endpoints = []
    for (x1, y1, x2, y2) in merged:
        # normalize so p_top has the smaller y (closer to top of image)
        if y1 <= y2:
            p_top, p_bot = (x1, y1 + y0), (x2, y2 + y0)
        else:
            p_top, p_bot = (x2, y2 + y0), (x1, y1 + y0)
        endpoints.append((p_top, p_bot))

    if not endpoints:
        return [], [], debug

    # reject dividers whose top/bottom endpoint doesn't line up with the
    # rest (e.g. clutter/equipment edges picked up above/below the real
    # grid) using a median-based outlier check on each row's y-values
    top_ys = np.array([p[0][1] for p in endpoints])
    bot_ys = np.array([p[1][1] for p in endpoints])
    top_ok = _mad_inlier_mask(top_ys)
    bot_ok = _mad_inlier_mask(bot_ys)
    endpoints = [p for p, ok_t, ok_b in zip(endpoints, top_ok, bot_ok) if ok_t and ok_b]

    top_row = _dedupe_points([p[0] for p in endpoints], min_dist=20)
    bottom_row = _dedupe_points([p[1] for p in endpoints], min_dist=20)
    top_row.sort(key=lambda p: p[0])
    bottom_row.sort(key=lambda p: p[0])

    # --- debug visualization ---
    vis = img.copy()
    counter =0
    for (x1, y1, x2, y2) in debug["merged_lines"]:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for (x, y) in top_row + bottom_row:
        cv2.circle(vis, (int(x), int(y)), 8, (counter*20, 255-counter*20, 0), -1)
    debug["vis"] = vis

    return top_row, bottom_row, debug


def _merge_lines(segments, angle_merge_deg, dist_merge_px):
    """Group Hough segments with similar angle+offset and fit one line per group."""
    def line_params(seg):
        x1, y1, x2, y2 = seg
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        theta = np.radians(angle)
        rho = x1 * np.sin(theta) - y1 * np.cos(theta)
        return angle, rho

    groups = []
    for seg in segments:
        angle, rho = line_params(seg)
        placed = False
        for g in groups:
            d_angle = min(abs(angle - g["angle"]), 180 - abs(angle - g["angle"]))
            if d_angle <= angle_merge_deg and abs(rho - g["rho"]) <= dist_merge_px:
                g["segs"].append(seg)
                placed = True
                break
        if not placed:
            groups.append({"angle": angle, "rho": rho, "segs": [seg]})

    merged = []
    for g in groups:
        pts = np.array([[s[0], s[1]] for s in g["segs"]] +
                        [[s[2], s[3]] for s in g["segs"]], dtype=np.float32)
        vx, vy, x0, y0_ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        t = np.dot(pts - [x0, y0_], [[vx], [vy]])
        t_min, t_max = t.min(), t.max()
        p1 = (x0 + vx * t_min, y0_ + vy * t_min)
        p2 = (x0 + vx * t_max, y0_ + vy * t_max)
        merged.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    return merged


def _dedupe_points(points, min_dist=20):
    kept = []
    for p in points:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > min_dist for q in kept):
            kept.append(p)
    return kept


def _mad_inlier_mask(values, thresh=3.5, min_n=3):
    """Median-absolute-deviation based outlier mask. With too few points
    to judge, everything is kept (nothing to compare against)."""
    values = np.asarray(values, dtype=float)
    if len(values) < min_n:
        return np.ones_like(values, dtype=bool)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.abs(values - med) < 5  # tight absolute fallback
    modified_z = 0.6745 * (values - med) / mad
    return np.abs(modified_z) < thresh


if __name__ == "__main__":
    top_row, bottom_row, debug = detect_grid_corners(
        '/mnt/user-data/uploads/1785534058928_image.png',
    )
    print(f"Top row ({len(top_row)} corners):")
    for c in top_row:
        print(" ", c)
    print(f"Bottom row ({len(bottom_row)} corners):")
    for c in bottom_row:
        print(" ", c)
    cv2.imwrite('/home/claude/grid_corners_vis.png', debug['vis'])
