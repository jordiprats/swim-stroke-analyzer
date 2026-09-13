"""
Post-processing pose data to fill in missed detections and filter implausible movements.

Since analysis is not real-time, we can look ahead and behind to:
  1. Interpolate frames where pose detection failed (pose=None).
  2. Apply kinematic constraints — discard or down-weight landmark positions
     that imply impossible joint movement between consecutive frames.
  3. Detect implausible landmark proximities (e.g., wrist too close to
     shoulder, elbows crossing) and flag them as low-visibility.

The output pose_data list preserves the same structure, but with fewer
pose=None entries and corrected landmark visibility values.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

# ── Kinematic limits (heuristic, tuned for swimming at ~30 fps after skip=2) ──

# Maximum pixel displacement per frame for a joint centre (based on frame
# dimensions ~1280×720).  At 30 fps a swimmer's hand can move ~200 px/frame
# during a fast recovery; anything above 300 px/frame is almost certainly a
# tracking glitch.
MAX_PIXEL_DELTA_PER_FRAME = 300

# Maximum change in joint angle per frame (degrees).  Elbow angle shouldn't
# change more than ~40° per frame during normal swimming; 60+ indicates a
# tracking jump.
MAX_ANGLE_DELTA_PER_FRAME = 50

# Minimum distance between two distinct landmarks that should never overlap
# (in pixels).  If e.g. wrist and shoulder are within 20 px, the wrist
# position is likely a hallucination.
MIN_LANDMARK_SEPARATION = 30

# Landmark pairs that should never be very close (they're far apart on the body)
CLOSE_CHECK_PAIRS = [
    ('left_wrist', 'left_shoulder'),
    ('right_wrist', 'right_shoulder'),
    ('left_wrist', 'left_hip'),
    ('right_wrist', 'right_hip'),
    ('left_ankle', 'left_shoulder'),
    ('right_ankle', 'right_shoulder'),
    ('left_elbow', 'right_elbow'),
    ('left_wrist', 'right_wrist'),  # In freestyle they shouldn't overlap
]

# Landmarks whose movement should be relatively smooth (hips, shoulders)
SMOOTH_LANDMARKS = [
    'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip',
]

MIN_VISIBILITY = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────

def _euclidean(a: Dict, b: Dict) -> float:
    """Euclidean distance between two landmarks (x, y)."""
    dx = a['x'] - b['x']
    dy = a['y'] - b['y']
    return float(np.sqrt(dx * dx + dy * dy))


def _lerp(a: Dict, b: Dict, t: float) -> Dict:
    """Linearly interpolate between two landmarks at blend factor t."""
    z = a.get('z', 0) + t * (b.get('z', 0) - a.get('z', 0))
    vis = a.get('visibility', 0) + t * (b.get('visibility', 0) - a.get('visibility', 0))
    return {
        'x': a['x'] + t * (b['x'] - a['x']),
        'y': a['y'] + t * (b['y'] - a['y']),
        'z': z,
        'visibility': float(vis),
    }


def _copy_landmark(src: Dict) -> Dict:
    """Deep-copy a landmark dict."""
    return {
        'x': src['x'],
        'y': src['y'],
        'z': src.get('z', 0),
        'visibility': src.get('visibility', 0),
    }


def _interpolate_pose(
    before: Dict,
    after: Dict,
    frame_before: int,
    frame_after: int,
    target_frame: int,
    frame_shape: Tuple[int, int],
) -> Dict:
    """Interpolate a pose dict for target_frame given before & after poses."""
    if frame_after == frame_before:
        t = 0
    else:
        t = (target_frame - frame_before) / (frame_after - frame_before)

    landmarks = {}
    for name in before['landmarks']:
        if name in before['landmarks'] and name in after['landmarks']:
            landmarks[name] = _lerp(
                before['landmarks'][name],
                after['landmarks'][name],
                t,
            )
        elif name in before['landmarks']:
            landmarks[name] = _copy_landmark(before['landmarks'][name])
        elif name in after['landmarks']:
            landmarks[name] = _copy_landmark(after['landmarks'][name])

    # Use frame_shape from before (or after) as fallback
    return {
        'landmarks': landmarks,
        'raw_landmarks': None,  # No raw MediaPipe data — it's synthetic
        'frame_shape': frame_shape,
    }


# ── Main smoothing pipeline ──────────────────────────────────────────────

def smooth_pose_data(pose_data: List[Dict]) -> List[Dict]:
    """
    Apply post-processing to improve pose data quality.

    Steps:
      1. Forward-fill + backward-fill missing detections, then interpolate.
      2. Detect implausible landmark proximities and reduce visibility.
      3. Apply kinematic constraints — zero-out landmarks that move
         impossibly fast between consecutive frames.
      4. Re-interpolate any landmarks that were zeroed out in step 3.

    Modifies pose_data in-place and returns it.
    """
    if not pose_data:
        return pose_data

    print("\nSmoothing pose data...")

    # ── Step 1: Interpolate missing frames ──
    _interpolate_missing_frames(pose_data)

    # ── Step 2: Fix implausible proximities ──
    _fix_implausible_proximities(pose_data)

    # ── Step 3: Kinematic filtering ──
    _apply_kinematic_filter(pose_data)

    # ── Step 4: Re-interpolate landmarks that were zeroed ──
    _reinterpolate_zeroed(pose_data)

    # Count how many frames now have valid pose
    valid_count = sum(1 for fd in pose_data if fd['pose'] is not None)
    print(f"Smoothing complete: {valid_count}/{len(pose_data)} frames have valid pose")
    return pose_data


# ── Step 1: Interpolate missing frames ──

def _interpolate_missing_frames(pose_data: List[Dict]):
    """Fill pose=None entries by interpolating from surrounding valid frames."""
    n = len(pose_data)
    if n < 2:
        return

    # First, forward-fill: find the first valid frame and fill backward
    first_valid = None
    for i in range(n):
        if pose_data[i]['pose'] is not None:
            first_valid = i
            break

    if first_valid is not None:
        # Fill frames before first valid with the first valid pose
        for i in range(first_valid):
            if pose_data[i]['pose'] is None:
                pose_data[i]['pose'] = _copy_pose(pose_data[first_valid]['pose'])

    # Forward-fill: propagate last valid pose forward
    last_valid = None
    for i in range(n):
        if pose_data[i]['pose'] is not None:
            last_valid = i

    if last_valid is not None:
        for i in range(last_valid + 1, n):
            if pose_data[i]['pose'] is None:
                pose_data[i]['pose'] = _copy_pose(pose_data[last_valid]['pose'])

    # Now interpolate between valid frames
    _interpolate_gaps(pose_data)


def _copy_pose(pose: Dict) -> Dict:
    """Deep-copy a pose dict (landmarks only, no raw_landmarks)."""
    if pose is None:
        return None
    landmarks = {}
    for name, lm in pose.get('landmarks', {}).items():
        landmarks[name] = _copy_landmark(lm)
    return {
        'landmarks': landmarks,
        'raw_landmarks': None,
        'frame_shape': pose.get('frame_shape', (0, 0)),
    }


def _interpolate_gaps(pose_data: List[Dict]):
    """
    For each gap between two valid frames, replace the copied pose
    with a proper interpolation.
    """
    n = len(pose_data)
    i = 0
    while i < n:
        if pose_data[i]['pose'] is not None:
            # Find next valid frame
            j = i + 1
            while j < n and pose_data[j]['pose'] is None:
                j += 1
            if j < n and pose_data[j]['pose'] is not None:
                # Gap from i+1 to j-1
                before = pose_data[i]['pose']
                after = pose_data[j]['pose']
                frame_shape = before.get('frame_shape', after.get('frame_shape', (0, 0)))
                for k in range(i + 1, j):
                    interpolated = _interpolate_pose(
                        before, after,
                        pose_data[i]['frame_number'],
                        pose_data[j]['frame_number'],
                        pose_data[k]['frame_number'],
                        frame_shape,
                    )
                    pose_data[k]['pose'] = interpolated
            i = j
        else:
            i += 1


# ── Step 2: Fix implausible proximities ──

def _fix_implausible_proximities(pose_data: List[Dict]):
    """Check landmark pairs that should never be close, reduce visibility if so."""
    for frame_data in pose_data:
        pose = frame_data.get('pose')
        if pose is None:
            continue
        landmarks = pose.get('landmarks', {})
        if not landmarks:
            continue

        for (a_name, b_name) in CLOSE_CHECK_PAIRS:
            a = landmarks.get(a_name)
            b = landmarks.get(b_name)
            if a is None or b is None:
                continue
            if a.get('visibility', 0) < MIN_VISIBILITY or b.get('visibility', 0) < MIN_VISIBILITY:
                continue

            dist = _euclidean(a, b)
            if dist < MIN_LANDMARK_SEPARATION:
                # One of them is likely a hallucination — drop visibility on the
                # one with lower confidence, or the distal one (wrist > shoulder)
                # The wrist/ankle are more likely to be wrong.
                # Reduce visibility to 0 so it gets re-interpolated later.
                a['visibility'] = 0.0


# ── Step 3: Kinematic filter ──

def _apply_kinematic_filter(pose_data: List[Dict]):
    """
    For each consecutive pair of frames, check landmark movement deltas.
    If any landmark moves more than MAX_PIXEL_DELTA_PER_FRAME, set its
    visibility to 0 (it will be re-interpolated later).
    """
    n = len(pose_data)
    if n < 2:
        return

    for i in range(1, n):
        prev = pose_data[i - 1]
        curr = pose_data[i]

        if prev['pose'] is None or curr['pose'] is None:
            continue

        prev_landmarks = prev['pose'].get('landmarks', {})
        curr_landmarks = curr['pose'].get('landmarks', {})

        if not prev_landmarks or not curr_landmarks:
            continue

        for name, curr_lm in curr_landmarks.items():
            if curr_lm.get('visibility', 0) < MIN_VISIBILITY:
                continue
            prev_lm = prev_landmarks.get(name)
            if prev_lm is None or prev_lm.get('visibility', 0) < MIN_VISIBILITY:
                continue

            # Pixel distance moved
            dx = curr_lm['x'] - prev_lm['x']
            dy = curr_lm['y'] - prev_lm['y']
            dist = float(np.sqrt(dx * dx + dy * dy))

            if dist > MAX_PIXEL_DELTA_PER_FRAME:
                curr_lm['visibility'] = 0.0

        # Also check angle changes for elbow and knee
        for side in ('left', 'right'):
            for joint in ('elbow',):
                p1 = f'{side}_shoulder'
                p2 = f'{side}_{joint}'
                p3 = f'{side}_wrist'
                if p1 in curr_landmarks and p2 in curr_landmarks and p3 in curr_landmarks:
                    curr_angle = _calculate_angle(curr_landmarks[p1], curr_landmarks[p2], curr_landmarks[p3])
                    prev_angle = _calculate_angle(
                        prev_landmarks.get(p1, {}),
                        prev_landmarks.get(p2, {}),
                        prev_landmarks.get(p3, {}),
                    )
                    if curr_angle is not None and prev_angle is not None:
                        delta = abs(curr_angle - prev_angle)
                        if delta > MAX_ANGLE_DELTA_PER_FRAME:
                            curr_landmarks[p2]['visibility'] = 0.0


# ── Step 4: Re-interpolate zeroed landmarks ──

def _reinterpolate_zeroed(pose_data: List[Dict]):
    """
    After kinematic filtering, some landmarks have visibility=0.
    Re-interpolate them from surrounding frames with valid visibility.
    """
    n = len(pose_data)
    if n < 3:
        return

    # For each frame, check each landmark
    for i in range(n):
        frame_data = pose_data[i]
        if frame_data['pose'] is None:
            continue
        landmarks = frame_data['pose'].get('landmarks', {})
        if not landmarks:
            continue

        for name, lm in landmarks.items():
            if lm['visibility'] > 0:
                continue  # Already valid

            # Find nearest valid before and after
            before_idx, after_idx = None, None
            for j in range(i - 1, -1, -1):
                if pose_data[j]['pose'] is not None:
                    prev_lm = pose_data[j]['pose']['landmarks'].get(name)
                    if prev_lm and prev_lm['visibility'] > MIN_VISIBILITY:
                        before_idx = j
                        break
            for j in range(i + 1, n):
                if pose_data[j]['pose'] is not None:
                    next_lm = pose_data[j]['pose']['landmarks'].get(name)
                    if next_lm and next_lm['visibility'] > MIN_VISIBILITY:
                        after_idx = j
                        break

            if before_idx is not None and after_idx is not None:
                before = pose_data[before_idx]['pose']['landmarks'][name]
                after = pose_data[after_idx]['pose']['landmarks'][name]
                t = (pose_data[i]['frame_number'] - pose_data[before_idx]['frame_number']) / \
                    (pose_data[after_idx]['frame_number'] - pose_data[before_idx]['frame_number'] + 1e-6)
                interpolated = _lerp(before, after, t)
                lm['x'] = interpolated['x']
                lm['y'] = interpolated['y']
                lm['z'] = interpolated['z']
                lm['visibility'] = interpolated['visibility']
            elif before_idx is not None:
                # Forward-fill from before
                before = pose_data[before_idx]['pose']['landmarks'][name]
                lm['x'] = before['x']
                lm['y'] = before['y']
                lm['z'] = before.get('z', 0)
                lm['visibility'] = before.get('visibility', 0) * 0.5
            elif after_idx is not None:
                # Backward-fill from after
                after = pose_data[after_idx]['pose']['landmarks'][name]
                lm['x'] = after['x']
                lm['y'] = after['y']
                lm['z'] = after.get('z', 0)
                lm['visibility'] = after.get('visibility', 0) * 0.5


# ── Angle calculation (same as pose_detector) ──

def _calculate_angle(p1: Dict, p2: Dict, p3: Dict) -> Optional[float]:
    """Angle at p2 in degrees."""
    if not all(k in p1 and k in p2 and k in p3 for k in ('x', 'y')):
        return None
    a = np.array([p1['x'], p1['y']])
    b = np.array([p2['x'], p2['y']])
    c = np.array([p3['x'], p3['y']])
    v1 = a - b
    v2 = c - b
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:
        return None
    cos = np.dot(v1, v2) / norm
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
