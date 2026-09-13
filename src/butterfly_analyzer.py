"""Analyzes butterfly stroke metrics and detects technique issues."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from src.models.butterfly_rules import (
    ARM_ENTRY_WIDTH_MIN, ARM_ENTRY_WIDTH_MAX,
    ELBOW_ANGLE_PULL_MIN, ELBOW_ANGLE_PULL_MAX, ELBOW_ANGLE_DROPPED,
    UNDULATION_MIN, UNDULATION_MAX, UNDULATION_TOO_FLAT, UNDULATION_EXCESSIVE,
    HEAD_LIFT_THRESHOLD, HEAD_LIFT_EXCESSIVE,
    STROKE_RATE_OPTIMAL_MIN, STROKE_RATE_OPTIMAL_MAX,
    STROKE_RATE_TOO_SLOW, STROKE_RATE_TOO_FAST,
    KNEE_ANGLE_OPTIMAL_MIN, KNEE_ANGLE_OPTIMAL_MAX,
    KNEE_ANGLE_TOO_STRAIGHT, KNEE_ANGLE_EXCESSIVE_BEND,
    SYNC_MAX_DELTA,
    MIN_VISIBILITY, ButterflyIssue, ISSUE_TYPES,
    SEVERITY_CRITICAL, SEVERITY_MODERATE, SEVERITY_MINOR,
)


class ButterflyAnalyzer:
    """Analyzes butterfly swimming technique from pose data."""

    def __init__(self):
        """Initialize butterfly analyzer."""
        self.metrics = {}
        self.issues = []

    def analyze_video(self, pose_data: List[Dict]) -> Dict:
        """
        Analyze complete video pose data for butterfly.

        Args:
            pose_data: List of frame data with pose information

        Returns:
            Dictionary containing metrics and detected issues
        """
        print("\nAnalyzing butterfly stroke mechanics...")

        # Filter frames with valid pose data
        valid_frames = [frame for frame in pose_data if frame['pose'] is not None]

        if len(valid_frames) == 0:
            return {
                'error': 'No valid pose data detected in video',
                'metrics': {},
                'issues': []
            }

        print(f"Valid frames: {len(valid_frames)}/{len(pose_data)}")

        # Analyze different aspects
        elbow_metrics = self._analyze_elbow_angles(valid_frames)
        entry_metrics = self._analyze_arm_entry_width(valid_frames)
        undulation_metrics = self._analyze_body_undulation(valid_frames)
        head_metrics = self._analyze_head_position(valid_frames)
        stroke_rate_metrics = self._analyze_stroke_rate(valid_frames)
        kick_metrics = self._analyze_kick(valid_frames)
        sync_metrics = self._analyze_arm_synchronization(valid_frames)

        # Combine all metrics
        self.metrics = {
            'elbow': elbow_metrics,
            'entry': entry_metrics,
            'undulation': undulation_metrics,
            'head': head_metrics,
            'stroke_rate': stroke_rate_metrics,
            'kick': kick_metrics,
            'synchronization': sync_metrics,
            'valid_frame_ratio': len(valid_frames) / len(pose_data)
        }

        # Detect issues based on metrics
        self.issues = self._detect_issues()

        return {
            'metrics': self.metrics,
            'issues': self.issues
        }

    def _analyze_elbow_angles(self, frames: List[Dict]) -> Dict:
        """Analyze elbow angles during the pull phase."""
        left_elbow_angles = []
        right_elbow_angles = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            # Check visibility
            if (landmarks['left_shoulder']['visibility'] < MIN_VISIBILITY or
                landmarks['left_elbow']['visibility'] < MIN_VISIBILITY or
                landmarks['left_wrist']['visibility'] < MIN_VISIBILITY):
                continue

            # Left elbow angle (shoulder → elbow → wrist)
            angle = self._calculate_angle(
                landmarks['left_shoulder'],
                landmarks['left_elbow'],
                landmarks['left_wrist']
            )
            left_elbow_angles.append(angle)

            # Right elbow angle
            if (landmarks['right_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_elbow']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_wrist']['visibility'] >= MIN_VISIBILITY):

                angle = self._calculate_angle(
                    landmarks['right_shoulder'],
                    landmarks['right_elbow'],
                    landmarks['right_wrist']
                )
                right_elbow_angles.append(angle)

        all_angles = left_elbow_angles + right_elbow_angles

        return {
            'avg_angle': np.mean(all_angles) if all_angles else None,
            'min_angle': np.min(all_angles) if all_angles else None,
            'max_angle': np.max(all_angles) if all_angles else None,
            'left_avg': np.mean(left_elbow_angles) if left_elbow_angles else None,
            'right_avg': np.mean(right_elbow_angles) if right_elbow_angles else None,
        }

    def _analyze_arm_entry_width(self, frames: List[Dict]) -> Dict:
        """
        Analyze arm entry width relative to shoulder width.

        In butterfly both arms enter simultaneously.  We measure the distance
        between the two wrists as a fraction of shoulder width — too narrow
        reduces pull length, too wide loses leverage.
        """
        entry_widths = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            # Need both wrists and both shoulders visible
            if (landmarks['left_wrist']['visibility'] < MIN_VISIBILITY or
                landmarks['right_wrist']['visibility'] < MIN_VISIBILITY or
                landmarks['left_shoulder']['visibility'] < MIN_VISIBILITY or
                landmarks['right_shoulder']['visibility'] < MIN_VISIBILITY):
                continue

            # Wrist span (entry width)
            wrist_span = abs(landmarks['left_wrist']['x'] - landmarks['right_wrist']['x'])
            shoulder_span = abs(landmarks['left_shoulder']['x'] - landmarks['right_shoulder']['x'])

            if shoulder_span < 1:
                continue

            # Ratio: 1.0 means wrist span = shoulder span
            ratio = wrist_span / shoulder_span
            entry_widths.append(ratio)

        if not entry_widths:
            return {
                'avg_entry_width': None,
                'min_entry_width': None,
                'max_entry_width': None,
            }

        return {
            'avg_entry_width': np.mean(entry_widths),
            'min_entry_width': np.min(entry_widths),
            'max_entry_width': np.max(entry_widths),
        }

    def _analyze_body_undulation(self, frames: List[Dict]) -> Dict:
        """
        Analyze dolphin-like body undulation.

        Butterfly uses vertical undulation — we measure the vertical
        displacement of the hips over the stroke cycle as a fraction of
        frame height.
        """
        hip_y_positions = []  # Average of left/right hip y

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            left_hip_y = landmarks['left_hip']['y']
            right_hip_y = landmarks['right_hip']['y']

            # Average both hips for a stable reference
            avg_y = (left_hip_y + right_hip_y) / 2
            hip_y_positions.append(avg_y)

        if len(hip_y_positions) < 5:
            return {
                'undulation_amplitude': None,
                'undulation_cycles': None,
                'avg_undulation': None,
            }

        frame_height = frames[0]['pose']['frame_shape'][0]

        # Normalise hip-y by frame height
        hip_y_norm = [y / frame_height for y in hip_y_positions]

        # Detect undulation cycles by counting zero-crossings of the
        # detrended signal (subtract mean).
        mean_y = np.mean(hip_y_norm)
        detrended = [y - mean_y for y in hip_y_norm]

        # Count sign changes (each up-down pair = one undulation cycle)
        crossings = 0
        for i in range(1, len(detrended)):
            if (detrended[i - 1] < 0 and detrended[i] >= 0) or \
               (detrended[i - 1] > 0 and detrended[i] <= 0):
                crossings += 1

        # Each undulation cycle = 2 crossings (up then down)
        undulation_cycles = crossings // 2
        amplitude = np.std(detrended) * 2  # Rough peak-to-peak estimate

        return {
            'undulation_amplitude': amplitude,
            'undulation_cycles': undulation_cycles,
            'avg_undulation': amplitude,
        }

    def _analyze_head_position(self, frames: List[Dict]) -> Dict:
        """Analyze head vertical movement for breathing pattern."""
        nose_y_positions = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']
            if landmarks['nose']['visibility'] >= MIN_VISIBILITY:
                nose_y_positions.append(landmarks['nose']['y'])

        if not nose_y_positions:
            return {
                'breathing_lift': None,
                'stability': None,
            }

        frame_height = frames[0]['pose']['frame_shape'][0]
        y_range = np.max(nose_y_positions) - np.min(nose_y_positions)
        normalized_range = y_range / frame_height

        # In butterfly, some head lift is normal for breathing.
        # We track the max upward displacement relative to the average.
        avg_y = np.mean(nose_y_positions)
        # Lift = how far above average the nose goes (up = smaller y)
        lift = avg_y - np.min(nose_y_positions)  # Positive = lifting up
        normalized_lift = lift / frame_height

        return {
            'vertical_movement': normalized_range,
            'breathing_lift': normalized_lift,
            'avg_y': np.mean(nose_y_positions),
            'stability': 1.0 - normalized_range,
        }

    def _analyze_stroke_rate(self, frames: List[Dict]) -> Dict:
        """
        Analyze stroke rate for butterfly.

        Butterfly has both arms moving together.  We average the left and
        right wrist x-coordinates and detect peaks in the combined signal.
        Each peak = one stroke cycle (both arms).
        """
        if len(frames) < 10:
            return {'spm': None}

        # Collect (timestamp, avg_wrist_x) samples
        samples = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            lv = landmarks['left_wrist']['visibility']
            rv = landmarks['right_wrist']['visibility']
            if lv >= MIN_VISIBILITY and rv >= MIN_VISIBILITY:
                avg_x = (landmarks['left_wrist']['x'] + landmarks['right_wrist']['x']) / 2
                samples.append((frame['timestamp'], avg_x))

        if len(samples) < 10:
            return {'spm': None}

        timestamps = [s[0] for s in samples]
        avg_wrist_x = [s[1] for s in samples]
        duration = timestamps[-1] - timestamps[0]

        if duration <= 0:
            return {'spm': None}

        # --- Smooth the trajectory ---
        window = max(3, len(avg_wrist_x) // 20)
        smoothed = self._moving_average(avg_wrist_x, window)

        # --- Estimate minimum gap between peaks ---
        # Max realistic butterfly SPM ~70 → ~0.86 s/stroke.
        fps_equiv = len(smoothed) / duration
        min_peak_gap = max(3, int(fps_equiv * 0.6))

        # --- Find peaks ---
        candidate_peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                candidate_peaks.append(i)

        # Greedy filter: keep peaks far enough apart
        filtered_peaks = []
        for peak in candidate_peaks:
            if not filtered_peaks or (peak - filtered_peaks[-1]) >= min_peak_gap:
                filtered_peaks.append(peak)

        total_strokes = len(filtered_peaks)  # Each peak = one butterfly stroke cycle
        spm = (total_strokes / duration) * 60

        # Sanity check: butterfly 15–80 SPM
        if spm < 15 or spm > 80:
            spm = None

        return {
            'spm': spm,
            'total_strokes': total_strokes if spm is not None else None,
            'duration': duration
        }

    @staticmethod
    def _moving_average(values: List[float], window: int) -> List[float]:
        """Apply a centred moving average to smooth a 1-D signal."""
        if window < 2 or len(values) < window:
            return list(values)
        half = window // 2
        smoothed = []
        for i in range(len(values)):
            lo = max(0, i - half)
            hi = min(len(values), i + half + 1)
            smoothed.append(sum(values[lo:hi]) / (hi - lo))
        return smoothed

    def _analyze_kick(self, frames: List[Dict]) -> Dict:
        """
        Analyze dolphin kick — both legs together.

        In butterfly the legs kick simultaneously.  We measure knee angles
        on both legs and look for synchronised bending.
        """
        knee_angles = []
        left_knee_angles = []
        right_knee_angles = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            # Left leg
            if (landmarks['left_hip']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_knee']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_ankle']['visibility'] >= MIN_VISIBILITY):

                angle = self._calculate_angle(
                    landmarks['left_hip'],
                    landmarks['left_knee'],
                    landmarks['left_ankle']
                )
                left_knee_angles.append(angle)
                knee_angles.append(angle)

            # Right leg
            if (landmarks['right_hip']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_knee']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_ankle']['visibility'] >= MIN_VISIBILITY):

                angle = self._calculate_angle(
                    landmarks['right_hip'],
                    landmarks['right_knee'],
                    landmarks['right_ankle']
                )
                right_knee_angles.append(angle)
                knee_angles.append(angle)

        return {
            'avg_knee_angle': np.mean(knee_angles) if knee_angles else None,
            'min_knee_angle': np.min(knee_angles) if knee_angles else None,
            'left_avg': np.mean(left_knee_angles) if left_knee_angles else None,
            'right_avg': np.mean(right_knee_angles) if right_knee_angles else None,
        }

    def _analyze_arm_synchronization(self, frames: List[Dict]) -> Dict:
        """
        Measure synchronisation of left and right arm movements using joint angles.

        In butterfly both arms should move together.  Instead of comparing
        wrist-x peak timestamps (which is sensitive to camera angle), we
        compare the **elbow angle** of each arm frame-by-frame.

        Rationale: joint angles are measured relative to the body and are
        much more robust to camera perspective than absolute x-coordinates.
        In synchronized butterfly, at any given frame the left and right
        elbow angles should be nearly equal.  If they consistently differ,
        the arms are out of sync.

        We also compute the **angle between the two arm vectors**
        (shoulder → wrist) — this should be near 0 when both arms are
        in the same phase and grows when they are offset.
        """
        if len(frames) < 5:
            return {'sync_delta': None, 'synchronized': None}

        elbow_diffs = []
        arm_angle_diffs = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            # --- Elbow angle comparison ---
            left_ok = all(
                landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
                for s in ('left_shoulder', 'left_elbow', 'left_wrist')
            )
            right_ok = all(
                landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
                for s in ('right_shoulder', 'right_elbow', 'right_wrist')
            )

            if left_ok and right_ok:
                left_angle = self._calculate_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_elbow'],
                    landmarks['left_wrist']
                )
                right_angle = self._calculate_angle(
                    landmarks['right_shoulder'],
                    landmarks['right_elbow'],
                    landmarks['right_wrist']
                )
                elbow_diffs.append(abs(left_angle - right_angle))

            # --- Arm-vector angle (shoulder → wrist) relative to body axis ---
            if all(
                landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
                for s in ('left_shoulder', 'right_shoulder',
                          'left_hip', 'right_hip',
                          'left_wrist', 'right_wrist')
            ):
                # Body axis (vertical): shoulder midpoint → hip midpoint
                sx = (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2
                sy = (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
                hx = (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2
                hy = (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
                body_vec = np.array([hx - sx, hy - sy])  # downward in image

                # Left arm vector (shoulder → wrist)
                lv = np.array([
                    landmarks['left_wrist']['x'] - landmarks['left_shoulder']['x'],
                    landmarks['left_wrist']['y'] - landmarks['left_shoulder']['y'],
                ])
                # Right arm vector
                rv = np.array([
                    landmarks['right_wrist']['x'] - landmarks['right_shoulder']['x'],
                    landmarks['right_wrist']['y'] - landmarks['right_shoulder']['y'],
                ])

                # Angle between left-arm vector and right-arm vector
                norm_l = np.linalg.norm(lv)
                norm_r = np.linalg.norm(rv)
                if norm_l > 1e-6 and norm_r > 1e-6:
                    cos_angle = np.dot(lv, rv) / (norm_l * norm_r)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    arm_angle = np.degrees(np.arccos(cos_angle))
                    # In synchronized butterfly arms are parallel → angle near 0
                    arm_angle_diffs.append(arm_angle)

        if not elbow_diffs:
            return {'sync_delta': None, 'synchronized': None}

        avg_elbow_diff = float(np.mean(elbow_diffs))
        avg_arm_angle = float(np.mean(arm_angle_diffs)) if arm_angle_diffs else None

        # Heuristic thresholds:
        #   - avg elbow diff < 20°  → synchronized
        #   - avg elbow diff 20-35° → marginal
        #   - avg elbow diff > 35°  → out of sync
        #
        # The arm-vector angle (angle between left and right arm vectors) is
        # more sensitive to stroke phase and camera perspective, so we use it
        # only as a supporting indicator, not a hard gate.
        sync_elbow = avg_elbow_diff <= 20.0

        # Overall sync: elbow-angle comparison is the primary signal.
        # If avg_arm_angle is available and very large (>60°) it hints at
        # asymmetry, but we don't gate on it.
        synchronized = sync_elbow

        return {
            'sync_delta': avg_elbow_diff,
            'synchronized': synchronized,
            'avg_elbow_diff': avg_elbow_diff,
            'avg_arm_angle': avg_arm_angle,
        }

    @staticmethod
    def _find_peak_timestamps(samples: List[Tuple[float, float]]) -> List[float]:
        """
        Find peak timestamps in a (timestamp, value) series.

        Returns timestamps of detected peaks.
        """
        timestamps = [s[0] for s in samples]
        values = [s[1] for s in samples]

        if len(values) < 3:
            return []

        # Smooth
        window = max(3, len(values) // 20)
        smoothed = ButterflyAnalyzer._moving_average(values, window)

        # Min peak gap (approx 0.5 s)
        fps_equiv = len(smoothed) / (timestamps[-1] - timestamps[0] + 1e-6)
        min_gap = max(3, int(fps_equiv * 0.4))

        # Detect peaks
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                peaks.append(i)

        # Filter by minimum gap
        filtered = []
        for p in peaks:
            if not filtered or (p - filtered[-1]) >= min_gap:
                filtered.append(p)

        return [timestamps[p] for p in filtered]

    def _detect_issues(self) -> List[ButterflyIssue]:
        """Detect technique issues based on analyzed metrics."""
        issues = []

        # --- Elbow angles ---
        if self.metrics['elbow']['avg_angle'] is not None:
            avg_elbow = self.metrics['elbow']['avg_angle']
            if avg_elbow > ELBOW_ANGLE_DROPPED:
                issues.append(ButterflyIssue(
                    'dropped_elbow',
                    ISSUE_TYPES['dropped_elbow']['severity'],
                    f"Dropped elbows during pull (avg {avg_elbow:.0f}° — should be {ELBOW_ANGLE_PULL_MIN}-{ELBOW_ANGLE_PULL_MAX}°)",
                    ISSUE_TYPES['dropped_elbow']['tip'],
                    avg_elbow
                ))

        # --- Arm entry width ---
        if self.metrics['entry']['avg_entry_width'] is not None:
            avg_width = self.metrics['entry']['avg_entry_width']
            # Convert from shoulder-span ratio to a more interpretable metric
            # We treat it as a normalised value; the rules file uses frame-width
            # thresholds, but here we compare to shoulder ratio equivalents.
            if avg_width < 0.6:
                issues.append(ButterflyIssue(
                    'arms_too_narrow',
                    ISSUE_TYPES['arms_too_narrow']['severity'],
                    f"Arms entering too narrow (wrist span {avg_width:.2f}x shoulder width — should be wider)",
                    ISSUE_TYPES['arms_too_narrow']['tip'],
                    avg_width
                ))
            elif avg_width > 1.4:
                issues.append(ButterflyIssue(
                    'arms_too_wide',
                    ISSUE_TYPES['arms_too_wide']['severity'],
                    f"Arms entering too wide (wrist span {avg_width:.2f}x shoulder width)",
                    ISSUE_TYPES['arms_too_wide']['tip'],
                    avg_width
                ))

        # --- Body undulation ---
        if self.metrics['undulation']['undulation_amplitude'] is not None:
            amp = self.metrics['undulation']['undulation_amplitude']
            if amp < UNDULATION_TOO_FLAT:
                issues.append(ButterflyIssue(
                    'flat_undulation',
                    ISSUE_TYPES['flat_undulation']['severity'],
                    f"Flat body — not enough dolphin undulation (amplitude {amp:.3f} — minimum {UNDULATION_MIN})",
                    ISSUE_TYPES['flat_undulation']['tip'],
                    amp
                ))
            elif amp > UNDULATION_EXCESSIVE:
                issues.append(ButterflyIssue(
                    'excessive_undulation',
                    ISSUE_TYPES['excessive_undulation']['severity'],
                    f"Excessive undulation (amplitude {amp:.3f} — optimal {UNDULATION_MIN}-{UNDULATION_MAX})",
                    ISSUE_TYPES['excessive_undulation']['tip'],
                    amp
                ))

        # --- Head position ---
        if self.metrics['head']['breathing_lift'] is not None:
            lift = self.metrics['head']['breathing_lift']
            if lift > HEAD_LIFT_EXCESSIVE:
                issues.append(ButterflyIssue(
                    'head_lifting_excessive',
                    ISSUE_TYPES['head_lifting_excessive']['severity'],
                    f"Lifting head too high to breathe (lift {lift:.2f} of frame height)",
                    ISSUE_TYPES['head_lifting_excessive']['tip'],
                    lift
                ))

        # --- Stroke rate ---
        if self.metrics['stroke_rate']['spm'] is not None:
            spm = self.metrics['stroke_rate']['spm']
            if spm < STROKE_RATE_TOO_SLOW:
                issues.append(ButterflyIssue(
                    'slow_stroke_rate',
                    ISSUE_TYPES['slow_stroke_rate']['severity'],
                    f"Stroke rate too slow ({spm:.0f} SPM — optimal {STROKE_RATE_OPTIMAL_MIN}-{STROKE_RATE_OPTIMAL_MAX})",
                    ISSUE_TYPES['slow_stroke_rate']['tip'],
                    spm
                ))
            elif spm > STROKE_RATE_TOO_FAST:
                issues.append(ButterflyIssue(
                    'fast_stroke_rate',
                    ISSUE_TYPES['fast_stroke_rate']['severity'],
                    f"Stroke rate too fast ({spm:.0f} SPM — optimal {STROKE_RATE_OPTIMAL_MIN}-{STROKE_RATE_OPTIMAL_MAX})",
                    ISSUE_TYPES['fast_stroke_rate']['tip'],
                    spm
                ))

        # --- Kick ---
        if self.metrics['kick']['avg_knee_angle'] is not None:
            avg_knee = self.metrics['kick']['avg_knee_angle']
            if avg_knee < KNEE_ANGLE_EXCESSIVE_BEND:
                issues.append(ButterflyIssue(
                    'excessive_knee_bend',
                    ISSUE_TYPES['excessive_knee_bend']['severity'],
                    f"Excessive knee bend in dolphin kick (avg {avg_knee:.0f}° — optimal {KNEE_ANGLE_OPTIMAL_MIN}-{KNEE_ANGLE_OPTIMAL_MAX}°)",
                    ISSUE_TYPES['excessive_knee_bend']['tip'],
                    avg_knee
                ))
            elif avg_knee > KNEE_ANGLE_TOO_STRAIGHT:
                issues.append(ButterflyIssue(
                    'knee_too_straight',
                    ISSUE_TYPES['knee_too_straight']['severity'],
                    f"Knees too straight — not enough kick (avg {avg_knee:.0f}°)",
                    ISSUE_TYPES['knee_too_straight']['tip'],
                    avg_knee
                ))

        # --- Arm synchronisation ---
        if self.metrics['synchronization']['sync_delta'] is not None:
            delta = self.metrics['synchronization']['sync_delta']
            if not self.metrics['synchronization']['synchronized']:
                issues.append(ButterflyIssue(
                    'arms_not_synchronised',
                    ISSUE_TYPES['arms_not_synchronised']['severity'],
                    f"Arms not synchronised (avg elbow-angle diff {delta:.1f}° — should be < 20°)",
                    ISSUE_TYPES['arms_not_synchronised']['tip'],
                    delta
                ))

        # Sort by severity
        severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_MODERATE: 1, SEVERITY_MINOR: 2}
        issues.sort(key=lambda x: severity_order[x.severity])

        return issues

    @staticmethod
    def _calculate_angle(point1: Dict, point2: Dict, point3: Dict) -> float:
        """Calculate angle between three points."""
        p1 = np.array([point1['x'], point1['y']])
        p2 = np.array([point2['x'], point2['y']])
        p3 = np.array([point3['x'], point3['y']])

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)
