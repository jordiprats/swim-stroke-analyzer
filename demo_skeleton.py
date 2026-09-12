"""
Generates a synthetic GIF demonstrating how the swim stroke analyzer
visualises pose detection — skeleton overlay, joint angles, stroke-cycle
motion, and butterfly-specific highlights.
"""

import io
import math
import numpy as np
from PIL import Image, ImageDraw

# ── helpers ──────────────────────────────────────────────────────────────

def _rotate(pivot, point, angle_deg):
    """Rotate `point` around `pivot` by `angle_deg` (clockwise)."""
    px, py = pivot
    ox, oy = point
    rad = math.radians(angle_deg)
    cos = math.cos(rad)
    sin = math.sin(rad)
    dx, dy = ox - px, oy - py
    return (px + dx * cos - dy * sin, py + dx * sin + dy * cos)


def _lerp(a, b, t):
    return a + (b - a) * t


# ── stroke cycle helpers ────────────────────────────────────────────────

def freestyle_pose(t, body_center=(160, 100)):
    """
    Returns landmark dict for a freestyle pose at normalised time t (0→1).
    Arms alternate: left arm pulls (0→0.5), right arm pulls (0.5→1).
    """
    cx, cy = body_center
    # Body length
    spine_len = 60

    # Shoulders (rotate with body roll)
    roll = math.sin(t * 2 * math.pi) * 15  # ±15° body roll
    shoulder_yoff = -10
    l_s = _rotate((cx, cy), (cx - 20, cy + shoulder_yoff), roll)
    r_s = _rotate((cx, cy), (cx + 20, cy + shoulder_yoff), roll)

    # Hips
    hip_yoff = spine_len
    l_h = (cx - 10, cy + hip_yoff)
    r_h = (cx + 10, cy + hip_yoff)

    # Elbows — arm recovery creates a characteristic "catch" bend
    # Left arm cycle: 0→0.5 pull, 0.5→1 recovery
    lt = t if t < 0.5 else 1 - t  # 0→0.5→0
    left_elbow_angle = _lerp(130, 80, lt * 2) if t < 0.5 else _lerp(80, 160, (t - 0.5) * 2)
    left_elbow = _rotate(l_s, (l_s[0] - 15, l_s[1] + 25), left_elbow_angle - 180)
    left_wrist = _rotate(left_elbow, (left_elbow[0] - 20, left_elbow[1] + 10), 20)

    # Right arm — opposite phase
    rt = (t + 0.5) % 1
    rlt = rt if rt < 0.5 else 1 - rt
    right_elbow_angle = _lerp(130, 80, rlt * 2) if rt < 0.5 else _lerp(80, 160, (rt - 0.5) * 2)
    right_elbow = _rotate(r_s, (r_s[0] + 15, r_s[1] + 25), right_elbow_angle - 180)
    right_wrist = _rotate(right_elbow, (right_elbow[0] + 20, right_elbow[1] + 10), 20)

    # Head (nose)
    head_y = cy - 25 + math.sin(t * 2 * math.pi) * 3
    nose = (cx, head_y)

    # Knees / ankles (simplified)
    l_k = (cx - 8, cy + spine_len + 30)
    r_k = (cx + 8, cy + spine_len + 30)
    l_a = (cx - 6, cy + spine_len + 55)
    r_a = (cx + 6, cy + spine_len + 55)

    return {
        'nose': {'x': nose[0], 'y': nose[1], 'visibility': 0.9},
        'left_shoulder': {'x': l_s[0], 'y': l_s[1], 'visibility': 0.9},
        'right_shoulder': {'x': r_s[0], 'y': r_s[1], 'visibility': 0.9},
        'left_elbow': {'x': left_elbow[0], 'y': left_elbow[1], 'visibility': 0.9},
        'right_elbow': {'x': right_elbow[0], 'y': right_elbow[1], 'visibility': 0.9},
        'left_wrist': {'x': left_wrist[0], 'y': left_wrist[1], 'visibility': 0.9},
        'right_wrist': {'x': right_wrist[0], 'y': right_wrist[1], 'visibility': 0.9},
        'left_hip': {'x': l_h[0], 'y': l_h[1], 'visibility': 0.9},
        'right_hip': {'x': r_h[0], 'y': r_h[1], 'visibility': 0.9},
        'left_knee': {'x': l_k[0], 'y': l_k[1], 'visibility': 0.9},
        'right_knee': {'x': r_k[0], 'y': r_k[1], 'visibility': 0.9},
        'left_ankle': {'x': l_a[0], 'y': l_a[1], 'visibility': 0.9},
        'right_ankle': {'x': r_a[0], 'y': r_a[1], 'visibility': 0.9},
    }


def butterfly_pose(t, body_center=(160, 100)):
    """
    Butterfly — both arms move together, hips undulate vertically.
    """
    cx, cy = body_center

    # Dolphin undulation: hips rise and fall
    undulation = math.sin(t * 2 * math.pi) * 20
    hip_y = cy + 60 + undulation

    # Shoulders (minimal roll)
    l_s = (cx - 20, cy - 10 + undulation * 0.3)
    r_s = (cx + 20, cy - 10 + undulation * 0.3)

    # Hips
    l_h = (cx - 10, hip_y)
    r_h = (cx + 10, hip_y)

    # Both arms together: pull phase (0→0.5) then recovery (0.5→1)
    # During pull, elbows bend; during recovery, arms swing forward
    phase = t if t < 0.5 else 1 - t  # 0→0.5→0
    elbow_bend = _lerp(150, 120, phase * 2) if t < 0.5 else _lerp(120, 160, (t - 0.5) * 2)

    left_elbow = _rotate(l_s, (l_s[0] - 15, l_s[1] + 30), elbow_bend - 180)
    left_wrist = _rotate(left_elbow, (left_elbow[0] - 22, left_elbow[1] + 10), 15)

    right_elbow = _rotate(r_s, (r_s[0] + 15, r_s[1] + 30), elbow_bend - 180)
    right_wrist = _rotate(right_elbow, (right_elbow[0] + 22, right_elbow[1] + 10), 15)

    # Head — lifts forward to breathe at the end of each cycle
    breath_lift = math.sin(t * 2 * math.pi + 1.5) * 8
    nose = (cx, cy - 25 + breath_lift)

    # Knees / ankles (dolphin kick — legs together)
    kick = math.sin(t * 2 * math.pi + 0.5) * 15
    l_k = (cx - 6, hip_y + 30 + kick)
    r_k = (cx + 6, hip_y + 30 + kick)
    l_a = (cx - 4, hip_y + 55 + kick * 0.5)
    r_a = (cx + 4, hip_y + 55 + kick * 0.5)

    return {
        'nose': {'x': nose[0], 'y': nose[1], 'visibility': 0.9},
        'left_shoulder': {'x': l_s[0], 'y': l_s[1], 'visibility': 0.9},
        'right_shoulder': {'x': r_s[0], 'y': r_s[1], 'visibility': 0.9},
        'left_elbow': {'x': left_elbow[0], 'y': left_elbow[1], 'visibility': 0.9},
        'right_elbow': {'x': right_elbow[0], 'y': right_elbow[1], 'visibility': 0.9},
        'left_wrist': {'x': left_wrist[0], 'y': left_wrist[1], 'visibility': 0.9},
        'right_wrist': {'x': right_wrist[0], 'y': right_wrist[1], 'visibility': 0.9},
        'left_hip': {'x': l_h[0], 'y': l_h[1], 'visibility': 0.9},
        'right_hip': {'x': r_h[0], 'y': r_h[1], 'visibility': 0.9},
        'left_knee': {'x': l_k[0], 'y': l_k[1], 'visibility': 0.9},
        'right_knee': {'x': r_k[0], 'y': r_k[1], 'visibility': 0.9},
        'left_ankle': {'x': l_a[0], 'y': l_a[1], 'visibility': 0.9},
        'right_ankle': {'x': r_a[0], 'y': r_a[1], 'visibility': 0.9},
    }


# ── drawing ──────────────────────────────────────────────────────────────

LANDMARK_CONNECTIONS = [
    ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
]

COLOR_SKELETON = (0, 200, 0)      # Green
COLOR_BOTH_ARMS = (0, 255, 255)   # Cyan
COLOR_ENTRY_LINE = (255, 0, 255)  # Magenta
COLOR_ANGLE = (255, 200, 0)       # Orange


def draw_skeleton_frame(draw, landmarks, is_butterfly=False):
    """Draw one frame of the skeleton overlay."""
    # Helper to get (x, y) as integers
    def xy(name):
        l = landmarks[name]
        return (int(l['x']), int(l['y']))

    # Draw connections (green skeleton)
    for src, dst in LANDMARK_CONNECTIONS:
        if landmarks[src]['visibility'] > 0.5 and landmarks[dst]['visibility'] > 0.5:
            draw.line([xy(src), xy(dst)], fill=COLOR_SKELETON, width=2)

    # Draw joint circles
    joint_names = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow',
                   'right_elbow', 'left_wrist', 'right_wrist',
                   'left_hip', 'right_hip', 'left_knee', 'right_knee',
                   'left_ankle', 'right_ankle']
    for name in joint_names:
        l = landmarks[name]
        if l['visibility'] > 0.5:
            draw.ellipse([l['x'] - 4, l['y'] - 4, l['x'] + 4, l['y'] + 4],
                         fill=COLOR_SKELETON)

    # ── butterfly-specific extras ──
    if is_butterfly:
        # Thick cyan arm overlay
        for arm in [('left_shoulder', 'left_elbow', 'left_wrist'),
                     ('right_shoulder', 'right_elbow', 'right_wrist')]:
            src, mid, dst = arm
            if all(landmarks[s]['visibility'] > 0.5 for s in arm):
                draw.line([xy(src), xy(mid)], fill=COLOR_BOTH_ARMS, width=5)
                draw.line([xy(mid), xy(dst)], fill=COLOR_BOTH_ARMS, width=5)

        # Entry-width line (magenta) between wrists
        if (landmarks['left_wrist']['visibility'] > 0.5 and
            landmarks['right_wrist']['visibility'] > 0.5):
            draw.line([xy('left_wrist'), xy('right_wrist')],
                      fill=COLOR_ENTRY_LINE, width=3)

        # SYNC label at wrist midpoint
        lx, ly = xy('left_wrist')
        rx, ry = xy('right_wrist')
        mid_x = (lx + rx) // 2
        mid_y = (ly + ry) // 2
        draw.text((mid_x - 18, mid_y - 12), "SYNC", fill=(0, 255, 0), font_size=12)

    # ── angle annotations (elbow) ──
    def draw_angle(p1, p2, p3, label):
        """Draw angle arc + label at p2."""
        if not (landmarks[p1]['visibility'] > 0.5 and
                landmarks[p2]['visibility'] > 0.5 and
                landmarks[p3]['visibility'] > 0.5):
            return
        a = xy(p1)
        b = xy(p2)
        c = xy(p3)

        # Simple angle arc
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        import math
        angle = math.degrees(math.acos(
            (v1[0]*v2[0] + v1[1]*v2[1]) /
            (math.hypot(*v1) * math.hypot(*v2) + 1e-6)
        ))
        draw.text((b[0] - 10, b[1] - 20), f"{angle:.0f}°",
                  fill=COLOR_ANGLE, font_size=11)

    draw_angle('left_shoulder', 'left_elbow', 'left_wrist', 'L')
    if is_butterfly:
        draw_angle('right_shoulder', 'right_elbow', 'right_wrist', 'R')


# ── build GIF ────────────────────────────────────────────────────────────

def make_gif(stroke_fn, frames=40, is_butterfly=False, filename='demo.gif'):
    """Create a looping GIF of the skeleton animation."""
    frames_list = []
    for i in range(frames):
        t = i / frames
        landmarks = stroke_fn(t)
        img = Image.new('RGB', (320, 200), (20, 20, 30))
        draw = ImageDraw.Draw(img)
        draw_skeleton_frame(draw, landmarks, is_butterfly)

        # Simple stats panel at top-left
        draw.rectangle([0, 0, 160, 80], fill=(0, 0, 0, 180))
        draw.text((6, 6), "STROKE ANALYSIS", fill=(255, 255, 255), font_size=13)

        # Elbow angle summary
        l_elbow = landmarks['left_elbow']
        r_elbow = landmarks['right_elbow']
        l_s = landmarks['left_shoulder']
        l_w = landmarks['left_wrist']

        import math
        v1 = (l_s['x'] - l_elbow['x'], l_s['y'] - l_elbow['y'])
        v2 = (l_w['x'] - l_elbow['x'], l_w['y'] - l_elbow['y'])
        angle = math.degrees(math.acos(
            (v1[0]*v2[0] + v1[1]*v2[1]) /
            (math.hypot(*v1) * math.hypot(*v2) + 1e-6)
        ))
        draw.text((6, 28), f"L-Elbow: {angle:.0f}°", fill=(255, 200, 0), font_size=11)

        if is_butterfly:
            draw.text((6, 44), "BUTTERFLY", fill=(0, 255, 255), font_size=11)
            draw.text((6, 60), "Both arms + undulation", fill=(200, 200, 200), font_size=10)
        else:
            draw.text((6, 44), "FREESTYLE", fill=(0, 255, 0), font_size=11)
            draw.text((6, 60), "Alternating arms + body roll", fill=(200, 200, 200), font_size=10)

        frames_list.append(img)

    # Save as GIF
    frames_list[0].save(
        filename,
        save_all=True,
        append_images=frames_list[1:],
        loop=0,
        duration=100,      # 10 fps
        optimize=False,
    )
    print(f"✓ Saved {filename} ({frames} frames)")


# ── run ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    make_gif(freestyle_pose, frames=40, is_butterfly=False, filename='demo_freestyle.gif')
    make_gif(butterfly_pose, frames=40, is_butterfly=True, filename='demo_butterfly.gif')
    print("\nDone — open the GIF files in your browser to view.")
