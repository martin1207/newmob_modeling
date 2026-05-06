#!/usr/bin/env python3
"""
NewMob Encounter Annotation Tool v3.0 — Workflow Efficiency
================================================================

Auto-detects encounters from detection CSV using track-based segmentation.
Includes ALL VRU types (pedestrian, cyclist, scooter). CONFIRM_INTERACTION
asked first. Multi-reference calibration supported.

Workflow:
  1. Launch: python annotate_encounters.py --video clip.mp4 --detections clip_detections.csv
  2. Tool auto-detects encounters per VRU track, prints summary to console
  3. ENCOUNTER_LIST: overlay lists all encounters with VRU type. Navigate with ./,  ENTER to open
  4. ENCOUNTER_VIEW: video plays encounter window. Timeline shows auto-markers. ENTER = code
  5. CODING: CONFIRM_INTERACTION first, then interaction-level variables
  6. REVIEW: confirm/edit, ENTER saves, returns to list
  7. TRIP_ANNOTATION: after all coded, prompt WEATHER/LIGHTING/SURFACE/SEGREGATION/COMPANION (once per clip)
  8. DONE: save CSVs

Controls (BEPO-safe — no letter keys in main flow):
  NAVIGATION
    SPACE         Play / pause
    .  or  >      Next frame
    ,  or  <      Previous frame
    BACKSPACE     Back 1 second (browse) / previous variable (coding)

  ENCOUNTER LIST
    .             Next encounter
    ,             Previous encounter
    ENTER         Open selected encounter
    TAB           Skip encounter (mark as skipped)
    j             Jump to next uncoded/pending encounter
    p             Jump to previous uncoded/pending encounter
    r             Replay current encounter from onset
    D             Batch-skip all density_secondary encounters

  ENCOUNTER VIEW
    ENTER         Start coding
    x             Quick-reject (CONFIRM=0, skip to next)
    7             Mark frame timestamp (optional annotation marker)
    5             Distance correction (click body parts, label 1-5, ENTER to compute)
    o             Obstacle point (click ground contact, label type 1-5/9, ENTER to confirm)
    BACKSPACE     Back to list

  CODING
    0-9           Select code for current variable
    ENTER         Confirm value
    TAB           Skip variable (code 9 = unknown)
    BACKSPACE     Previous variable (undo: restores previous value)

  REVIEW
    ENTER         Save encounter, return to list
    TAB           Add/edit notes (free text via terminal)
    BACKSPACE     Back to coding

  SYSTEM
    3             Toggle YOLO overlay
    6             Calibrate (click HEAD+FOOT | 2=Ref 3=Mark 4=Multi-ped)
    ESC           Save and quit

Usage:
    python annotate_encounters.py \\
        --video ./segment.mp4 \\
        --detections ./segment_detections.csv \\
        [--output ./encounters.csv] \\
        [--zones ./pedestrian_zone_segments.csv] \\
        [--suggestions ./vlm_suggestions.csv] \\
        [--trip_id TRIP_ID] [--city CITY] [--rater 1]
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import math
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# MANUAL VARIABLE DEFINITIONS — V4.0 codebook
# Sources: SHRP2, UDRIVE D41.1, BikeSAFE, DOCTOR, Zhang 2023, Boufous 2018
#
# Two coding levels (V4.0):
#   Interaction-level (MANUAL_VARIABLES): CONFIRM, VRU_TYPE,
#       INTERACTION_TYPE, VRU_AGE_GROUP,
#       AWARE_BEFORE_MINDIST (1=Yes, 0=No, 9=Can't tell)
#   Clip-level (TRIP_VARIABLES): WEATHER, LIGHTING, SURFACE, ZONE_TYPE,
#       VISUAL_SEGREGATION, RIDING_COMPANION
#
# V3.5 changes:
#   INTERACT removed (supervisor feedback: every detected track is an encounter)
#   VRU_REACTION moved back to post-processing (V3.7)
#   INTERACTION_TYPE codes updated: Same direction, Opposite direction, Crossing, Stationary
#   OBSTACLE_PRESENT, INFRASTRUCTURE_TYPE, EFFECTIVE_WIDTH removed (V3.0 simplification)
#   ZONE_TYPE: Pedestrian street / Park-greenway / Shared road / Motorised road (Zhang 2023 + ext)
#   AWARE_BEFORE_MINDIST: binary categorical (1=Yes, 0=No, 9=Can't tell)
#   Distance correction suggestion (occlusion/truncation/non-pedestrian)
#   Nearby steering segments merged (min 5s gap)
#
# Post-processing (NOT manually coded):
#     VRU_MOVEMENT_DIR    → trajectory heading relative to rider's path
#     EVASIVE_RIDER       → steering from rider segmentation + braking from AccX
#     RIDER_COMMUNICATION → requires audio analysis (deferred)
#     EVASION_SEQUENCE    → derived from rider/VRU evasion onsets
#     VRU_GROUP_SIZE      → derive GROUP_FLAG in post-processing
#   Rider segmentation: steering only (braking from AccX in post-processing).
#   GROUP_VARIABLES empty (all moved to interaction level in V2.4).
# ═══════════════════════════════════════════════════════════════════

MANUAL_VARIABLES = OrderedDict([
    # ── 0. Gating ──
    ("CONFIRM", {
        "type": "categorical",
        "codes": {1: "Accept", 0: "Reject", 2: "Review later"},
        "prompt": "CONFIRM INTERACTION?",
        "group": "confirm",
    }),
    # ── 1. Interaction geometry ──
    # VRU type — auto-filled from YOLO majority type, rater confirms or overrides
    ("VRU_TYPE", {
        "type": "categorical",
        "codes": {1: "Pedestrian", 2: "Cyclist", 3: "E-scooterist",
                  4: "Other MMV", 5: "Motorised vehicle",
                  6: "Animal", 7: "Stationary obstacle", 9: "Unknown"},
        "prompt": "VRU TYPE",
        "group": "interaction",
        "auto_fill": "vru_type_code",
        "suggested_frame": "perception",
        "carry_forward": True,
    }),
    # DOCTOR directional classification (Kraay & van der Horst 1986; Liang et al. 2021)
    # Consistent with BikeSAFE (Dozza & Werneke 2014, Table 2)
    # V3.5: Following+Overtaking merged → "Same direction" (supervisor feedback)
    ("INTERACTION_TYPE", {
        "type": "categorical",
        "codes": {1: "Same direction", 2: "Opposite direction",
                  3: "Crossing", 4: "Stationary", 9: "Unknown"},
        "prompt": "INTERACTION TYPE",
        "group": "interaction",
        "auto_fill": True,
        "suggested_frame": "perception",
        "carry_forward": True,
    }),

    # ── 2. VRU characteristics ──
    # VRU gait — manually coded (auto-estimation from speed unreliable)
    # Walk/run threshold: 2.0 m/s (Hreljac 1993; Rotstein 2005)
    # Gated: only prompted when VRU_TYPE == 1 (pedestrian)
    ("VRU_GAIT", {
        "type": "categorical",
        "codes": {1: "Stationary", 2: "Walking", 3: "Running/jogging",
                  9: "Unknown"},
        "prompt": "VRU GAIT (pedestrian only)",
        "group": "vru",
        "suggested_frame": "perception",
        "carry_forward": False,
        "gated": True,
    }),
    # B17 — SHRP2 "Other Road User Age", simplified per video resolution limits
    ("VRU_AGE_GROUP", {
        "type": "categorical",
        "codes": {1: "Child", 2: "Adult", 3: "Elderly", 9: "Unknown"},
        "prompt": "VRU AGE GROUP",
        "group": "vru",
        "suggested_frame": "perception",
        "carry_forward": True,
        "default": 2,  # Most VRUs are adults; rater ENTER to confirm or number to override
    }),
    # ── 3. Context ──
    # OBSTACLE_PRESENT removed (V3.0) — obstacle zones captured in pre-encounter phase
    # SIGHT_OBSTRUCTION removed (V3.7) — captured via obstacle point marking (key 'o')
    # VRU_GROUP_SIZE removed (V4.1) — derive GROUP_FLAG in post-processing
    # VRU_REACTION removed (V3.7) — moved back to post-processing
    # VRU awareness — simplified from frame_mark to binary categorical (V4.1)
    # 1=Yes (VRU showed awareness before closest pass), 0=No, 9=Can't tell
    # Resolution: reliable <8m, marginal 8-12m (Kim et al. 2025; Rasouli et al. 2017/2019)
    ("AWARE_BEFORE_MINDIST", {
        "type": "categorical",
        "codes": {1: "Yes", 0: "No", 9: "Unknown/Can't tell"},
        "prompt": "VRU aware before closest pass? (1=Yes, 0=No, 9=Can't tell)",
        "description": "Did the VRU show awareness of the rider before the minimum distance frame?",
        "default": 9,
        "carry_forward": False,
        "optional": True,
    }),

    # ── Post-processing variables (NOT manually coded) ──
    # VRU_MOVEMENT_DIR  → trajectory heading (post-processing)
    # EVASIVE_RIDER     → steering segmentation + AccX (post-processing)
    # RIDER_COMMUNICATION → audio analysis (deferred)
    # EVASION_SEQUENCE  → derived from onset comparison (post-processing)
])

# ── Group-level variables (coded once per interaction group) ──
# An interaction group = encounters happening simultaneously that the rider
# responds to as a unit. Rider behavior and aggregate VRU behavior coded here.
GROUP_VARIABLES = OrderedDict([
    # All former group variables moved to interaction level (V2.4)
    # VRU_ATTENTION → MANUAL, VRU_MOVEMENT_DIR → MANUAL
    # EVASIVE_MANEUVER → EVASIVE_RIDER (MANUAL), RIDER_PATH → removed
])

# Trip-level variables (prompted once per clip after all encounters coded)
# HELMET removed — not reliably observable from first-person camera
TRIP_VARIABLES = OrderedDict([
    ("WEATHER", {
        "type": "categorical",
        "codes": {1: "No adverse conditions", 2: "Adverse conditions",
                  9: "Unknown"},
        "prompt": "WEATHER (1=No adverse, 2=Adverse, 9=Unknown)",
        "default": 1,
    }),
    ("LIGHTING", {
        "type": "categorical",
        "codes": {1: "Daylight", 2: "Dawn/Dusk", 3: "Dark+lit", 9: "Unknown"},
        "prompt": "LIGHTING (1=Daylight, 2=Dawn/Dusk, 3=Dark+lit, 9=Unknown)",
        "default": 1,
    }),
    ("SURFACE_CONDITION", {
        "type": "categorical",
        "codes": {1: "Dry", 2: "Wet", 3: "Gravel/unpaved",
                  4: "Uneven/potholes", 9: "Unknown"},
        "prompt": "SURFACE (1=Dry, 2=Wet, 3=Gravel, 4=Uneven, 9=Unknown)",
        "default": 1,
    }),
    # Context — UDRIVE D41.1 (road type), adapted for French non-motorized spaces
    # Zhang 2023 terminology: pedestrian streets, parks/greenways, shared roads
    ("ZONE_TYPE", {
        "type": "categorical",
        "codes": {1: "Pedestrian street", 2: "Park/greenway",
                  3: "Shared road", 4: "Motorised road", 9: "Unknown"},
        "prompt": "Zone type (1=Ped street, 2=Park/greenway, 3=Shared road, 4=Motorised road, 9=Unk)",
        "description": "Type of space (including brief motorised road segments)",
        "default": 1,
    }),
    # Visual/physical segregation between cycling and pedestrian areas
    # Boufous et al. 2018: OR=3.9 for speed increase with visual segregation
    ("VISUAL_SEGREGATION", {
        "type": "categorical",
        "codes": {1: "No", 2: "Yes", 9: "Unknown"},
        "prompt": "Visual segregation? (1=No, 2=Yes painted/marked, 9=Unknown)",
        "description": "Painted line or marking separating cycling and pedestrian areas",
        "default": 1,
    }),
    # Whether the ego-rider is riding alone or with a companion
    ("RIDING_COMPANION", {
        "type": "categorical",
        "codes": {1: "Solo", 2: "With companion", 9: "Unknown"},
        "prompt": "Riding companion? (1=Solo, 2=With companion, 9=Unknown)",
        "description": "Whether the ego-rider is alone or riding with a companion",
        "default": 1,
    }),
    # PATH_WIDTH_M removed (V3.7) — will use lane marking + OSM data post-hoc
    # ZONE_WIDTH_INDEX removed (V2.4) — will use OSM data post-hoc
    # PED_COUNT_CLIP removed (V2.5) — count available from coded data (CONFIRM=1)
    # INFRASTRUCTURE_TYPE removed (V3.0) — redundant with ZONE_TYPE
    # EFFECTIVE_WIDTH removed (V3.0) — will use OSM data post-hoc
])

# ── Rider-segment variables (coded per temporal segment of the clip) ──
# Two independent segmentations of the full clip:
#   1. Acceleration phase: accelerating / decelerating / constant speed
#   2. Steering phase: steering (left/right) / straight
# Phases can overlap (e.g., braking while turning).
# IMU signals (AccX, GyrZ, GPS speed) are displayed on video to help.
RIDER_ACCEL_CODES = {1: "Accelerating", 2: "Decelerating", 3: "Constant speed",
                     9: "Unknown"}
RIDER_STEER_CODES = {1: "Steering left", 2: "Steering right", 3: "Straight",
                     9: "Unknown"}

# Perception delay constant — time for visual info to reach rider's brain
# ~200ms is the typical visual processing latency (Thorpe et al. 1996)
# The perception frame = interaction_onset + PERCEPTION_DELAY_S
PERCEPTION_DELAY_S = 0.200

# GPS speed offset — confirmed by NewMob: GPS is recorded at 1 Hz with
# a ~2-second DELAY relative to IMU and video (value repeated ~100x in file).
# IMU data (acc_x_g, yaw_rate_dps) is at ~100 Hz with NO delay.
# Negative value = GPS lags behind video reality.
DEFAULT_SPEED_OFFSET_S = -2.0


def _build_spline_speed(det_df, fps, speed_offset_s):
    """Build per-frame speed dict using cubic spline interpolation of 1Hz GPS.

    GPS speed is recorded at 1Hz with ~2s delay.  We:
      1. Collect unique GPS speed samples, apply the offset correction
      2. Remove glitches (speed=0 with bearing=0, duplicate epochs)
      3. Cubic-spline interpolate to every video frame
      4. Extrapolate edges with boundary values

    Returns dict {frame_number: speed_kmh}.
    Falls back to linear interp if < 4 GPS samples (spline needs >= 4).
    """
    from scipy.interpolate import CubicSpline

    frame_speed = {}
    if 'speed_kmh' not in det_df.columns:
        return frame_speed

    speed_offset_frames = int(speed_offset_s * fps)

    # Step 1: collect raw GPS samples (offset-corrected)
    gps_samples = {}
    for frame_num, group in det_df.groupby('frame'):
        spd = group.iloc[0]['speed_kmh']
        corrected_frame = int(frame_num) + speed_offset_frames
        if pd.notna(spd):
            gps_samples[corrected_frame] = float(spd)

    all_video_frames = sorted(det_df['frame'].unique())
    if len(gps_samples) < 2:
        # Only one GPS sample — constant speed
        if len(gps_samples) == 1:
            s0 = next(iter(gps_samples.values()))
            for vf in all_video_frames:
                frame_speed[int(vf)] = s0
        return frame_speed

    sorted_gps = sorted(gps_samples.items())
    gps_frames = np.array([f for f, _ in sorted_gps])
    gps_speeds = np.array([s for _, s in sorted_gps])

    # Step 2: remove glitches — duplicate frames or zero-speed-with-zero-bearing
    # (bearing check requires raw data; here we just remove duplicates and zeros
    # that are bracketed by non-zero speeds, which are GPS cold-start artifacts)
    if len(gps_frames) >= 3:
        keep = np.ones(len(gps_frames), dtype=bool)
        for i in range(1, len(gps_frames) - 1):
            if gps_speeds[i] == 0 and gps_speeds[i - 1] > 2 and gps_speeds[i + 1] > 2:
                keep[i] = False  # isolated zero = glitch
        gps_frames = gps_frames[keep]
        gps_speeds = gps_speeds[keep]

    v_min = int(all_video_frames[0])
    v_max = int(all_video_frames[-1])
    target_frames = np.arange(v_min, v_max + 1)

    # Step 3: interpolate
    if len(gps_frames) >= 4:
        # Cubic spline (not-a-knot boundary)
        cs = CubicSpline(gps_frames, gps_speeds, bc_type='not-a-knot')
        interp_speeds = cs(target_frames)
        # Clip: spline can undershoot slightly
        interp_speeds = np.maximum(interp_speeds, 0.0)
    else:
        # < 4 points: fall back to linear interpolation
        interp_speeds = np.interp(target_frames, gps_frames, gps_speeds)

    for i, f in enumerate(target_frames):
        frame_speed[int(f)] = float(interp_speeds[i])

    return frame_speed


# Distance cap — pinhole estimates above this are unreliable
# (1px foot error ≈ 0.5m at 15m vs 0.05m at 5m)
DISTANCE_CAP_M = 30.0

# Edge margin — monocular distance unreliable at frame edges
EDGE_MARGIN_PX = 80

# ── Per-vehicle camera defaults ──
# Derived from IMU-calibrated distance data:
#   E-bike: camera on handlebar, rider leans forward → slight upward tilt
#   E-scooter: camera on handlebar, rider stands upright → slight downward tilt
#
# Resolution note: Samsung Galaxy S8 records at 1920x1080 (1080p).
# NewMob clips were downscaled to 1280x720 (720p) during extraction.
# Default focal length: 1445px @1080p (Samsung Galaxy S8 calibrated),
# auto-scaled to actual video resolution.
# E.g. 1445 * 720/1080 = 963px @720p.  Pass --focal_length to override.
VEHICLE_CAMERA_DEFAULTS = {
    'bike': {
        'camera_height': 1.20,   # e-bike handlebar mount; typical range 1.10-1.50m
        'pitch': 0.0,            # calibrate per-clip with key 6; expect 0 +/-5°
        'focal_length_1080p': 1445,  # Samsung Galaxy S8 calibrated; auto-scaled in __init__
    },
    'escooter': {
        'camera_height': 1.25,   # e-scooter handlebar mount
        'pitch': 0.0,            # calibrate per-clip with key 6; expect 0 +/-5°
        'focal_length_1080p': 1445,  # Samsung Galaxy S8 calibrated; auto-scaled in __init__
    },
}


def detect_vehicle_type(video_path):
    """Auto-detect vehicle type from video filename.

    Convention: v1/v2/v3 = e-bike, t1/t2/t3 = e-scooter.
    Falls back to 'bike' if unrecognized.
    """
    stem = Path(video_path).stem.lower()
    if stem.startswith('t') and len(stem) > 1 and stem[1] in '0123456789':
        return 'escooter'
    if stem.startswith('v') and len(stem) > 1 and stem[1] in '0123456789':
        return 'bike'
    # Check for keywords in path
    path_lower = str(video_path).lower()
    if 'trottinette' in path_lower or 'scooter' in path_lower or 'escooter' in path_lower:
        return 'escooter'
    return 'bike'  # default


# Severity suggestion thresholds — aligned with codebook V2.5 Section 9.1
# AccX (m/s²): 1.0 / 2.0 / 3.5 boundaries
# GyrZ (°/s): 15 / 30 boundaries
# Informed by Huertas-Leyva et al. (2018), Strauss et al. (2017)
SEVERITY_THRESHOLDS = {
    "decel": [(3.5, 4), (2.0, 3), (1.0, 2)],
    "yaw": [(30.0, 3), (15.0, 2)],
}


def _decel_color(decel_ms2):
    """Return BGR color for deceleration value display."""
    if decel_ms2 > 3.5:
        return (0, 0, 255)      # Red — near-miss
    elif decel_ms2 > 2.0:
        return (0, 165, 255)    # Orange — incident
    elif decel_ms2 > 1.0:
        return (0, 255, 255)    # Yellow — proximity
    else:
        return (0, 255, 0)      # Green — non-conflict


# ═══════════════════════════════════════════════════════════════════
# RTS KALMAN SMOOTHER (constant-velocity model)
# ═══════════════════════════════════════════════════════════════════

def rts_smooth_track(frames, distances, q=1.0, r=3.0):
    """Smooth a single track's distance timeseries using Kalman + RTS.

    State: [position, velocity]  (constant-velocity model)
    Handles frame gaps by predicting through them (dt = frame gap).

    Parameters:
        frames: array of frame numbers (may have gaps)
        distances: array of distance values (same length)
        q: process noise — lower = smoother (default 1.0)
        r: measurement noise — higher = smoother (default 3.0)

    Returns:
        smoothed distances (same length as input)

    Note: This smoother only operates on existing observations. It does NOT
    add new frames, extrapolate beyond the track, or fill gaps with synthetic detections.
    """
    n = len(frames)
    if n < 8:
        return distances.copy()  # Too few points for reliable RTS smoothing

    obs = np.array(distances, dtype=float)
    frm = np.array(frames, dtype=float)

    # Forward Kalman pass
    x_fwd = np.zeros((n, 2))  # [position, velocity]
    P_fwd = np.zeros((n, 2, 2))

    x_fwd[0] = [obs[0], 0.0]
    P_fwd[0] = np.eye(2) * 10.0

    H = np.array([[1.0, 0.0]])
    R_mat = np.array([[r]])

    for i in range(1, n):
        dt = frm[i] - frm[i - 1]  # Frame gap (usually 1, can be larger)
        F = np.array([[1, dt], [0, 1]])
        Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])

        x_pred = F @ x_fwd[i - 1]
        P_pred = F @ P_fwd[i - 1] @ F.T + Q

        if np.isfinite(obs[i]) and obs[i] > 0:
            y_inn = obs[i] - H @ x_pred
            S = H @ P_pred @ H.T + R_mat
            K = P_pred @ H.T / S[0, 0]
            x_fwd[i] = x_pred + (K * y_inn).flatten()
            P_fwd[i] = (np.eye(2) - K @ H) @ P_pred
        else:
            x_fwd[i] = x_pred
            P_fwd[i] = P_pred

    # Backward RTS pass
    x_sm = x_fwd.copy()
    P_sm = P_fwd.copy()
    for i in range(n - 2, -1, -1):
        dt = frm[i + 1] - frm[i]
        F = np.array([[1, dt], [0, 1]])
        Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        P_pred = F @ P_fwd[i] @ F.T + Q
        det = np.linalg.det(P_pred)
        if abs(det) < 1e-12:
            continue
        C = P_fwd[i] @ F.T @ np.linalg.inv(P_pred)
        x_sm[i] = x_fwd[i] + C @ (x_sm[i + 1] - F @ x_fwd[i])
        P_sm[i] = P_fwd[i] + C @ (P_sm[i + 1] - P_pred) @ C.T

    # Clamp to positive distances
    result = np.maximum(x_sm[:, 0], 0.3)
    return result


def _dedup_detections(det_df):
    """Remove duplicate detections per (track_id, frame).

    When a tracker produces two detections for the same track in one frame
    (e.g., bbox splits or track switch), keep the one most similar to the
    previous frame's detection (by foot_x and bbox_height proximity).
    """
    has_fx = 'foot_x' in det_df.columns
    has_bh = 'bbox_height' in det_df.columns
    if not has_fx and not has_bh:
        # No spatial info to compare — keep first occurrence
        return det_df.drop_duplicates(subset=['track_id', 'frame'], keep='first')

    # Find (track_id, frame) pairs with duplicates
    dup_mask = det_df.duplicated(subset=['track_id', 'frame'], keep=False)
    if not dup_mask.any():
        return det_df

    n_dups = 0
    drop_indices = []
    for (tid, frame), group in det_df[dup_mask].groupby(['track_id', 'frame']):
        if len(group) <= 1:
            continue
        n_dups += 1
        # Find previous frame's detection for this track
        prev = det_df[(det_df['track_id'] == tid) & (det_df['frame'] < frame)]
        if len(prev) > 0:
            prev_row = prev.sort_values('frame').iloc[-1]
            # Score each duplicate by similarity to previous frame
            best_idx = None
            best_score = float('inf')
            for idx, row in group.iterrows():
                score = 0
                if has_fx and pd.notna(row.get('foot_x')) and pd.notna(prev_row.get('foot_x')):
                    score += abs(float(row['foot_x']) - float(prev_row['foot_x']))
                if has_bh and pd.notna(row.get('bbox_height')) and pd.notna(prev_row.get('bbox_height')):
                    score += abs(float(row['bbox_height']) - float(prev_row['bbox_height']))
                if score < best_score:
                    best_score = score
                    best_idx = idx
            # Drop all except best
            for idx in group.index:
                if idx != best_idx:
                    drop_indices.append(idx)
        else:
            # No previous frame — keep first row
            for idx in group.index[1:]:
                drop_indices.append(idx)

    if drop_indices:
        det_df = det_df.drop(drop_indices)
        print(f"  [DEDUP] Removed {len(drop_indices)} duplicate detections from {n_dups} (track,frame) pairs")
    return det_df


def _interpolate_1frame_gaps(det_df):
    """Interpolate 1-frame gaps in each track's bounding box sequence.

    For tracks where frame N-1 and N+1 exist but frame N is missing,
    create an interpolated row at frame N using linear interpolation
    of foot_x, foot_y, bbox_height, bbox_width, distance_m, lateral_m.
    Interpolated rows are marked with is_interpolated=True.
    """
    interp_cols = ['foot_x', 'foot_y', 'bbox_height', 'bbox_width',
                   'distance_m', 'lateral_m', 'angle_deg']
    # Only interpolate columns that exist
    interp_cols = [c for c in interp_cols if c in det_df.columns]
    if not interp_cols:
        return det_df

    new_rows = []
    for tid in det_df['track_id'].unique():
        tdf = det_df[det_df['track_id'] == tid].sort_values('frame')
        if len(tdf) < 2:
            continue
        frames = tdf['frame'].values.astype(int)
        # Find 1-frame gaps
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap == 2:  # exactly 1 missing frame
                missing_frame = frames[i] + 1
                row_before = tdf[tdf['frame'] == frames[i]].iloc[0]
                row_after = tdf[tdf['frame'] == frames[i + 1]].iloc[0]
                new_row = row_before.copy()
                new_row['frame'] = missing_frame
                new_row['is_interpolated'] = True
                # Linear interpolation for numeric columns
                for col in interp_cols:
                    v_before = row_before.get(col)
                    v_after = row_after.get(col)
                    if pd.notna(v_before) and pd.notna(v_after):
                        new_row[col] = (float(v_before) + float(v_after)) / 2.0
                new_rows.append(new_row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        det_df = pd.concat([det_df, new_df], ignore_index=True)
        det_df = det_df.sort_values(['track_id', 'frame']).reset_index(drop=True)
        # Fill NaN in is_interpolated for original rows that lacked the column.
        # Without this, bool(NaN) == True causes all original detections to be
        # incorrectly skipped as interpolated in _draw_yolo_overlay.
        mask = det_df['is_interpolated'].isna()
        if mask.any():
            det_df.loc[mask, 'is_interpolated'] = False
        print(f"  [INTERP] Interpolated {len(new_rows)} 1-frame gaps across tracks")
    return det_df


def _flag_position_jumps(det_df, min_jump_m=1.0, jump_frac=0.15, max_gap_frames=2):
    """Flag or remove detections with implausible position jumps within a track.

    If within max_gap_frames the distance_m changes by more than the threshold,
    the less coherent detection (the one that disagrees more with the overall
    track trend) is marked as an outlier and removed.

    Threshold is distance-adaptive: max(min_jump_m, distance * jump_frac).
    At 5m: 1.0m threshold. At 10m: 1.5m. At 20m: 3.0m.

    Parameters:
        det_df: detection DataFrame
        min_jump_m: minimum threshold for position jump in meters (default 1.0)
        jump_frac: fraction of distance for adaptive threshold (default 0.15)
        max_gap_frames: max frame gap to check (default 2 = within 2 frames)
    """
    if 'distance_m' not in det_df.columns:
        return det_df

    drop_indices = []
    n_flagged = 0

    for tid in det_df['track_id'].unique():
        tdf = det_df[det_df['track_id'] == tid].sort_values('frame')
        if len(tdf) < 3:
            continue

        frames = tdf['frame'].values.astype(int)
        dists = tdf['distance_m'].values.astype(float)
        indices = tdf.index.values

        # Check consecutive pairs within max_gap_frames
        i = 0
        while i < len(frames) - 1:
            frame_gap = frames[i + 1] - frames[i]
            if frame_gap > max_gap_frames:
                i += 1
                continue

            dist_jump = abs(dists[i + 1] - dists[i])
            # Distance-adaptive threshold: stricter at close range
            avg_dist = (dists[i] + dists[i + 1]) / 2.0
            threshold = max(min_jump_m, avg_dist * jump_frac)
            if dist_jump <= threshold:
                i += 1
                continue

            # Jump detected — determine which point is the outlier
            # Compare each point against its other neighbor (if available)
            # The point that differs most from its other neighbor is the outlier
            score_i = 0.0  # penalty for keeping point i
            score_ip1 = 0.0  # penalty for keeping point i+1

            # Check point i against its predecessor
            if i > 0 and (frames[i] - frames[i - 1]) <= max_gap_frames:
                score_i = abs(dists[i] - dists[i - 1])
            # Check point i+1 against its successor
            if i + 1 < len(frames) - 1 and (frames[i + 2] - frames[i + 1]) <= max_gap_frames:
                score_ip1 = abs(dists[i + 1] - dists[i + 2])

            # Remove the one with higher score (more inconsistent with neighbors)
            if score_i > score_ip1:
                drop_indices.append(indices[i])
                n_flagged += 1
                i += 1  # skip past the removed point
            elif score_ip1 > 0:
                drop_indices.append(indices[i + 1])
                n_flagged += 1
                i += 2  # skip past the removed point
            else:
                # Can't determine — skip (e.g., both are boundary points)
                i += 1

    if drop_indices:
        det_df = det_df.drop(drop_indices)
        print(f"  [JUMP] Flagged and removed {n_flagged} position jump outliers (>max({min_jump_m}m, {jump_frac:.0%}*d) in {max_gap_frames}f)")
    return det_df


def smooth_detections(det_df):
    """Apply RTS smoothing to distance_m per track in the detection DataFrame.

    Adds 'distance_raw_m' (original) and overwrites 'distance_m' with smoothed.
    Returns modified DataFrame.
    """
    if 'distance_m' not in det_df.columns:
        return det_df

    det_df = det_df.copy()

    # Remove duplicate detections before smoothing
    det_df = _dedup_detections(det_df)

    # Flag and remove position jumps (>1m in 2 frames)
    det_df = _flag_position_jumps(det_df)

    # Interpolate 1-frame gaps in each track (fill short tracker dropouts)
    det_df = _interpolate_1frame_gaps(det_df)

    # If already smoothed before (e.g., recalibration), restore raw first
    if 'distance_raw_m' in det_df.columns:
        det_df['distance_m'] = det_df['distance_raw_m'].copy()
    else:
        det_df['distance_raw_m'] = det_df['distance_m'].copy()

    n_smoothed = 0
    for tid in det_df['track_id'].unique():
        mask = (det_df['track_id'] == tid) & (det_df['distance_m'] > 0)
        if mask.sum() < 8:
            continue
        frames = det_df.loc[mask, 'frame'].values
        dists = det_df.loc[mask, 'distance_m'].values
        smoothed = rts_smooth_track(frames, dists)
        det_df.loc[mask, 'distance_m'] = smoothed
        n_smoothed += 1

    # Also smooth lateral_m for more stable VRU speed estimation
    n_lat_smoothed = 0
    if 'lateral_m' in det_df.columns:
        if 'lateral_raw_m' not in det_df.columns:
            det_df['lateral_raw_m'] = det_df['lateral_m'].copy()
        else:
            det_df['lateral_m'] = det_df['lateral_raw_m'].copy()
        for tid in det_df['track_id'].unique():
            mask = (det_df['track_id'] == tid) & (det_df['lateral_m'].notna())
            if mask.sum() < 8:
                continue
            frames = det_df.loc[mask, 'frame'].values
            lats = det_df.loc[mask, 'lateral_m'].values
            # Use slightly higher r for lateral (more measurement noise in pixel-to-meter)
            smoothed_lat = rts_smooth_track(frames, np.abs(lats), q=1.0, r=5.0)
            # Restore sign
            signs = np.sign(lats)
            signs[signs == 0] = 1
            det_df.loc[mask, 'lateral_m'] = smoothed_lat * signs
            n_lat_smoothed += 1

    print(f"  [RTS] Smoothed {n_smoothed} dist + {n_lat_smoothed} lateral tracks (q=1.0, r=3.0/5.0)")

    # Add longitudinal distance column: sqrt(distance² - lateral²)
    if 'lateral_m' in det_df.columns:
        d = det_df['distance_m'].values
        l = det_df['lateral_m'].values
        det_df['distance_longitudinal_m'] = np.sqrt(np.maximum(d**2 - l**2, 0.0))

    # Distance confidence based on bbox height — sigmoid falloff below 30px.
    # At 30px bbox height, 1px foot error ≈ 0.5m distance error.
    # Below ~20px the pinhole model is essentially guessing.
    # sigmoid(x) = 1 / (1 + exp(-k*(x - x0))) with k=0.3, x0=30
    if 'bbox_height' in det_df.columns:
        bh = det_df['bbox_height'].values.astype(float)
        det_df['distance_confidence'] = np.round(
            1.0 / (1.0 + np.exp(-0.3 * (bh - 30.0))), 3
        )
        n_low_conf = int((det_df['distance_confidence'] < 0.5).sum())
        if n_low_conf > 0:
            print(f"  [CONF] {n_low_conf} detections with distance_confidence < 0.5 (bbox_height < 30px)")

    # Per-detection visibility score (0-1) from YOLO visibility_status + frame edges
    EDGE_MARGIN = 80
    # NewMob: recorded 1080p (S8), clips downscaled to 720p. Infer from data.
    frame_w, frame_h = 1280, 720
    if 'foot_x' in det_df.columns and det_df['foot_x'].max() > 1800:
        frame_w, frame_h = 1920, 1080
    vis = np.ones(len(det_df))
    if 'visibility_status' in det_df.columns:
        vis_map = {'FULL': 1.0, 'PARTIAL': 0.7, 'OCCLUDED': 0.4}
        vis = det_df['visibility_status'].map(vis_map).fillna(0.7).values
    if 'foot_x' in det_df.columns and 'foot_y' in det_df.columns and 'bbox_height' in det_df.columns:
        fx = det_df['foot_x'].values.astype(float)
        fy = det_df['foot_y'].values.astype(float)
        bh = det_df['bbox_height'].values.astype(float)
        edge_trunc = (
            (fx < EDGE_MARGIN) | (fx > frame_w - EDGE_MARGIN) |
            ((fy - bh) < 5) | (fy > frame_h - 5)
        )
        vis[edge_trunc] = np.minimum(vis[edge_trunc], 0.5)
    det_df['visibility'] = np.round(vis, 2)

    return det_df


# ═══════════════════════════════════════════════════════════════════
# VRU SPEED ESTIMATION
# ═══════════════════════════════════════════════════════════════════

def _central_diff(signal, frames, fps, half_window=2):
    """5-frame central difference for speed estimation."""
    n = len(signal)
    speed = np.zeros(n)
    for i in range(n):
        left = min(i, half_window)
        right = min(n - 1 - i, half_window)
        if left == 0 and right == 0:
            continue
        if left == 0:
            dt = (frames[i + right] - frames[i]) / fps
        elif right == 0:
            dt = (frames[i] - frames[i - left]) / fps
        else:
            dt = (frames[i + right] - frames[i - left]) / fps
        if dt < 1e-6:
            continue
        if left == 0:
            speed[i] = (signal[i + right] - signal[i]) / dt
        elif right == 0:
            speed[i] = (signal[i] - signal[i - left]) / dt
        else:
            speed[i] = (signal[i + right] - signal[i - left]) / dt
    return speed


def _trimmed_mean(arr, lo_pct=10, hi_pct=90):
    """Trimmed mean: average values between lo and hi percentiles."""
    if len(arr) == 0:
        return 0.0
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    mask = (arr >= lo) & (arr <= hi)
    if mask.sum() == 0:
        return float(np.median(arr))
    return float(np.mean(arr[mask]))


def estimate_vru_speed(track_df, fps=30.0):
    """Estimate VRU ground speed from track data.

    Uses 5-frame central difference, median filtering, trimmed mean,
    and lateral-speed capping for robust classification.

    Returns dict or None if insufficient data.
    """
    track_df = track_df.sort_values('frame').copy()
    n = len(track_df)
    if n < 5:
        return None

    frames = track_df['frame'].values.astype(float)
    dist_m = track_df['distance_m'].values.astype(float)
    if 'lateral_m' in track_df.columns:
        lat_m = track_df['lateral_m'].fillna(0.0).values.astype(float)
    else:
        lat_m = np.zeros(n)
    # Replace NaN distances with 0 to prevent cascading NaN in derivatives
    dist_m = np.where(np.isfinite(dist_m), dist_m, 0.0)

    ego_speeds = np.zeros(n)
    if 'speed_kmh' in track_df.columns:
        for j, (_, row) in enumerate(track_df.iterrows()):
            ego_speeds[j] = float(row['speed_kmh']) if pd.notna(row['speed_kmh']) else 0.0

    long_m = np.sqrt(np.maximum(dist_m**2 - lat_m**2, 0.01))

    # half_window=3 (7-frame window) for smoother speed estimates
    # Works well with RTS-smoothed lateral and distance data
    lat_speed = _central_diff(lat_m, frames, fps, half_window=3)
    closing_rate = -_central_diff(long_m, frames, fps, half_window=3)
    ego_mps = ego_speeds / 3.6
    vru_long_speed = closing_rate - ego_mps

    vru_total_speed = np.sqrt(lat_speed**2 + vru_long_speed**2)
    vru_lateral_only = np.abs(lat_speed)

    # Median filter
    try:
        from scipy.ndimage import median_filter
        med_win = min(5, n if n % 2 == 1 else n - 1)
        if med_win < 3:
            med_win = 3 if n >= 3 else 1
        if med_win >= 3:
            vru_total_speed = median_filter(vru_total_speed, size=med_win)
            vru_lateral_only = median_filter(vru_lateral_only, size=med_win)
    except ImportError:
        pass  # scipy optional

    valid = vru_total_speed < 10.0
    if valid.sum() < 2:
        return None

    total_est = _trimmed_mean(vru_total_speed[valid])
    lat_est = _trimmed_mean(vru_lateral_only[valid])
    if lat_est > 0.3:
        # Crossing VRU — cap by lateral to prevent longitudinal noise inflation
        # Use 1.2× instead of 1.5× for conservative classification:
        # monocular lateral speed has ~30% noise, so a walker at 1.5 m/s
        # can appear as 2.0 m/s; 1.2× prevents inflating into runner range
        cls_speed = min(total_est, 1.2 * lat_est)
    else:
        # Non-crossing VRU (longitudinal/parallel motion)
        # Longitudinal speed is noisy, so be conservative:
        # cap at 3.0 m/s (allows runner detection for clearly fast VRUs
        # while still filtering extreme outliers from monocular noise)
        cls_speed = min(total_est, 3.0)

    if cls_speed < 0.5:
        cls = 'stationary'
    elif cls_speed < 2.0:
        cls = 'walker'
    else:
        cls = 'runner'

    return {
        'vru_speed_mps': round(total_est, 2),
        'vru_speed_kmh': round(total_est * 3.6, 1),
        'classification_speed_mps': round(cls_speed, 2),
        'classification': cls,
    }


# ═══════════════════════════════════════════════════════════════════
# AUTO-DETECTION ENGINE (Phase 1)
# ═══════════════════════════════════════════════════════════════════

def compute_track_flags(det_df, track_id):
    """Flag tracks with quality issues.

    Returns list of flag strings. Empty list = clean track.
    Flags: 'short' (< 10 frames), 'size_jump' (bbox change > 40%),
           'pos_jump' (position jump > 50px between consecutive frames),
           'swap' (probable track ID switch — direction reversal + speed change).
    """
    t = det_df[det_df['track_id'] == track_id].sort_values('frame')
    flags = []
    if len(t) < 10:
        flags.append('short')
    if len(t) >= 2 and 'bbox_height' in t.columns:
        heights = t['bbox_height'].dropna().values
        if len(heights) >= 2:
            max_h = heights.max()
            min_h = heights.min()
            if min_h > 0 and (max_h - min_h) / min_h > 0.4:
                flags.append('size_jump')
    swap_frame = None
    if len(t) >= 2 and 'foot_x' in t.columns and 'foot_y' in t.columns:
        fx = t['foot_x'].values
        fy = t['foot_y'].values
        frames = t['frame'].values
        for i in range(1, len(fx)):
            if pd.notna(fx[i]) and pd.notna(fx[i-1]) and pd.notna(fy[i]) and pd.notna(fy[i-1]):
                dx = abs(float(fx[i]) - float(fx[i-1]))
                dy = abs(float(fy[i]) - float(fy[i-1]))
                if max(dx, dy) > 50:
                    flags.append('pos_jump')
                    break
        # Track swap detection: look for direction reversal in foot_x velocity
        # Compute smoothed velocity over 5-frame windows and detect sign changes
        if len(fx) >= 10:
            window = 5
            vx = []
            for i in range(window, len(fx)):
                if pd.notna(fx[i]) and pd.notna(fx[i - window]):
                    vx.append((float(fx[i]) - float(fx[i - window]), int(frames[i])))
            # Find sign changes in smoothed velocity (ignore small movements < 3px/window)
            for i in range(1, len(vx)):
                v_prev, _ = vx[i - 1]
                v_curr, f_curr = vx[i]
                if abs(v_prev) > 3 and abs(v_curr) > 3 and v_prev * v_curr < 0:
                    # Direction reversal detected — likely a track swap
                    if 'swap' not in flags:
                        flags.append('swap')
                        swap_frame = f_curr
                    break
    return flags, swap_frame


def detect_same_user_links(encounters, det_df, fps, max_gap_s=3.0, max_dist_px=150):
    """Detect encounters that likely involve the same physical person.

    Two tracks are linked if:
    - Track A ends within max_gap_s of Track B starting (or vice versa)
    - The endpoint of A is spatially close to the startpoint of B (foot_x/foot_y < max_dist_px)
    - OR they temporally overlap with similar foot positions

    Returns dict: track_id -> set of linked track_ids
    """
    if 'foot_x' not in det_df.columns:
        return {}

    # Build per-track summaries: first/last frame, first/last foot position
    track_info = {}
    for tid in det_df['track_id'].unique():
        t = det_df[det_df['track_id'] == tid].sort_values('frame')
        if len(t) < 3:
            continue
        first = t.iloc[0]
        last = t.iloc[-1]
        track_info[tid] = {
            'frame_start': int(first['frame']),
            'frame_end': int(last['frame']),
            'start_x': float(first['foot_x']) if pd.notna(first.get('foot_x')) else None,
            'start_y': float(first['foot_y']) if pd.notna(first.get('foot_y')) else None,
            'end_x': float(last['foot_x']) if pd.notna(last.get('foot_x')) else None,
            'end_y': float(last['foot_y']) if pd.notna(last.get('foot_y')) else None,
        }

    # Only check tracks that appear in encounters
    enc_tracks = set(e['primary_track'] for e in encounters)
    max_gap_frames = int(max_gap_s * fps)

    links = {}  # track_id -> set of linked track_ids
    for tid_a in enc_tracks:
        if tid_a not in track_info:
            continue
        a = track_info[tid_a]
        for tid_b in enc_tracks:
            if tid_b <= tid_a or tid_b not in track_info:
                continue
            b = track_info[tid_b]

            # Check temporal adjacency: A ends near B starts (or overlap)
            gap_ab = b['frame_start'] - a['frame_end']
            gap_ba = a['frame_start'] - b['frame_end']
            temporally_close = (abs(gap_ab) < max_gap_frames or abs(gap_ba) < max_gap_frames
                                or (a['frame_start'] <= b['frame_end'] and b['frame_start'] <= a['frame_end']))

            if not temporally_close:
                continue

            # Check spatial proximity at the junction
            linked = False
            # A ends -> B starts
            if (a['end_x'] is not None and b['start_x'] is not None
                    and abs(gap_ab) < max_gap_frames):
                dx = abs(a['end_x'] - b['start_x'])
                dy = abs(a['end_y'] - b['start_y']) if a['end_y'] and b['start_y'] else 0
                if max(dx, dy) < max_dist_px:
                    linked = True
            # B ends -> A starts
            if (not linked and b['end_x'] is not None and a['start_x'] is not None
                    and abs(gap_ba) < max_gap_frames):
                dx = abs(b['end_x'] - a['start_x'])
                dy = abs(b['end_y'] - a['start_y']) if b['end_y'] and a['start_y'] else 0
                if max(dx, dy) < max_dist_px:
                    linked = True
            # Overlap: check mid-track positions
            if not linked and a['frame_start'] <= b['frame_end'] and b['frame_start'] <= a['frame_end']:
                overlap_start = max(a['frame_start'], b['frame_start'])
                overlap_end = min(a['frame_end'], b['frame_end'])
                mid_frame = (overlap_start + overlap_end) // 2
                a_mid = det_df[(det_df['track_id'] == tid_a) & (abs(det_df['frame'] - mid_frame) < 5)]
                b_mid = det_df[(det_df['track_id'] == tid_b) & (abs(det_df['frame'] - mid_frame) < 5)]
                if len(a_mid) > 0 and len(b_mid) > 0:
                    dx = abs(float(a_mid.iloc[0]['foot_x']) - float(b_mid.iloc[0]['foot_x']))
                    dy = abs(float(a_mid.iloc[0]['foot_y']) - float(b_mid.iloc[0]['foot_y']))
                    if max(dx, dy) < max_dist_px:
                        linked = True

            if linked:
                links.setdefault(tid_a, set()).add(tid_b)
                links.setdefault(tid_b, set()).add(tid_a)

    return links


def _split_on_gaps(frames, gap_threshold=10):
    """Split a sorted list of frame numbers into segments separated by gaps >= gap_threshold.

    Returns list of lists, each a contiguous segment of frame numbers.
    """
    if not frames:
        return []
    segments = [[frames[0]]]
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] >= gap_threshold:
            segments.append([])
        segments[-1].append(frames[i])
    return segments


def _speed_adaptive_threshold(speed_kmh, fallback=10.0):
    """Compute THW-based interaction zone threshold.

    D_entry = v_ego(m/s) × THW_THRESHOLD_S (default 2s).
    Equivalent to THW < 2s (Minderhoud & Bovy 2001, Svensson 1998).
    Floor 5m: ensures encounters are detected even at very low speeds
    (e.g., stopped rider with VRU at 4m). No cap — THW is self-normalizing.
    When speed is unavailable (<=0.5 km/h), returns fallback (default 10m).
    """
    THW_THRESHOLD_S = 2.0
    if speed_kmh is None or speed_kmh <= 0.5:
        return fallback  # no reliable speed data → conservative default
    speed_mps = speed_kmh / 3.6
    return max(5.0, speed_mps * THW_THRESHOLD_S)


def auto_detect_encounters(det_df, fps, d_threshold=None, min_track_frames=3,
                           speed_offset_s=None, max_angle_deg=60.0,
                           max_lateral_m=None, thw_threshold=None,
                           max_distance=15.0, camera_height=1.1,
                           min_ego_speed_kmh=0.0,
                           dense_scene_k=5, dense_scene_n=3):
    """Auto-detect encounters from detection CSV.

    Algorithm (v4 — THW-based, RTS-smoothed, track-based):
    1. Apply RTS smoothing to distance_m per track
    2. Include all VRU tracks, discard transient tracks (< min_track_frames)
    3. Trigger: THW = d / v_ego < 2s (Minderhoud & Bovy 2001), floor 5m
    4. ONE encounter per track (the window around global min distance)
    5. Same track = same interaction (no duplicate encounters)
    6. Onset/offset = first/last detection of VRU (full track span)
    7. Perception frame = onset + 200ms (visual processing delay, suggested only)
    8. Speed: GPS 1Hz cubic-spline interpolated, offset-corrected (-2.0s)

    d_threshold: if None, uses THW-based threshold (recommended).
                 If a float, uses that fixed value for backward compatibility.

    Returns (encounters, track_summary) tuple:
        encounters: list of dicts with encounter data
        track_summary: list of dicts with per-track stats (all tracks, including filtered)
    """
    use_adaptive_threshold = (d_threshold is None)
    if d_threshold is None:
        d_threshold = 10.0  # fallback for tracks with no speed data
    if speed_offset_s is None:
        speed_offset_s = DEFAULT_SPEED_OFFSET_S

    # Note: RTS smoothing is applied to det_df BEFORE this function is called
    # (in AnnotationTool.__init__). distance_m is already smoothed here.

    # YOLO only produces 'pedestrian' and 'cyclist' (bicycle) for VRUs
    # Everything else (car, truck, bus, motorcycle) is excluded
    motor_types = {'car', 'truck', 'bus', 'motorcycle', 'motor_vehicle'}

    # Height assumptions per VRU type
    # E-scooter: rider stands on deck (~20-25cm above ground), total height ~1.80m.
    # YOLO detects e-scooter riders as 'pedestrian'; height correction applies
    # via manual VRU_TYPE override or auto-MMV classification.
    VRU_HEIGHT_MAP = {
        'pedestrian': 1.70,
        'person': 1.70,
        'cyclist': 1.40,     # seated on bicycle, handlebars to head
        'e-scooter': 1.80,   # standing on deck adds ~10cm
        'mmv_rider': 1.80,   # generic micro-mobility vehicle rider
    }
    DEFAULT_ASSUMED_HEIGHT = 1.70  # pipeline default

    # E-scooter deck height: the rider's feet stand ~10cm above ground on the
    # deck. This causes foot_y to be higher in the image than true ground
    # contact, systematically overestimating pinhole distance.
    # Correction: d_true = d_measured * (h_cam - deck_h) / h_cam
    ESCOOTER_DECK_HEIGHT_M = 0.10

    # EMA class probabilities (alpha=0.9) — reduces class flickering.
    # Computes per-frame EMA probabilities forward through the track, then
    # averages the probabilities across ALL frames. This differs from majority
    # voting by downweighting isolated class flips (which decay quickly in EMA)
    # while preserving the overall class proportion for sustained runs.
    EMA_ALPHA = 0.9
    track_majority_type = {}
    track_class_prob = {}  # tid → {class: probability}
    if 'user_type' in det_df.columns:
        vru_tracks = set()
        for tid in det_df['track_id'].unique():
            track_df = det_df[det_df['track_id'] == tid].sort_values('frame')
            types = [t for t in track_df['user_type'].values
                     if isinstance(t, str) and not pd.isna(t)]
            if not types:
                continue
            # Forward EMA: compute per-frame probabilities, collect snapshots
            ema_state = {}
            frame_probs = []  # list of {class: prob} at each frame
            for t in types:
                for cls in ema_state:
                    ema_state[cls] *= (1 - EMA_ALPHA)
                ema_state[t] = ema_state.get(t, 0.0) * (1 - EMA_ALPHA) + EMA_ALPHA
                frame_probs.append(dict(ema_state))
            # Average per-frame EMA probabilities across all frames
            all_classes = set()
            for fp in frame_probs:
                all_classes.update(fp.keys())
            class_probs = {c: np.mean([fp.get(c, 0.0) for fp in frame_probs])
                           for c in all_classes}
            best_class = max(class_probs, key=class_probs.get)
            # Break ties: prefer 'pedestrian' (safer default for encounter analysis)
            if best_class != 'pedestrian' and 'pedestrian' in class_probs:
                if abs(class_probs['pedestrian'] - class_probs[best_class]) < 0.05:
                    best_class = 'pedestrian'
            if best_class in motor_types:
                continue
            if best_class == 'cyclist':
                track_majority_type[tid] = 'cyclist'
            else:
                track_majority_type[tid] = 'pedestrian'
            track_class_prob[tid] = class_probs
            vru_tracks.add(tid)
    else:
        vru_tracks = set(det_df['track_id'].unique())
        track_majority_type = {tid: 'pedestrian' for tid in vru_tracks}

    # Filter transient tracks: require >= min_track_frames with valid distance
    # This eliminates single-frame false positives / YOLO noise
    stable_tracks = set()
    for tid in vru_tracks:
        track_df = det_df[(det_df['track_id'] == tid) & (det_df['distance_m'] > 0)]
        if len(track_df) >= min_track_frames:
            stable_tracks.add(tid)
        else:
            print(f"  [AUTO] Excluded T{tid}: only {len(track_df)} valid frames (need {min_track_frames}+)")

    vru_tracks = stable_tracks
    vru_df = det_df[det_df['track_id'].isin(vru_tracks)].copy()

    # Exclude interpolated detections from encounter detection (not reliable for distance)
    if 'is_interpolated' in vru_df.columns:
        n_interp = vru_df['is_interpolated'].fillna(False).astype(bool).sum()
        if n_interp > 0:
            vru_df = vru_df[~vru_df['is_interpolated'].fillna(False).astype(bool)]
            print(f"  [AUTO] Excluded {n_interp} interpolated detections from encounter detection")

    # Exclude ego-rider false positives: tracks stuck at frame bottom with constant distance
    # Pattern: foot_y near bottom of frame + distance std < 0.02m (rider body/handlebars)
    if 'foot_y' in det_df.columns:
        frame_h = det_df['foot_y'].max() + 1  # approximate frame height
        ego_tracks = set()
        for tid in vru_tracks:
            t = det_df[(det_df['track_id'] == tid) & (det_df['distance_m'] > 0)]
            if len(t) < 10:
                continue
            if t['foot_y'].mean() > frame_h - 15 and t['distance_m'].std() < 0.02:
                ego_tracks.add(tid)
        if ego_tracks:
            vru_tracks -= ego_tracks
            vru_df = vru_df[~vru_df['track_id'].isin(ego_tracks)]
            print(f"  [AUTO] Excluded {len(ego_tracks)} ego-rider tracks (body/handlebars at frame bottom)")

    if len(vru_df) == 0:
        print("  [AUTO] No stable VRU tracks found.")
        return [], []

    zone_desc = "all tracks (no threshold)"
    print(f"  [AUTO] {len(vru_tracks)} VRU tracks (>= {min_track_frames} frames, zone: {zone_desc})")

    # Pre-index speed per frame (cubic spline interpolation of 1Hz GPS)
    frame_speed = _build_spline_speed(det_df, fps, speed_offset_s)

    # Pre-index sensor data per frame
    frame_sensor = {}
    for frame_num, group in det_df.groupby('frame'):
        row0 = group.iloc[0]
        sensor = {}
        if 'acc_x_g' in group.columns and pd.notna(row0['acc_x_g']):
            sensor['decel'] = abs(float(row0['acc_x_g'])) * 9.81
        if 'acc_y_g' in group.columns and pd.notna(row0['acc_y_g']):
            sensor['lateral_accel'] = abs(float(row0['acc_y_g'])) * 9.81
        if 'yaw_rate_dps' in group.columns and pd.notna(row0['yaw_rate_dps']):
            sensor['yaw'] = abs(float(row0['yaw_rate_dps']))
        frame_sensor[int(frame_num)] = sensor

    # ── Auto-merge fragmented tracks (same person, different track IDs) ──
    # When a tracker loses a person and picks them back up with a new ID,
    # the two tracks are spatially similar (foot_x, bbox_height, distance_m)
    # and temporally close (gap < 8 frames). Merge shorter into longer.
    has_foot = 'foot_x' in vru_df.columns
    has_bbox = 'bbox_height' in vru_df.columns
    has_dist = 'distance_m' in vru_df.columns
    merge_boundary_frames = {}  # dst_tid → set of frames near merge boundaries
    if has_foot and has_bbox and len(vru_tracks) > 1:
        # Build track summary: boundary frame stats (last 3 / first 3 frames)
        track_summary = {}
        for tid in sorted(vru_tracks):
            tdf = vru_df[vru_df['track_id'] == tid].sort_values('frame')
            if len(tdf) < 3:
                continue
            frames = tdf['frame'].values
            last3 = tdf.tail(3)
            first3 = tdf.head(3)
            track_summary[tid] = {
                'f_start': int(frames[0]),
                'f_end': int(frames[-1]),
                'n_frames': len(tdf),
                'end_foot_x': float(last3['foot_x'].median()) if last3['foot_x'].notna().any() else None,
                'end_bbox_h': float(last3['bbox_height'].median()) if last3['bbox_height'].notna().any() else None,
                'end_dist': float(last3['distance_m'].median()) if has_dist and last3['distance_m'].notna().any() else None,
                'end_dist_last': float(tdf.iloc[-1]['distance_m']) if has_dist and pd.notna(tdf.iloc[-1]['distance_m']) else None,
                'start_foot_x': float(first3['foot_x'].median()) if first3['foot_x'].notna().any() else None,
                'start_bbox_h': float(first3['bbox_height'].median()) if first3['bbox_height'].notna().any() else None,
                'start_dist': float(first3['distance_m'].median()) if has_dist and first3['distance_m'].notna().any() else None,
                'start_dist_first': float(tdf.iloc[0]['distance_m']) if has_dist and pd.notna(tdf.iloc[0]['distance_m']) else None,
            }
        # Find merge pairs: track A ends near where track B starts
        merge_map = {}  # src_tid → dst_tid (src gets merged into dst)
        sorted_tids = sorted(track_summary.keys(), key=lambda t: track_summary[t]['f_start'])
        for i in range(len(sorted_tids)):
            for j in range(i + 1, len(sorted_tids)):
                tid_a = sorted_tids[i]
                tid_b = sorted_tids[j]
                sa = track_summary[tid_a]
                sb = track_summary[tid_b]
                # tid_b starts within 8 frames of tid_a ending (or overlaps up to 5 frames)
                temporal_gap = sb['f_start'] - sa['f_end']
                if temporal_gap > 8 or temporal_gap < -5:
                    continue
                # Spatial: foot_x < 40px, bbox_height < 40px, distance < 2m
                if (sa['end_foot_x'] is None or sb['start_foot_x'] is None or
                    sa['end_bbox_h'] is None or sb['start_bbox_h'] is None):
                    continue
                fx_diff = abs(sa['end_foot_x'] - sb['start_foot_x'])
                bh_diff = abs(sa['end_bbox_h'] - sb['start_bbox_h'])
                if fx_diff >= 40 or bh_diff >= 50:
                    continue
                # Distance consistency: distance at boundary must be within 3m
                # Check both median (robust) and actual boundary frame (catches spikes)
                if (sa['end_dist'] is not None and sb['start_dist'] is not None):
                    dist_diff = abs(sa['end_dist'] - sb['start_dist'])
                    if dist_diff > 3.0:
                        continue
                # Overlap check: if both tracks have detections at the same
                # frame, check whether foot_x positions are close. If close
                # (< 60px median), they are the same person re-detected with a
                # new ID — allow merge. If far apart, they are genuinely
                # different people — reject merge.
                if temporal_gap < 0:
                    overlap_start = sb['f_start']
                    overlap_end = sa['f_end']
                    a_overlap = vru_df[(vru_df['track_id'] == tid_a) &
                                      (vru_df['frame'] >= overlap_start) &
                                      (vru_df['frame'] <= overlap_end)]
                    b_overlap = vru_df[(vru_df['track_id'] == tid_b) &
                                      (vru_df['frame'] >= overlap_start) &
                                      (vru_df['frame'] <= overlap_end)]
                    a_frames = set(a_overlap['frame'].values)
                    b_frames = set(b_overlap['frame'].values)
                    shared_frames = sorted(a_frames & b_frames)
                    if len(shared_frames) >= 2:
                        # Compare foot_x at each shared frame
                        fx_diffs = []
                        for sf in shared_frames:
                            fa = a_overlap[a_overlap['frame'] == sf]['foot_x'].values
                            fb = b_overlap[b_overlap['frame'] == sf]['foot_x'].values
                            if len(fa) > 0 and len(fb) > 0 and not (np.isnan(fa[0]) or np.isnan(fb[0])):
                                fx_diffs.append(abs(float(fa[0]) - float(fb[0])))
                        if fx_diffs:
                            median_fx_diff = float(np.median(fx_diffs))
                            if median_fx_diff >= 60:
                                print(f"  [AUTO] Reject merge T{tid_a}↔T{tid_b}: "
                                      f"{len(shared_frames)} shared frames, "
                                      f"median foot_x diff={median_fx_diff:.0f}px (different people)")
                                continue
                            else:
                                print(f"  [AUTO] Overlap merge T{tid_a}↔T{tid_b}: "
                                      f"{len(shared_frames)} shared frames, "
                                      f"median foot_x diff={median_fx_diff:.0f}px (same person)")
                        else:
                            # No valid foot_x comparisons — reject to be safe
                            print(f"  [AUTO] Reject merge T{tid_a}↔T{tid_b}: "
                                  f"{len(shared_frames)} shared frames, no valid foot_x data")
                            continue
                # Merge shorter into longer
                if sa['n_frames'] >= sb['n_frames']:
                    merge_map[tid_b] = tid_a
                else:
                    merge_map[tid_a] = tid_b
                src = tid_b if sa['n_frames'] >= sb['n_frames'] else tid_a
                dst = tid_a if sa['n_frames'] >= sb['n_frames'] else tid_b
                print(f"  [AUTO] Merge T{src} → T{dst}"
                      f" (fx={fx_diff:.0f}px, bh={bh_diff:.0f}px, gap={temporal_gap}f)")
        # Apply merges and record boundary frames for track-switch detection
        if merge_map:
            # Resolve transitive merges: A→B, B→C → A→C
            def resolve_dst(tid):
                visited = set()
                while tid in merge_map and tid not in visited:
                    visited.add(tid)
                    tid = merge_map[tid]
                return tid
            for src in list(merge_map.keys()):
                merge_map[src] = resolve_dst(src)
            vru_df = vru_df.copy()
            for src, dst in merge_map.items():
                # Record the boundary frames: src end and dst start (or vice versa)
                src_end = track_summary[src]['f_end'] if src in track_summary else None
                dst_start = track_summary[dst]['f_start'] if dst in track_summary else None
                src_start = track_summary[src]['f_start'] if src in track_summary else None
                dst_end = track_summary[dst]['f_end'] if dst in track_summary else None
                if dst not in merge_boundary_frames:
                    merge_boundary_frames[dst] = set()
                # Mark frames near the merge boundary (±2 frames)
                for bf in [src_end, src_start, dst_start, dst_end]:
                    if bf is not None:
                        for offset in range(-2, 3):
                            merge_boundary_frames[dst].add(bf + offset)
                vru_df.loc[vru_df['track_id'] == src, 'track_id'] = dst
                vru_tracks.discard(src)
                track_majority_type.pop(src, None)
            # Dedup: after overlap merges, a track may have 2 rows per frame.
            # Keep the row with the smaller distance_m (closer = more reliable).
            before_dedup = len(vru_df)
            if has_dist:
                vru_df = vru_df.sort_values(['track_id', 'frame', 'distance_m'])
            vru_df = vru_df.drop_duplicates(subset=['track_id', 'frame'], keep='first')
            n_dedup = before_dedup - len(vru_df)
            if n_dedup > 0:
                print(f"  [AUTO] Dedup: removed {n_dedup} duplicate frame rows after merge")

    # ── Second merge pass: heavily overlapping tracks (same person, long overlap) ──
    # The first pass catches short gaps/slight overlaps. This pass catches cases
    # where ByteTrack creates a second ID while the first is still active, resulting
    # in extensive overlap (e.g., 5+ shared frames). Merge if median foot_x AND
    # median bbox_height at shared frames are both close.
    if has_foot and len(vru_tracks) > 1:
        remaining_tids = sorted(vru_tracks)
        overlap_merge_map = {}
        for i in range(len(remaining_tids)):
            for j in range(i + 1, len(remaining_tids)):
                tid_a = remaining_tids[i]
                tid_b = remaining_tids[j]
                if tid_a in overlap_merge_map or tid_b in overlap_merge_map:
                    continue
                tdf_a = vru_df[vru_df['track_id'] == tid_a]
                tdf_b = vru_df[vru_df['track_id'] == tid_b]
                shared = sorted(set(tdf_a['frame'].values) & set(tdf_b['frame'].values))
                if len(shared) < 5:
                    continue  # need substantial overlap
                fx_diffs = []
                for sf in shared:
                    fa = tdf_a[tdf_a['frame'] == sf]['foot_x'].values
                    fb = tdf_b[tdf_b['frame'] == sf]['foot_x'].values
                    if len(fa) > 0 and len(fb) > 0 and not (np.isnan(fa[0]) or np.isnan(fb[0])):
                        fx_diffs.append(abs(float(fa[0]) - float(fb[0])))
                if len(fx_diffs) < 3:
                    continue
                med_fx = float(np.median(fx_diffs))
                if med_fx < 30:
                    # Same person — merge shorter into longer
                    na = len(tdf_a)
                    nb = len(tdf_b)
                    src = tid_b if na >= nb else tid_a
                    dst = tid_a if na >= nb else tid_b
                    overlap_merge_map[src] = dst
                    print(f"  [AUTO] Overlap-merge T{src} → T{dst}: "
                          f"{len(shared)} shared frames, "
                          f"median foot_x diff={med_fx:.0f}px (same person)")
        if overlap_merge_map:
            vru_df = vru_df.copy()
            for src, dst in overlap_merge_map.items():
                vru_df.loc[vru_df['track_id'] == src, 'track_id'] = dst
                vru_tracks.discard(src)
                track_majority_type.pop(src, None)
                if dst not in merge_boundary_frames:
                    merge_boundary_frames[dst] = set()
            # Dedup again after overlap merges
            before_dedup = len(vru_df)
            if has_dist:
                vru_df = vru_df.sort_values(['track_id', 'frame', 'distance_m'])
            vru_df = vru_df.drop_duplicates(subset=['track_id', 'frame'], keep='first')
            n_dedup = before_dedup - len(vru_df)
            if n_dedup > 0:
                print(f"  [AUTO] Dedup: removed {n_dedup} duplicate frame rows after overlap-merge")

    # Infer frame dimensions from detection data (for edge/truncation filters)
    # NewMob: Samsung Galaxy S8 records 1080p, clips downscaled to 720p
    _frame_w = 1280
    _frame_h = 720
    if 'foot_x' in det_df.columns:
        fx_max = det_df['foot_x'].max()
        if fx_max > 1800:
            _frame_w, _frame_h = 1920, 1080
        elif fx_max > 1200:
            _frame_w, _frame_h = 1280, 720

    # Per VRU track: ONE encounter (window around the global min distance)
    encounters = []

    for tid in sorted(vru_tracks):
        track_df = vru_df[vru_df['track_id'] == tid]
        track_type = track_majority_type.get(tid, 'pedestrian')

        # Build per-frame data for this track (quality-filtered)
        track_frame_data = {}
        for _, row in track_df.iterrows():
            frame_num = int(row['frame'])
            dist = row.get('distance_m', 0)
            if dist <= 0:
                continue
            # Edge filter: monocular distance unreliable at frame edges
            foot_x = row.get('foot_x', None)
            if foot_x is not None and pd.notna(foot_x):
                fx = float(foot_x)
                if fx < EDGE_MARGIN_PX or fx > _frame_w - EDGE_MARGIN_PX:
                    continue
            # Distance cap removed — max_distance filter handles encounter
            # selection; keeping all frames lets tracks with spikes still
            # qualify when their minimum distance is within threshold.
            # Lateral filter: exclude VRUs beyond max_lateral_m (e.g. sidewalk pedestrians)
            if max_lateral_m is not None and 'lateral_m' in row.index:
                lat = row.get('lateral_m')
                if pd.notna(lat) and abs(float(lat)) > max_lateral_m:
                    continue
            # Keep minimum distance per frame (after merge, a track may have
            # multiple detections at the same frame from the absorbed track)
            new_dist = float(dist)
            if frame_num in track_frame_data:
                if new_dist >= track_frame_data[frame_num]['dist']:
                    continue  # already have a closer detection for this frame
            # Confidence and bbox for quality metrics
            conf = float(row.get('distance_confidence', 1.0)) if 'distance_confidence' in row.index and pd.notna(row.get('distance_confidence')) else 1.0
            bh = float(row.get('bbox_height', 0)) if 'bbox_height' in row.index and pd.notna(row.get('bbox_height')) else 0.0
            fy = float(row.get('foot_y', 0)) if 'foot_y' in row.index and pd.notna(row.get('foot_y')) else 0.0
            fx_val = float(foot_x) if foot_x is not None and pd.notna(foot_x) else 0.0
            lat_val = None
            if 'lateral_m' in row.index and pd.notna(row.get('lateral_m')):
                lat_val = float(row['lateral_m'])
            track_frame_data[frame_num] = {
                'dist': new_dist,
                'lateral_m': lat_val,
                'speed_kmh': frame_speed.get(frame_num, 0.0),
                'confidence': conf,
                'bbox_height': bh,
                'foot_x': fx_val,
                'foot_y': fy,
            }

        if not track_frame_data:
            continue

        # Find first detection of this track (any distance) — simply the earliest frame
        all_track_frames = sorted(det_df[det_df['track_id'] == tid]['frame'].values)
        frame_first_detection = int(all_track_frames[0]) if len(all_track_frames) > 0 else None

        # Auto-detect last valid frame: flag track switches where the tracker
        # jumps to a different person. Detects via appearance only:
        #   - Large foot_x jump (>80px) + bbox_height change (>60px) in same frame
        #     (tracker switched to a nearby but different person)
        # NOTE: distance-based detection disabled — spikes from vehicle rolling
        # are expected and handled in post-processing.
        frame_last_valid = int(all_track_frames[-1])
        # Build foot_x / bbox_height index for track-switch detection
        track_footx = {}
        track_bboxh = {}
        for _, row in track_df.iterrows():
            fn = int(row['frame'])
            if 'foot_x' in row and pd.notna(row.get('foot_x')):
                track_footx[fn] = float(row['foot_x'])
            if 'bbox_height' in row and pd.notna(row.get('bbox_height')):
                track_bboxh[fn] = float(row['bbox_height'])

        # Get merge boundary frames for this track (if any merges occurred)
        tid_merge_boundaries = merge_boundary_frames.get(tid, set())

        if len(track_frame_data) >= 2:
            sorted_fd = sorted(track_frame_data.items())
            for k in range(1, len(sorted_fd)):
                f_prev, d_prev = sorted_fd[k - 1]
                f_curr, d_curr = sorted_fd[k]
                # Skip track-switch detection at merge boundaries
                if f_prev in tid_merge_boundaries or f_curr in tid_merge_boundaries:
                    continue
                # Distance-based track switch DISABLED: distance spikes are
                # expected due to vehicle rolling and will be handled in
                # post-processing. Only use appearance-based detection.
                # Track switch via appearance: foot_x jumps >80px AND bbox_height
                # changes >60px in a single frame → tracker switched to different person
                fx_prev = track_footx.get(f_prev)
                fx_curr = track_footx.get(f_curr)
                bh_prev = track_bboxh.get(f_prev)
                bh_curr = track_bboxh.get(f_curr)
                if (fx_prev is not None and fx_curr is not None and
                    bh_prev is not None and bh_curr is not None):
                    fx_jump = abs(fx_curr - fx_prev)
                    bh_jump = abs(bh_curr - bh_prev)
                    if fx_jump > 80 and bh_jump > 60:
                        frame_last_valid = f_prev
                        break

        # Find global minimum distance for this track
        min_dist = float('inf')
        mindist_frame = None
        for f, fd in track_frame_data.items():
            if f > frame_last_valid:
                continue  # skip post-jump frames
            if fd['dist'] < min_dist:
                min_dist = fd['dist']
                mindist_frame = f

        # No threshold filtering (V2.5): every VRU track = one encounter.
        # Compute threshold for display/metadata only (not for filtering).
        if use_adaptive_threshold and mindist_frame is not None:
            mindist_speed = track_frame_data.get(mindist_frame, {}).get('speed_kmh', 0.0)
            effective_threshold = _speed_adaptive_threshold(mindist_speed)
        else:
            effective_threshold = d_threshold

        # All valid frames for this track are active (no zone filtering)
        seg_frames = sorted(f for f in track_frame_data if f <= frame_last_valid)
        if not seg_frames:
            continue

        seg_min_dist = min_dist
        seg_mindist_frame = mindist_frame

        if mindist_frame is None:
            # No valid distance for this track — skip encounter
            continue

        if True:  # single block, replaces segment loop
            # START = first detection of this track, END = last valid detection
            frame_start = frame_first_detection if frame_first_detection is not None else seg_frames[0]
            frame_end = frame_last_valid

            # Count unique VRU tracks visible in this window
            all_vru_in_window = set()
            for f in range(frame_start, frame_end + 1):
                frame_vrus = vru_df[(vru_df['frame'] == f) & (vru_df['distance_m'] > 0)]
                all_vru_in_window.update(frame_vrus['track_id'].unique())

            # Auto-compute speed: median of non-zero speeds in encounter window
            enc_speeds = []
            for f in range(frame_start, frame_end + 1):
                spd = frame_speed.get(f, 0.0)
                if spd > 0.5:
                    enc_speeds.append(spd)
            if enc_speeds:
                speed_at_encounter = round(float(np.median(enc_speeds)), 1)
            else:
                direct_speeds = []
                for f in range(frame_start, frame_end + 1):
                    fgroup = det_df[det_df['frame'] == f]
                    if len(fgroup) > 0:
                        s = fgroup.iloc[0].get('speed_kmh', 0)
                        if pd.notna(s) and float(s) > 0.5:
                            direct_speeds.append(float(s))
                speed_at_encounter = round(float(np.median(direct_speeds)), 1) if direct_speeds else 0.0

            peak_decel = 0.0
            peak_lateral_accel = 0.0
            peak_yaw = 0.0
            for f in range(frame_start, frame_end + 1):
                s = frame_sensor.get(f, {})
                peak_decel = max(peak_decel, s.get('decel', 0.0))
                peak_lateral_accel = max(peak_lateral_accel, s.get('lateral_accel', 0.0))
                peak_yaw = max(peak_yaw, s.get('yaw', 0.0))

            duration_s = (frame_end - frame_start) / fps

            # ── Rider behavior metrics ──
            reaction_time_s = None
            for f in range(frame_start, frame_end + 1):
                s = frame_sensor.get(f, {})
                if s.get('decel', 0) > 0.1 * 9.81:
                    reaction_time_s = round((f - frame_start) / fps, 2)
                    break

            speed_at_entry = frame_speed.get(frame_start, 0.0)
            pre_entry_frame = frame_start - int(3.0 * fps)
            speed_before = frame_speed.get(pre_entry_frame, speed_at_entry)
            anticipatory_reduction_kmh = round(speed_before - speed_at_entry, 1)

            acc_values = []
            acc_frames = []
            for f in range(frame_start, frame_end + 1):
                s = frame_sensor.get(f, {})
                if 'decel' in s:
                    acc_values.append(s['decel'])
                    acc_frames.append(f)
            if len(acc_values) >= 3:
                acc_arr = np.array(acc_values)
                dt_arr = np.diff(acc_frames) / fps
                dt_arr[dt_arr < 1e-6] = 1.0 / fps
                jerk_arr = np.abs(np.diff(acc_arr) / dt_arr)
                mean_jerk = round(float(np.mean(jerk_arr)), 2)
            else:
                mean_jerk = None

            frame_perception = frame_start + int(PERCEPTION_DELAY_S * fps)
            frame_perception = min(frame_perception, seg_mindist_frame)

            # Compute minimum lateral distance across encounter frames
            _lat_vals = [abs(track_frame_data[f]['lateral_m'])
                         for f in seg_frames
                         if f in track_frame_data
                         and track_frame_data[f].get('lateral_m') is not None]
            _min_lat = round(min(_lat_vals), 2) if _lat_vals else None

            enc = {
                'idx': len(encounters),
                'frame_start': frame_start,
                'frame_perception': frame_perception,
                'frame_mindist': seg_mindist_frame,
                'frame_end': frame_end,
                'ts_start': frame_start / fps,
                'ts_perception': frame_perception / fps,
                'ts_mindist': seg_mindist_frame / fps,
                'ts_end': frame_end / fps,
                'min_dist': round(seg_min_dist, 2),
                'min_lateral_m': _min_lat,
                'primary_track': tid,
                'primary_type': track_type,
                'vru_type_code': {'pedestrian': 1, 'cyclist': 2}.get(track_type, 9),
                'vru_count': len(all_vru_in_window),
                'speed_kmh': speed_at_encounter,
                'frame_first_detection': frame_first_detection,
                'frame_last_valid': frame_last_valid,
                'peak_decel_ms2': round(peak_decel, 2),
                'peak_lateral_accel_ms2': round(peak_lateral_accel, 2),
                'peak_yaw_deg_s': round(peak_yaw, 1),
                'duration_s': round(duration_s, 1),
                'reaction_time_s': reaction_time_s,
                'anticipatory_reduction_kmh': anticipatory_reduction_kmh,
                'mean_jerk_ms3': mean_jerk,
                'status': 'pending',
                'codes': OrderedDict([
                    (k, v.get("default")) for k, v in MANUAL_VARIABLES.items()
                ]),
                'notes': '',
                'note_timestamps': [],
                'interaction_zone_m': round(effective_threshold, 1),
                '_assumed_height_m': VRU_HEIGHT_MAP.get(track_type, DEFAULT_ASSUMED_HEIGHT),
                '_distance_bias_pct': round(
                    (DEFAULT_ASSUMED_HEIGHT / VRU_HEIGHT_MAP.get(track_type, DEFAULT_ASSUMED_HEIGHT) - 1) * 100, 1
                ),
            }
            # Pre-fill VRU_TYPE from auto-detected type
            enc['codes']['VRU_TYPE'] = {'pedestrian': 1, 'cyclist': 2}.get(track_type, 9)
            # EMA class probability: how confident is the classification?
            cp = track_class_prob.get(tid, {})
            enc['class_prob'] = round(cp.get(track_type, 1.0), 3)

            # IMU-based confidence: encounter has kinematic confirmation if
            # peak deceleration > 0.5 m/s² or peak yaw > 5°/s within window
            enc['imu_confirmed'] = (peak_decel > 0.5 or peak_yaw > 5.0)

            # VRU speed estimation
            vru_spd = estimate_vru_speed(track_df, fps)
            if vru_spd:
                enc['vru_speed_mps'] = vru_spd['vru_speed_mps']
                enc['vru_speed_kmh'] = vru_spd['vru_speed_kmh']
                enc['vru_movement'] = vru_spd['classification']
                # VRU_GAIT: post-processing metadata for pedestrians
                # Walk-run transition at 2.0 m/s (Hreljac 1993; Rotstein 2005)
                # Gait stored as metadata; VRU_TYPE no longer splits walk/run
                gait_map = {'stationary': 1, 'walker': 2, 'runner': 3}
                enc['vru_gait'] = gait_map.get(vru_spd['classification'], 9)
            else:
                enc['vru_speed_mps'] = None
                enc['vru_speed_kmh'] = None
                enc['vru_movement'] = 'unknown'
                enc['vru_gait'] = 9

            # Auto-classify MMV (motorized micro-vehicle) riders:
            # YOLO detects e-scooter riders as 'pedestrian', but their speed
            # gives them away. Requires BOTH:
            #   - VRU speed > 20 km/h (high enough to filter noisy pedestrian estimates)
            #   - Track has >= 80 frames (~2.7s at 30fps, sustained approach)
            # Shorter tracks often have inflated VRU speed from noisy
            # monocular distance derivatives — these are walking pedestrians.
            enc['auto_vru_class'] = track_type  # default: YOLO majority type
            n_track_frames = len(track_df)
            if (enc.get('vru_speed_kmh') and enc['vru_speed_kmh'] > 20.0
                    and track_type == 'pedestrian' and n_track_frames >= 80):
                enc['auto_vru_class'] = 'mmv_rider'
                enc['_assumed_height_m'] = 1.80  # e-scooter adds ~10cm
                # Apply e-scooter deck foot offset correction (10cm above ground)
                # d_true = d_measured * (h_cam - deck_h) / h_cam
                deck_corr = (camera_height - ESCOOTER_DECK_HEIGHT_M) / camera_height
                enc['min_dist'] = round(enc['min_dist'] * deck_corr, 2)
                if enc.get('min_lateral_m') is not None:
                    enc['min_lateral_m'] = round(enc['min_lateral_m'] * deck_corr, 2)
                enc['_escooter_deck_correction'] = round(deck_corr, 4)
                enc['_distance_bias_pct'] = round((deck_corr - 1) * 100, 1)
            # Store average bbox height for manual review
            bh_vals = track_df['bbox_height'].dropna().values if 'bbox_height' in track_df.columns else []
            enc['_avg_bbox_height'] = round(float(np.mean(bh_vals)), 1) if len(bh_vals) > 0 else None
            enc['max_bbox_height'] = round(float(np.max(bh_vals)), 1) if len(bh_vals) > 0 else 0.0

            # ── Auto-suggest INTERACTION_TYPE (Beitel et al. 2018: 30-degree rule) ──
            # Classify dominant approach geometry from approach-phase kinematics.
            # Uses foot_x drift (crossing), closing rate vs ego speed (same/opposite),
            # and VRU stationarity. The suggestion is shown to the rater who confirms
            # or overrides via number key.
            #
            # Approach phase = frames from encounter start to min-distance frame.
            approach_frames = sorted(
                f for f in seg_frames
                if f <= seg_mindist_frame and f in track_frame_data
            )
            suggested_itype = 9  # default: Unknown
            if len(approach_frames) >= 3:
                # 1. Check stationarity from VRU speed estimation
                if enc.get('vru_movement') == 'stationary':
                    suggested_itype = 4  # Stationary VRU
                else:
                    # 2. Foot_x drift: detect crossing motion
                    # If foot_x moves >100px across the approach phase, VRU
                    # is traversing the ego path laterally (crossing geometry).
                    fx_vals = [track_frame_data[f]['foot_x'] for f in approach_frames
                               if track_frame_data[f].get('foot_x', 0) > 0]
                    fx_drift = 0.0
                    if len(fx_vals) >= 3:
                        fx_arr = np.array(fx_vals)
                        fx_drift = abs(float(fx_arr[-1] - fx_arr[0]))

                    # 3. Closing rate: how fast is distance decreasing?
                    # d_dot = (d_end - d_start) / dt, negative = closing
                    d_start = track_frame_data[approach_frames[0]]['dist']
                    d_end = track_frame_data[approach_frames[-1]]['dist']
                    dt_approach = (approach_frames[-1] - approach_frames[0]) / fps
                    if dt_approach > 0.05:
                        closing_rate_mps = -(d_end - d_start) / dt_approach
                    else:
                        closing_rate_mps = 0.0

                    # Median ego speed during approach (m/s)
                    ego_speeds_approach = [
                        track_frame_data[f].get('speed_kmh', 0.0)
                        for f in approach_frames
                    ]
                    median_ego_mps = (float(np.median(ego_speeds_approach)) / 3.6
                                      if ego_speeds_approach else 0.0)

                    # Classification:
                    # Crossing: large lateral drift (foot_x moves >100px)
                    if fx_drift > 100:
                        suggested_itype = 3  # Crossing
                    # Opposite direction: closing rate significantly exceeds
                    # ego speed (VRU moving toward ego).
                    elif closing_rate_mps > 0 and median_ego_mps > 0.1:
                        if closing_rate_mps > median_ego_mps + 0.5:
                            suggested_itype = 2  # Opposite direction
                        else:
                            suggested_itype = 1  # Same direction
                    elif closing_rate_mps <= 0 and median_ego_mps > 0.5:
                        # Distance increasing: ego passed VRU or both same dir
                        suggested_itype = 1  # Same direction
                    # else: insufficient signal -> Unknown (9)
            elif enc.get('vru_movement') == 'stationary':
                # Short track but VRU confirmed stationary
                suggested_itype = 4

            enc['suggested_interaction_type'] = suggested_itype
            # Pre-fill INTERACTION_TYPE from auto-suggestion
            enc['codes']['INTERACTION_TYPE'] = suggested_itype

            # Distance confidence: min confidence across encounter frames
            # Low confidence = small bbox → unreliable pinhole distance
            seg_confs = [track_frame_data[f]['confidence'] for f in seg_frames if f in track_frame_data]
            enc['min_distance_confidence'] = round(min(seg_confs), 3) if seg_confs else None
            enc['mean_confidence'] = round(float(np.mean(seg_confs)), 4) if seg_confs else 0.0

            # Truncation flag: VRU bbox touches frame edge at any encounter frame
            # Truncated VRUs have unreliable bbox height → unreliable distance
            FRAME_W, FRAME_H = _frame_w, _frame_h
            TRUNC_MARGIN = 5
            n_truncated = 0
            for f in seg_frames:
                fd = track_frame_data.get(f)
                if fd is None:
                    continue
                bbox_top = fd['foot_y'] - fd['bbox_height']
                if (bbox_top < TRUNC_MARGIN or fd['foot_y'] > FRAME_H - TRUNC_MARGIN
                        or fd['foot_x'] < TRUNC_MARGIN or fd['foot_x'] > FRAME_W - TRUNC_MARGIN):
                    n_truncated += 1
            enc['n_truncated_frames'] = n_truncated
            enc['truncated'] = n_truncated > 0

            # TTC_approx: time-to-collision at perception frame
            # TTC = d(t) / |ḋ(t)| when ḋ < 0 (closing approach)
            # Uses 3-frame central difference on smoothed distance
            ttc_approx = None
            perc_f = enc['frame_perception']
            seg_sorted = sorted(seg_frames)
            # Find frames around perception for derivative
            perc_idx = None
            for pi, sf in enumerate(seg_sorted):
                if sf >= perc_f:
                    perc_idx = pi
                    break
            if perc_idx is not None and perc_idx >= 1 and perc_idx < len(seg_sorted) - 1:
                f_before = seg_sorted[perc_idx - 1]
                f_after = seg_sorted[perc_idx + 1]
                f_now = seg_sorted[perc_idx]
                d_before = track_frame_data.get(f_before, {}).get('dist')
                d_after = track_frame_data.get(f_after, {}).get('dist')
                d_now = track_frame_data.get(f_now, {}).get('dist')
                if d_before is not None and d_after is not None and d_now is not None:
                    dt = (f_after - f_before) / fps
                    if dt > 0:
                        d_dot = (d_after - d_before) / dt  # m/s, negative = closing
                        if d_dot < -0.1:  # only compute TTC for closing approach
                            ttc_approx = round(d_now / abs(d_dot), 2)
            enc['ttc_approx_s'] = ttc_approx

            # DRAC: Deceleration Rate to Avoid Collision
            # DRAC = v_closing² / (2 * d)  [m/s²]
            # Computed at perception frame using the same closing rate as TTC.
            # Interpretation: required constant deceleration to stop before
            # reaching the VRU. Higher = more dangerous.
            # Reference: Archer (2005), Hydén (1987)
            drac = None
            # Reuse the d_dot computed above for TTC (at perception frame)
            if perc_idx is not None and perc_idx >= 1 and perc_idx < len(seg_sorted) - 1:
                f_before_d = seg_sorted[perc_idx - 1]
                f_after_d = seg_sorted[perc_idx + 1]
                f_now_d = seg_sorted[perc_idx]
                db = track_frame_data.get(f_before_d, {}).get('dist')
                da = track_frame_data.get(f_after_d, {}).get('dist')
                dn = track_frame_data.get(f_now_d, {}).get('dist')
                if db is not None and da is not None and dn is not None:
                    dt_d = (f_after_d - f_before_d) / fps
                    if dt_d > 0 and dn > 0.01:
                        d_dot_d = (da - db) / dt_d
                        if d_dot_d < -0.1:  # closing approach
                            v_closing = abs(d_dot_d)
                            drac = round(v_closing ** 2 / (2.0 * dn), 2)
            enc['drac_ms2'] = drac
            enc['drac_capped'] = drac is not None and drac > 20.0

            # THW: Time Headway = d / v_ego  [seconds]
            # Computed per frame, store min and value at perception.
            thw_perc = None
            if speed_at_encounter and speed_at_encounter > 0.5:
                speed_mps = speed_at_encounter / 3.6
                perc_dist = track_frame_data.get(frame_perception, {}).get('dist')
                if perc_dist is not None and speed_mps > 0.1:
                    thw_perc = round(perc_dist / speed_mps, 2)

            min_thw = float('inf')
            for f in seg_frames:
                fd = track_frame_data.get(f)
                if fd and fd.get('speed_kmh') and fd['speed_kmh'] > 0.5:
                    spd = fd['speed_kmh'] / 3.6
                    if spd > 0.1:
                        min_thw = min(min_thw, fd['dist'] / spd)
            min_thw = round(min_thw, 2) if min_thw < float('inf') else None

            enc['thw_perception_s'] = thw_perc
            enc['min_thw_s'] = min_thw

            # ── Crossing direction from lateral_m trajectory ──
            # Determines if VRU crossed the rider's path (R_to_L, L_to_R, same_side)
            # based on lateral displacement between track start and end.
            lat_vals = [track_frame_data[f]['lateral_m']
                        for f in sorted(track_frame_data.keys())
                        if track_frame_data[f].get('lateral_m') is not None]
            if len(lat_vals) >= 3:
                lat_start = float(np.mean(lat_vals[:3]))
                lat_end = float(np.mean(lat_vals[-3:]))
                delta_lat = lat_end - lat_start
                if abs(delta_lat) > 1.0:  # >1m lateral shift = crossing
                    crossing_dir = 'R_to_L' if delta_lat < 0 else 'L_to_R'
                else:
                    crossing_dir = 'same_side'
            else:
                crossing_dir = 'unknown'
            enc['crossing_dir'] = crossing_dir

            encounters.append(enc)

    # ── Trim far-VRU encounters ──
    # If a closer VRU has its mindist within a farther VRU's active window,
    # trim the farther VRU's encounter to start after the closer VRU's mindist.
    # Rationale: the rider's attention is on the nearest VRU first.
    encounters.sort(key=lambda e: e['min_dist'])  # closest first
    trimmed = []
    for i, enc in enumerate(encounters):
        # Find the latest mindist frame of any closer encounter that overlaps
        trim_after = None
        for j in range(i):
            closer = encounters[j]
            # Does the closer encounter's mindist fall within this encounter's window?
            if closer['frame_mindist'] >= enc['frame_start'] and closer['frame_mindist'] <= enc['frame_end']:
                if trim_after is None or closer['frame_mindist'] > trim_after:
                    trim_after = closer['frame_mindist']
        if trim_after is not None and trim_after >= enc['frame_mindist']:
            # The closer VRU's mindist is AFTER this VRU's mindist — no trim needed
            pass
        elif trim_after is not None:
            # Trim: this encounter starts after the closer VRU's mindist
            enc['frame_start'] = trim_after + 1
            enc['duration_s'] = round((enc['frame_end'] - enc['frame_start']) / fps, 1)
            if enc['frame_start'] > enc['frame_end']:
                continue  # trimmed to nothing — drop
        trimmed.append(enc)
    encounters = trimmed

    # Fragment filter removed (V2.5): all tracks shown, rater decides via CONFIRM_INTERACTION

    # Compute track flags for each encounter
    for enc in encounters:
        enc['flags'], enc['swap_frame'] = compute_track_flags(det_df, enc['primary_track'])

    # No VRU cap — let the annotator see all encounters and skip irrelevant ones

    # DENSITY_SECONDARY flag (V3.6): per-frame density filtering.
    # When >K VRUs simultaneously within max_distance, only nearest N for
    # manual coding. Rest auto-filled. If >50% frames filtered -> secondary.
    # density_rank = average distance rank across frames (1=nearest).
    _fvd_idx = {}
    for _, _dr in vru_df.iterrows():
        _dfr = int(_dr['frame'])
        _ddi = _dr.get('distance_m', 0)
        _dti = _dr['track_id']
        if _ddi > 0 and (max_distance is None or _ddi <= max_distance):
            _fvd_idx.setdefault(_dfr, {})
            if _dti not in _fvd_idx[_dfr] or _ddi < _fvd_idx[_dfr][_dti]:
                _fvd_idx[_dfr][_dti] = _ddi
    for enc in encounters:
        _et = enc['primary_track']
        _es, _ee = enc['frame_start'], enc['frame_end']
        _nft, _nff, _rs, _rc, _ms = 0, 0, 0.0, 0, 0
        for _fr in range(_es, _ee + 1):
            _fd = _fvd_idx.get(_fr, {})
            _ns = len(_fd)
            _ms = max(_ms, _ns)
            if _et not in _fd:
                continue
            _nft += 1
            _sd = sorted(_fd.items(), key=lambda x: x[1])
            _rk = next(i + 1 for i, (t, _) in enumerate(_sd) if t == _et)
            _rs += _rk
            _rc += 1
            if _ns > dense_scene_k and _rk > dense_scene_n:
                _nff += 1
        enc['n_simultaneous_vrus'] = _ms
        enc['density_rank'] = round(_rs / _rc, 1) if _rc > 0 else 1.0
        enc['density_secondary'] = (_nft > 0 and _nff / _nft > 0.5)
    _nsec = sum(1 for e in encounters if e.get('density_secondary'))
    if _nsec > 0:
        print(f"  [DENSE] {_nsec}/{len(encounters)} encounters marked as "
              f"density_secondary (K={dense_scene_k}, N={dense_scene_n}), "
              f"{len(encounters) - _nsec} for manual coding")

    # Detect same-user links across tracks
    links = detect_same_user_links(encounters, det_df, fps)
    for enc in encounters:
        tid = enc['primary_track']
        if tid in links:
            enc['linked_tracks'] = sorted(links[tid])
        else:
            enc['linked_tracks'] = []

    # ── Contextual vehicles ──
    # For each VRU encounter, find motor vehicle tracks present in the same
    # frame window.  These are NOT interaction targets but obstacles that may
    # explain the ego-rider's behaviour (swerving, braking, path change).
    vehicle_df = det_df[det_df['user_type'].isin(motor_types)].copy() if 'user_type' in det_df.columns else pd.DataFrame()
    if len(vehicle_df) > 0:
        # Pre-index vehicle frames for speed
        veh_by_frame = {}
        for _, vrow in vehicle_df.iterrows():
            f = int(vrow['frame'])
            veh_by_frame.setdefault(f, []).append(vrow)

        for enc in encounters:
            ctx_vehicles = []
            seen_vtids = set()
            for f in range(enc['frame_start'], enc['frame_end'] + 1):
                for vrow in veh_by_frame.get(f, []):
                    vtid = int(vrow['track_id'])
                    if vtid in seen_vtids:
                        continue
                    seen_vtids.add(vtid)
                    vt = vehicle_df[vehicle_df['track_id'] == vtid]
                    vt_in_window = vt[(vt['frame'] >= enc['frame_start']) & (vt['frame'] <= enc['frame_end'])]
                    if len(vt_in_window) > 0:
                        ctx_vehicles.append({
                            'track_id': vtid,
                            'type': vrow.get('user_type', 'vehicle'),
                            'min_dist': round(float(vt_in_window['distance_m'].min()), 2),
                            'n_frames': len(vt_in_window),
                        })
            enc['contextual_vehicles'] = ctx_vehicles
    else:
        for enc in encounters:
            enc['contextual_vehicles'] = []

    # Build track summary (one row per VRU track) before filtering
    track_summary = []
    _summary_by_tid = {}
    for enc in encounters:
        tid = enc['primary_track']
        row = {
            'track_id': tid,
            'frame_start': enc['frame_start'],
            'frame_end': enc['frame_end'],
            'num_frames': enc.get('num_frames', enc['frame_end'] - enc['frame_start'] + 1),
            'min_distance': round(enc['min_dist'], 2),
            'min_THW': enc.get('min_thw_s'),
            'mean_confidence': round(enc.get('mean_confidence', 0.0), 4),
            'max_bbox_height': round(enc.get('max_bbox_height', 0.0), 1),
            'passed_filter': True,
        }
        track_summary.append(row)
        _summary_by_tid[tid] = row

    # THW pre-filter: exclude encounters where min(THW) > threshold
    if thw_threshold is not None:
        before = len(encounters)
        filtered = []
        for enc in encounters:
            thw = enc.get('min_thw_s')
            if thw is None or thw <= thw_threshold:
                enc['thw_filtered'] = False
                filtered.append(enc)
            else:
                enc['thw_filtered'] = True
                tid = enc['primary_track']
                if tid in _summary_by_tid:
                    _summary_by_tid[tid]['passed_filter'] = False
        encounters = filtered
        n_removed = before - len(encounters)
        if n_removed > 0:
            print(f"  [THW] Filtered {n_removed}/{before} encounters (THW > {thw_threshold}s)")

    # Distance pre-filter: exclude encounters where min_dist > max_distance
    if max_distance is not None:
        before_dist = len(encounters)
        filtered_dist = []
        for enc in encounters:
            if enc['min_dist'] <= max_distance:
                filtered_dist.append(enc)
            else:
                tid = enc['primary_track']
                if tid in _summary_by_tid:
                    _summary_by_tid[tid]['passed_filter'] = False
        encounters = filtered_dist
        n_dist_removed = before_dist - len(encounters)
        if n_dist_removed > 0:
            print(f"  [DIST] Filtered {n_dist_removed}/{before_dist} encounters "
                  f"(min_dist > {max_distance}m)")

    # Ego speed pre-filter: exclude encounters where rider speed < threshold
    if min_ego_speed_kmh and min_ego_speed_kmh > 0:
        before_spd = len(encounters)
        filtered_spd = []
        for enc in encounters:
            if enc.get('speed_kmh', 0.0) >= min_ego_speed_kmh:
                filtered_spd.append(enc)
            else:
                tid = enc['primary_track']
                if tid in _summary_by_tid:
                    _summary_by_tid[tid]['passed_filter'] = False
        encounters = filtered_spd
        n_spd_removed = before_spd - len(encounters)
        if n_spd_removed > 0:
            print(f"  [EGO-SPD] Filtered {n_spd_removed}/{before_spd} encounters "
                  f"(ego speed < {min_ego_speed_kmh} km/h)")

    # Fleeting-far filter: remove short encounters at long range.
    # These are typically edge-of-frame VRUs appearing for a few frames
    # at far distances — noisy, hard to annotate, rarely interacting.
    # Rule: duration < 0.5s AND min_dist > 12m → drop
    before_ff = len(encounters)
    filtered_ff = []
    for enc in encounters:
        if enc['duration_s'] < 0.5 and enc['min_dist'] > 12.0:
            tid = enc['primary_track']
            if tid in _summary_by_tid:
                _summary_by_tid[tid]['passed_filter'] = False
        else:
            filtered_ff.append(enc)
    encounters = filtered_ff
    n_ff_removed = before_ff - len(encounters)
    if n_ff_removed > 0:
        print(f"  [FLEETING] Filtered {n_ff_removed}/{before_ff} encounters "
              f"(duration < 0.5s AND min_dist > 12m)")

    # Sort by frame_mindist and re-index
    encounters.sort(key=lambda e: e['frame_mindist'])
    for i, enc in enumerate(encounters):
        enc['idx'] = i

    return encounters, track_summary


def flag_constrained_zones(encounters, constrained_zones_path, fps=30.0):
    """Flag encounters occurring within manually-marked constrained path zones.

    Constrained zones CSV format:
        start_frame,end_frame,zone_type,description
        1200,1450,terrasse,Café terrace narrows path
        2800,3100,bollard,Bollard chicane

    Each encounter gets:
        constrained_path: bool — True if mindist frame is inside a zone
        constrained_zone_type: str — zone_type label (or '' if not constrained)
        constrained_zone_desc: str — description (or '')
    """
    zones = []
    if constrained_zones_path and os.path.exists(constrained_zones_path):
        with open(constrained_zones_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                zones.append({
                    'start': int(row['start_frame']),
                    'end': int(row['end_frame']),
                    'type': row.get('zone_type', 'constrained'),
                    'desc': row.get('description', ''),
                })
        print(f"  [ZONES] Loaded {len(zones)} constrained zone(s) from {constrained_zones_path}")

    for enc in encounters:
        mf = enc['frame_mindist']
        matched = None
        for z in zones:
            if z['start'] <= mf <= z['end']:
                matched = z
                break
        enc['constrained_path'] = matched is not None
        enc['constrained_zone_type'] = matched['type'] if matched else ''
        enc['constrained_zone_desc'] = matched['desc'] if matched else ''


def suggest_severity(peak_decel, min_dist, peak_yaw=None):
    """Suggest severity code from deceleration and yaw rate.

    Codebook V2.5 Section 9.1: SEVERITY = max(severity_from_AccX, severity_from_GyrZ).
    Distance branch removed — distance is not a reliable severity proxy
    (confounded by path width, VRU speed, and approach geometry).
    """
    sev = 1
    if peak_decel is not None:
        for thresh, s in SEVERITY_THRESHOLDS["decel"]:
            if peak_decel >= thresh:
                sev = max(sev, s)
                break
    if peak_yaw is not None:
        for thresh, s in SEVERITY_THRESHOLDS["yaw"]:
            if peak_yaw >= thresh:
                sev = max(sev, s)
                break
    return sev


# ═══════════════════════════════════════════════════════════════════
# ZONE LOOKUP (Phase 5)
# ═══════════════════════════════════════════════════════════════════

def load_zone_segments(zones_path, trip_id=None):
    """Load pedestrian zone segments CSV and return zone info.

    Returns dict: {trip_id: [{'zone_name', 'zone_type', 'city', ...}, ...]}
    """
    if not zones_path or not os.path.exists(zones_path):
        return {}

    df = pd.read_csv(zones_path)
    zones = {}
    for _, row in df.iterrows():
        tid = row.get('trip_id', '')
        if tid not in zones:
            zones[tid] = []
        zone_type_str = str(row.get('zone_type', '')).lower()
        # Map zone_type string to code
        zt_map = {
            'pedestrian': 1, 'shared_path': 2, 'shared path': 2,
            'living_street': 3, 'living street': 3,
            'park': 4, 'quay': 5, 'berges': 5,
            'sidewalk': 6,
        }
        zone_code = zt_map.get(zone_type_str, 9)
        zones[tid].append({
            'zone_name': row.get('zone_name', ''),
            'zone_type': zone_code,
            'zone_type_raw': zone_type_str,
            'city': row.get('city', ''),
            'osm_id': row.get('osm_id', ''),
        })
    return zones


# ═══════════════════════════════════════════════════════════════════
# VIDEO FILENAME PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_video_filename(video_path):
    """Extract trip info from NewMob video filename.

    Expected pattern: v3_326-336_2023-04-27__19_14_03.mp4
    Returns (trip_id_prefix, clip_offset_s, clip_start_epoch_ms).
    """
    stem = Path(video_path).stem
    clip_offset_s = 0
    m_offset = re.search(r'_(\d+)-(\d+)_', stem)
    if m_offset:
        clip_offset_s = int(m_offset.group(1))

    clip_start_ms = None
    m = re.search(r'(\d{4}-\d{2}-\d{2})__(\d{2}_\d{2}_\d{2})', stem)
    if m:
        date_str = m.group(1)
        time_str = m.group(2).replace('_', ':')
        try:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            trip_start_ms = int(dt.timestamp() * 1000)
            clip_start_ms = trip_start_ms + clip_offset_s * 1000
        except ValueError:
            pass

    return stem, clip_offset_s, clip_start_ms


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION (Phase 6)
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# MULTI-REFERENCE CALIBRATION (Nelder-Mead solver)
# ═══════════════════════════════════════════════════════════════════

# Road marking standards (French NF P 98-011)
MARKING_LENGTHS = {
    1: (3.00, "Dashed line (3.0m)"),
    2: (0.50, "Crossing stripe (0.5m)"),
    3: (3.00, "Lane width (3.0m)"),
}


def pixel_to_ground(u, v, f_px, cx, cy, h, pitch_deg):
    """Map pixel (u, v) to ground coordinates (X, Y) in meters."""
    horizon_v = cy - f_px * np.tan(np.radians(pitch_deg))
    dv = v - horizon_v
    if dv <= 0:
        return float('inf'), float('inf')
    Y = f_px * h / dv
    X = (u - cx) * h / dv
    return X, Y


def ground_distance_between(p1, p2, f_px, cx, cy, h, pitch_deg):
    """Compute ground distance between two pixel points."""
    X1, Y1 = pixel_to_ground(p1[0], p1[1], f_px, cx, cy, h, pitch_deg)
    X2, Y2 = pixel_to_ground(p2[0], p2[1], f_px, cx, cy, h, pitch_deg)
    if Y1 == float('inf') or Y2 == float('inf'):
        return float('inf')
    return np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)


def solve_marking_calibration(line_data, f_px, cx, cy):
    """Solve for (camera_height, pitch) from reference marking measurements.

    line_data: list of ((u1,v1), (u2,v2), target_length_m)
    Returns: (h, pitch_deg, rmse) or None on failure.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("    [CAL] ERROR: scipy not installed. Run: pip install scipy")
        return None

    def objective(params):
        h, pitch_deg = params
        if h < 0.75 or h > 1.35:
            return 1e10
        if pitch_deg < -5 or pitch_deg > 15:
            return 1e10
        total = 0.0
        for p1, p2, target in line_data:
            dist = ground_distance_between(p1, p2, f_px, cx, cy, h, pitch_deg)
            if dist == float('inf'):
                return 1e10
            total += (dist - target) ** 2
        return total

    best_result = None
    best_error = float('inf')
    for h_init in [0.85, 0.95, 1.05, 1.15, 1.25]:
        for pitch_init in [1, 3, 5, 7]:
            result = minimize(objective, x0=[h_init, pitch_init],
                              method='Nelder-Mead',
                              options={'xatol': 0.001, 'fatol': 0.0001})
            if result.fun < best_error:
                best_error = result.fun
                best_result = result

    if best_result is None or best_error > 1e8:
        return None

    h, pitch = best_result.x
    rmse = np.sqrt(best_error / len(line_data))
    return h, pitch, rmse


# ═══════════════════════════════════════════════════════════════════
# MAIN ANNOTATION TOOL
# ═══════════════════════════════════════════════════════════════════

class EncounterAnnotator:
    """OpenCV-based encounter annotation tool v3 with auto-detection."""

    # States
    ENCOUNTER_LIST = "encounter_list"
    ENCOUNTER_VIEW = "encounter_view"
    CODING = "coding"
    REVIEW = "review"
    INTERACTION_GROUPING = "interaction_grouping"
    GROUP_CODING = "group_coding"
    TRIP_ANNOTATION = "trip_annotation"
    RIDER_SEGMENT = "rider_segment"
    RIDER_SEGMENT_CODING = "rider_segment_coding"
    DONE = "done"
    OBSTACLE_MARKING = "obstacle_marking"
    CALIBRATION_PHASE = "calibration_phase"
    # Keep old name as alias for compatibility
    SAME_USER_GROUPING = INTERACTION_GROUPING

    def __init__(self, video_path, detections_path, output_path=None,
                 zones_path=None, trip_id="", city="", rater_id=1,
                 rider_id=None,
                 camera_height=1.1, focal_length=None, pitch=0.0,
                 ped_height=1.70, calibration_path=None,
                 speed_offset_s=None, recompute_distances=True,
                 max_lateral_m=None, d_threshold=None,
                 constrained_zones_path=None, record_path=None,
                 no_smooth=False, thw_threshold=None, max_distance=15.0,
                 min_ego_speed_kmh=0.0,
                 no_resume=False, annotation_fps=None, fps_display=None,
                 min_zone_gap_s=2.0, basket_mask=None,
                 dense_scene_k=5, dense_scene_n=3,
                 max_encounters=None):
        # Session resume
        self.no_resume = no_resume
        # Zone merge gap threshold (seconds)
        self.min_zone_gap_s = min_zone_gap_s if min_zone_gap_s is not None else 2.0
        # Basket/handlebar mask region (x1, y1, x2, y2) or None
        self.basket_mask = basket_mask
        # Max encounters to show (quality-sorted, None = all)
        self.max_encounters = max_encounters

        # Video
        self.record_path = record_path
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Auto-scale focal length from 1080p base to actual video resolution
        if focal_length is None:
            focal_length = 1445 * self.height / 1080

        self.duration_total = self.total_frames / self.fps
        # Annotation FPS: step multiple frames per keypress to reduce workload
        if annotation_fps is None or annotation_fps >= self.fps:
            self.annotation_fps = self.fps
            self.frame_step = 1
        else:
            self.annotation_fps = annotation_fps
            self.frame_step = max(1, round(self.fps / annotation_fps))

        # Display FPS: reduce display framerate for annotation comfort (vibration reduction)
        # When set, only display every Nth frame; all frames still processed internally
        self.fps_display = fps_display
        if fps_display and fps_display < self.fps:
            self.frame_step = max(1, round(self.fps / fps_display))
            self.annotation_fps = fps_display

        if self.frame_step > 1:
            print(f"  Annotation FPS: {self.annotation_fps:.0f} (step={self.frame_step} frames)")
        self._strong_steering_episodes = []  # Will be computed from IMU after sensor load

        # Camera calibration
        self.camera_height_m = camera_height
        self.focal_length_px = focal_length
        self.pitch_deg = pitch
        self.ped_height_m = ped_height
        self.calibration_factor = 1.0

        # Auto-discover calibration: look in output_dir first, then next to video
        # This allows working with read-only video directories (external disks)
        if not calibration_path:
            stem = Path(video_path).stem
            video_dir = os.path.dirname(os.path.abspath(video_path))

            # Priority 1: output directory (for read-only video locations)
            if output_path:
                output_dir = os.path.dirname(os.path.abspath(output_path))
                output_cal = os.path.join(output_dir, f"{stem}_calibration.json")
                if os.path.exists(output_cal):
                    calibration_path = output_cal
                    print(f"  Auto-discovered calibration (output dir): {output_cal}")

            # Priority 2: next to video (original behavior)
            if not calibration_path:
                auto_cal = os.path.join(video_dir, f"{stem}_calibration.json")
                if os.path.exists(auto_cal):
                    calibration_path = auto_cal
                    print(f"  Auto-discovered calibration (video dir): {auto_cal}")

        self.calibration_from_file = False  # Track if calibration was loaded from JSON
        self.calibration_source = None       # Path of loaded calibration file
        if calibration_path and os.path.exists(calibration_path):
            with open(calibration_path) as f:
                cal = json.load(f)
            self.camera_height_m = cal.get('camera_height_m', camera_height)
            self.pitch_deg = cal.get('pitch_deg', pitch)
            self.calibration_factor = cal.get('distance_calibration_factor', 1.0)
            if 'focal_length_px' in cal:
                self.focal_length_px = cal['focal_length_px']
            if 'assumed_height_m' in cal:
                self.ped_height_m = cal['assumed_height_m']
            self.calibration_from_file = True
            self.calibration_source = calibration_path
            cal_method = cal.get('calibration_method', 'unknown')
            cal_ts = cal.get('timestamp', '')
            print(f"  Calibration loaded from file: h={self.camera_height_m:.2f}m, "
                  f"pitch={self.pitch_deg:.1f}deg, f={self.focal_length_px:.0f}px "
                  f"(method={cal_method}, saved={cal_ts})")

        # Parse video filename
        self.video_stem, self.clip_offset_s, self.clip_start_ms = parse_video_filename(video_path)
        # Sanitize stem: Private Use Area chars (\uf020-\uf029) cause Windows file I/O errors
        self.video_stem = re.sub(r'[\uf020-\uf029]', '_', self.video_stem)

        print(f"Video: {video_path}")
        print(f"  {self.width}x{self.height} @ {self.fps:.1f}fps, "
              f"{self.total_frames} frames, {self.duration_total:.1f}s")
        print(f"  Camera: h={self.camera_height_m:.2f}m, pitch={self.pitch_deg:.1f}°, "
              f"f={self.focal_length_px:.0f}px")

        # Load detections and apply RTS smoothing
        self.det_df = pd.read_csv(detections_path)

        # Keep homography distance_m as-is (even when uncalibrated).
        # Previous behaviour switched to height-based when no calibration JSON,
        # but user wants homography consistently so that calibration (key 6)
        # improves it rather than silently switching methods.
        if ('distance_height_m' in self.det_df.columns
                and not self.calibration_from_file):
            n_valid = self.det_df['distance_height_m'].notna().sum()
            if n_valid > 0:
                print(f"  [DIST] Keeping homography distance_m (no calibration file). "
                      f"distance_height_m available as backup ({n_valid} rows). "
                      f"Calibrate pitch with key 6 for best accuracy.")

        # Recompute distances from foot_y if explicitly requested
        # (Default: keep CSV distances — they may come from IMU pipeline with per-frame pitch)
        if recompute_distances and 'foot_y' in self.det_df.columns:
            h = self.camera_height_m
            f_px = self.focal_length_px
            cx = self.width / 2.0
            cy = self.height / 2.0
            # Always use ground-plane distance: d = f * h_cam / (foot_y - horizon_y)
            # This does NOT depend on assumed person height (h_person=1.70m),
            # so it works correctly for children, adults, cyclists, etc.
            # Default calibration (h_cam=1.20m, pitch=0°) is used when no
            # calibration JSON exists — horizon sits at frame centre (cy).
            #
            # Note: T2/T3 clips (Nov 2022 app config) lack Pitch column entirely.
            # They use static calibrated pitch. See codebook V4.0 section on sensor limitations.
            #
            # IMPORTANT: The CSV 'pitch_deg' column contains raw IMU pitch in the
            # phone's sensor reference frame, NOT the optical pitch used by the
            # pinhole distance model. When a calibration JSON exists, its pitch_deg
            # is the optically-calibrated value (from ground markings). Using raw
            # IMU pitch directly causes severe distance overestimation because the
            # IMU reference frame differs from the camera optical axis.
            # -> Always use static calibrated pitch when calibration was loaded.
            has_pitch_col = False  # Disabled: raw IMU pitch != optical pitch
            imu_baseline = None    # Median IMU pitch over clip (sensor frame)
            # Re-enable dynamic pitch ONLY when we have both a calibrated optical
            # pitch (from JSON) AND per-frame IMU pitch in the CSV.  The IMU value
            # is used as a *relative* offset from its own median, so the sensor-
            # frame bias cancels out and only frame-to-frame bounce remains.
            if (self.calibration_from_file
                    and 'pitch_deg' in self.det_df.columns):
                valid_pitches = self.det_df['pitch_deg'].dropna()
                if len(valid_pitches) > 10:  # need enough samples for a stable median
                    imu_baseline = float(valid_pitches.median())
                    self._imu_pitch_baseline = imu_baseline
                    pitch_deltas = valid_pitches - imu_baseline
                    pitch_std = float(pitch_deltas.std())
                    # Sanity check: if IMU pitch is too noisy (std > 2°), the
                    # sensor data is unreliable (phone rotating in mount, etc.)
                    # → fall back to static calibrated pitch only.
                    if pitch_std > 2.0:
                        print(f"  [CAL] IMU pitch too noisy (std={pitch_std:.1f}°, "
                              f"range={pitch_deltas.min():.1f} to {pitch_deltas.max():.1f}°) "
                              f"→ using static calibrated pitch only")
                    else:
                        has_pitch_col = True
                        print(f"  [CAL] Dynamic pitch: calibrated={self.pitch_deg:.1f}° "
                              f"+ IMU delta (baseline={imu_baseline:.1f}°, "
                              f"range={pitch_deltas.min():.1f} to {pitch_deltas.max():.1f}°)")
            if not has_pitch_col:
                print(f"  [CAL] Ground-plane distance: pitch={self.pitch_deg:.1f}°, "
                      f"h_cam={h:.2f}m (height-independent — works for children/adults)")
            # Precompute static horizon for fallback
            static_horizon_v = cy - f_px * np.tan(np.radians(self.pitch_deg))

            # ------ Roll correction ------
            # When the phone rotates around the optical axis (roll), VRUs at
            # frame edges get shifted vertically in the image.  A VRU at the
            # left edge moves UP when the camera rolls right (and vice versa),
            # introducing distance error.  Correction:
            #   effective_foot_y = foot_y - (foot_x - cx) * tan(roll_rad)
            # Roll angle is recovered by integrating gyroscope roll_rate_dps
            # and applying a high-pass Butterworth filter to remove drift.
            has_roll = False
            _roll_by_frame = {}
            if 'roll_rate_dps' in self.det_df.columns:
                # Extract per-frame roll rate (one value per frame)
                roll_per_frame = self.det_df.groupby('frame')['roll_rate_dps'].first().sort_index()
                roll_vals = roll_per_frame.dropna()
                if len(roll_vals) > 20:
                    # Interpolate to all frames (detection CSVs have gaps)
                    f_min, f_max = int(roll_vals.index.min()), int(roll_vals.index.max())
                    all_frames = np.arange(f_min, f_max + 1)
                    roll_interp = np.interp(all_frames, roll_vals.index.values, roll_vals.values)
                    dt = 1.0 / self.fps
                    # Integrate roll rate to get roll angle (on dense grid)
                    roll_integrated = np.cumsum(roll_interp * dt)
                    # High-pass Butterworth filter (2nd order)
                    # fc adapts to clip length: at least 2 full cycles needed
                    clip_duration = len(all_frames) * dt
                    try:
                        from scipy.signal import butter, filtfilt
                        nyq = self.fps / 2.0
                        fc = max(0.1, 2.0 / clip_duration)  # at least 2 cycles
                        if fc < nyq and len(roll_integrated) > 12:
                            b, a = butter(2, fc / nyq, btype='high')
                            roll_filtered = filtfilt(b, a, roll_integrated)
                        else:
                            roll_filtered = roll_integrated - np.median(roll_integrated)
                    except ImportError:
                        roll_filtered = roll_integrated - np.median(roll_integrated)
                    roll_std = float(np.std(roll_filtered))
                    roll_range = (float(np.min(roll_filtered)), float(np.max(roll_filtered)))
                    # Sanity: if roll is tiny (< 0.3° std), skip correction
                    if roll_std > 0.3:
                        has_roll = True
                        for i, fn in enumerate(all_frames):
                            _roll_by_frame[int(fn)] = float(roll_filtered[i])
                        print(f"  [CAL] Roll correction: std={roll_std:.2f}°, "
                              f"range=[{roll_range[0]:.1f}, {roll_range[1]:.1f}]°")
                    else:
                        print(f"  [CAL] Roll negligible (std={roll_std:.2f}°) — skipping correction")
            self._roll_by_frame = _roll_by_frame  # store for visualization

            n_recomp = 0
            n_lat_recomp = 0
            n_roll_corr = 0
            has_foot_x = 'foot_x' in self.det_df.columns
            has_bbox_h = 'bbox_height' in self.det_df.columns
            for idx, row in self.det_df.iterrows():
                fy = row.get('foot_y')
                # Always use ground-plane: d = f * h_cam / (foot_y - horizon_y)
                # Use per-frame effective pitch when available, else static
                # effective_pitch = calibrated optical + (IMU_frame - IMU_median)
                if has_pitch_col and pd.notna(row.get('pitch_deg')):
                    imu_delta = float(row['pitch_deg']) - imu_baseline
                    imu_delta = max(-3.0, min(3.0, imu_delta))  # clamp to ±3°
                    effective_pitch = self.pitch_deg + imu_delta
                    horizon_v = cy - f_px * np.tan(np.radians(effective_pitch))
                else:
                    horizon_v = static_horizon_v
                if pd.notna(fy) and float(fy) > horizon_v + 1:
                    dv = float(fy) - horizon_v
                    # Roll correction: shift foot_y based on horizontal position
                    # Clamp to ±5° (beyond this, ground-plane model breaks down)
                    if has_roll and has_foot_x:
                        fx = row.get('foot_x')
                        frame_num = int(row.get('frame', 0))
                        roll_deg = _roll_by_frame.get(frame_num, 0.0)
                        roll_deg = max(-5.0, min(5.0, roll_deg))  # clamp
                        if pd.notna(fx) and abs(roll_deg) > 0.1:
                            roll_shift = (float(fx) - cx) * np.tan(np.radians(roll_deg))
                            dv = dv - roll_shift  # correct for roll-induced vertical shift
                            n_roll_corr += 1
                            if dv <= 1:  # safety: don't allow negative/zero dv
                                dv = float(fy) - horizon_v  # revert
                                n_roll_corr -= 1
                    new_d = f_px * h / dv
                    if 0.5 < new_d < 100:
                        self.det_df.at[idx, 'distance_m'] = new_d
                        n_recomp += 1
                        if has_foot_x:
                            fx = row.get('foot_x')
                            if pd.notna(fx):
                                new_lat = (float(fx) - cx) * new_d / f_px
                                self.det_df.at[idx, 'lateral_m'] = new_lat
                                n_lat_recomp += 1
            if has_pitch_col and has_roll:
                method = "ground-plane (dynamic pitch + roll)"
            elif has_pitch_col:
                method = "ground-plane (dynamic pitch)"
            elif has_roll:
                method = "ground-plane (static pitch + roll)"
            else:
                method = "ground-plane (static pitch)"
            roll_info = f", roll-corrected {n_roll_corr}" if n_roll_corr > 0 else ""
            print(f"  Recomputed {n_recomp} distances + {n_lat_recomp} lateral via {method} "
                  f"(h={h:.2f}m, pitch={self.pitch_deg:.1f}°{roll_info})")

        if no_smooth:
            print(f"  Smoothing SKIPPED (--no_smooth): using raw distances")
        else:
            self.det_df = smooth_detections(self.det_df)
        n_tracks = self.det_df['track_id'].nunique() if 'track_id' in self.det_df.columns else 0
        n_vrus = n_tracks  # All tracks in detection CSVs are VRUs
        if 'user_type' in self.det_df.columns:
            type_counts = self.det_df.groupby('track_id')['user_type'].agg(
                lambda x: x.value_counts().index[0]).value_counts()
            type_str = ", ".join(f"{cnt} {tp}" for tp, cnt in type_counts.items())
        else:
            type_str = f"{n_tracks} unknown"
        print(f"  Detections: {len(self.det_df)} rows, {n_tracks} tracks "
              f"({type_str}), from {detections_path}")

        # GPS/IMU speed offset (GPS data lags video by ~1.5-2s; GPS reports position after it occurred)
        self.speed_offset_s = speed_offset_s if speed_offset_s is not None else DEFAULT_SPEED_OFFSET_S
        self.max_lateral_m = max_lateral_m  # None = no lateral filter
        self.d_threshold = d_threshold  # None = speed-adaptive, float = fixed threshold
        self.thw_threshold = thw_threshold  # None = no filter, float = THW ceiling in seconds
        self.max_distance = max_distance  # Max distance in m for interaction (default 15.0)
        self.min_ego_speed_kmh = min_ego_speed_kmh  # Min ego speed for interaction (default 0 = no filter)
        self.dense_scene_k = dense_scene_k  # Dense scene threshold (default 5)
        self.dense_scene_n = dense_scene_n  # Nearest VRUs to keep for manual coding (default 3)
        speed_offset_frames = int(self.speed_offset_s * self.fps)
        print(f"  Speed offset: {self.speed_offset_s:.1f}s ({speed_offset_frames} frames)")

        # Pre-index detections by frame
        self._det_by_frame = {}
        self._sensor_by_frame = {}
        for frame_num, group in self.det_df.groupby('frame'):
            self._det_by_frame[int(frame_num)] = group

        # Build spline-interpolated speed for all frames
        _spline_speed = _build_spline_speed(self.det_df, self.fps, self.speed_offset_s)

        # Build sensor index (speed from spline, IMU direct)
        for frame_num, group in self.det_df.groupby('frame'):
            row = group.iloc[0]
            sensor = {}
            fn = int(frame_num)
            if fn in _spline_speed:
                sensor['speed_kmh'] = _spline_speed[fn]
            for col in ('yaw_rate_dps', 'roll_rate_dps', 'acc_x_g'):
                if col in group.columns and pd.notna(row[col]):
                    sensor[col] = float(row[col])
            # Min VRU distance (all non-motor-vehicle types)
            motor_types = {'car', 'truck', 'bus', 'motorcycle', 'motor_vehicle'}
            if 'user_type' in group.columns:
                vrus = group[~group['user_type'].isin(motor_types)]
            else:
                vrus = group
            valid_vrus = vrus[vrus['distance_m'] > 0]
            if len(valid_vrus) > 0:
                sensor['min_dist_m'] = float(valid_vrus['distance_m'].min())
            self._sensor_by_frame[fn] = sensor

        # Build GPS trajectory if lat/lon available in detection CSV
        if 'gps_lat' in self.det_df.columns and 'gps_lon' in self.det_df.columns:
            gps_per_frame = self.det_df.groupby('frame')[['gps_lat', 'gps_lon']].first()
            gps_per_frame = gps_per_frame.dropna()
            if len(gps_per_frame) > 5:
                self._gps_trajectory = {
                    int(f): (row['gps_lat'], row['gps_lon'])
                    for f, row in gps_per_frame.iterrows()
                }
                print(f"  [GPS] Loaded {len(self._gps_trajectory)} GPS positions")

        # Fill sensor data for ALL frames (IMU signals needed for steering detection)
        self._fill_sensor_all_frames()

        # Steering episode detection disabled — IMU pitch corrections pending
        # Will re-enable after dynamic pitch calibration is validated
        self._strong_steering_episodes = []
        print(f"  [STEER-VIZ] Steering detection disabled (pending pitch calibration)")

        # Shakiness warning: compute mean absolute yaw rate from IMU if available
        _yaw_values = [abs(s.get('yaw_rate_dps', 0.0) or 0.0)
                       for s in self._sensor_by_frame.values()
                       if 'yaw_rate_dps' in s]
        if _yaw_values:
            mean_abs_yaw = sum(_yaw_values) / len(_yaw_values)
            if mean_abs_yaw > 15.0:
                print(f"\n  WARNING: Shaky video detected (mean |gyro_yaw| = {mean_abs_yaw:.1f} deg/s)")
                print(f"  Consider: --fps_display 7 for annotation comfort")
                if fps_display is None and mean_abs_yaw > 25.0:
                    print(f"  Auto-setting fps_display=7 (very shaky)")
                    self.fps_display = 7
                    self.frame_step = max(1, round(self.fps / 7))
                    self.annotation_fps = 7

        # Defer encounter detection until after pre-encounter phase
        # (steering -> obstacles -> calibration)
        self.encounters = []
        self._det_ready = True  # Detection data loaded, ready to detect when needed
        self._constrained_zones_path = constrained_zones_path

        # Output paths — include rater_id for double-coding support
        if output_path:
            self.output_path = output_path
        else:
            self.output_path = f"{self.video_stem}_rater{rater_id}_encounters.csv"
        self.trip_output_path = str(Path(self.output_path).parent / "trip_annotations.csv")
        self.detections_path = detections_path

        # Admin
        self.trip_id = trip_id or self.video_stem
        self.rater_id = rater_id

        # Rider ID — auto-derive from video filename if not provided
        if rider_id is None:
            m = re.match(r'^([a-z]\d+)_', Path(video_path).stem)
            self.rider_id = f"R_{m.group(1)}" if m else "R_UNKNOWN"
        else:
            self.rider_id = rider_id

        # Zone lookup
        self.zone_info = {}
        self.city = city
        if zones_path:
            all_zones = load_zone_segments(zones_path, trip_id)
            if trip_id and trip_id in all_zones:
                self.zone_info = all_zones[trip_id]
                if self.zone_info and not self.city:
                    self.city = self.zone_info[0].get('city', '')
                print(f"  Zone info: {len(self.zone_info)} zones loaded for {trip_id}")
                # Pre-fill trip-level ZONE_TYPE from zone data
                if self.zone_info:
                    zt = self.zone_info[0].get('zone_type', 9)
                    self.trip_codes['ZONE_TYPE'] = zt
            else:
                # Try matching any trip_id
                for tid, zlist in all_zones.items():
                    if zlist and not self.city:
                        self.city = zlist[0].get('city', '')
                print(f"  Zone info: loaded {sum(len(v) for v in all_zones.values())} zones "
                      f"(no exact trip_id match)")

        # State
        self.current_frame = 0
        self.playing = False
        self.state = self.ENCOUNTER_LIST
        self.selected_enc_idx = 0
        self.show_yolo = True
        self.show_trajectory = False
        self.show_signals = False
        self.show_density_secondary = True  # Show secondary encounters in list (toggle with 'd')

        # Pre-encounter phase state
        self.clip_obstacle_zones = []      # List of {frame_start, frame_end, time_start, time_end, type}
        self.clip_obstacle_open = None     # Frame number of open obstacle start, or None
        self.clip_obstacle_open_type = None  # Zone type for currently open zone
        self.pre_encounter_phase = True    # True until encounters are detected

        # Auto-save state
        self.coded_since_last_autosave = 0
        self.auto_save_interval = 3  # save every N coded encounters as safety net
        self._last_autosave_time = time.time()  # Timer-based auto-save (every 60s)
        self._autosave_flash_until = 0  # Show [SAVED] indicator until this time

        # Coding state
        self.coding_var_idx = 0
        self.coding_var_names = list(MANUAL_VARIABLES.keys())
        self.input_buffer = ""
        self._undo_stack = []  # [(var_name, old_value)] for undo during coding

        # Distance correction state
        self.dist_correction_mode = False
        self.dist_correction_quick_foot = False  # Quick-foot mode: clicks auto-label as foot
        self.dist_correction_points = []  # [(x, y, part_id, part_name), ...]
        self.dist_correction_pending_click = None  # (x, y) waiting for part label
        self.dist_corrections = {}  # {(frame, track_id): corrected_dist}
        self.dist_correction_history = {}  # {frame: [(x, y, part_id, part_name), ...]}
        self.dist_correction_last_result = None  # (frame, dist_m, foot_x, foot_y) for overlay
        # Body part positions as fraction of total height from top (VRU-type-aware)
        self.BODY_PART_POS_BY_TYPE = {
            1: {  # Pedestrian (standing)
                1: ('head', 0.00), 2: ('shoulder', 0.20), 3: ('hip', 0.50),
                4: ('knee', 0.75), 5: ('foot', 1.00),
            },
            2: {  # Cyclist (seated, leaning forward)
                1: ('head', 0.00), 2: ('shoulder', 0.18), 3: ('hip', 0.55),
                4: ('knee', 0.78), 5: ('wheel', 1.00),
            },
            3: {  # E-scooterist (standing on platform)
                1: ('head', 0.00), 2: ('shoulder', 0.22), 3: ('hip', 0.48),
                4: ('knee', 0.73), 5: ('foot', 1.00),
            },
            4: {  # Other MMV — use pedestrian model as default
                1: ('head', 0.00), 2: ('shoulder', 0.20), 3: ('hip', 0.50),
                4: ('knee', 0.75), 5: ('foot', 1.00),
            },
        }
        self.BODY_PART_POS = self.BODY_PART_POS_BY_TYPE[1]  # Default pedestrian

        # Obstacle point marking state (key 'o' in ENCOUNTER_VIEW/CODING)
        # Supports 1-3 points per obstacle for wide obstacle footprints (V3.7)
        self.obstacle_point_mode = False
        self.obstacle_point_pending_click = None  # (x, y) waiting for type label
        self.obstacle_point_last_result = None    # (frame, dist_m, x, y, type_code) for overlay
        self._obs_pt_staged = None                # staged point dict awaiting type label key
        self._obs_pt_multi = []                   # accumulated points for current obstacle (max 3)
        # Obstacle click-to-measure in OBSTACLE_MARKING phase (key 5)
        self.obs_click_mode = False
        self.clip_obstacle_points = []  # [{frame, distance_m, px, py}]
        self.OBSTACLE_TYPE_CODES = {
            1: "Bollard/post",
            2: "Bench/furniture",
            3: "Parked vehicle",
            4: "Construction/barrier",
            5: "Vegetation",
            9: "Other",
        }
        self.ZONE_TYPE_CODES = {
            1: 'pedestrian_area',
            2: 'shared_space',
            3: 'non_motorised_path',
            4: 'crosswalk',
            5: 'park',
            6: 'obstacle',
            7: 'dismounted',
        }

        # Lane marking state (key 'l' — define left/right path edges)
        # Supports multiple lane segments: re-press 'l' when road direction changes
        self.lane_marking_mode = False
        self.lane_marking_clicks = []  # Accumulates up to 4 clicks
        self.clip_lane_lines = None    # Legacy single-segment (kept for session restore compat)
        self.clip_lane_lines_list = [] # List of dicts: [{'left': ..., 'right': ..., 'frame': int}, ...]

        # Zoom state (mouse-wheel zoom for precise clicking)
        self.zoom_level = 1.0       # 1.0 = no zoom, 2.0 = 2x, etc.
        self.zoom_cx = self.width // 2   # zoom center x (in original coords)
        self.zoom_cy = self.height // 2  # zoom center y (in original coords)
        self._hover_pos = None     # (x, y) of current mouse position for live preview
        self.show_magnifier = True  # Show magnifier loupe in precision-click modes

        # Manual track creation state
        self.manual_track_mode = False
        self.manual_track_id = None       # Current manual track ID being created
        self.manual_track_points = {}     # {frame: (foot_x, foot_y)} for current manual track
        self.manual_tracks_created = []   # List of all manually created track IDs

        # Interaction grouping + group coding state
        self.interaction_groups = {}  # group_id -> set of encounter indices
        self.next_group_id = 1
        self.grouping_selected = 0
        self.group_coding_idx = 0  # which group is being coded
        self.group_var_idx = 0     # which variable within the group
        self.group_var_names = list(GROUP_VARIABLES.keys())
        self.group_codes = {}      # group_id -> {var_name: value}

        # Trip annotation state
        self.trip_var_idx = 0
        self.trip_var_names = list(TRIP_VARIABLES.keys())
        self.trip_codes = OrderedDict([(k, None) for k in TRIP_VARIABLES])

        # Carry-forward
        self.carry_forward = {}

        # Notes editing state (in-window, no terminal input needed)
        self.notes_editing = False
        self.notes_buffer = ""

        # Rider segment state — two independent segmentations
        self.rider_accel_segments = []    # [{frame_start, frame_end, code}]
        self.rider_steer_segments = []    # [{frame_start, frame_end, code}]
        self.rider_boundaries = [0]       # Working boundaries for current pass
        self.rider_pass = 'accel'         # 'accel' or 'steer'
        self.rider_seg_idx = 0            # Current segment being coded
        self.show_imu_overlay = False     # Toggle IMU signal graphs on video
        self.show_bev_minimap = False     # BEV minimap off by default
        self.show_gps_minimap = False    # GPS map off by default
        self._gps_trajectory = None      # Pre-computed GPS trajectory (lat, lon) per frame
        # _strong_steering_episodes: computed earlier from IMU data (~line 2202)
        # Do NOT re-initialize here — it would overwrite detected episodes

        # Calibration state
        self.cal_state = None  # None, 'head', 'foot', 'ref_p1', 'ref_p2', 'ref_input',
                               # 'marking_p1', 'marking_p2', 'marking_type', 'marking_custom',
                               # 'multi_head', 'multi_foot'
        self.cal_head_xy = None
        self.cal_ref_p1 = None
        self.cal_ref_p2 = None
        self.cal_ref_input = ""
        self.cal_marking_pairs = []  # For multi-reference: list of ((x1,y1),(x2,y2))
        self.cal_marking_p1 = None
        self.cal_height_input = ""   # Custom height typed by user (digits/period)
        self.cal_ped_pairs = []      # Multi-ped: list of (head_xy, foot_xy, height_m)
        self.cal_history = []        # List of (h, pitch, method, rmse)

        # VLM pre-annotation suggestions (loaded from --suggestions CSV)
        self._vlm_suggestions = {}  # {track_id: {var_name: code}}

        # Coding speed tracker (encounters per minute, for progress HUD)
        self._coding_timestamps = []  # list of (encounter_idx, timestamp) when coded
        self._session_start_time = time.time()

        # Saved encounters (written to CSV)
        self.saved_encounters = []

    # ─── Session persistence ───

    def _session_state_path(self):
        """Return path to session state JSON file.

        Placed next to the output CSV (not the video) so it works
        when the video is on a read-only external drive.
        """
        output_dir = Path(self.output_path).parent
        stem = Path(self.video_path).stem
        return output_dir / f"{stem}_rater{self.rater_id}_session.json"

    def _save_session_state(self, silent=False):
        """Save full annotation state for session resume.

        Writes atomically (tmp + rename) to prevent corruption on crash.
        Called after each encounter is coded/skipped and on clean exit.
        """
        state = {
            'version': '3.6',
            'video_path': str(self.video_path),
            'rater_id': self.rater_id,
            'rider_id': getattr(self, 'rider_id', None),
            'selected_enc_idx': self.selected_enc_idx,
            'state': self.state,
            'encounters': [],
            'dist_corrections': {},
            'clip_obstacle_zones': [],
            'trip_codes': {},
            'calibration': {
                'cy': self.height / 2.0 if hasattr(self, 'height') else None,
                'f': getattr(self, 'focal_length_px', None),
                'h_cam': getattr(self, 'camera_height_m', None),
                'pitch_deg': getattr(self, 'pitch_deg', None),
            },
            'timestamp': datetime.now().isoformat(),
        }

        # Serialize encounters (only essential fields)
        for enc in self.encounters:
            enc_data = {
                'idx': enc.get('idx'),
                'primary_track': enc.get('primary_track'),
                'status': enc.get('status', 'pending'),
                'codes': enc.get('codes', {}),
                'notes': enc.get('notes', ''),
                'note_timestamps': enc.get('note_timestamps', []),
                'coding_start_ts': enc.get('coding_start_ts', ''),
                'coding_end_ts': enc.get('coding_end_ts', ''),
            }
            if enc.get('obstacle_points'):
                enc_data['obstacle_points'] = enc['obstacle_points']
            state['encounters'].append(enc_data)

        # Serialize distance corrections
        if hasattr(self, 'dist_corrections') and self.dist_corrections:
            for key, val in self.dist_corrections.items():
                state['dist_corrections'][f"{key[0]}_{key[1]}"] = val

        # Serialize distance correction history (click points)
        if hasattr(self, 'dist_correction_history') and self.dist_correction_history:
            history = {}
            for frame, pts in self.dist_correction_history.items():
                history[str(frame)] = [
                    [p[0], p[1], p[2], p[3]] for p in pts
                ]
            state['dist_correction_history'] = history

        # Serialize obstacle zones
        if hasattr(self, 'clip_obstacle_zones') and self.clip_obstacle_zones:
            state['clip_obstacle_zones'] = list(self.clip_obstacle_zones)

        # Serialize obstacle points (click-to-measure)
        if hasattr(self, 'clip_obstacle_points') and self.clip_obstacle_points:
            state['clip_obstacle_points'] = list(self.clip_obstacle_points)

        # Serialize lane lines (multi-segment)
        if self.clip_lane_lines_list:
            state['clip_lane_lines_list'] = self.clip_lane_lines_list
        elif self.clip_lane_lines:
            state['clip_lane_lines'] = self.clip_lane_lines

        # Serialize trip codes
        if hasattr(self, 'trip_codes') and self.trip_codes:
            state['trip_codes'] = {
                k: v for k, v in self.trip_codes.items()
                if v is not None
            }

        # Write atomically (write to temp, then rename)
        session_path = self._session_state_path()
        tmp_path = session_path.with_suffix('.tmp')
        try:
            os.makedirs(session_path.parent, exist_ok=True)
            # Backup rotation: copy existing session to _backup.json before overwriting
            backup_path = str(session_path).replace('.json', '_backup.json')
            if os.path.exists(session_path):
                shutil.copy2(session_path, backup_path)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str, ensure_ascii=False)
            tmp_path.rename(session_path)
            if not silent:
                n_done = len([e for e in self.encounters
                              if e.get('status') in ('coded', 'skipped')])
                print(f"  [SESSION] State saved "
                      f"({n_done}/{len(self.encounters)} coded/skipped)")
        except Exception as e:
            print(f"  [SESSION] Failed to save state: {e}")

    def _load_session_state(self):
        """Load session state if available. Returns True if restored.

        Matches saved encounters to current encounters by primary_track.
        Restores codes, status, notes, distance corrections, trip codes.
        """
        if self.no_resume:
            print("  [SESSION] Resume disabled (--no_resume)")
            return False

        session_path = self._session_state_path()
        if not session_path.exists():
            return False

        try:
            with open(session_path) as f:
                state = json.load(f)

            saved_encounters = {
                e['primary_track']: e
                for e in state.get('encounters', [])
            }
            restored = 0
            for enc in self.encounters:
                pt = enc.get('primary_track')
                if pt in saved_encounters:
                    saved = saved_encounters[pt]
                    if saved.get('status') in ('coded', 'skipped'):
                        enc['status'] = saved['status']
                        enc['codes'] = saved.get('codes', {})
                        enc['notes'] = saved.get('notes', '')
                        _nts = saved.get('note_timestamps', [])
                        enc['note_timestamps'] = (
                            _nts if isinstance(_nts, list) else [])
                        if saved.get('coding_start_ts'):
                            enc['coding_start_ts'] = saved['coding_start_ts']
                        if saved.get('coding_end_ts'):
                            enc['coding_end_ts'] = saved['coding_end_ts']
                        if saved.get('obstacle_points'):
                            enc['obstacle_points'] = (
                                saved['obstacle_points'])
                        restored += 1

            # Restore distance corrections
            if state.get('dist_corrections'):
                for key_str, val in state[
                        'dist_corrections'].items():
                    parts = key_str.split('_')
                    if len(parts) == 2:
                        try:
                            frame = int(parts[0])
                            track_id = int(parts[1])
                            self.dist_corrections[
                                (frame, track_id)] = val
                        except (ValueError, TypeError):
                            pass

            # Restore distance correction history
            if state.get('dist_correction_history'):
                for frame_str, pts in state[
                        'dist_correction_history'].items():
                    try:
                        frame = int(frame_str)
                        self.dist_correction_history[frame] = [
                            (p[0], p[1], p[2], p[3]) for p in pts
                        ]
                    except (ValueError, TypeError, IndexError):
                        pass

            # Restore calibration parameters from session state (with sanity checks)
            saved_cal = state.get('calibration', {})
            if saved_cal:
                cal_changed = False
                h_cam = saved_cal.get('h_cam')
                if h_cam is not None and 0.8 <= h_cam <= 2.0:
                    self.camera_height_m = h_cam
                    cal_changed = True
                elif h_cam is not None:
                    print(f"  [SESSION] WARNING: ignoring invalid h_cam={h_cam:.2f}m (must be 0.8-2.0m)")
                pitch = saved_cal.get('pitch_deg')
                if pitch is not None and -5 <= pitch <= 15:
                    self.pitch_deg = pitch
                    cal_changed = True
                elif pitch is not None:
                    print(f"  [SESSION] WARNING: ignoring invalid pitch={pitch:.1f}deg (must be -5 to 15)")
                f_px = saved_cal.get('f')
                if f_px is not None and 700 <= f_px <= 1800:
                    self.focal_length_px = f_px
                    cal_changed = True
                elif f_px is not None:
                    print(f"  [SESSION] WARNING: ignoring invalid f={f_px:.0f}px (must be 700-1800)")
                # Restore horizon pixel (cy) with sanity bounds
                cy = saved_cal.get('cy')
                if (cy is not None and hasattr(self, 'height')
                        and 0 < cy < self.height):
                    # cy should be near frame center; reject wild values
                    self._session_cy = cy
                    cal_changed = True
                if cal_changed:
                    cal_parts = [f"h={self.camera_height_m:.2f}m",
                                 f"pitch={self.pitch_deg:.1f}deg",
                                 f"f={self.focal_length_px:.0f}px"]
                    if hasattr(self, '_session_cy'):
                        cal_parts.append(f"cy={self._session_cy:.0f}px")
                    print(f"  [SESSION] Calibration restored: "
                          f"{', '.join(cal_parts)}")

            # Restore clip obstacle zones
            if state.get('clip_obstacle_zones'):
                self.clip_obstacle_zones = set(state['clip_obstacle_zones'])

            # Restore clip obstacle points
            if state.get('clip_obstacle_points'):
                self.clip_obstacle_points = list(state['clip_obstacle_points'])
                print(f"  [SESSION] Restored {len(self.clip_obstacle_points)} obstacle points")

            # Restore trip codes
            if state.get('trip_codes'):
                for k, v in state['trip_codes'].items():
                    if k in self.trip_codes:
                        self.trip_codes[k] = v

            # Restore lane lines (multi-segment or legacy single)
            if state.get('clip_lane_lines_list'):
                self.clip_lane_lines_list = state['clip_lane_lines_list']
                self.clip_lane_lines = self.clip_lane_lines_list[-1] if self.clip_lane_lines_list else None
            elif state.get('clip_lane_lines'):
                self.clip_lane_lines = state['clip_lane_lines']
                self.clip_lane_lines_list = [self.clip_lane_lines]

            # Restore position
            saved_idx = state.get('selected_enc_idx', 0)
            if self.encounters:
                self.selected_enc_idx = min(
                    saved_idx, len(self.encounters) - 1)
            else:
                self.selected_enc_idx = 0

            # Add restored encounters to saved list
            for enc in self.encounters:
                if (enc.get('status') == 'coded'
                        and enc not in self.saved_encounters):
                    self.saved_encounters.append(enc)

            ts = state.get('timestamp', 'unknown')
            print(f"  [SESSION] Restored {restored}/"
                  f"{len(self.encounters)} encounters "
                  f"from previous session")
            print(f"  [SESSION] Session saved at: {ts}")
            print(f"  [SESSION] State file: {session_path}")
            return restored > 0
        except Exception as e:
            print(f"  [SESSION] Failed to load state: {e}")
            return False

    # ─── Sensor data fill ───

    def _fill_sensor_all_frames(self):
        """Fill _sensor_by_frame for ALL video frames via nearest-neighbor.

        Detection CSV only has IMU values for frames with detections.
        Steering auto-detection needs GyrZ for every frame. This fills
        gaps by carrying forward the nearest known IMU value.
        """
        if not self._sensor_by_frame:
            return
        known_frames = sorted(self._sensor_by_frame.keys())
        if not known_frames:
            return
        for f in range(self.total_frames):
            if f in self._sensor_by_frame:
                continue
            # Find nearest known frame
            idx = np.searchsorted(known_frames, f)
            if idx == 0:
                nearest = known_frames[0]
            elif idx >= len(known_frames):
                nearest = known_frames[-1]
            else:
                before = known_frames[idx - 1]
                after = known_frames[idx]
                nearest = before if (f - before) <= (after - f) else after
            # Copy IMU signals only (not distance/speed which are detection-specific)
            src = self._sensor_by_frame[nearest]
            filled = {}
            for k in ('yaw_rate_dps', 'roll_rate_dps', 'acc_x_g'):
                if k in src:
                    filled[k] = src[k]
            if filled:
                self._sensor_by_frame[f] = filled

    def _detect_steering_episodes(self):
        """Detect strong camera-motion episodes from IMU for timeline display.

        Flags sustained harsh steering (yaw) OR rolling (GyrX) that degrades
        VRU trajectories and distance estimation.
        Returns list of (onset_frame, offset_frame, motion_type) tuples.
        Severity filter (peak * duration >= 600) removes brief e-scooter jitter.
        """
        if not self._sensor_by_frame:
            return []

        # Detect episodes for each signal independently
        signals = [
            ('yaw_rate_dps', 40.0, 50.0, 'STEER'),   # Only very harsh steering
            ('roll_rate_dps', 30.0, 40.0, 'ROLL'),    # Only very harsh rolling
        ]
        min_severity = 900  # peak * duration threshold (stricter)

        all_episodes = []
        for sig_key, onset_thresh, peak_thresh, label in signals:
            # Check if signal exists in data
            has_signal = any(sig_key in self._sensor_by_frame.get(f, {})
                            for f in range(min(30, self.total_frames)))
            if not has_signal:
                continue
            # Find onset-level frames
            onset_frames = []
            for f in range(self.total_frames):
                val = abs(self._sensor_by_frame.get(f, {}).get(sig_key, 0.0) or 0.0)
                if val > onset_thresh:
                    onset_frames.append(f)
            if not onset_frames:
                continue
            # Group into episodes (merge gaps < 5 frames)
            episodes = []
            ep_start = onset_frames[0]
            ep_end = onset_frames[0]
            for f in onset_frames[1:]:
                if f - ep_end <= 5:
                    ep_end = f
                else:
                    episodes.append((ep_start, ep_end))
                    ep_start = f
                    ep_end = f
            episodes.append((ep_start, ep_end))
            # Keep only episodes with sufficient severity
            for onset, offset in episodes:
                duration = offset - onset + 1
                peak = max(abs(self._sensor_by_frame.get(f, {}).get(sig_key, 0.0) or 0.0)
                           for f in range(onset, offset + 1))
                severity = peak * duration
                if peak >= peak_thresh and severity >= min_severity:
                    all_episodes.append((max(0, onset - 5),
                                         min(self.total_frames - 1, offset + 5),
                                         label))

        # Merge overlapping episodes (keep the more descriptive label)
        if not all_episodes:
            return []
        all_episodes.sort(key=lambda x: x[0])
        merged = [all_episodes[0]]
        for onset, offset, label in all_episodes[1:]:
            prev_on, prev_off, prev_label = merged[-1]
            if onset <= prev_off + 5:  # Overlapping or adjacent
                # Merge: extend range, combine labels
                new_label = prev_label if prev_label == label else prev_label + '+' + label
                merged[-1] = (min(prev_on, onset), max(prev_off, offset), new_label)
            else:
                merged.append((onset, offset, label))
        return merged

    # ─── Frame reading ───

    def _get_frame_image(self, frame_num):
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    # ─── YOLO overlay (Phase 4: improved distance display) ───

    # VRU type display config
    VRU_TYPE_LABELS = {'pedestrian': 'PED', 'cyclist': 'CYC', 'scooter': 'SCO'}
    # Colorblind-safe palette (no red-green contrast)
    VRU_TYPE_COLORS = {
        'pedestrian': (255, 255, 255),  # White
        'cyclist':    (255, 255, 0),    # Cyan
        'scooter':    (0, 165, 255),    # Orange
    }

    def _draw_yolo_overlay(self, img, frame_num, primary_track=None):
        """Draw detection overlay with primary track highlighted, others dimmed.
        Shows ALL VRU types (pedestrian, cyclist, scooter) with type labels.
        """
        if not self.show_yolo:
            return img
        dets = self._det_by_frame.get(frame_num)
        if dets is None:
            return img

        n_drawn = 0
        for _, det in dets.iterrows():
            # Skip interpolated detections — they are synthetic and not reliable
            # Use pd.notna guard: bool(NaN) is True, which would skip real detections
            _interp = det.get('is_interpolated', False)
            if pd.notna(_interp) and bool(_interp):
                continue
            # Skip ego-rider false positives (foot at very bottom of frame, constant distance)
            fy = det.get('foot_y', 0)
            if pd.notna(fy) and float(fy) > self.height - 15:
                continue
            dist = det.get('distance_m', 0)
            tid = det.get('track_id', -1)
            user_type = str(det.get('user_type', '')).lower()
            is_occluded = bool(det.get('is_occluded', False))

            is_primary = (primary_track is not None and tid == primary_track)

            # All VRUs default to PED — YOLO misclassifies frequently
            # Rater corrects type in Step 2 (track-by-track coding)
            type_label = 'PED'
            type_color = (255, 255, 255)  # White for all VRUs

            # Get bounding box coordinates
            # Prefer original YOLO bbox when available (accurate shape)
            # Fall back to foot-based reconstruction (narrower approximation)
            if 'bbox_x1' in det.index and pd.notna(det.get('bbox_x1')):
                x1 = int(det['bbox_x1'])
                y1 = int(det['bbox_y1'])
                x2 = int(det['bbox_x2'])
                y2 = int(det['bbox_y2'])
            elif 'foot_x' in det.index and 'foot_y' in det.index and pd.notna(det['foot_x']):
                fx = int(det['foot_x'])
                fy_val = int(det['foot_y'])
                bh = int(det.get('bbox_height', 50))
                bw = max(bh // 2, 20)  # approximate width from height
                x1 = fx - bw // 2
                y1 = fy_val - bh
                x2 = fx + bw // 2
                y2 = fy_val
            else:
                continue

            if is_primary:
                # Primary track: thick colored box + large distance text
                # Colorblind-safe palette (no red-green)
                if dist < 1.5:
                    color = (255, 0, 255)    # Magenta — danger close
                elif dist < 3.0:
                    color = (0, 165, 255)    # Orange — close
                elif dist < 5.0:
                    color = (0, 255, 255)    # Yellow — moderate
                else:
                    color = (255, 255, 0)    # Cyan — far

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                # Foot marker
                if 'foot_x' in det.index and pd.notna(det['foot_x']):
                    cv2.drawMarker(img, (int(det['foot_x']), int(det['foot_y'])),
                                   color, cv2.MARKER_CROSS, 12, 2)

                # Type label at top-left of box
                cv2.putText(img, type_label, (x1 + 2, y1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, type_color, 2)

                # Large distance text at top of frame
                corrected = self.dist_corrections.get((frame_num, tid))
                if corrected is not None:
                    dist_text = f"T{tid} {type_label}  {dist:.2f}m -> *{corrected:.2f}m*"
                    text_color = (255, 0, 255)  # Magenta for corrected
                elif is_occluded:
                    dist_text = f"T{tid} {type_label}  [{dist:.2f}m]"
                    text_color = color
                else:
                    dist_text = f"T{tid} {type_label}  {dist:.2f}m"
                    text_color = color
                font_scale = 1.2
                thickness = 3
                (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, thickness)
                tx = (self.width - tw) // 2
                ty = 50
                # Background rectangle
                cv2.rectangle(img, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (0, 0, 0), -1)
                cv2.putText(img, dist_text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                n_drawn += 1
            else:
                # Other VRUs: thin type-colored box, small text
                box_color = tuple(c // 2 for c in type_color)  # Dimmed type color
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 1)
                # Distance display: brackets for occluded
                if is_occluded:
                    label = f"{type_label} T{tid} [{dist:.1f}m]"
                    label_color = box_color
                else:
                    label = f"{type_label} T{tid} {dist:.1f}m"
                    label_color = box_color
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)
                n_drawn += 1

        # Debug indicator: bright bar at top-right showing overlay is active
        tag = f"YOLO: {n_drawn}/{len(dets)} F{frame_num}"
        cv2.rectangle(img, (self.width - 200, 0), (self.width, 20), (0, 0, 0), -1)
        cv2.putText(img, tag, (self.width - 195, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Auto-save indicator: show [SAVED] for 2s after each auto-save
        if hasattr(self, '_autosave_flash_until') and time.time() < self._autosave_flash_until:
            cv2.putText(img, "[SAVED]", (self.width - 120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        return img

    # ─── Signal Charts ───

    def _draw_signal_charts(self, img, enc):
        """Draw mini time-series charts on right side during ENCOUNTER_VIEW.

        Four stacked charts (200x60px): acc_x_g, yaw_rate_dps, speed_kmh, distance_m.
        Toggle with 's' key.
        """
        if not getattr(self, 'show_signals', False):
            return img
        if enc is None:
            return img

        chart_w, chart_h = 200, 60
        margin = 5
        x0 = self.width - chart_w - 10
        y0 = 10

        fs = enc['frame_start']
        fe = enc['frame_end']
        cur_f = self.current_frame

        # Collect data arrays
        frames_range = range(fs, fe + 1)
        t_arr = np.array([(f - fs) / self.fps for f in frames_range])

        signals = [
            ('acc_x_g', 'Accel X (g)', (-0.5, 0.5), (0, 200, 255)),
            ('yaw_rate_dps', 'Yaw rate (deg/s)', (-30, 30), (255, 200, 0)),
            ('roll_rate_dps', 'Roll rate (deg/s)', (-30, 30), (0, 140, 255)),
            ('speed_kmh', 'Speed (km/h)', (0, 30), (0, 255, 100)),
            ('min_dist_m', 'Distance (m)', (0, 15), (255, 100, 100)),
        ]

        for si, (key, title, (vmin, vmax), color) in enumerate(signals):
            cy = y0 + si * (chart_h + margin)

            # Background
            overlay = img.copy()
            cv2.rectangle(overlay, (x0, cy), (x0 + chart_w, cy + chart_h),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
            cv2.rectangle(img, (x0, cy), (x0 + chart_w, cy + chart_h),
                          (60, 60, 60), 1)

            # Title
            cv2.putText(img, title, (x0 + 3, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

            # Grid lines
            for frac in (0.25, 0.5, 0.75):
                gy = int(cy + 14 + frac * (chart_h - 18))
                cv2.line(img, (x0, gy), (x0 + chart_w, gy), (40, 40, 40), 1)

            # Zero line for signed signals
            if vmin < 0:
                zero_frac = vmax / (vmax - vmin)
                zy = int(cy + 14 + zero_frac * (chart_h - 18))
                cv2.line(img, (x0, zy), (x0 + chart_w, zy), (80, 80, 80), 1)

            # Collect values
            vals = []
            for f in frames_range:
                sensor = self._sensor_by_frame.get(f, {})
                vals.append(sensor.get(key, np.nan))
            vals = np.array(vals)

            # Build polyline
            plot_y_start = cy + 14
            plot_h = chart_h - 18
            pts = []
            for i, (t, v) in enumerate(zip(t_arr, vals)):
                if np.isnan(v):
                    continue
                px = int(x0 + (t / max(t_arr[-1], 0.001)) * chart_w)
                frac = np.clip((vmax - v) / max(vmax - vmin, 0.001), 0, 1)
                py = int(plot_y_start + frac * plot_h)
                pts.append((px, py))

            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)],
                              False, color, 1, cv2.LINE_AA)

            # Cursor for current frame
            if fs <= cur_f <= fe and t_arr[-1] > 0:
                cx_pos = int(x0 + ((cur_f - fs) / self.fps / t_arr[-1]) * chart_w)
                cv2.line(img, (cx_pos, cy), (cx_pos, cy + chart_h),
                         (255, 255, 255), 1)

                # Current value label
                cur_sensor = self._sensor_by_frame.get(cur_f, {})
                cur_val = cur_sensor.get(key)
                if cur_val is not None:
                    cv2.putText(img, f"{cur_val:.1f}", (x0 + chart_w - 40, cy + chart_h - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

        return img

    # ─── Timeline ───

    def _draw_timeline(self, img, enc=None):
        """Draw timeline bar. If enc provided, show encounter window markers."""
        bar_y = self.height - 100
        bar_h = 8
        bar_x1 = 20
        bar_x2 = self.width - 20
        bar_w = bar_x2 - bar_x1

        # Background
        cv2.rectangle(img, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (60, 60, 60), -1)

        # Encounter window highlight
        if enc is not None and self.total_frames > 0:
            ws = bar_x1 + int(enc['frame_start'] / self.total_frames * bar_w)
            we = bar_x1 + int(enc['frame_end'] / self.total_frames * bar_w)
            cv2.rectangle(img, (ws, bar_y - 2), (we, bar_y + bar_h + 2), (80, 80, 40), -1)

            # Markers: F=first detection, S=onset(enters zone), P=perception(+200ms), M=mindist, E=offset(leaves zone)
            markers = [
                (enc['frame_end'], (0, 0, 255), "E"),    # Red — offset
            ]
            # First detection marker (before entering interaction zone)
            ffd = enc.get('frame_first_detection')
            if ffd is not None and ffd < enc['frame_start']:
                markers.insert(0, (ffd, (180, 180, 180), "F"))  # Grey — first detection
            markers.insert(-1, (enc['frame_start'], (0, 255, 0), "S"))  # Green — onset
            markers.insert(-1, (enc.get('frame_perception', enc['frame_start']), (255, 0, 255), "P"))  # Magenta — perception
            markers.insert(-1, (enc['frame_mindist'], (0, 255, 255), "M"))  # Cyan — mindist
            # Awareness markers (onset + optional offset, marked by key 7)
            aware_f = enc.get('_aware_frame')
            if aware_f is not None:
                markers.insert(-1, (aware_f, (255, 200, 0), "A"))  # Orange — awareness onset
            aware_off = enc.get('_aware_offset_frame')
            if aware_off is not None:
                markers.insert(-1, (aware_off, (255, 200, 0), "A2"))  # Orange — awareness offset
            for frame, color, label in markers:
                mx = bar_x1 + int(frame / self.total_frames * bar_w)
                cv2.line(img, (mx, bar_y - 10), (mx, bar_y + bar_h + 10), color, 2)
                cv2.putText(img, label, (mx - 4, bar_y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Camera-motion zones: colored rectangles on timeline from auto-detected IMU
        if self._strong_steering_episodes and self.total_frames > 0:
            for onset, offset, label in self._strong_steering_episodes:
                sx0 = bar_x1 + int(onset / self.total_frames * bar_w)
                sx1 = bar_x1 + int(offset / self.total_frames * bar_w)
                sx1 = max(sx1, sx0 + 8)  # minimum width
                # Blue for steering, orange for roll, mixed for both
                if 'ROLL' in label and 'STEER' not in label:
                    fill_color = (0, 140, 255)   # Orange for roll
                elif 'STEER' in label and 'ROLL' not in label:
                    fill_color = (255, 140, 0)   # Blue for steering
                else:
                    fill_color = (200, 0, 200)   # Purple for combined
                short_label = label.replace('+', '/')
                cv2.rectangle(img, (sx0, bar_y - 35), (sx1, bar_y + bar_h),
                              fill_color, -1)
                cv2.rectangle(img, (sx0, bar_y - 35), (sx1, bar_y + bar_h),
                              (255, 255, 255), 2)
                mid = (sx0 + sx1) // 2
                cv2.putText(img, short_label, (mid - 20, bar_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

        # Current position
        if self.total_frames > 0:
            pos_x = bar_x1 + int(self.current_frame / self.total_frames * bar_w)
            cv2.line(img, (pos_x, bar_y - 5), (pos_x, bar_y + bar_h + 5), (255, 255, 255), 2)

        return img

    # ─── Status bar ───

    def _draw_status_bar(self, img):
        """Draw bottom status bar."""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, self.height - 90), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        t = self.current_frame / self.fps
        sensor = self._sensor_by_frame.get(self.current_frame, {})

        # Line 1: frame info + sensor readout
        play_str = "PLAY" if self.playing else "PAUSE"
        fps_tag = f" @{self.annotation_fps:.0f}fps" if self.frame_step > 1 else ""
        parts = [f"[{play_str}] F{self.current_frame}/{self.total_frames} t={t:.2f}s{fps_tag}"]
        if 'speed_kmh' in sensor:
            parts.append(f"v={sensor['speed_kmh']:.1f}km/h")
        if 'min_dist_m' in sensor:
            parts.append(f"d={sensor['min_dist_m']:.1f}m")
        line1 = " | ".join(parts)

        # Line 2: state-specific
        line2 = ""
        line3 = ""

        if self.state == self.ENCOUNTER_LIST:
            n_total = len(self.encounters)
            n_coded = sum(1 for e in self.encounters if e['status'] == 'coded')
            n_skip = sum(1 for e in self.encounters if e['status'] == 'skipped')
            if n_total > 0:
                enc = self.encounters[self.selected_enc_idx]
                line2 = (f"ENCOUNTER LIST [{self.selected_enc_idx+1}/{n_total}] "
                         f"({n_coded} coded, {n_skip} skipped) | "
                         f"E{enc['idx']+1:03d} minD={enc['min_dist']:.1f}m "
                         f"T{enc['primary_track']} [{enc['status']}]")
            else:
                line2 = "NO ENCOUNTERS DETECTED"
            line3 = "./,=navigate ENTER=open TAB=skip j/p=next/prev r=replay n=new_track d=density D=batch-skip | 6=cal 3=YOLO | ESC=quit"

        elif self.state == self.ENCOUNTER_VIEW:
            enc = self.encounters[self.selected_enc_idx]
            # Compute speed-adaptive zone display
            cur_speed = sensor.get('speed_kmh', enc['speed_kmh'])
            d_entry = max(5, min(20, cur_speed / 3.6 * 3.0))
            obs_list = enc.get('obstacles', [])
            open_obs = [o for o in obs_list if o['type'] == 'obstacle' and o.get('frame_end') is None]
            obs_tag = ""
            if open_obs:
                obs_tag = f" OBS:OPEN@F{open_obs[-1]['frame']}"
            elif obs_list:
                obs_tag = f" OBS:{len(obs_list)}"
            # Awareness status badge (binary: 1=Yes, 0=No, 9=Unknown)
            _aw_val = enc.get('codes', {}).get('AWARE_BEFORE_MINDIST')
            if _aw_val == 1:
                aware_tag = " A:yes"
            elif _aw_val == 0:
                aware_tag = " A:no"
            elif _aw_val == 9:
                aware_tag = " A:?"
            else:
                aware_tag = ""
            line2 = (f"ENCOUNTER E{enc['idx']+1:03d} | "
                     f"frames {enc['frame_start']}-{enc['frame_end']} "
                     f"({enc['duration_s']:.1f}s) | "
                     f"minDist={enc['min_dist']:.1f}m v={enc['speed_kmh']:.0f}km/h "
                     f"decel={enc['peak_decel_ms2']:.1f}m/s2 "
                     f"yaw={enc['peak_yaw_deg_s']:.0f}deg/s | "
                     f"Zone: D={d_entry:.1f}m (v={cur_speed:.0f}) "
                     f"THW={enc.get('min_thw_s', '?')}s{obs_tag}{aware_tag}")
            # V3.0: distance correction suggestion
            suggest, reason = self._should_suggest_distance_correction(enc)
            if suggest:
                line3 = f"[5] DIST CORRECTION RECOMMENDED: {reason} | ENTER=code 7=mark 0=not-aware BACK=list"
            else:
                line3 = "SPACE=play ./,=frame ENTER=code 7=mark 0=not-aware BACK=list | 5=dist o=obs-pt 8=obs 9=note t=traj 3=YOLO 6=cal | ESC=quit"

        elif self.state == self.CODING:
            enc = self.encounters[self.selected_enc_idx]
            var_name = self.coding_var_names[self.coding_var_idx]
            var_def = MANUAL_VARIABLES[var_name]
            current_val = enc['codes'].get(var_name)
            if current_val is not None:
                code_label = var_def.get('codes', {}).get(current_val, '')
                val_str = f"= {current_val} [{code_label}]" if code_label else f"= {current_val}"
            else:
                val_str = ""
            line2 = (f"CODING E{enc['idx']+1:03d} [{self.coding_var_idx+1}/"
                     f"{len(self.coding_var_names)}] {var_def['prompt']} {val_str}")
            if var_def["type"] == "frame_mark":
                # V3.0: Show suggested awareness frame (~8m approach distance)
                af = self._suggest_awareness_frame(enc)
                if af is not None:
                    # Get distance at suggested frame for hint
                    af_dist = ""
                    if self.det_df is not None:
                        tid = enc.get('primary_track')
                        af_row = self.det_df[(self.det_df['track_id'] == tid) &
                                             (self.det_df['frame'] == af)]
                        if not af_row.empty:
                            d = af_row.iloc[0].get('distance_m')
                            if pd.notna(d):
                                af_dist = f" ~{d:.1f}m"
                    line3 = f"Suggested: F{af}{af_dist} | 7=mark 0=not aware ENTER=confirm TAB=can't tell | 5=dist"
                else:
                    line3 = "7=mark 0=not aware ENTER=confirm TAB=can't tell BACK=prev | 5=dist | SPACE=play"
            elif var_def["type"] in ("integer", "float"):
                line3 = "0-9/.=type ENTER=confirm TAB=skip BACK=del | 5=dist | SPACE=play ./,=frame"
            else:
                line3 = "0-9=code TAB=skip BACK=prev ENTER=confirm | 5=dist | SPACE=play ./,=frame"

        elif self.state == self.REVIEW:
            enc = self.encounters[self.selected_enc_idx]
            notes_str = f" [notes: {enc['notes'][:30]}...]" if enc.get('notes') else ""
            line2 = f"REVIEW E{enc['idx']+1:03d}{notes_str} | ENTER=save TAB=add notes BACK=edit"
            warnings = self._validate_encounter(enc)
            if warnings:
                line2 += f" | WARNINGS: {'; '.join(warnings)}"
            line3 = "ENTER=save TAB=notes BACK=edit | ESC=quit"

        elif self.state == self.INTERACTION_GROUPING:
            sel = self.grouping_selected
            if 0 <= sel < len(self.encounters):
                enc = self.encounters[sel]
                grp = enc.get('interaction_group', '-')
                line2 = (f"INTERACTION GROUP | E{enc['idx']+1:03d} T{enc['primary_track']} "
                         f"({enc.get('primary_type','?')[:3]}) group={grp}")
            else:
                line2 = "INTERACTION GROUPING"
            line3 = "./, = navigate | 1-9 = set group | 0 = clear | ENTER = done | BACK = list"

        elif self.state == self.GROUP_CODING:
            if self.groups_to_code:
                gid = self.groups_to_code[self.group_coding_idx]
                var_name = self.group_var_names[self.group_var_idx]
                var_def = GROUP_VARIABLES[var_name]
                members = self.interaction_groups.get(gid, set())
                encs_str = "+".join(f"E{self.encounters[m]['idx']+1:03d}" for m in sorted(members))
                line2 = (f"GROUP {gid} ({encs_str}) [{self.group_var_idx+1}/"
                         f"{len(self.group_var_names)}] {var_def['prompt']}")
                line3 = "0-9=code TAB=skip BACK=prev | ESC=quit"

        elif self.state == self.TRIP_ANNOTATION:
            var_name = self.trip_var_names[self.trip_var_idx]
            var_def = TRIP_VARIABLES[var_name]
            current_val = self.trip_codes.get(var_name)
            if current_val is not None:
                code_label = var_def.get('codes', {}).get(current_val, '')
                val_str = f"= {current_val} [{code_label}]" if code_label else f"= {current_val}"
            else:
                val_str = ""
            line2 = (f"TRIP ANNOTATION [{self.trip_var_idx+1}/"
                     f"{len(self.trip_var_names)}] {var_def['prompt']} {val_str}")
            line3 = "0-9=code TAB=skip BACK=prev/back to list | ESC=back to list"

        elif self.state == self.RIDER_SEGMENT:
            pass_label = "ACCELERATION" if self.rider_pass == 'accel' else "STEERING"
            n_bounds = len(self.rider_boundaries) - 1
            line2 = (f"RIDER {pass_label} — {n_bounds} boundary(ies). "
                     f"r=add c=clear ENTER=code ESC=skip")
            line3 = "SPACE=play r=boundary c=clear BACK=undo ENTER=finalize ESC=skip"

        elif self.state == self.RIDER_SEGMENT_CODING:
            segments = (self.rider_accel_segments if self.rider_pass == 'accel'
                        else self.rider_steer_segments)
            codes = RIDER_ACCEL_CODES if self.rider_pass == 'accel' else RIDER_STEER_CODES
            seg = segments[self.rider_seg_idx]
            pass_label = "ACCEL" if self.rider_pass == 'accel' else "STEER"
            auto_str = ""
            if seg.get('code') is not None:
                auto_str = f" (auto: {codes.get(seg['code'], '?')} — ENTER=accept)"
            codes_str = " | ".join(f"{k}={v}" for k, v in codes.items())
            line2 = (f"RIDER {pass_label} S{self.rider_seg_idx+1}/{len(segments)} "
                     f"(F{seg['frame_start']}-F{seg['frame_end']}){auto_str}")
            line3 = f"{codes_str} | TAB=unknown BACK=prev"

        elif self.state == self.OBSTACLE_MARKING:
            n_zones = len(self.clip_obstacle_zones)
            n_pts = len(self.clip_obstacle_points)
            if getattr(self, '_waiting_zone_type', False):
                line2 = f"SELECT ZONE TYPE at F{self.current_frame} | {n_zones} zone(s)"
                line3 = "1=Pedestrian  2=Shared  3=Non-motor  4=Crosswalk  5=Park  6=Obstacle  7=Dismounted  ESC=Cancel"
            elif self.clip_obstacle_open is not None:
                zone_type_display = self.clip_obstacle_open_type or 'obstacle'
                open_str = f" [ZONE: {zone_type_display} F{self.clip_obstacle_open}-...]"
                click_str = " [CLICK]" if self.obs_click_mode else ""
                n_lane_segs = len(self.clip_lane_lines_list)
                lane_str = f" [LANE x{n_lane_segs}]" if n_lane_segs > 0 else ""
                lane_mode_str = " [LANE MARKING]" if self.lane_marking_mode else ""
                line2 = f"ZONE MARKING | {n_zones} zone(s) {n_pts} pt(s){open_str}{click_str}{lane_str}{lane_mode_str}"
                line3 = "8=close zone  5=click dist  ./,=frame  SPACE=play  ENTER=done  ESC=skip  BACK=undo"
            else:
                open_str = ""
                click_str = " [CLICK]" if self.obs_click_mode else ""
                n_lane_segs = len(self.clip_lane_lines_list)
                lane_str = f" [LANE x{n_lane_segs}]" if n_lane_segs > 0 else ""
                lane_mode_str = " [LANE MARKING]" if self.lane_marking_mode else ""
                line2 = f"ZONE MARKING | {n_zones} zone(s) {n_pts} pt(s){open_str}{click_str}{lane_str}{lane_mode_str}"
                line3 = "8=start/end zone  5=click dist  ./,=frame  SPACE=play  ENTER=done  ESC=skip  BACK=undo"
        elif self.state == self.CALIBRATION_PHASE:
            cal_tag = "[SAVED]" if self.calibration_from_file else "[DEFAULT]"
            line2 = (f"CALIBRATION {cal_tag} | h={self.camera_height_m:.2f}m  "
                     f"pitch={self.pitch_deg:.1f} deg  f={self.focal_length_px:.0f}px")
            if self.calibration_from_file:
                line3 = "ENTER=use saved  6=recalibrate  ESC=skip  ./,=frame  SPACE=play"
            else:
                line3 = "6=calibrate  ENTER=accept+detect  ESC=skip+detect  ./,=frame  SPACE=play"

        elif self.state == self.DONE:
            line2 = "ALL DONE. Encounters saved."
            line3 = "ESC=quit"

        # Calibration overlay
        if self.cal_state is not None:
            if self.cal_state == 'head':
                h_str = self.cal_height_input or f"{self.ped_height_m:.2f}"
                line2 = f"CALIBRATE: Click HEAD [h={h_str}m] | 2=Ref 3=Mark 4=Multi-ped"
                line3 = "Type digits=height | ESC=cancel"
            elif self.cal_state == 'foot':
                h_str = self.cal_height_input or f"{self.ped_height_m:.2f}"
                line2 = f"CALIBRATE: Click FOOT of same person [h={h_str}m]"
                line3 = "Right-click=undo head | ESC=cancel"
            elif self.cal_state == 'ref_p1':
                line2 = "REFERENCE: Click first endpoint of known-length object"
                line3 = "ESC=cancel"
            elif self.cal_state == 'ref_p2':
                line2 = "REFERENCE: Click second endpoint"
                line3 = "ESC=cancel"
            elif self.cal_state == 'ref_input':
                line2 = f"REFERENCE: Type length in meters: {self.cal_ref_input}_"
                line3 = "ENTER=confirm ESC=cancel"
            elif self.cal_state == 'marking_p1':
                n = len(self.cal_marking_pairs)
                line2 = f"MARKINGS: Click endpoint 1 of marking #{n+1} ({n} pairs done)"
                line3 = "Click pairs | ENTER=done (need 2+) | ESC=cancel"
            elif self.cal_state == 'marking_p2':
                line2 = "MARKINGS: Click endpoint 2"
                line3 = "ESC=cancel"
            elif self.cal_state == 'marking_type':
                line2 = f"MARKINGS: {len(self.cal_marking_pairs)} pairs. Select type:"
                line3 = "1=Dash(3m) 2=Crossing(0.5m) 3=Lane(3m) 4=Custom | ESC=cancel"
            elif self.cal_state == 'marking_custom':
                line2 = f"MARKINGS: Custom length: {self.cal_ref_input}_"
                line3 = "ENTER=confirm ESC=cancel"
            elif self.cal_state == 'multi_head':
                n = len(self.cal_ped_pairs)
                h_str = self.cal_height_input or f"{self.ped_height_m:.2f}"
                line2 = f"MULTI-PED: Click HEAD of person #{n+1} [h={h_str}m] ({n} pairs done)"
                line3 = "Type digits=height | ENTER=solve (2+) | Right-click=undo | ESC=cancel"
            elif self.cal_state == 'multi_foot':
                h_str = self.cal_height_input or f"{self.ped_height_m:.2f}"
                line2 = f"MULTI-PED: Click FOOT of same person [h={h_str}m]"
                line3 = "Right-click=undo head | ESC=cancel"

        y = self.height - 68
        for line in [line1, line2, line3]:
            max_chars = self.width // 7
            if len(line) > max_chars:
                line = line[:max_chars - 3] + "..."
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1)
            y += 20

        # Color-coded deceleration indicator (ENCOUNTER_VIEW / CODING)
        if self.state in (self.ENCOUNTER_VIEW, self.CODING) and self.encounters:
            enc = self.encounters[self.selected_enc_idx]
            decel = enc['peak_decel_ms2']
            dc = _decel_color(decel)
            decel_text = f"decel={decel:.1f}m/s2  d_min={enc['min_dist']:.2f}m"
            cv2.putText(img, decel_text, (self.width - 350, self.height - 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, dc, 1)

        return img

    # ─── Encounter list overlay ───

    def _draw_encounter_list(self, img):
        """Draw encounter list overlay on the left side of screen."""
        if self.state != self.ENCOUNTER_LIST or not self.encounters:
            return img

        overlay = img.copy()
        panel_w = min(450, self.width // 2)
        cv2.rectangle(overlay, (0, 0), (panel_w, self.height - 100), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        # Progress counter: confirmed / total / remaining
        _n_coded = sum(1 for e in self.encounters if e['status'] == 'coded')
        _n_skipped = sum(1 for e in self.encounters if e['status'] == 'skipped')
        _n_remaining = len(self.encounters) - _n_coded - _n_skipped

        # Dense scene HUD count
        _n_sec = sum(1 for e in self.encounters if e.get('density_secondary'))
        _n_man = len(self.encounters) - _n_sec
        if _n_sec > 0:
            cv2.putText(img, f"ENCOUNTERS ({len(self.encounters)})  "
                        f"DENSE: {_n_man}/{len(self.encounters)} manual, {_n_sec} auto",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(img, f"ENCOUNTERS ({len(self.encounters)})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Progress bar text: Viewing X/Y | Coded: N | Remaining: R | Speed
        _speed_str = ""
        if self._coding_timestamps and len(self._coding_timestamps) >= 2:
            # Compute coding speed from last 5 encounters
            recent = self._coding_timestamps[-5:]
            dt = recent[-1][1] - recent[0][1]
            if dt > 0:
                enc_per_min = (len(recent) - 1) / (dt / 60.0)
                _speed_str = f" | {enc_per_min:.1f} enc/min"
                # Estimate remaining time
                if _n_remaining > 0:
                    est_min = _n_remaining / enc_per_min
                    _speed_str += f" (~{est_min:.0f}min left)"
        elif self._coding_timestamps and len(self._coding_timestamps) == 1:
            elapsed = time.time() - self._session_start_time
            if elapsed > 10:
                _speed_str = f" | 1 coded in {elapsed:.0f}s"
        progress_text = (f"Viewing: {self.selected_enc_idx + 1}/{len(self.encounters)} | "
                         f"Coded: {_n_coded} | Remaining: {_n_remaining}{_speed_str}")
        cv2.putText(img, progress_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (180, 255, 180), 1)

        # Fatigue warning (amber text at top-right)
        fatigue_warn = self._check_fatigue()
        if fatigue_warn:
            cv2.putText(img, fatigue_warn, (self.width - 500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

        # Filter encounters for display based on show_density_secondary toggle
        _display_encounters = self.encounters
        if not self.show_density_secondary:
            _display_encounters = [e for e in self.encounters
                                   if not e.get('density_secondary')]

        # Show encounters around selected index
        visible_start = max(0, self.selected_enc_idx - 10)
        visible_end = min(len(self.encounters), visible_start + 22)

        y = 60
        for i in range(visible_start, visible_end):
            enc = self.encounters[i]
            is_selected = (i == self.selected_enc_idx)

            # Status color
            status_colors = {
                'pending': (180, 180, 180),
                'coding': (0, 200, 255),
                'coded': (255, 255, 0),      # Cyan
                'skipped': (100, 100, 100),
                'review_later': (0, 165, 255),  # Orange
            }
            color = status_colors.get(enc['status'], (180, 180, 180))

            # Selection indicator
            prefix = ">>" if is_selected else "  "
            marker = {
                'pending': "[ ]",
                'coding': "[~]",
                'coded': "[+]",
                'skipped': "[x]",
                'review_later': "[?]",
            }.get(enc['status'], "[ ]")

            # Distance color (colorblind-safe)
            dist = enc['min_dist']
            if dist < 1.5:
                dist_color = (255, 0, 255)    # Magenta
            elif dist < 3.0:
                dist_color = (0, 165, 255)    # Orange
            elif dist < 5.0:
                dist_color = (0, 255, 255)    # Yellow
            else:
                dist_color = (255, 255, 0)    # Cyan

            ptype = self.VRU_TYPE_LABELS.get(enc.get('primary_type', ''), 'UNK')
            yaw = enc.get('peak_yaw_deg_s', 0)
            mvmt = enc.get('vru_movement', '?')
            mvmt_short = {'stationary': 'STAT', 'walker': 'WALK', 'runner': 'RUN', 'unknown': '?'}.get(mvmt, '?')
            spd_str = f"{enc['vru_speed_kmh']:.0f}km/h" if enc.get('vru_speed_kmh') else ""
            line = f"{prefix} {marker} E{enc['idx']+1:03d} {dist:.1f}m T{enc['primary_track']}({ptype}) {mvmt_short} {spd_str} {enc['duration_s']:.1f}s"
            text_color = (255, 255, 255) if is_selected else color
            font_scale = 0.45 if is_selected else 0.4
            thickness = 2 if is_selected else 1

            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, text_color, thickness)

            # Show specific flag reasons in red + linked tracks in yellow
            flag_parts = []
            if enc.get('flags'):
                flag_map = {'short': '[SHORT]', 'size_jump': '[SIZE]', 'pos_jump': '[JUMP]', 'swap': '[SWAP]'}
                flag_parts = [flag_map.get(f, f"[{f}]") for f in enc['flags']]
            if enc.get('density_secondary'):
                flag_parts.append(f"[DENSE R{enc.get('density_rank', '?')}]")
            if enc.get('in_steering_zone'):
                flag_parts.append("[STEER]")
            if enc.get('in_obstacle_zone'):
                flag_parts.append("[OBS]")
            linked = enc.get('linked_tracks', [])
            if linked:
                flag_parts.append("=T" + ",".join(str(t) for t in linked))
            if flag_parts:
                flag_str = " ".join(flag_parts)
                flag_x = 10 + len(line) * 8
                flag_color = (0, 200, 255) if linked else (0, 0, 255)  # Yellow if linked, red for flags
                cv2.putText(img, flag_str, (min(flag_x, panel_w - 100), y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, flag_color, 1)

            # Distance dot
            cv2.circle(img, (panel_w - 20, y - 5), 5, dist_color, -1)

            y += 22

        return img

    # ─── Coding overlay ───

    def _draw_coding_overlay(self, img):
        """Show coding variable options at top of screen."""
        if self.state not in (self.CODING, self.TRIP_ANNOTATION,
                              self.RIDER_SEGMENT_CODING):
            return img

        if self.state == self.CODING:
            var_name = self.coding_var_names[self.coding_var_idx]
            var_def = MANUAL_VARIABLES[var_name]
            enc = self.encounters[self.selected_enc_idx]
            current_val = enc['codes'].get(var_name)
        elif self.state == self.RIDER_SEGMENT_CODING:
            # Build a synthetic var_def for the rider segment
            codes = RIDER_ACCEL_CODES if self.rider_pass == 'accel' else RIDER_STEER_CODES
            segments = (self.rider_accel_segments if self.rider_pass == 'accel'
                        else self.rider_steer_segments)
            seg = segments[self.rider_seg_idx]
            pass_label = "ACCELERATION" if self.rider_pass == 'accel' else "STEERING"
            var_name = pass_label
            var_def = {"type": "categorical", "codes": codes,
                       "prompt": f"{pass_label} PHASE"}
            current_val = seg.get('code')
        else:
            var_name = self.trip_var_names[self.trip_var_idx]
            var_def = TRIP_VARIABLES[var_name]
            current_val = self.trip_codes.get(var_name)

        # Semi-transparent panel
        overlay = img.copy()
        panel_h = max(60, 30 + (len(var_def.get("codes", {})) // 3 + 1) * 22 + 10)
        panel_h = min(panel_h, 180)
        cv2.rectangle(overlay, (0, 0), (self.width, panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        if self.state == self.CODING:
            n_total = len(self.coding_var_names)
            idx = self.coding_var_idx
            label = "CODING"
        elif self.state == self.RIDER_SEGMENT_CODING:
            segments = (self.rider_accel_segments if self.rider_pass == 'accel'
                        else self.rider_steer_segments)
            n_total = len(segments)
            idx = self.rider_seg_idx
            pass_name = "ACCEL" if self.rider_pass == 'accel' else "STEER"
            label = f"RIDER {pass_name}"
        else:
            n_total = len(self.trip_var_names)
            idx = self.trip_var_idx
            label = "TRIP"
        cv2.putText(img, f"[{label} {idx+1}/{n_total}] {var_def['prompt']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if var_def["type"] == "categorical":
            x, y = 10, 50
            for code, label_text in var_def["codes"].items():
                text = f"{code} = {label_text}"
                color = (0, 255, 0) if current_val == code else (200, 200, 200)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                x += len(text) * 9 + 20
                if x > self.width - 200:
                    x = 10
                    y += 22
            # Show ENTER hint for pre-filled categorical variables (auto-fill or default)
            if current_val is not None and (var_def.get("auto_fill") or var_def.get("default") is not None):
                cv2.putText(img, "ENTER=accept | or press number to override",
                            (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            # Show VLM suggestion hint if available and not already pre-filled
            if (self.state == self.CODING and current_val is not None
                    and self.encounters):
                enc_ref = self.encounters[self.selected_enc_idx]
                vlm = self._vlm_suggestions.get(enc_ref.get('primary_track'), {})
                if var_name in vlm:
                    vlm_code = vlm[var_name]
                    vlm_label = var_def.get("codes", {}).get(vlm_code, "?")
                    vlm_match = " (matches)" if current_val == vlm_code else ""
                    hint_y = y + 10 if not var_def.get("auto_fill") else y + 26
                    cv2.putText(img, f"VLM: {vlm_code}={vlm_label}{vlm_match}",
                                (10, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 200, 0), 1)
            # Show auto-suggestion hint for trip variables
            if self.state == self.TRIP_ANNOTATION:
                auto_val = self._get_trip_auto_value(var_name)
                if auto_val is not None and auto_val in var_def.get("codes", {}):
                    suggestion_label = var_def["codes"][auto_val]
                    cv2.putText(img, f"Suggested: {auto_val}={suggestion_label} (ENTER=accept)",
                                (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        elif var_def["type"] == "multiselect":
            x, y = 10, 50
            for code, label_text in var_def["codes"].items():
                selected = str(code) in self.input_buffer.split(",") if self.input_buffer else False
                color = (0, 255, 0) if selected else (200, 200, 200)
                cv2.putText(img, f"{code}={label_text}", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                x += len(f"{code}={label_text}") * 9 + 15
                if x > self.width - 200:
                    x = 10
                    y += 22
            cv2.putText(img, f"Selected: [{self.input_buffer}]  ENTER to confirm",
                        (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        elif var_def["type"] == "frame_mark":
            # Show mark instructions + current marked value
            enc_ref = self.encounters[self.selected_enc_idx] if self.state == self.CODING else None
            if current_val == 0 or current_val == 0.0:
                # Key-0: VRU never showed awareness
                cv2.putText(img, "NO AWARENESS OBSERVED (coded: VRU unaware)",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 255), 2)
                cv2.putText(img, "ENTER=confirm | 7=re-mark | 0=toggle off | TAB=can't tell",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            elif current_val is not None and current_val != "":
                aware_frame = enc_ref.get('_aware_frame', '?') if enc_ref else '?'
                off_frame = enc_ref.get('_aware_offset_frame') if enc_ref else None
                if off_frame is not None:
                    dur = round(enc_ref.get('ts_vru_awareness_offset', 0) - current_val, 2)
                    mark_text = f"ONSET F{aware_frame} -> OFFSET F{off_frame} ({dur:.2f}s)"
                else:
                    mark_text = f"ONSET at F{aware_frame} ({current_val:.2f}s) — 7=offset"
                cv2.putText(img, mark_text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                hint = "ENTER=confirm | 7=offset" if off_frame is None else "ENTER=confirm | 7=re-mark onset"
                cv2.putText(img, f"{hint} | TAB=can't tell",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                cv2.putText(img, "Play video, press 7 when VRU becomes aware (onset)",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                cv2.putText(img, "7=onset  0=not aware  ENTER=confirm  TAB=can't tell | SPACE=play ./,=frame",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        elif var_def["type"] in ("integer", "float"):
            type_label = "integer" if var_def["type"] == "integer" else "decimal"
            range_str = ""
            if "min" in var_def or "max" in var_def:
                lo = var_def.get("min", 0)
                hi = var_def.get("max", "")
                range_str = f"  [{lo}-{hi}]"
            # Show auto-fill hint if available
            auto_hint = ""
            if var_def.get("auto_fill"):
                if self.state == self.CODING:
                    enc = self.encounters[self.selected_enc_idx]
                    auto_val = enc.get(var_def["auto_fill"])
                elif self.state == self.TRIP_ANNOTATION:
                    auto_val = self._get_trip_auto_value(var_name)
                else:
                    auto_val = None
                if auto_val is not None:
                    auto_hint = f"  (auto={auto_val}, ENTER=accept)"
            hint = f"Type {type_label}{range_str}{auto_hint} | ENTER=confirm TAB=skip BACK=delete"
            cv2.putText(img, hint, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            # Show current value (pre-filled or typed)
            if not self.input_buffer and current_val is not None and current_val != "":
                display_val = str(current_val)
                val_color = (0, 200, 255)  # Cyan = auto-filled
            else:
                display_val = self.input_buffer if self.input_buffer else "_"
                val_color = (0, 255, 0)    # Green = manual
            cv2.putText(img, f"Value: {display_val}_", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, val_color, 2)

        return img

    # ─── Review overlay ───

    def _draw_review_overlay(self, img):
        """Show review summary of coded encounter."""
        if self.state != self.REVIEW:
            return img

        enc = self.encounters[self.selected_enc_idx]

        overlay = img.copy()
        panel_w = min(500, self.width // 2)
        cv2.rectangle(overlay, (0, 0), (panel_w, self.height - 100), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        cv2.putText(img, f"REVIEW E{enc['idx']+1:03d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        y = 60
        # Auto fields
        perc_f = enc.get('frame_perception', '?')
        auto_fields = [
            f"Frames: S={enc['frame_start']} P={perc_f} M={enc['frame_mindist']} E={enc['frame_end']}",
            f"Duration: {enc['duration_s']:.1f}s | Perception: {enc.get('ts_perception', 0):.2f}s",
            f"Speed (GPS corrected): {enc['speed_kmh']:.1f} km/h",
            f"Min dist: {enc['min_dist']:.2f}m",
            f"Peak decel: {enc['peak_decel_ms2']:.2f} m/s2",
            f"Peak yaw: {enc['peak_yaw_deg_s']:.1f} deg/s",
            f"VRU count: {enc['vru_count']} (auto)",
            f"Primary track: T{enc['primary_track']} ({enc.get('primary_type', 'unknown')})",
            f"VRU movement: {enc.get('vru_movement', '?')} ({enc.get('vru_speed_kmh', '?')} km/h)",
        ]
        for line in auto_fields:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (150, 200, 200), 1)
            y += 18

        y += 10
        cv2.putText(img, "--- Manual codes ---", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y += 22

        for var_name, var_def in MANUAL_VARIABLES.items():
            val = enc['codes'].get(var_name)
            if val is not None and var_def["type"] == "categorical":
                label = var_def["codes"].get(val, str(val))
                line = f"{var_name}: {val} ({label})"
            elif val is not None:
                line = f"{var_name}: {val}"
            else:
                line = f"{var_name}: --"
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1)
            y += 18

        # Show notes (existing or being edited)
        y += 10
        if self.notes_editing:
            cv2.putText(img, "EDITING NOTES (ENTER=save, ESC=cancel):", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y += 20
            # Show cursor with blinking effect
            display_text = self.notes_buffer + "_"
            cv2.putText(img, display_text[:70], (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 18
            if len(self.notes_buffer) > 70:
                cv2.putText(img, self.notes_buffer[70:140], (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y += 18
            # Category hints (only show when buffer is empty or short)
            if len(self.notes_buffer) < 3:
                hint = "[TRACK] [OCCLUSION] [DISTANCE] [CHANGE] [GROUP] [INFRA] [SENSOR] [ZONE] [VRU] [OTHER]"
                cv2.putText(img, hint, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (128, 128, 128), 1)
                y += 16
        elif enc.get('notes'):
            cv2.putText(img, f"NOTES: {enc['notes'][:60]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            y += 18

        return img

    # ─── Grouping overlay ───

    def _draw_grouping_overlay(self, img):
        """Show interaction grouping or group coding panel on screen."""
        if self.state not in (self.INTERACTION_GROUPING, self.GROUP_CODING):
            return img

        overlay = img.copy()
        panel_w = min(500, self.width // 2)
        panel_h = min(self.height - 100, 60 + len(self.encounters) * 22 + 80)
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        if self.state == self.INTERACTION_GROUPING:
            cv2.putText(img, "INTERACTION GROUPING", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            y = 50
            for i, enc in enumerate(self.encounters):
                if enc.get('status') not in ('coded',):
                    continue
                grp = enc.get('interaction_group')
                grp_str = f"G{grp}" if grp else " -"
                selected = (i == self.grouping_selected)
                color = (0, 255, 0) if selected else (200, 200, 200)
                marker = ">>>" if selected else "   "
                line = (f"{marker} E{enc['idx']+1:03d} T{enc['primary_track']:3d} "
                        f"({enc.get('primary_type','?')[:3]}) {grp_str}")
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            color, 1)
                y += 20

            y += 10
            for gid, members in sorted(self.interaction_groups.items()):
                encs = ", ".join(f"E{self.encounters[m]['idx']+1:03d}"
                                for m in sorted(members))
                cv2.putText(img, f"Group {gid}: {encs}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                y += 18

        elif self.state == self.GROUP_CODING and self.groups_to_code:
            gid = self.groups_to_code[self.group_coding_idx]
            members = self.interaction_groups.get(gid, set())
            encs_str = ", ".join(f"E{self.encounters[m]['idx']+1:03d}"
                                for m in sorted(members))
            cv2.putText(img, f"GROUP {gid} CODING ({encs_str})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            y = 55
            var_name = self.group_var_names[self.group_var_idx]
            var_def = GROUP_VARIABLES[var_name]
            cv2.putText(img, f"[{self.group_var_idx+1}/{len(self.group_var_names)}] "
                        f"{var_def['prompt']}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25

            if var_def["type"] == "categorical":
                for code, label in var_def["codes"].items():
                    cv2.putText(img, f"  {code} = {label}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y += 18

            # Show already-coded variables for this group
            y += 10
            for vn in self.group_var_names[:self.group_var_idx]:
                val = self.group_codes[gid].get(vn)
                if val is not None:
                    vd = GROUP_VARIABLES[vn]
                    if vd["type"] == "categorical":
                        lbl = vd["codes"].get(val, str(val))
                    else:
                        lbl = str(val)
                    cv2.putText(img, f"  {vn}: {val} ({lbl})", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 200, 150), 1)
                    y += 16

        return img

    def _draw_imu_overlay(self, img):
        """Draw IMU signal graphs on the video: Speed, AccX, GyrZ."""
        h = self.height
        w = self.width

        # Graph area: right side, 3 stacked mini-graphs
        gx0 = w - 280
        gw = 250
        gh = 50
        gy_start = 10
        signals = [
            ('speed_kmh', 'Speed (km/h)', (0, 255, 255), 0, 30),
            ('acc_x_g', 'AccX (g)', (0, 200, 255), -0.5, 0.5),
            ('yaw_rate_dps', 'GyrZ (deg/s)', (255, 100, 255), -30, 30),
            ('roll_rate_dps', 'GyrX (deg/s)', (0, 140, 255), -30, 30),
        ]

        # Collect values over a window of ±60 frames around current
        half_win = 60
        f_lo = max(0, self.current_frame - half_win)
        f_hi = min(self.total_frames - 1, self.current_frame + half_win)

        for si, (key, label, color, vmin, vmax) in enumerate(signals):
            gy = gy_start + si * (gh + 15)

            # Semi-transparent background
            overlay = img.copy()
            cv2.rectangle(overlay, (gx0 - 5, gy - 5), (gx0 + gw + 5, gy + gh + 15),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

            # Label
            cv2.putText(img, label, (gx0, gy + gh + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            # Current value
            cur_sensor = self._sensor_by_frame.get(self.current_frame, {})
            cur_val = cur_sensor.get(key)
            if cur_val is not None and not (isinstance(cur_val, float) and np.isnan(cur_val)):
                cv2.putText(img, f"{cur_val:.2f}", (gx0 + gw - 50, gy + gh + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Zero line
            if vmin < 0 < vmax:
                zero_y = gy + int(gh * (vmax / (vmax - vmin)))
                cv2.line(img, (gx0, zero_y), (gx0 + gw, zero_y), (60, 60, 60), 1)

            # Plot signal
            pts = []
            for f in range(f_lo, f_hi + 1):
                sensor = self._sensor_by_frame.get(f, {})
                v = sensor.get(key)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                px = gx0 + int((f - f_lo) / max(1, f_hi - f_lo) * gw)
                v_clamped = max(vmin, min(vmax, v))
                py = gy + int(gh * (1.0 - (v_clamped - vmin) / max(0.001, vmax - vmin)))
                pts.append((px, py))

            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(img, pts[i], pts[i + 1], color, 1)

            # Playhead vertical line
            px_cur = gx0 + int((self.current_frame - f_lo) / max(1, f_hi - f_lo) * gw)
            px_cur = max(gx0, min(gx0 + gw, px_cur))
            cv2.line(img, (px_cur, gy), (px_cur, gy + gh), (255, 255, 255), 1)

        return img

    def _draw_bev_minimap(self, img):
        """Draw a bird's-eye view minimap showing ego + VRU positions,
        obstacle points, and lane lines.

        Uses detection data (distance_m + lateral_m) to plot VRU positions
        relative to the ego-rider in a top-down view.
        """
        # Minimap dimensions and position (top-right corner)
        map_w, map_h = 260, 260
        margin = 10
        mx0 = self.width - map_w - margin
        my0 = margin

        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (mx0 - 2, my0 - 2),
                      (mx0 + map_w + 2, my0 + map_h + 2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Map range: ±15m lateral, 0-20m forward
        lat_range = 15.0   # meters each side
        fwd_range = 20.0   # meters forward

        def world_to_map(d_fwd, d_lat):
            """Convert (forward, lateral) in meters to map pixel coords."""
            px = mx0 + int(map_w * (0.5 + d_lat / (2 * lat_range)))
            py = my0 + map_h - int(map_h * d_fwd / fwd_range)
            return px, py

        # Draw grid
        for d in (5, 10, 15, 20):
            _, gy = world_to_map(d, 0)
            if my0 <= gy <= my0 + map_h:
                cv2.line(img, (mx0, gy), (mx0 + map_w, gy), (40, 40, 40), 1)
                cv2.putText(img, f"{d}m", (mx0 + 2, gy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)
        # Center vertical line (ego path)
        cx_map = mx0 + map_w // 2
        cv2.line(img, (cx_map, my0), (cx_map, my0 + map_h), (40, 40, 40), 1)

        # Draw ego marker at bottom-center (triangle pointing up)
        ego_x, ego_y = world_to_map(0, 0)
        pts_ego = np.array([
            [ego_x, ego_y - 8],
            [ego_x - 5, ego_y + 4],
            [ego_x + 5, ego_y + 4],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts_ego], (0, 200, 0))

        # Draw VRU positions from detection data
        colors_bev = [
            (255, 100, 100), (100, 100, 255), (100, 255, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 100),
            (200, 200, 200), (255, 180, 100), (100, 200, 255), (200, 100, 255),
        ]

        if self.det_df is not None:
            # Get all tracks visible in a window around current frame
            trail_frames = 15  # show trail of last N frames
            f_lo = max(0, self.current_frame - trail_frames)
            f_hi = self.current_frame

            # Get unique tracks
            mask = (self.det_df['frame'] >= f_lo) & (self.det_df['frame'] <= f_hi)
            visible = self.det_df[mask]

            for ti, (tid, grp) in enumerate(visible.groupby('track_id')):
                color = colors_bev[ti % len(colors_bev)]
                trail_pts = []

                for _, row in grp.sort_values('frame').iterrows():
                    d_m = row.get('distance_m')
                    lat_m = row.get('lateral_m', 0.0)
                    if pd.isna(d_m) or d_m <= 0 or d_m > fwd_range:
                        continue
                    if pd.isna(lat_m):
                        lat_m = 0.0

                    px, py = world_to_map(d_m, lat_m)
                    # Clamp to minimap bounds
                    px = max(mx0, min(mx0 + map_w, px))
                    py = max(my0, min(my0 + map_h, py))
                    trail_pts.append((px, py, int(row['frame'])))

                # Draw trail
                if len(trail_pts) >= 2:
                    for i in range(len(trail_pts) - 1):
                        age = f_hi - trail_pts[i][2]
                        alpha = max(0.3, 1.0 - age / max(1, trail_frames))
                        c = tuple(int(v * alpha) for v in color)
                        cv2.line(img, trail_pts[i][:2], trail_pts[i+1][:2], c, 1)

                # Draw current-frame position as filled circle
                if trail_pts:
                    last = trail_pts[-1]
                    if last[2] == self.current_frame:
                        cv2.circle(img, last[:2], 4, color, -1)
                        cv2.putText(img, f"T{tid}", (last[0] + 5, last[1] - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

        # ── Obstacle points on BEV ──
        cx = self.width / 2.0
        obs_colors = {
            1: (0, 0, 200),    # Bollard — red
            2: (0, 140, 255),  # Bench — orange
            3: (200, 200, 0),  # Vehicle — cyan
            4: (0, 200, 255),  # Construction — yellow
            5: (0, 180, 0),    # Vegetation — green
            9: (180, 180, 180),  # Other — grey
        }
        # Clip-level obstacle points
        for op in self.clip_obstacle_points:
            d_fwd = op.get('distance_m', 0)
            if d_fwd <= 0 or d_fwd > fwd_range:
                continue
            # Compute lateral from pixel x: lat = d * (px - cx) / f
            op_lat = d_fwd * (op.get('px', cx) - cx) / max(1, self.focal_length_px)
            bx, by = world_to_map(d_fwd, op_lat)
            bx = max(mx0, min(mx0 + map_w, bx))
            by = max(my0, min(my0 + map_h, by))
            oc = obs_colors.get(op.get('type', 9), (180, 180, 180))
            cv2.drawMarker(img, (bx, by), oc, cv2.MARKER_DIAMOND, 8, 2)
        # Per-encounter obstacle points
        if (self.selected_enc_idx is not None
                and 0 <= self.selected_enc_idx < len(self.encounters)):
            enc = self.encounters[self.selected_enc_idx]
            for op in enc.get('obstacle_points', []):
                d_fwd = op.get('distance_m', 0)
                if d_fwd <= 0 or d_fwd > fwd_range:
                    continue
                op_lat = d_fwd * (op.get('px', cx) - cx) / max(1, self.focal_length_px)
                bx, by = world_to_map(d_fwd, op_lat)
                bx = max(mx0, min(mx0 + map_w, bx))
                by = max(my0, min(my0 + map_h, by))
                oc = obs_colors.get(op.get('type', 9), (180, 180, 180))
                cv2.drawMarker(img, (bx, by), oc, cv2.MARKER_TILTED_CROSS, 8, 2)

        # ── Lane lines on BEV ──
        horizon_v = self.height / 2.0 - self.focal_length_px * np.tan(
            np.radians(self.pitch_deg))
        # Determine which lane segment is active for current frame
        active_lane = self._get_lane_for_frame(self.current_frame)
        # Color palette for lane segments (left, right) per segment index
        lane_palettes = [
            ((0, 255, 0), (0, 200, 255)),     # Seg 1: green / orange
            ((255, 255, 0), (255, 100, 255)),  # Seg 2: cyan / magenta
            ((100, 255, 255), (255, 180, 100)),  # Seg 3: yellow / light blue
            ((200, 200, 200), (150, 150, 255)),  # Seg 4: grey / pink
        ]
        def _pixel_to_ground(lx, ly):
            """Convert image pixel to (forward_m, lateral_m) on ground plane."""
            dv = ly - horizon_v
            if dv < 5:
                return None, None
            d = self.focal_length_px * self.camera_height_m / dv
            lat = d * (lx - cx) / max(1, self.focal_length_px)
            return d, lat

        for si, lane_seg in enumerate(self.clip_lane_lines_list):
            is_active = (lane_seg is active_lane)
            thickness = 3 if is_active else 1
            pal = lane_palettes[si % len(lane_palettes)]
            for side_idx, side in enumerate(('left', 'right')):
                color_lane = pal[side_idx]
                if not is_active:
                    color_lane = tuple(c // 2 for c in color_lane)
                pts_lane = lane_seg.get(side, [])
                if len(pts_lane) < 2:
                    continue
                # Project 2 clicked points to ground plane
                gp = []
                for (lx, ly) in pts_lane:
                    d, lat = _pixel_to_ground(lx, ly)
                    if d is not None:
                        gp.append((d, lat))
                if len(gp) < 2:
                    continue
                # Extrapolate line from 2 ground-plane points to 1m..fwd_range
                d0, lat0 = gp[0]
                d1, lat1 = gp[1]
                dd = d1 - d0
                if abs(dd) < 0.01:
                    # Nearly same depth — draw horizontal line
                    ext_pts = [world_to_map(d0, lat0), world_to_map(d1, lat1)]
                else:
                    # Parametric line: lat(d) = lat0 + (lat1-lat0)*(d-d0)/(d1-d0)
                    slope = (lat1 - lat0) / dd
                    d_near = max(1.0, min(d0, d1) - 2.0)
                    d_far = min(fwd_range, max(d0, d1) + 5.0)
                    n_steps = 12
                    ext_pts = []
                    for step in range(n_steps + 1):
                        d_s = d_near + (d_far - d_near) * step / n_steps
                        lat_s = lat0 + slope * (d_s - d0)
                        bpx, bpy = world_to_map(d_s, lat_s)
                        bpx = max(mx0, min(mx0 + map_w, bpx))
                        bpy = max(my0, min(my0 + map_h, bpy))
                        ext_pts.append((bpx, bpy))
                # Draw extrapolated line
                if len(ext_pts) >= 2:
                    for i in range(len(ext_pts) - 1):
                        cv2.line(img, ext_pts[i], ext_pts[i + 1], color_lane, thickness)
                # Mark the actual clicked points as small circles
                for (d_g, lat_g) in gp:
                    cpx, cpy = world_to_map(d_g, lat_g)
                    cpx = max(mx0, min(mx0 + map_w, cpx))
                    cpy = max(my0, min(my0 + map_h, cpy))
                    cv2.circle(img, (cpx, cpy), 3, color_lane, -1)
            # Label active segment with lane width
            if is_active:
                left_gp = []
                right_gp = []
                for side_name, gp_list in [('left', left_gp), ('right', right_gp)]:
                    for (lx, ly) in lane_seg.get(side_name, []):
                        d, lat = _pixel_to_ground(lx, ly)
                        if d is not None:
                            gp_list.append((d, lat))
                if len(left_gp) >= 1 and len(right_gp) >= 1:
                    # Compute width at ego position (d=2m) using extrapolation
                    ref_d = 2.0
                    lat_at_ref = []
                    for gpts in (left_gp, right_gp):
                        if len(gpts) >= 2:
                            dd = gpts[1][0] - gpts[0][0]
                            if abs(dd) > 0.01:
                                slope = (gpts[1][1] - gpts[0][1]) / dd
                                lat_at_ref.append(gpts[0][1] + slope * (ref_d - gpts[0][0]))
                            else:
                                lat_at_ref.append(gpts[0][1])
                        else:
                            lat_at_ref.append(gpts[0][1])
                    if len(lat_at_ref) == 2:
                        w = abs(lat_at_ref[1] - lat_at_ref[0])
                        ego_off_l = abs(lat_at_ref[0])
                        ego_off_r = abs(lat_at_ref[1])
                        cv2.putText(img, f"L{si+1} w={w:.1f}m",
                                    (mx0 + 2, my0 + map_h - 4 - si * 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                    (255, 255, 255), 1)
                        cv2.putText(img, f"ego: L{ego_off_l:.1f} R{ego_off_r:.1f}",
                                    (mx0 + 2, my0 + map_h - 4 - si * 14 + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                                    (180, 180, 180), 1)

        # Label
        n_lanes = len(self.clip_lane_lines_list)
        label = f"BEV" + (f" [L{n_lanes}]" if n_lanes > 0 else "")
        cv2.putText(img, label, (mx0 + 2, my0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return img

    def _draw_gps_minimap(self, img):
        """Draw a GPS trajectory minimap showing ego position on a top-down map."""
        if not self._gps_trajectory:
            return img

        map_w, map_h = 260, 260
        margin = 10
        # Place to the left of BEV if BEV is shown, otherwise top-right
        if self.show_bev_minimap:
            mx0 = self.width - 2 * map_w - 2 * margin - 4
        else:
            mx0 = self.width - map_w - margin
        my0 = margin

        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (mx0 - 2, my0 - 2),
                      (mx0 + map_w + 2, my0 + map_h + 2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Get all GPS points and compute bounds
        all_frames = sorted(self._gps_trajectory.keys())
        all_lats = [self._gps_trajectory[f][0] for f in all_frames]
        all_lons = [self._gps_trajectory[f][1] for f in all_frames]

        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)

        # Add 10% padding
        lat_pad = max((lat_max - lat_min) * 0.1, 0.00005)
        lon_pad = max((lon_max - lon_min) * 0.1, 0.00005)
        lat_min -= lat_pad
        lat_max += lat_pad
        lon_min -= lon_pad
        lon_max += lon_pad

        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min

        def gps_to_map(lat, lon):
            px = mx0 + int(map_w * (lon - lon_min) / lon_span) if lon_span > 0 else mx0 + map_w // 2
            py = my0 + map_h - int(map_h * (lat - lat_min) / lat_span) if lat_span > 0 else my0 + map_h // 2
            return max(mx0, min(mx0 + map_w, px)), max(my0, min(my0 + map_h, py))

        # Draw full trajectory as thin grey line
        pts = []
        for f in all_frames:
            lat, lon = self._gps_trajectory[f]
            pts.append(gps_to_map(lat, lon))
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], (60, 60, 60), 1)

        # Draw past trajectory (up to current frame) in green
        past_pts = []
        for f in all_frames:
            if f <= self.current_frame:
                lat, lon = self._gps_trajectory[f]
                past_pts.append(gps_to_map(lat, lon))
        if len(past_pts) > 1:
            for i in range(len(past_pts) - 1):
                cv2.line(img, past_pts[i], past_pts[i + 1], (0, 180, 0), 2)

        # Draw encounter positions as colored dots
        for enc in self.encounters:
            enc_frame = enc.get('frame_mindist', enc.get('frame_start', 0))
            # Find nearest GPS frame
            nearest_f = min(all_frames, key=lambda f: abs(f - enc_frame))
            lat, lon = self._gps_trajectory[nearest_f]
            epx, epy = gps_to_map(lat, lon)
            color = (0, 200, 255) if enc.get('status') == 'coded' else (100, 100, 255)
            cv2.circle(img, (epx, epy), 4, color, -1)

        # Draw current ego position
        nearest_f = min(all_frames, key=lambda f: abs(f - self.current_frame))
        lat, lon = self._gps_trajectory[nearest_f]
        ego_px, ego_py = gps_to_map(lat, lon)
        cv2.circle(img, (ego_px, ego_py), 6, (0, 255, 0), -1)
        cv2.circle(img, (ego_px, ego_py), 8, (255, 255, 255), 1)

        # Scale bar (approximate)
        import math
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(sum(all_lats) / len(all_lats)))
        map_width_m = lon_span * meters_per_deg_lon
        if map_width_m > 0:
            bar_m = 10 if map_width_m < 50 else 50 if map_width_m < 200 else 100
            bar_px = int(map_w * (bar_m / map_width_m))
            bar_px = min(bar_px, map_w - 20)
            cv2.line(img, (mx0 + 5, my0 + map_h - 8),
                     (mx0 + 5 + bar_px, my0 + map_h - 8), (200, 200, 200), 2)
            cv2.putText(img, f"{bar_m}m", (mx0 + 5, my0 + map_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # Label
        cv2.putText(img, "GPS", (mx0 + 2, my0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return img

    def _draw_rider_segment_overlay(self, img):
        """Draw rider segment boundaries and current segment highlight."""
        if self.state not in (self.RIDER_SEGMENT, self.RIDER_SEGMENT_CODING,
                              self.OBSTACLE_MARKING):
            return img

        h = self.height
        w = self.width

        # Draw a thin timeline bar at the bottom (above status bar)
        bar_y = h - 60
        bar_h = 8
        bar_x0 = 30
        bar_x1 = w - 30
        bar_w = bar_x1 - bar_x0
        cv2.rectangle(img, (bar_x0, bar_y), (bar_x1, bar_y + bar_h),
                      (80, 80, 80), -1)

        # Playhead position
        if self.total_frames > 1:
            px = bar_x0 + int(self.current_frame / (self.total_frames - 1) * bar_w)
            cv2.line(img, (px, bar_y - 5), (px, bar_y + bar_h + 5),
                     (0, 255, 255), 2)

        # Steering/roll episode bars (from IMU auto-detection)
        if self._strong_steering_episodes and self.total_frames > 1:
            for onset, offset, label in self._strong_steering_episodes:
                sx0 = bar_x0 + int(onset / (self.total_frames - 1) * bar_w)
                sx1 = bar_x0 + int(offset / (self.total_frames - 1) * bar_w)
                sx1 = max(sx1, sx0 + 8)
                # Color by type: blue=STEER, orange=ROLL, purple=combined
                if 'ROLL' in label and 'STEER' not in label:
                    fill_color = (0, 140, 255)   # Orange (BGR)
                elif 'STEER' in label and 'ROLL' not in label:
                    fill_color = (255, 140, 0)   # Blue (BGR)
                else:
                    fill_color = (200, 0, 200)   # Purple
                cv2.rectangle(img, (sx0, bar_y - 25), (sx1, bar_y + bar_h),
                              fill_color, -1)
                cv2.rectangle(img, (sx0, bar_y - 25), (sx1, bar_y + bar_h),
                              (255, 255, 255), 2)  # White border
                mid = (sx0 + sx1) // 2
                short_label = label.replace('+', '/').replace('STEER', 'STR').replace('ROLL', 'ROL')
                cv2.putText(img, short_label, (mid - 20, bar_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

        # Draw boundary markers (red vertical lines)
        for f in self.rider_boundaries:
            bx = bar_x0 + int(f / max(1, self.total_frames - 1) * bar_w)
            cv2.line(img, (bx, bar_y - 8), (bx, bar_y + bar_h + 8),
                     (0, 0, 255), 2)

        # In CODING mode: highlight current segment and show color-coded segments
        if self.state == self.RIDER_SEGMENT_CODING:
            segments = (self.rider_accel_segments if self.rider_pass == 'accel'
                        else self.rider_steer_segments)
            # Color map for segment codes
            if self.rider_pass == 'accel':
                code_colors = {1: (0, 200, 0), 2: (0, 0, 255), 3: (180, 180, 180)}
            else:
                code_colors = {1: (255, 200, 0), 2: (200, 0, 255), 3: (180, 180, 180)}

            overlay = img.copy()
            for i, seg in enumerate(segments):
                sx0 = bar_x0 + int(seg['frame_start'] / max(1, self.total_frames - 1) * bar_w)
                sx1 = bar_x0 + int(seg['frame_end'] / max(1, self.total_frames - 1) * bar_w)
                color = code_colors.get(seg.get('code'), (80, 80, 80))
                cv2.rectangle(overlay, (sx0, bar_y - 2), (sx1, bar_y + bar_h + 2),
                              color, -1)
                # Highlight current segment
                if i == self.rider_seg_idx:
                    cv2.rectangle(overlay, (sx0, bar_y - 4), (sx1, bar_y + bar_h + 4),
                                  (255, 255, 255), 2)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        # Draw zones on timeline (colored rectangles by type)
        if self.state == self.OBSTACLE_MARKING:
            zone_colors = {
                'pedestrian_area': (255, 200, 100),  # Light blue
                'shared_space': (0, 255, 255),       # Yellow
                'non_motorised_path': (255, 0, 255), # Magenta
                'crosswalk': (255, 255, 0),          # Cyan
                'park': (0, 255, 0),                 # Green
                'obstacle': (0, 0, 255),             # Red
                'dismounted': (128, 0, 255),         # Purple
            }
            for z in self.clip_obstacle_zones:
                zx0 = bar_x0 + int(z['frame_start'] / max(1, self.total_frames - 1) * bar_w)
                zx1 = bar_x0 + int(z['frame_end'] / max(1, self.total_frames - 1) * bar_w)
                zone_type = z.get('type', 'obstacle')
                color = zone_colors.get(zone_type, (0, 0, 255))
                cv2.rectangle(img, (zx0, bar_y - 3), (zx1, bar_y + bar_h + 3),
                              color, -1)
            # Draw open zone range (dashed, using zone type color)
            if self.clip_obstacle_open is not None:
                ox0 = bar_x0 + int(self.clip_obstacle_open / max(1, self.total_frames - 1) * bar_w)
                ox1 = bar_x0 + int(self.current_frame / max(1, self.total_frames - 1) * bar_w)
                zone_type = self.clip_obstacle_open_type or 'obstacle'
                color = zone_colors.get(zone_type, (0, 0, 255))
                cv2.rectangle(img, (ox0, bar_y - 3), (ox1, bar_y + bar_h + 3),
                              color, 2)
            # Draw obstacle points (clicked distances)
            for op in self.clip_obstacle_points:
                color = (0, 200, 255) if op['frame'] == self.current_frame else (0, 140, 180)
                cv2.drawMarker(img, (op['px'], op['py']), color,
                               cv2.MARKER_CROSS, 14, 2)
                cv2.putText(img, f"{op['distance_m']:.1f}m",
                            (op['px'] + 10, op['py'] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            # Click mode indicator
            if self.obs_click_mode:
                cv2.putText(img, "CLICK: obstacle bottom for distance | 5=done",
                            (self.width // 2 - 250, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        return img

    # ─── Calibration markers ───

    def _draw_calibration(self, img):
        """Draw calibration markers."""
        if self.cal_state == 'foot' and self.cal_head_xy is not None:
            hx, hy = self.cal_head_xy
            cv2.drawMarker(img, (hx, hy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, "HEAD", (hx + 12, hy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.cal_state == 'ref_p2' and self.cal_ref_p1 is not None:
            p1x, p1y = self.cal_ref_p1
            cv2.drawMarker(img, (p1x, p1y), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, "P1", (p1x + 12, p1y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Multi-reference marking pairs
        if self.cal_state in ('marking_p1', 'marking_p2', 'marking_type', 'marking_custom'):
            for i, (p1, p2) in enumerate(self.cal_marking_pairs):
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                         (0, 255, 255), 2)
                cv2.drawMarker(img, (int(p1[0]), int(p1[1])),
                               (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
                cv2.drawMarker(img, (int(p2[0]), int(p2[1])),
                               (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
                mid_x = int((p1[0] + p2[0]) / 2)
                mid_y = int((p1[1] + p2[1]) / 2)
                cv2.putText(img, f"#{i+1}", (mid_x + 5, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Show pending first point
            if self.cal_state == 'marking_p2' and self.cal_marking_p1 is not None:
                px, py = self.cal_marking_p1
                cv2.drawMarker(img, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(img, "P1", (px + 8, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Multi-ped calibration: show collected pairs
        if self.cal_state in ('multi_head', 'multi_foot'):
            for i, (head_xy, foot_xy, ped_h) in enumerate(self.cal_ped_pairs):
                hx, hy = int(head_xy[0]), int(head_xy[1])
                fx, fy = int(foot_xy[0]), int(foot_xy[1])
                cv2.drawMarker(img, (hx, hy), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                cv2.drawMarker(img, (fx, fy), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                cv2.line(img, (hx, hy), (fx, fy), (0, 255, 255), 1)
                cv2.putText(img, f"#{i+1} {ped_h:.2f}m", (hx + 10, hy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            # Show pending head click
            if self.cal_state == 'multi_foot' and self.cal_head_xy is not None:
                hx, hy = self.cal_head_xy
                cv2.drawMarker(img, (hx, hy), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img, "HEAD", (hx + 12, hy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return img

    # ─── Validation ───

    def _validate_encounter(self, enc):
        """Return list of warning strings for display before saving."""
        warnings = []
        codes = enc.get('codes', {})

        # 1. CONFIRM=1 but VRU_TYPE unknown
        if codes.get('CONFIRM') == 1 and codes.get('VRU_TYPE') == 9:
            warnings.append("VRU_TYPE is Unknown (9) — consider reviewing video")

        # 2. Stationary but VRU was moving
        if codes.get('INTERACTION_TYPE') == 4:
            spd = enc.get('vru_speed_kmh', 0) or 0
            if spd > 2.0:
                warnings.append(f"INTERACTION_TYPE=Stationary but VRU speed={spd:.1f} km/h")

        # 3. Awareness (binary categorical — no timestamp validation needed)
        # AWARE_BEFORE_MINDIST: 1=Yes, 0=No, 9=Unknown — no range check required

        # 4. Distance correction changed distance by > 50%
        orig = enc.get('min_dist', 0) or 0
        corr = enc.get('min_dist_corrected', None)
        if corr is not None and orig > 0:
            pct_change = abs(corr - orig) / orig * 100
            if pct_change > 50:
                warnings.append(f"Distance correction: {orig:.1f}m → {corr:.1f}m ({pct_change:.0f}% change)")

        # 5. Child but large bbox (suggesting adult)
        if codes.get('VRU_AGE_GROUP') == 1:
            bh = enc.get('bbox_height_at_mindist', 0) or 0
            if bh > 200:  # bbox > 200px suggests close adult, not child
                warnings.append(f"VRU_AGE_GROUP=Child but bbox height={bh:.0f}px (large — verify)")

        return warnings

    # ─── Magnifier loupe ───

    def _draw_magnifier(self, img, orig_img):
        """Draw a magnifier loupe in the top-right corner showing a 4x zoomed
        crop around the current mouse position, with a crosshair overlay.
        Active during calibration, obstacle marking, distance correction, and
        lane marking modes for pixel-precise clicking."""
        if not self.show_magnifier or self._hover_pos is None:
            return
        # Only show in precision-click modes
        in_precision = (self.dist_correction_mode or self.obstacle_point_mode
                        or self.obs_click_mode or self.lane_marking_mode
                        or self.state == self.CALIBRATION_PHASE)
        if not in_precision:
            return
        mx, my = self._hover_pos  # original image coords
        mag = 4  # magnification factor
        loupe_sz = 180  # loupe display size (px)
        half_src = loupe_sz // (2 * mag)  # source half-window

        # Clamp source crop to image bounds
        h, w = orig_img.shape[:2]
        x0 = max(0, min(mx - half_src, w - 2 * half_src))
        y0 = max(0, min(my - half_src, h - 2 * half_src))
        x1 = min(w, x0 + 2 * half_src)
        y1 = min(h, y0 + 2 * half_src)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return

        crop = orig_img[y0:y1, x0:x1]
        loupe = cv2.resize(crop, (loupe_sz, loupe_sz), interpolation=cv2.INTER_NEAREST)

        # Crosshair at center
        cx, cy = loupe_sz // 2, loupe_sz // 2
        cv2.line(loupe, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
        cv2.line(loupe, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)
        cv2.circle(loupe, (cx, cy), 3, (0, 255, 255), 1)

        # Border
        cv2.rectangle(loupe, (0, 0), (loupe_sz - 1, loupe_sz - 1), (0, 255, 255), 2)

        # Place in top-right corner of display image
        pad = 10
        dy = 30  # below top bar
        rx = img.shape[1] - loupe_sz - pad
        ry = dy
        if rx < 0 or ry < 0:
            return
        img[ry:ry + loupe_sz, rx:rx + loupe_sz] = loupe

    # ─── Mouse callback ───

    def _zoom_to_orig(self, x, y):
        """Convert display coordinates back to original image coordinates."""
        if self.zoom_level <= 1.0:
            return x, y
        # Visible region in original coords
        vw = int(self.width / self.zoom_level)
        vh = int(self.height / self.zoom_level)
        x0 = max(0, min(self.zoom_cx - vw // 2, self.width - vw))
        y0 = max(0, min(self.zoom_cy - vh // 2, self.height - vh))
        # Map display pixel → original pixel
        ox = x0 + int(x * vw / self.width)
        oy = y0 + int(y * vh / self.height)
        return ox, oy

    def _mouse_callback(self, event, x, y, flags, param):
        # ── Track mouse position for real-time distance preview ──
        if event == cv2.EVENT_MOUSEMOVE:
            if self.zoom_level > 1.0:
                ox, oy = self._zoom_to_orig(x, y)
            else:
                ox, oy = x, y
            self._hover_pos = (ox, oy)
            return

        # ── Mouse wheel zoom ──
        if event == cv2.EVENT_MOUSEWHEEL:
            # Scroll up = zoom in, scroll down = zoom out
            if flags > 0:
                self.zoom_level = min(self.zoom_level * 1.3, 6.0)
            else:
                self.zoom_level = max(self.zoom_level / 1.3, 1.0)
            if self.zoom_level <= 1.01:
                self.zoom_level = 1.0
            else:
                # Center zoom on cursor position (convert display → original first)
                self.zoom_cx, self.zoom_cy = self._zoom_to_orig(x, y)
            return

        # ── Translate coordinates when zoomed ──
        if self.zoom_level > 1.0:
            x, y = self._zoom_to_orig(x, y)

        # ── Manual track creation mode: click foot positions ──
        if self.manual_track_mode and event == cv2.EVENT_LBUTTONDOWN:
            frame = self.current_frame
            self.manual_track_points[frame] = (x, y)
            # Compute distance from foot_y
            horizon_v = self.height / 2.0 - self.focal_length_px * np.tan(
                np.radians(self.pitch_deg))
            dv = y - horizon_v
            dist_m = self.focal_length_px * self.camera_height_m / dv if dv > 5 else 0
            print(f"    [MANUAL] F{frame}: foot=({x},{y}), d={dist_m:.2f}m  "
                  f"({len(self.manual_track_points)} pts total)")
            return

        if self.manual_track_mode and event == cv2.EVENT_RBUTTONDOWN:
            # Undo: remove current frame's point
            frame = self.current_frame
            if frame in self.manual_track_points:
                del self.manual_track_points[frame]
                print(f"    [MANUAL] Removed F{frame}. {len(self.manual_track_points)} pts remain.")
            return

        # ── Lane marking mode: click 4 points for left/right path edges ──
        if self.lane_marking_mode and event == cv2.EVENT_LBUTTONDOWN:
            self.lane_marking_clicks.append((x, y))
            n = len(self.lane_marking_clicks)
            if n == 1:
                print(f"    [LANE] Left edge point 1: ({x}, {y}). Click point 2.")
            elif n == 2:
                print(f"    [LANE] Left edge point 2: ({x}, {y}). Now click 2 points for RIGHT edge.")
            elif n == 3:
                print(f"    [LANE] Right edge point 3: ({x}, {y}). Click point 4 to complete.")
            elif n == 4:
                new_lane = {
                    'left': [self.lane_marking_clicks[0], self.lane_marking_clicks[1]],
                    'right': [self.lane_marking_clicks[2], self.lane_marking_clicks[3]],
                    'frame': self.current_frame,
                }
                self.clip_lane_lines = new_lane  # keep legacy field updated
                self.clip_lane_lines_list.append(new_lane)
                self.lane_marking_clicks = []
                self.lane_marking_mode = False
                n_segs = len(self.clip_lane_lines_list)
                print(f"    [LANE] Segment {n_segs} saved at F{self.current_frame}.")
                print(f"    [LANE] >> Press 'l' to add another lane pair, or ENTER to finish. <<")
            return

        if self.lane_marking_mode and event == cv2.EVENT_RBUTTONDOWN:
            if self.lane_marking_clicks:
                removed = self.lane_marking_clicks.pop()
                print(f"    [LANE] Undid click ({removed[0]}, {removed[1]}). {len(self.lane_marking_clicks)}/4 clicks.")
            elif self.clip_lane_lines_list:
                removed = self.clip_lane_lines_list.pop()
                # Update legacy field to last remaining or None
                self.clip_lane_lines = self.clip_lane_lines_list[-1] if self.clip_lane_lines_list else None
                print(f"    [LANE] Removed segment at F{removed['frame']}. "
                      f"{len(self.clip_lane_lines_list)} segment(s) remain.")
                if not self.clip_lane_lines_list:
                    self.lane_marking_mode = False
            else:
                self.lane_marking_mode = False
                print("    [LANE] Cancelled.")
            return

        # ── Obstacle click-to-measure (OBSTACLE_MARKING phase, key 5) ──
        if self.obs_click_mode and self.state == self.OBSTACLE_MARKING:
            if event == cv2.EVENT_LBUTTONDOWN:
                cy_val = self.height / 2.0
                horizon_v = cy_val - self.focal_length_px * np.tan(
                    np.radians(self.pitch_deg))
                dv = y - horizon_v
                if dv <= 1:
                    print(f"    [OBS-5] Click at/above horizon — can't compute distance.")
                    return
                obs_dist = round(self.focal_length_px * self.camera_height_m / dv, 2)
                self.clip_obstacle_points.append({
                    'frame': self.current_frame,
                    'distance_m': obs_dist,
                    'px': x, 'py': y,
                })
                print(f"    [OBS-5] Obstacle at ({x},{y}) → d={obs_dist:.2f}m "
                      f"(F{self.current_frame}). Total: {len(self.clip_obstacle_points)}. "
                      f"Click more, right-click=undo, 5=done.")
                return
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.clip_obstacle_points:
                    removed = self.clip_obstacle_points.pop()
                    print(f"    [OBS-5] Undid last obstacle "
                          f"({removed['distance_m']:.2f}m). "
                          f"{len(self.clip_obstacle_points)} remaining.")
                else:
                    self.obs_click_mode = False
                    print(f"    [OBS-5] No points to undo. Click mode off.")
                return

        # ── Distance correction mode: click body parts ──
        if self.dist_correction_mode and event == cv2.EVENT_LBUTTONDOWN:
            if self.dist_correction_pending_click is not None:
                # Already have an unlabeled click — warn
                print(f"    [DIST] Label previous click first (1-5), or right-click to undo.")
                return
            # Quick-foot mode: auto-label as foot and compute immediately
            if getattr(self, 'dist_correction_quick_foot', False):
                part_name, part_pos = self.BODY_PART_POS[5]  # foot
                self.dist_correction_points.append((x, y, 5, part_name))
                # Compute distance immediately from foot click
                cy_val = self.height / 2.0
                horizon_v = cy_val - self.focal_length_px * np.tan(
                    np.radians(self.pitch_deg))
                dv = y - horizon_v
                if dv > 1:
                    quick_dist = round(self.focal_length_px * self.camera_height_m / dv, 2)
                    enc = self.encounters[self.selected_enc_idx]
                    tid = enc['primary_track']
                    frame = self.current_frame
                    self.dist_corrections[(frame, tid)] = quick_dist
                    self.dist_correction_history[frame] = list(self.dist_correction_points)
                    self.dist_correction_last_result = (frame, quick_dist, x, y)
                    all_corr = [d for (f, t), d in self.dist_corrections.items() if t == tid]
                    enc['min_dist_corrected'] = round(min(all_corr), 2)
                    print(f"    [DIST] Quick-foot at ({x},{y}) → {quick_dist:.2f}m (F{frame} T{tid})")
                    # Stay in correction mode for next frame
                    self.dist_correction_points = []
                else:
                    print(f"    [DIST] Foot at/above horizon — can't compute.")
                    self.dist_correction_points = []
                return
            self.dist_correction_pending_click = (x, y)
            print(f"    [DIST] Click at ({x}, {y}). Press: 1=Head 2=Shoulder 3=Hip 4=Knee 5=Foot")
            return

        if self.dist_correction_mode and event == cv2.EVENT_RBUTTONDOWN:
            if self.dist_correction_pending_click is not None:
                # Undo the unlabeled click
                self.dist_correction_pending_click = None
                print("    [DIST] Right-click: undid unlabeled click.")
            elif self.dist_correction_points:
                # Undo last labeled point
                removed = self.dist_correction_points.pop()
                print(f"    [DIST] Right-click: removed {removed[3]}. {len(self.dist_correction_points)} points remain.")
            else:
                # No points → cancel mode
                self.dist_correction_mode = False
                print("    [DIST] Cancelled.")
            return

        # ── Obstacle point marking mode: click ground contact point (1-3 per obstacle) ──
        if self.obstacle_point_mode and event == cv2.EVENT_LBUTTONDOWN:
            if len(self._obs_pt_multi) >= 3:
                print(f"    [OBS-PT] Max 3 points per obstacle. Press type (1-5/9) or ENTER to confirm.")
                return
            # Compute distance immediately from click y (pinhole: d = f * h_cam / (y - horizon))
            cy_val = self.height / 2.0
            horizon_v = cy_val - self.focal_length_px * np.tan(
                np.radians(self.pitch_deg))
            dv = y - horizon_v
            if dv <= 1:
                print(f"    [OBS-PT] Click at/above horizon — can't compute distance.")
                return
            obs_dist = round(self.focal_length_px * self.camera_height_m / dv, 2)
            # Compute lateral position for width calculation
            cx_val = self.width / 2.0
            obs_lateral = obs_dist * (x - cx_val) / max(1, self.focal_length_px)
            pt = {'px': x, 'py': y, 'distance_m': obs_dist, 'lateral_m': round(obs_lateral, 3)}
            self._obs_pt_multi.append(pt)
            n_pts = len(self._obs_pt_multi)
            # Stage for type label (update with each new click)
            self._obs_pt_staged = {
                'frame': self.current_frame,
                'type': 9,
                'type_name': 'Other',
            }
            self.obstacle_point_pending_click = (x, y)
            enc = self.encounters[self.selected_enc_idx]
            n_saved = len(enc.get('obstacle_points', []))
            if n_pts < 3:
                print(f"    [OBS-PT] Point {n_pts}/3 at ({x},{y}) d={obs_dist:.2f}m. "
                      f"Click more points or press type (1-5/9) / ENTER to confirm.")
            else:
                print(f"    [OBS-PT] Point 3/3 at ({x},{y}) d={obs_dist:.2f}m (max). "
                      f"Press type: 1=Bollard 2=Bench 3=Vehicle 4=Constr 5=Veg 9=Other | ENTER=Other")
            return

        if self.obstacle_point_mode and event == cv2.EVENT_RBUTTONDOWN:
            if self._obs_pt_multi:
                # Undo last multi-point click
                removed = self._obs_pt_multi.pop()
                n_remain = len(self._obs_pt_multi)
                print(f"    [OBS-PT] Right-click: undid point at ({removed['px']},{removed['py']}). "
                      f"{n_remain} point(s) for current obstacle.")
                if not self._obs_pt_multi:
                    self.obstacle_point_pending_click = None
                    self._obs_pt_staged = None
            else:
                enc = self.encounters[self.selected_enc_idx]
                obs_pts = enc.get('obstacle_points', [])
                if obs_pts:
                    removed = obs_pts.pop()
                    print(f"    [OBS-PT] Undid last obstacle ({removed.get('type_name', '?')} "
                          f"at {removed.get('distance_m', 0):.2f}m). {len(obs_pts)} remaining.")
                else:
                    self.obstacle_point_mode = False
                    print("    [OBS-PT] Cancelled.")
            return

        # Right-click: cancel/undo last calibration click
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.cal_state == 'foot':
                self.cal_state = 'head'
                self.cal_head_xy = None
                print("    [CAL] Right-click: undid head click. Click HEAD again.")
            elif self.cal_state == 'ref_p2':
                self.cal_state = 'ref_p1'
                self.cal_ref_p1 = None
                print("    [CAL] Right-click: undid P1 click. Click first endpoint again.")
            elif self.cal_state == 'marking_p2':
                self.cal_state = 'marking_p1'
                self.cal_marking_p1 = None
                print("    [CAL] Right-click: undid marking P1. Click endpoint 1 again.")
            elif self.cal_state == 'marking_p1' and self.cal_marking_pairs:
                self.cal_marking_pairs.pop()
                n = len(self.cal_marking_pairs)
                print(f"    [CAL] Right-click: removed last pair. {n} pairs remain.")
            elif self.cal_state == 'multi_foot':
                self.cal_state = 'multi_head'
                self.cal_head_xy = None
                print("    [CAL] Right-click: undid head click. Click HEAD again.")
            elif self.cal_state == 'multi_head' and self.cal_ped_pairs:
                removed = self.cal_ped_pairs.pop()
                n = len(self.cal_ped_pairs)
                print(f"    [CAL] Right-click: removed last pair. {n} pairs remain.")
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.cal_state == 'head':
            self.cal_head_xy = (x, y)
            self.cal_state = 'foot'
            print(f"    [CAL] Head at ({x}, {y}). Now click FOOT.")

        elif self.cal_state == 'foot':
            head_x, head_y = self.cal_head_xy
            foot_x, foot_y = x, y
            height_px = foot_y - head_y
            if height_px <= 10:
                print(f"    [CAL] Invalid: foot must be below head.")
                self.cal_state = 'head'
                return

            # Use custom height if typed, otherwise default
            ped_h = self.ped_height_m
            if self.cal_height_input:
                try:
                    ped_h = float(self.cal_height_input)
                    if ped_h < 0.5 or ped_h > 2.5:
                        print(f"    [CAL] Height {ped_h}m out of range [0.5-2.5], using {self.ped_height_m:.2f}m")
                        ped_h = self.ped_height_m
                except ValueError:
                    print(f"    [CAL] Invalid height '{self.cal_height_input}', using {self.ped_height_m:.2f}m")
                    ped_h = self.ped_height_m

            cy = self.height / 2
            f_px = self.focal_length_px
            h_cam = self.camera_height_m  # Keep current camera height (physically constrained)

            # Solve for pitch given fixed camera height:
            #   height_px = ped_h * dv / h_cam
            #   dv = foot_y - horizon_v = foot_y - cy + f*tan(pitch)
            #   => pitch = atan((h_cam * height_px / ped_h - foot_y + cy) / f)
            dv = h_cam * height_px / ped_h
            tan_pitch = (dv - foot_y + cy) / f_px
            pitch_deg = np.degrees(np.arctan(tan_pitch))

            # Sanity check pitch
            if pitch_deg < -5 or pitch_deg > 15:
                print(f"    [CAL] WARNING: solved pitch={pitch_deg:.1f}° is unusual (expected 0-8°).")
                print(f"    [CAL] Check your clicks or try multi-ped calibration (2+ people).")

            distance = f_px * ped_h / height_px
            old_pitch = self.pitch_deg
            self.pitch_deg = pitch_deg
            self.calibration_factor = 1.0

            print(f"    [CAL] Person height: {ped_h:.2f}m, pixels: {height_px}")
            print(f"    [CAL] Distance: {distance:.1f}m")
            print(f"    [CAL] Camera height: {h_cam:.2f}m (kept fixed)")
            print(f"    [CAL] Pitch: {old_pitch:.1f}° -> {pitch_deg:.2f}° (solved)")

            # Re-run auto-detection with corrected distances
            self._recalibrate_and_redetect()

            # Record in history
            self.cal_history.append((h_cam, pitch_deg, 'HEAD+FOOT', 0.0))
            self._print_cal_history()

            self.cal_state = None
            self.cal_head_xy = None
            self.cal_height_input = ""

        elif self.cal_state == 'marking_p1':
            self.cal_marking_p1 = (x, y)
            self.cal_state = 'marking_p2'
            print(f"    [CAL] Marking P1 at ({x}, {y}). Click endpoint 2.")
            return

        elif self.cal_state == 'marking_p2':
            p1 = self.cal_marking_p1
            p2 = (x, y)
            self.cal_marking_pairs.append((p1, p2))
            self.cal_marking_p1 = None
            self.cal_state = 'marking_p1'
            n = len(self.cal_marking_pairs)
            print(f"    [CAL] Marking #{n}: ({p1[0]},{p1[1]}) -> ({x},{y}). "
                  f"Click more or press ENTER when done ({n} pairs).")
            return

        elif self.cal_state == 'ref_p1':
            self.cal_ref_p1 = (x, y)
            self.cal_state = 'ref_p2'
            print(f"    [CAL] Point 1 at ({x}, {y}). Now click second endpoint.")

        elif self.cal_state == 'ref_p2':
            p1x, p1y = self.cal_ref_p1
            p2x, p2y = x, y
            pixel_dist = np.sqrt((p2x - p1x)**2 + (p2y - p1y)**2)
            print(f"    [CAL] Point 2 at ({x}, {y}). Pixel distance: {pixel_dist:.1f}")
            print(f"    [CAL] Type the real-world length in meters and press ENTER.")
            self.cal_state = 'ref_input'
            self.cal_pixel_dist = pixel_dist
            self.cal_ref_p2 = (x, y)
            self.cal_ref_input = ""

        elif self.cal_state == 'multi_head':
            self.cal_head_xy = (x, y)
            self.cal_state = 'multi_foot'
            h_str = self.cal_height_input or f"{self.ped_height_m:.2f}"
            print(f"    [CAL] Head at ({x}, {y}) [h={h_str}m]. Now click FOOT.")

        elif self.cal_state == 'multi_foot':
            head_xy = self.cal_head_xy
            foot_xy = (x, y)
            height_px = foot_xy[1] - head_xy[1]
            if height_px <= 10:
                print(f"    [CAL] Invalid: foot must be below head.")
                self.cal_state = 'multi_head'
                return
            # Parse height for this person
            ped_h = self.ped_height_m
            if self.cal_height_input:
                try:
                    ped_h = float(self.cal_height_input)
                    if ped_h < 0.5 or ped_h > 2.5:
                        print(f"    [CAL] Height {ped_h}m out of range, using {self.ped_height_m:.2f}m")
                        ped_h = self.ped_height_m
                except ValueError:
                    ped_h = self.ped_height_m
            self.cal_ped_pairs.append((head_xy, foot_xy, ped_h))
            n = len(self.cal_ped_pairs)
            self.cal_state = 'multi_head'
            self.cal_head_xy = None
            self.cal_height_input = ""
            print(f"    [CAL] Person #{n}: {height_px:.0f}px [{ped_h:.2f}m]. "
                  f"Click more or ENTER when done ({n} pairs, need 2+).")

    def _solve_marking_calibration(self, target_len):
        """Solve for camera height and pitch from marking pairs."""
        pairs = self.cal_marking_pairs
        if len(pairs) < 2:
            print(f"    [CAL] Need at least 2 pairs.")
            self.cal_state = None
            return

        f_px = self.focal_length_px
        cx = self.width / 2.0
        cy = self.height / 2.0
        line_data = [(p1, p2, target_len) for p1, p2 in pairs]

        print(f"    [CAL] Solving for (h, pitch) from {len(pairs)} markings, each {target_len:.2f}m...")
        result = solve_marking_calibration(line_data, f_px, cx, cy)
        if result is None:
            print(f"    [CAL] Solver failed. Check your clicks.")
            self.cal_state = None
            self.cal_marking_pairs = []
            return

        h, pitch, rmse = result

        # Verify each marking
        print(f"\n    [CAL] SOLUTION: h={h:.3f}m  pitch={pitch:.2f}deg  RMSE={rmse*100:.1f}cm")
        for i, (p1, p2) in enumerate(pairs):
            dist = ground_distance_between(p1, p2, f_px, cx, cy, h, pitch)
            err = dist - target_len
            print(f"      Marking {i+1}: {dist:.3f}m (error: {err*100:+.1f}cm)")

        # Apply
        old_h = self.camera_height_m
        old_pitch = self.pitch_deg
        self.camera_height_m = h
        self.pitch_deg = pitch
        print(f"    [CAL] Applied: h={old_h:.2f}->{h:.3f}m  pitch={old_pitch:.1f}->{pitch:.2f}deg")

        self.cal_state = None
        self.cal_marking_pairs = []
        self.cal_marking_p1 = None

        # Record in history
        self.cal_history.append((h, pitch, 'MARKINGS', rmse))
        self._print_cal_history()

        # Re-detect encounters
        self._recalibrate_and_redetect()

    def _solve_single_reference(self, target_m):
        """Solve for camera height h such that ground distance between ref points == target_m.

        Uses brentq root-finding on h, keeping pitch fixed.
        """
        try:
            from scipy.optimize import brentq
        except ImportError:
            print("    [CAL] ERROR: scipy not installed. Run: pip install scipy")
            return

        p1 = self.cal_ref_p1
        p2_x, p2_y = None, None
        # Recover p2 from pixel_dist and p1 — we stored cal_pixel_dist but not p2.
        # We need the actual pixel coordinates. Store them when clicking.
        if not hasattr(self, 'cal_ref_p2') or self.cal_ref_p2 is None:
            print("    [CAL] Reference point data missing.")
            return

        p2 = self.cal_ref_p2
        f_px = self.focal_length_px
        cx = self.width / 2.0
        cy = self.height / 2.0
        pitch = self.pitch_deg

        print(f"    [CAL] Solving for h: ground_dist(P1,P2) == {target_m:.2f}m (pitch={pitch:.1f}deg fixed)")

        def residual(h):
            d = ground_distance_between(p1, p2, f_px, cx, cy, h, pitch)
            if d == float('inf'):
                return 1e6
            return d - target_m

        # Search for h in [0.5, 2.0]
        try:
            # Check bracket
            r_lo = residual(0.5)
            r_hi = residual(2.0)
            if r_lo * r_hi > 0:
                # No sign change — try broader range or Nelder-Mead fallback
                from scipy.optimize import minimize_scalar
                res = minimize_scalar(lambda h: residual(h)**2, bounds=(0.5, 2.0), method='bounded')
                h_sol = res.x
                err = abs(residual(h_sol))
                if err > 0.1:
                    print(f"    [CAL] Solver couldn't converge (residual={err:.2f}m). Check your clicks.")
                    return
            else:
                h_sol = brentq(residual, 0.5, 2.0, xtol=0.001)
        except Exception as e:
            print(f"    [CAL] Solver failed: {e}")
            return

        dist = ground_distance_between(p1, p2, f_px, cx, cy, h_sol, pitch)
        old_h = self.camera_height_m
        self.camera_height_m = h_sol
        self.calibration_factor = 1.0
        print(f"    [CAL] Reference: {target_m:.2f}m = {self.cal_pixel_dist:.0f}px")
        print(f"    [CAL] Solved h: {old_h:.3f}m -> {h_sol:.3f}m (verified dist={dist:.3f}m)")

        # Record in history
        rmse = abs(dist - target_m)
        self.cal_history.append((h_sol, pitch, 'REFERENCE', rmse))
        self._print_cal_history()

        self._recalibrate_and_redetect()

    def _solve_multi_ped(self):
        """Solve for (h, pitch) from multiple pedestrian HEAD+FOOT pairs."""
        pairs = self.cal_ped_pairs
        if len(pairs) < 2:
            print(f"    [CAL] Need at least 2 pedestrian pairs (have {len(pairs)}).")
            return

        try:
            from scipy.optimize import minimize
        except ImportError:
            print("    [CAL] ERROR: scipy not installed. Run: pip install scipy")
            return

        f_px = self.focal_length_px
        cx = self.width / 2.0
        cy = self.height / 2.0

        def objective(params):
            h, pitch_deg = params
            if h < 0.5 or h > 2.0:
                return 1e10
            if pitch_deg < -5 or pitch_deg > 15:
                return 1e10
            horizon_v = cy - f_px * np.tan(np.radians(pitch_deg))
            total = 0.0
            for head_xy, foot_xy, ped_h in pairs:
                foot_y = foot_xy[1]
                head_y = head_xy[1]
                height_px = foot_y - head_y
                if height_px <= 0:
                    return 1e10
                dv = foot_y - horizon_v
                if dv <= 0:
                    return 1e10
                # Predicted height in pixels: ped_h * f_px / Y where Y = f_px * h / dv
                # = ped_h * dv / h
                predicted_px = ped_h * dv / h
                total += (predicted_px - height_px) ** 2
            return total

        print(f"    [CAL] Solving for (h, pitch) from {len(pairs)} pedestrians...")

        best_result = None
        best_error = float('inf')
        for h_init in [0.85, 0.95, 1.05, 1.15, 1.25]:
            for pitch_init in [1, 3, 5, 7]:
                result = minimize(objective, x0=[h_init, pitch_init],
                                  method='Nelder-Mead',
                                  options={'xatol': 0.001, 'fatol': 0.0001})
                if result.fun < best_error:
                    best_error = result.fun
                    best_result = result

        if best_result is None or best_error > 1e8:
            print(f"    [CAL] Solver failed. Check your clicks.")
            self.cal_state = None
            self.cal_ped_pairs = []
            return

        h, pitch = best_result.x
        rmse_px = np.sqrt(best_error / len(pairs))

        # Verify each person
        horizon_v = cy - f_px * np.tan(np.radians(pitch))
        print(f"\n    [CAL] SOLUTION: h={h:.3f}m  pitch={pitch:.2f}deg  RMSE={rmse_px:.1f}px")
        for i, (head_xy, foot_xy, ped_h) in enumerate(pairs):
            height_px = foot_xy[1] - head_xy[1]
            dv = foot_xy[1] - horizon_v
            predicted_px = ped_h * dv / h
            err = predicted_px - height_px
            print(f"      Person {i+1} [{ped_h:.2f}m]: {height_px:.0f}px actual, {predicted_px:.0f}px predicted (err: {err:+.1f}px)")

        # Apply
        old_h = self.camera_height_m
        old_pitch = self.pitch_deg
        self.camera_height_m = h
        self.pitch_deg = pitch
        self.calibration_factor = 1.0
        print(f"    [CAL] Applied: h={old_h:.2f}->{h:.3f}m  pitch={old_pitch:.1f}->{pitch:.2f}deg")

        self.cal_state = None
        self.cal_ped_pairs = []
        self.cal_height_input = ""

        # Record in history
        self.cal_history.append((h, pitch, 'MULTI-PED', rmse_px))
        self._print_cal_history()

        self._recalibrate_and_redetect()

    def _print_cal_history(self):
        """Print calibration history summary and auto-save to JSON."""
        if not self.cal_history:
            return
        print(f"\n    [CAL] ─── Calibration History ({len(self.cal_history)} entries) ───")
        for i, (h, pitch, method, rmse) in enumerate(self.cal_history):
            rmse_str = f"RMSE={rmse:.3f}" if rmse > 0 else ""
            print(f"      #{i+1} {method:12s}: h={h:.3f}m  pitch={pitch:.2f}deg  {rmse_str}")
        # Show current values
        print(f"      Current: h={self.camera_height_m:.3f}m  pitch={self.pitch_deg:.2f}deg")
        # Auto-save calibration to JSON (same directory as video)
        self._save_calibration()

    def _save_calibration(self):
        """Save current calibration parameters to a JSON file in output directory."""
        cal_data = {
            'camera_height_m': round(self.camera_height_m, 4),
            'pitch_deg': round(self.pitch_deg, 3),
            'focal_length_px': round(self.focal_length_px, 1),
            'assumed_height_m': round(self.ped_height_m, 3),
            'distance_calibration_factor': round(self.calibration_factor, 4),
            'video': os.path.basename(self.video_path),
            # V3.6 skeleton fields (sec 13.3)
            'clip_id': self.video_stem,
            'cy': round(self.height / 2.0 - self.focal_length_px * np.tan(
                np.radians(self.pitch_deg)), 1),
            'cx': round(self.width / 2.0, 1),
            'f': round(self.focal_length_px, 1),
            'h_cam': round(self.camera_height_m, 4),
            'calibration_method': (self.cal_history[-1][2]
                                   if self.cal_history else 'default'),
            'rater_id': self.rater_id,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'history': [
                {'method': method, 'h': round(h, 4), 'pitch': round(p, 3),
                 'rmse': round(rmse, 4)}
                for h, p, method, rmse in self.cal_history
            ],
        }
        # Save to output directory (not next to video - external disk may be read-only)
        output_dir = os.path.dirname(os.path.abspath(self.output_path))
        cal_path = os.path.join(output_dir, f"{self.video_stem}_calibration.json")
        with open(cal_path, 'w', encoding='utf-8') as f:
            json.dump(cal_data, f, indent=2, ensure_ascii=False)
        print(f"      Calibration saved: {cal_path}")

    def _run_encounter_detection(self):
        """Run encounter detection and flag zone-overlapping encounters."""
        print("\n  [AUTO-DETECTION] Running encounter detection...")
        self.encounters, track_summary = auto_detect_encounters(
            self.det_df, self.fps,
            d_threshold=self.d_threshold,
            speed_offset_s=self.speed_offset_s,
            max_lateral_m=self.max_lateral_m,
            thw_threshold=self.thw_threshold,
            max_distance=self.max_distance,
            camera_height=self.camera_height_m,
            min_ego_speed_kmh=self.min_ego_speed_kmh,
            dense_scene_k=self.dense_scene_k,
            dense_scene_n=self.dense_scene_n)
        # Apply constrained zone flags if available
        if hasattr(self, '_constrained_zones_path') and self._constrained_zones_path:
            flag_constrained_zones(self.encounters, self._constrained_zones_path, self.fps)

        # Merge adjacent same-type zones before flagging encounters
        self._merge_adjacent_zones()

        # Flag encounters overlapping with steering/obstacle zones
        self._flag_zone_encounters()

        # Flag encounters with detections in basket mask region
        self._flag_basket_occluded()

        # ── Quality score for encounter prioritization ──
        # Score based on: min_dist (closer=better, 3x), duration_frames (longer=better, 2x),
        # max_bbox_height (taller=more reliable, 1x), mean_confidence (higher=better, 1x).
        # Normalized to 0-100 scale. Used for sorting and --max_encounters filtering.
        if self.encounters:
            # Collect raw values for normalization
            _qs_dists = [e['min_dist'] for e in self.encounters]
            _qs_durs = [(e['frame_end'] - e['frame_start'] + 1) for e in self.encounters]
            _qs_bboxs = [e.get('max_bbox_height', 0.0) for e in self.encounters]
            _qs_confs = [e.get('mean_confidence', 0.0) for e in self.encounters]

            # Min/max for normalization (avoid division by zero)
            _d_min, _d_max = min(_qs_dists), max(_qs_dists)
            _dur_min, _dur_max = min(_qs_durs), max(_qs_durs)
            _bh_min, _bh_max = min(_qs_bboxs), max(_qs_bboxs)
            _cf_min, _cf_max = min(_qs_confs), max(_qs_confs)

            for i, enc in enumerate(self.encounters):
                # Normalize each component to 0-1 (higher = better)
                d_range = _d_max - _d_min if _d_max > _d_min else 1.0
                dur_range = _dur_max - _dur_min if _dur_max > _dur_min else 1.0
                bh_range = _bh_max - _bh_min if _bh_max > _bh_min else 1.0
                cf_range = _cf_max - _cf_min if _cf_max > _cf_min else 1.0

                # Distance: closer = higher score (invert)
                dist_score = 1.0 - (enc['min_dist'] - _d_min) / d_range
                # Duration: longer = higher score
                dur_frames = enc['frame_end'] - enc['frame_start'] + 1
                dur_score = (dur_frames - _dur_min) / dur_range
                # Bbox height: taller = higher score
                bh_score = (enc.get('max_bbox_height', 0.0) - _bh_min) / bh_range
                # Confidence: higher = higher score
                cf_score = (enc.get('mean_confidence', 0.0) - _cf_min) / cf_range

                # Weighted sum: dist(3x) + duration(2x) + bbox(1x) + conf(1x) = 7 total
                raw_score = (dist_score * 3.0 + dur_score * 2.0
                             + bh_score * 1.0 + cf_score * 1.0) / 7.0
                enc['quality_score'] = round(raw_score * 100, 1)

            # Sort by quality_score descending (best first)
            self.encounters.sort(key=lambda e: e.get('quality_score', 0), reverse=True)

            # Apply --max_encounters cap if set
            if self.max_encounters is not None and len(self.encounters) > self.max_encounters:
                n_before = len(self.encounters)
                self.encounters = self.encounters[:self.max_encounters]
                print(f"  [QUALITY] Kept top {self.max_encounters}/{n_before} encounters "
                      f"by quality score (range {self.encounters[-1]['quality_score']:.0f}"
                      f"-{self.encounters[0]['quality_score']:.0f})")

            # Re-sort by frame_mindist for chronological annotation order
            # and re-index after quality filtering
            self.encounters.sort(key=lambda e: e['frame_mindist'])
            for i, enc in enumerate(self.encounters):
                enc['idx'] = i

        # Write track summary CSV (all tracks, including filtered)
        if track_summary:
            output_dir = os.path.dirname(os.path.abspath(self.output_path))
            summary_path = os.path.join(output_dir, f"{self.video_stem}_autodetect.csv")
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'track_id', 'frame_start', 'frame_end', 'num_frames',
                    'min_distance', 'min_THW', 'mean_confidence',
                    'max_bbox_height', 'passed_filter'])
                writer.writeheader()
                writer.writerows(track_summary)
            n_passed = sum(1 for r in track_summary if r['passed_filter'])
            n_total = len(track_summary)
            print(f"  [AUTO-DETECTION] Track summary: {n_passed}/{n_total} passed filter → {summary_path}")

        print(f"  [AUTO-DETECTION] Found {len(self.encounters)} encounters")
        for enc in self.encounters:
            ptype = enc.get('primary_type', 'unk')[:3].upper()
            ffd = enc.get('frame_first_detection', '?')
            flv = enc.get('frame_last_valid', '?')
            flags_str = ""
            all_flags = list(enc.get('flags', []))
            if enc.get('in_steering_zone'):
                all_flags.append('STEER-ZONE')
            if enc.get('in_obstacle_zone'):
                all_flags.append('OBS-ZONE')
            if all_flags:
                flags_str = " " + " ".join(f"[{f.upper()}]" for f in all_flags)
            linked = enc.get('linked_tracks', [])
            link_str = " =T" + ",T".join(str(t) for t in linked) if linked else ""
            ctx_v = enc.get('contextual_vehicles', [])
            ctx_str = f" veh:[{','.join('T'+str(v['track_id']) for v in ctx_v)}]" if ctx_v else ""
            cz_str = f" [{enc['constrained_zone_type']}]" if enc.get('constrained_path') else ""
            print(f"    E{enc['idx']+1:03d}: frames {enc['frame_start']}-{enc['frame_end']} "
                  f"({enc['duration_s']:.1f}s) | minDist={enc['min_dist']:.2f}m "
                  f"| T{enc['primary_track']}({ptype}) | VRUs={enc['vru_count']} "
                  f"| v={enc['speed_kmh']:.0f}km/h | 1st={ffd} last_valid={flv}{flags_str}{link_str}{ctx_str}{cz_str}")

    def _flag_zone_encounters(self):
        """Flag encounters whose frames fall in steering/obstacle zones.

        Encounters are FLAGGED (not excluded) if their min-distance frame
        falls within an obstacle zone and/or a strong-steering zone.
        The flags steering_overlap and obstacle_zone_overlap are stored
        on the encounter dict and written to the output CSV.
        """
        steer_ranges = []
        for seg in getattr(self, 'rider_steer_segments', []):
            code = seg.get('code')
            if code is not None and code != 3:  # Not straight
                steer_ranges.append((seg['frame_start'], seg['frame_end']))

        obs_ranges = [(z['frame_start'], z['frame_end'])
                      for z in self.clip_obstacle_zones]

        for enc in self.encounters:
            mf = enc['frame_mindist']
            enc['in_steering_zone'] = any(s <= mf <= e for s, e in steer_ranges)
            enc['in_obstacle_zone'] = any(s <= mf <= e for s, e in obs_ranges)
            enc['steering_overlap'] = enc['in_steering_zone']
            enc['obstacle_zone_overlap'] = enc['in_obstacle_zone']
            enc['zone_flagged'] = enc['in_steering_zone'] or enc['in_obstacle_zone']

        n_flagged = sum(1 for e in self.encounters if e.get('zone_flagged'))
        if n_flagged > 0:
            print(f"  [ZONES] {n_flagged} encounters flagged "
                  f"(min-distance in steering/obstacle zone)")

    def _flag_basket_occluded(self):
        """Flag encounters where VRU foot position falls within the basket mask region.

        When --basket_mask is set, detections whose foot_x,foot_y falls within
        the mask rectangle (expanded by 20px margin) are considered occluded by
        the basket/handlebar. The encounter is flagged if ANY of its frames have
        the primary track inside the mask region.
        """
        if not self.basket_mask:
            return
        bx1, by1, bx2, by2 = self.basket_mask
        margin = 20
        mx1 = bx1 - margin
        my1 = by1 - margin
        mx2 = bx2 + margin
        my2 = by2 + margin
        n_flagged = 0
        for enc in self.encounters:
            tid = enc['primary_track']
            f_start = enc['frame_start']
            f_end = enc['frame_end']
            track_mask = ((self.det_df['track_id'] == tid)
                          & (self.det_df['frame'] >= f_start)
                          & (self.det_df['frame'] <= f_end))
            track_rows = self.det_df[track_mask]
            basket_hit = False
            n_basket_frames = 0
            for _, r in track_rows.iterrows():
                fx = r.get('foot_x')
                fy = r.get('foot_y')
                if pd.notna(fx) and pd.notna(fy):
                    if mx1 <= float(fx) <= mx2 and my1 <= float(fy) <= my2:
                        basket_hit = True
                        n_basket_frames += 1
            enc['basket_occluded'] = basket_hit
            enc['n_basket_occluded_frames'] = n_basket_frames
            if basket_hit:
                n_flagged += 1
        if n_flagged > 0:
            print(f"  [BASKET] {n_flagged} encounter(s) flagged as basket-occluded")

    def _recalibrate_and_redetect(self):
        """Recompute all distances using calibrated camera model, then re-detect."""
        print("    [CAL] Recomputing distances with calibrated camera model...")

        h = self.camera_height_m
        pitch = self.pitch_deg
        f_px = self.focal_length_px
        cx = self.width / 2.0
        cy = self.height / 2.0
        # Use delta-from-baseline approach for IMU pitch (same as __init__).
        # Raw IMU pitch_deg is in the phone's sensor frame, NOT optical pitch.
        # effective_pitch = calibrated_optical + clamp(IMU_frame - IMU_median, ±3°)
        has_pitch_col = False
        imu_baseline = None
        if 'pitch_deg' in self.det_df.columns:
            valid_pitches = self.det_df['pitch_deg'].dropna()
            if len(valid_pitches) > 10:
                imu_baseline = float(valid_pitches.median())
                pitch_std = float((valid_pitches - imu_baseline).std())
                if pitch_std > 2.0:
                    print(f"    [CAL] IMU pitch too noisy (std={pitch_std:.1f}°) "
                          f"→ using static calibrated pitch only")
                else:
                    has_pitch_col = True
                    print(f"    [CAL] Dynamic pitch: calibrated={pitch:.1f}° "
                          f"+ IMU delta (baseline={imu_baseline:.1f}°, "
                          f"std={pitch_std:.1f}°)")
            else:
                print(f"    [CAL] Static pitch: {pitch:.1f}° "
                      f"(pitch_deg column has < 10 valid samples)")
        else:
            print(f"    [CAL] Static pitch: {pitch:.1f}° "
                  f"(no pitch_deg column in CSV)")
        # Precompute static horizon for fallback
        static_horizon_v = cy - f_px * np.tan(np.radians(pitch))

        # Save old distances for comparison
        old_dist = self.det_df['distance_m'].copy()

        # Recompute distance_m and lateral_m from foot_y using ground-plane model
        # Always uses ground-plane: d = f * h_cam / (foot_y - horizon_y)
        # This is height-independent (works for children/adults/cyclists)
        n_recomputed = 0
        has_foot_x = 'foot_x' in self.det_df.columns
        if 'foot_y' in self.det_df.columns:
            for idx, row in self.det_df.iterrows():
                fy = row.get('foot_y')
                # Use per-frame IMU pitch delta when available, else static
                if has_pitch_col and pd.notna(row.get('pitch_deg')):
                    imu_delta = float(row['pitch_deg']) - imu_baseline
                    imu_delta = max(-3.0, min(3.0, imu_delta))
                    effective_pitch = pitch + imu_delta
                    horizon_v = cy - f_px * np.tan(np.radians(effective_pitch))
                else:
                    horizon_v = static_horizon_v
                if pd.notna(fy) and float(fy) > horizon_v + 1:
                    dv = float(fy) - horizon_v
                    new_d = f_px * h / dv
                    if 0.5 < new_d < 100:
                        self.det_df.at[idx, 'distance_m'] = new_d
                        n_recomputed += 1
                        if has_foot_x:
                            fx = row.get('foot_x')
                            if pd.notna(fx):
                                self.det_df.at[idx, 'lateral_m'] = (float(fx) - cx) * new_d / f_px

        # Update distance_raw_m so smooth_detections doesn't restore old values
        self.det_df['distance_raw_m'] = self.det_df['distance_m'].copy()

        # Re-apply RTS smoothing
        self.det_df = smooth_detections(self.det_df)
        if has_pitch_col:
            method = "ground-plane (dynamic pitch)"
        else:
            method = "ground-plane (static pitch)"
        print(f"    [CAL] Recomputed {n_recomputed} distances via {method}, re-smoothed")

        # Compare distance changes
        new_dist = self.det_df['distance_m']
        ratio = (new_dist / old_dist.replace(0, np.nan)).dropna()
        if len(ratio) > 0:
            print(f"    [CAL] Distance change: median ratio = {ratio.median():.3f}x "
                  f"(range {ratio.min():.3f} - {ratio.max():.3f})")

        # Rebuild sensor index (spline speed + IMU)
        self._sensor_by_frame = {}
        _spline_speed = _build_spline_speed(self.det_df, self.fps, self.speed_offset_s)
        for frame_num, group in self.det_df.groupby('frame'):
            self._det_by_frame[int(frame_num)] = group
            row = group.iloc[0]
            sensor = {}
            fn = int(frame_num)
            if fn in _spline_speed:
                sensor['speed_kmh'] = _spline_speed[fn]
            for col in ('yaw_rate_dps', 'roll_rate_dps', 'acc_x_g'):
                if col in group.columns and pd.notna(row[col]):
                    sensor[col] = float(row[col])
            motor_types = {'car', 'truck', 'bus', 'motorcycle', 'motor_vehicle'}
            if 'user_type' in group.columns:
                vrus = group[~group['user_type'].isin(motor_types)]
            else:
                vrus = group
            valid_vrus = vrus[vrus['distance_m'] > 0]
            if len(valid_vrus) > 0:
                sensor['min_dist_m'] = float(valid_vrus['distance_m'].min())
            self._sensor_by_frame[fn] = sensor

        # Fill sensor data for ALL frames
        self._fill_sensor_all_frames()

        # During pre-encounter phase, don't re-detect yet
        if getattr(self, 'pre_encounter_phase', False):
            print("    [CAL] Distance recomputed. Encounters will be "
                  "detected after calibration phase.")
            return

        # Re-detect encounters
        print("    [CAL] Re-running encounter detection...")
        old_encounters = self.encounters
        self.encounters, _ = auto_detect_encounters(self.det_df, self.fps,
                                                     d_threshold=self.d_threshold,
                                                     speed_offset_s=self.speed_offset_s,
                                                     max_lateral_m=self.max_lateral_m,
                                                     thw_threshold=self.thw_threshold,
                                                     max_distance=self.max_distance,
                                                     camera_height=self.camera_height_m,
                                                     min_ego_speed_kmh=self.min_ego_speed_kmh,
                                                     dense_scene_k=self.dense_scene_k,
                                                     dense_scene_n=self.dense_scene_n)

        # Preserve coding status from old encounters (match by frame range overlap)
        for new_enc in self.encounters:
            for old_enc in old_encounters:
                if (old_enc['status'] in ('coded', 'coding', 'review_later') and
                    abs(new_enc['frame_mindist'] - old_enc['frame_mindist']) < 15):
                    new_enc['codes'] = old_enc['codes']
                    new_enc['status'] = old_enc['status']
                    new_enc['notes'] = old_enc['notes']
                    break

        # Print before/after comparison
        print(f"\n    [CAL] ── Before/After Calibration ──")
        print(f"    [CAL] {'Enc':>4} {'Track':>6} {'Old dist':>9} {'New dist':>9} {'Change':>8}")
        for new_enc in self.encounters:
            tid = new_enc['primary_track']
            old_match = None
            for old_enc in old_encounters:
                if old_enc['primary_track'] == tid:
                    old_match = old_enc
                    break
            old_d = f"{old_match['min_dist']:.2f}m" if old_match else "new"
            new_d = f"{new_enc['min_dist']:.2f}m"
            if old_match:
                delta = new_enc['min_dist'] - old_match['min_dist']
                chg = f"{delta:+.2f}m"
            else:
                chg = "new"
            print(f"    [CAL] E{new_enc['idx']+1:03d}  T{tid:>4}  {old_d:>9} {new_d:>9} {chg:>8}")

        print(f"    [CAL] Re-detected {len(self.encounters)} encounters")
        print(f"\n    ╔══════════════════════════════════════════════╗")
        print(f"    ║  CALIBRATION COMPLETE                        ║")
        print(f"    ║  h={self.camera_height_m:.3f}m  pitch={self.pitch_deg:.2f}deg  f={self.focal_length_px:.0f}px  ║")
        print(f"    ║  {len(self.encounters)} encounters after recalibration       ║")
        if self.pitch_deg == 0.0:
            print(f"    ║  Note: pitch=0° — distance changes may be    ║")
            print(f"    ║  subtle. Use lane markings (key 6→3) for     ║")
            print(f"    ║  pitch estimation.                            ║")
        print(f"    ╚══════════════════════════════════════════════╝\n")
        # Try to preserve current encounter selection after recalibration
        # Match by primary_track of the encounter the user was viewing
        old_tid = None
        if hasattr(self, '_pre_cal_track'):
            old_tid = self._pre_cal_track
        if old_tid is not None and self.encounters:
            for i, enc in enumerate(self.encounters):
                if enc['primary_track'] == old_tid:
                    self.selected_enc_idx = i
                    self.current_frame = enc['frame_mindist']
                    break
            else:
                self.selected_enc_idx = 0
                if self.encounters:
                    self.current_frame = self.encounters[0]['frame_mindist']
        else:
            self.selected_enc_idx = 0
            if self.encounters:
                self.current_frame = self.encounters[0]['frame_mindist']

    # ─── Coding logic ───

    def _handle_coding_key(self, key):
        """Handle keypress during CODING state."""
        enc = self.encounters[self.selected_enc_idx]
        var_name = self.coding_var_names[self.coding_var_idx]
        var_def = MANUAL_VARIABLES[var_name]

        if var_def["type"] == "categorical":
            if key == 9:  # TAB = skip
                old_val = enc['codes'].get(var_name)
                self._undo_stack.append((var_name, old_val))
                enc['codes'][var_name] = 9
                self._advance_coding()
                return
            # ENTER = accept pre-filled value (for auto-fill variables like VRU_TYPE)
            if (key == 13 or key == 10) and enc['codes'].get(var_name) is not None:
                old_val = enc['codes'].get(var_name)
                self._undo_stack.append((var_name, old_val))
                self._advance_coding()
                return
            if ord('0') <= key <= ord('9'):
                code = key - ord('0')
                if code in var_def["codes"]:
                    old_val = enc['codes'].get(var_name)
                    self._undo_stack.append((var_name, old_val))
                    enc['codes'][var_name] = code
                    if var_def.get("carry_forward"):
                        self.carry_forward[var_name] = code
                    self._advance_coding()
                else:
                    print(f"    Invalid code {code} for {var_name}")

        elif var_def["type"] == "multiselect":
            if key == 13 or key == 10:  # ENTER = confirm
                enc['codes'][var_name] = self.input_buffer if self.input_buffer else "0"
                self.input_buffer = ""
                self._advance_coding()
                return
            if key == 9:  # TAB = skip
                enc['codes'][var_name] = "9"
                self.input_buffer = ""
                self._advance_coding()
                return
            if ord('0') <= key <= ord('9'):
                code = chr(key)
                if self.input_buffer:
                    codes = self.input_buffer.split(",")
                    if code in codes:
                        codes.remove(code)
                    else:
                        codes.append(code)
                    self.input_buffer = ",".join(codes)
                else:
                    self.input_buffer = code

        elif var_def["type"] == "frame_mark":
            # Press mark_key (7): first press = onset, second press = offset
            mark_key = var_def.get("mark_key", ord('7'))
            if key == mark_key:
                ts = round(self.current_frame / self.fps, 2)
                # If onset already marked and no offset yet → mark offset
                if enc.get('_aware_frame') is not None and enc.get('_aware_offset_frame') is None:
                    if self.current_frame > enc['_aware_frame']:
                        enc['_aware_offset_frame'] = self.current_frame
                        enc['ts_vru_awareness_offset'] = ts
                        dur = ts - enc['ts_vru_awareness']
                        print(f"    [AWARE] Offset at F{self.current_frame} ({ts:.2f}s) "
                              f"duration={dur:.2f}s")
                    else:
                        # Pressed on same or earlier frame → re-mark onset
                        enc['codes'][var_name] = ts
                        enc['ts_vru_awareness'] = ts
                        enc['_aware_frame'] = self.current_frame
                        enc.pop('_aware_offset_frame', None)
                        enc.pop('ts_vru_awareness_offset', None)
                        print(f"    [AWARE] Re-marked onset at F{self.current_frame} ({ts:.2f}s)")
                else:
                    # First press or re-mark: set onset, clear offset
                    enc['codes'][var_name] = ts
                    enc['ts_vru_awareness'] = ts
                    enc['_aware_frame'] = self.current_frame
                    enc.pop('_aware_offset_frame', None)
                    enc.pop('ts_vru_awareness_offset', None)
                    print(f"    [AWARE] Onset at F{self.current_frame} ({ts:.2f}s) — press 7 again for offset")
            elif key == ord('0'):  # 0 = "No awareness observed" (VRU unaware)
                enc['codes'][var_name] = 0  # Numeric 0 = coded unaware
                enc['ts_vru_awareness'] = 0
                enc.pop('_aware_frame', None)
                enc.pop('_aware_offset_frame', None)
                enc.pop('ts_vru_awareness_offset', None)
                print(f"    [AWARE] No awareness observed (VRU unaware)")
            elif key == 13 or key == 10:  # ENTER = confirm marked value or skip
                current = enc['codes'].get(var_name)
                if current is not None and current != "":
                    # Confirm marked timestamp or key-0 "not aware"
                    self._advance_coding()
                elif var_def.get("optional"):
                    # ENTER with nothing marked = same as TAB (can't determine)
                    enc['codes'][var_name] = ""
                    self._advance_coding()
                return
            elif key == 9:  # TAB = "Cannot determine" (missing data)
                enc['codes'][var_name] = ""
                enc['ts_vru_awareness'] = ""
                enc.pop('_aware_frame', None)
                enc.pop('_aware_offset_frame', None)
                enc.pop('ts_vru_awareness_offset', None)
                print(f"    [AWARE] Cannot determine (missing)")
                self._advance_coding()
                return

        elif var_def["type"] in ("integer", "float"):
            if key == 13 or key == 10:  # ENTER = confirm
                if self.input_buffer:
                    try:
                        if var_def["type"] == "integer":
                            val = int(self.input_buffer)
                            lo = var_def.get("min", 0)
                            hi = var_def.get("max", 9999)
                            if val < lo or val > hi:
                                print(f"    Value {val} out of range [{lo}-{hi}] for {var_name}")
                                return
                        else:
                            val = float(self.input_buffer)
                            if val < 0:
                                print(f"    Value must be >= 0 for {var_name}")
                                return
                        enc['codes'][var_name] = val
                        if var_def.get("carry_forward"):
                            self.carry_forward[var_name] = val
                        self.input_buffer = ""
                        self._advance_coding()
                    except ValueError:
                        print(f"    Invalid number: {self.input_buffer}")
                elif enc['codes'].get(var_name) not in (None, ""):
                    # Accept pre-filled / auto-filled value
                    self.input_buffer = ""
                    self._advance_coding()
                elif var_def.get("optional"):
                    # Optional field: ENTER with no input = skip (leave blank)
                    enc['codes'][var_name] = ""
                    self.input_buffer = ""
                    self._advance_coding()
                return
            if key == 9:  # TAB = skip
                enc['codes'][var_name] = ""
                self.input_buffer = ""
                self._advance_coding()
                return
            # BACKSPACE handled by main loop (deletes from input_buffer or retreats)
            if ord('0') <= key <= ord('9'):
                self.input_buffer += chr(key)
            elif key == ord('.') and var_def["type"] == "float":
                if '.' not in self.input_buffer:
                    self.input_buffer += '.'

    def _advance_coding(self):
        """Move to next variable. Handles CONFIRM_INTERACTION skip/review logic."""
        enc = self.encounters[self.selected_enc_idx]

        # Check if we just coded CONFIRM_INTERACTION
        if self.coding_var_idx < len(self.coding_var_names):
            var_name = self.coding_var_names[self.coding_var_idx]
            if var_name == "CONFIRM":
                confirm_val = enc['codes'].get("CONFIRM")
                if confirm_val == 0:
                    # No — skip all remaining variables, mark as skipped
                    enc['status'] = 'skipped'
                    enc['coding_end_ts'] = datetime.now().isoformat()
                    self._save_session_state()
                    self.coding_var_idx = 0
                    self.input_buffer = ""
                    print(f"    CONFIRM=No -> skipped E{enc['idx']+1:03d}")
                    # Check if all encounters are now coded or skipped
                    all_done = all(e['status'] in ('coded', 'skipped')
                                   for e in self.encounters)
                    if all_done:
                        self._save_encounters()
                        self._enter_interaction_grouping()
                    else:
                        # Move to next pending encounter (skip review_later)
                        self._reset_modal_flags()
                        self.state = self.ENCOUNTER_LIST
                        found_next = False
                        for i in range(self.selected_enc_idx + 1, len(self.encounters)):
                            if self.encounters[i]['status'] == 'pending':
                                self.selected_enc_idx = i
                                self.current_frame = max(
                                    0, self.encounters[i]['frame_start'] - 30)
                                found_next = True
                                break
                        if not found_next:
                            # Wrap around from start
                            for i in range(0, self.selected_enc_idx):
                                if self.encounters[i]['status'] == 'pending':
                                    self.selected_enc_idx = i
                                    self.current_frame = max(
                                        0, self.encounters[i]['frame_start'] - 30)
                                    found_next = True
                                    break
                        if not found_next:
                            self._save_encounters()
                            self._enter_interaction_grouping()
                    return
                elif confirm_val == 2:
                    # Review later — advance to next pending encounter (skip review_later)
                    enc['status'] = 'review_later'
                    self._reset_modal_flags()
                    self.coding_var_idx = 0
                    print(f"    CONFIRM=Review later -> next pending")
                    # Move to next pending encounter only
                    self.state = self.ENCOUNTER_LIST
                    found_next = False
                    for i in range(self.selected_enc_idx + 1, len(self.encounters)):
                        if self.encounters[i]['status'] == 'pending':
                            self.selected_enc_idx = i
                            self.current_frame = max(
                                0, self.encounters[i]['frame_start'] - 30)
                            found_next = True
                            break
                    if not found_next:
                        for i in range(0, self.selected_enc_idx):
                            if self.encounters[i]['status'] == 'pending':
                                self.selected_enc_idx = i
                                self.current_frame = max(
                                    0, self.encounters[i]['frame_start'] - 30)
                                found_next = True
                                break
                    if not found_next:
                        self._save_encounters()
                        self._enter_interaction_grouping()
                    return

        self.coding_var_idx += 1
        self.input_buffer = ""
        if self.coding_var_idx >= len(self.coding_var_names):
            self.state = self.REVIEW
            self.coding_var_idx = 0
            print(f"    Coding complete -> REVIEW")
        else:
            self._handle_gated_variable_skip()
            self._navigate_to_suggested_frame()

    def _handle_gated_variable_skip(self):
        """Skip gated variables that don't apply to current encounter.

        Also auto-advances past high-confidence auto-filled variables
        (VRU_TYPE, INTERACTION_TYPE) to reduce keystrokes.

        VRU_GAIT is gated: only prompt when VRU_TYPE == 1 (pedestrian).
        For non-pedestrians, auto-fill with 9 (Unknown/not applicable) and skip.

        AWARE_BEFORE_MINDIST auto-skipped when:
          - VRU_TYPE != 1 (non-pedestrian: body language not visible)
        """
        enc = self.encounters[self.selected_enc_idx]
        while self.coding_var_idx < len(self.coding_var_names):
            var_name = self.coding_var_names[self.coding_var_idx]
            var_def = MANUAL_VARIABLES[var_name]

            # Gated variables: skip when condition not met
            if var_def.get("gated"):
                if var_name == "VRU_GAIT":
                    vru_type = enc['codes'].get('VRU_TYPE')
                    if vru_type != 1:
                        enc['codes'][var_name] = 9
                        print(f"    VRU_GAIT auto-filled: 9 (not a pedestrian)")
                        self.coding_var_idx += 1
                        continue

            # Auto-advance high-confidence auto-filled variables
            if var_def.get("auto_fill") and enc['codes'].get(var_name) is not None:
                val = enc['codes'][var_name]
                skip = False
                if var_name == "VRU_TYPE":
                    cp = enc.get('class_prob', 0)
                    if cp >= 0.85 and val != 9:
                        print(f"    VRU_TYPE auto-accepted: {val} "
                              f"({var_def['codes'].get(val, '?')}, conf={cp:.0%})")
                        skip = True
                elif var_name == "INTERACTION_TYPE":
                    pass  # Show auto-suggestion, require ENTER to confirm
                if skip:
                    if var_def.get("carry_forward"):
                        self.carry_forward[var_name] = val
                    self.coding_var_idx += 1
                    continue

            # Auto-skip AWARE_BEFORE_MINDIST for non-pedestrians
            if var_name == "AWARE_BEFORE_MINDIST" and var_def.get("optional"):
                vru_type = enc['codes'].get('VRU_TYPE')
                if vru_type != 1:
                    enc['codes'][var_name] = 9  # Unknown/not applicable
                    print(f"    AWARENESS auto-skipped (VRU_TYPE={vru_type}, not a pedestrian)")
                    self.coding_var_idx += 1
                    continue

            break

        if self.coding_var_idx >= len(self.coding_var_names):
            self.state = self.REVIEW
            self.coding_var_idx = 0
            print(f"    Coding complete -> REVIEW")

    def _retreat_coding(self):
        """Go back to previous variable, restoring the undo stack value."""
        if self.coding_var_idx > 0:
            # Restore previous value from undo stack if available
            if self._undo_stack:
                var_name, old_val = self._undo_stack.pop()
                enc = self.encounters[self.selected_enc_idx]
                enc['codes'][var_name] = old_val
                print(f"    [UNDO] {var_name} -> {old_val}")
            self.coding_var_idx -= 1
            # Skip back over gated variables that were auto-filled
            while self.coding_var_idx > 0:
                vn = self.coding_var_names[self.coding_var_idx]
                vd = MANUAL_VARIABLES[vn]
                if vd.get("gated"):
                    enc = self.encounters[self.selected_enc_idx]
                    if vn == "VRU_GAIT" and enc['codes'].get('VRU_TYPE') != 1:
                        # This was auto-skipped, go back further
                        if self._undo_stack:
                            un, uv = self._undo_stack.pop()
                            enc['codes'][un] = uv
                        self.coding_var_idx -= 1
                        continue
                break
            self.input_buffer = ""
            self._navigate_to_suggested_frame()

    def _quick_reject_encounter(self):
        """Quick-reject current encounter: set CONFIRM=0, skip all variables, advance.

        Bound to key 'x' in ENCOUNTER_VIEW and CODING states.
        Saves a minimal row (EVENT_ID, TRIP_ID, CONFIRM=0, auto-computed variables).
        """
        if not self.encounters:
            return
        enc = self.encounters[self.selected_enc_idx]
        enc['codes']['CONFIRM'] = 0
        enc['status'] = 'skipped'
        enc['coding_end_ts'] = datetime.now().isoformat()
        self.coding_var_idx = 0
        self.input_buffer = ""
        self._save_session_state()
        print(f"  [QUICK-REJECT] E{enc['idx']+1:03d} T{enc['primary_track']} -> CONFIRM=0 (skipped)")

        # Check if all encounters are now coded or skipped
        all_done = all(e['status'] in ('coded', 'skipped')
                       for e in self.encounters)
        if all_done:
            self._save_encounters()
            self._enter_interaction_grouping()
        else:
            # Move to next pending encounter
            self._reset_modal_flags()
            self.state = self.ENCOUNTER_LIST
            found_next = False
            for i in range(self.selected_enc_idx + 1, len(self.encounters)):
                if self.encounters[i]['status'] == 'pending':
                    self.selected_enc_idx = i
                    self.current_frame = max(
                        0, self.encounters[i]['frame_start'] - 30)
                    found_next = True
                    break
            if not found_next:
                # Wrap around from start
                for i in range(0, self.selected_enc_idx):
                    if self.encounters[i]['status'] == 'pending':
                        self.selected_enc_idx = i
                        self.current_frame = max(
                            0, self.encounters[i]['frame_start'] - 30)
                        found_next = True
                        break
            if not found_next:
                self._save_encounters()
                self._enter_interaction_grouping()

    def _suggest_awareness_frame(self, enc):
        """Suggest frame where VRU awareness is most likely observable.

        Strategy: Find frame where VRU is ~8m away (reliable head detection
        range at 720p) and approaching. Falls back to frame where distance
        first drops below 10m, then to frame_mindist.
        """
        track_id = enc.get('primary_track')
        if track_id is None or self.det_df is None:
            return enc.get('frame_mindist')
        trk = self.det_df[self.det_df['track_id'] == track_id].sort_values('frame')
        if trk.empty:
            return enc.get('frame_mindist')
        target_dist = 8.0
        best_frame = None
        best_diff = float('inf')
        mindist_frame = enc.get('frame_mindist', int(trk['frame'].iloc[-1]))
        approach = trk[trk['frame'] <= mindist_frame]
        for _, row in approach.iterrows():
            d = row.get('distance_m')
            if pd.notna(d) and d > 0:
                diff = abs(d - target_dist)
                if diff < best_diff:
                    best_diff = diff
                    best_frame = int(row['frame'])
        return best_frame if best_frame is not None else enc.get('frame_mindist')

    def _reset_modal_flags(self):
        """Reset all modal interaction flags. Call on every state transition."""
        self.dist_correction_mode = False
        self.dist_correction_quick_foot = False
        self.dist_correction_points = []
        self.dist_correction_pending_click = None
        self.obstacle_point_mode = False
        self.obstacle_point_pending_click = None
        self._obs_pt_staged = None
        self._obs_pt_multi = []
        self.lane_marking_mode = False
        self.lane_marking_clicks = []
        self.manual_track_mode = False
        self.input_buffer = ""

    def _check_fatigue(self):
        """Check coding duration and speed degradation. Returns warning or None."""
        elapsed_s = time.time() - self._session_start_time
        # Time-based warning (every 45 min)
        if elapsed_s > 45 * 60:
            mins = int(elapsed_s / 60)
            if not getattr(self, '_fatigue_warned_at', 0) or \
               (elapsed_s - self._fatigue_warned_at) > 30 * 60:
                self._fatigue_warned_at = elapsed_s
                return f"BREAK? {mins}min session. 10-15min break recommended."
        # Speed degradation check (need at least 10 coded encounters)
        if len(self._coding_timestamps) >= 10:
            # Baseline: first 5 intervals
            base_times = self._coding_timestamps[:6]
            base_dt = base_times[-1][1] - base_times[0][1]
            base_speed = 5.0 / max(1, base_dt) if base_dt > 0 else 0
            # Current: last 5 intervals
            recent = self._coding_timestamps[-6:]
            recent_dt = recent[-1][1] - recent[0][1]
            recent_speed = 5.0 / max(1, recent_dt) if recent_dt > 0 else 0
            if base_speed > 0 and recent_speed < 0.5 * base_speed:
                return f"SLOWDOWN: coding speed dropped to {recent_speed/base_speed:.0%} of baseline"
        return None

    def _should_suggest_distance_correction(self, enc):
        """Check if distance correction is recommended for the encounter.

        Suggests correction when:
        1. VRU is partially occluded near mindist frame
        2. VRU is close enough to matter (<10m)
        3. bbox is truncated at frame edge
        4. Non-pedestrian VRU (foot point may not match ground contact)
        Returns (should_suggest: bool, reason: str).
        """
        track_id = enc.get('primary_track')
        if track_id is None:
            return False, ""
        min_dist = enc.get('min_dist', enc.get('min_dist_m', 99))
        if min_dist > 10.0:
            return False, ""
        mindist_frame = enc.get('frame_mindist')
        if mindist_frame is None or self.det_df is None:
            return False, ""
        trk = self.det_df[self.det_df['track_id'] == track_id]
        near_frames = trk[abs(trk['frame'] - mindist_frame) <= 5]
        for _, row in near_frames.iterrows():
            vis = str(row.get('visibility_status', '')).upper()
            occluded = row.get('is_occluded', False)
            foot_x = row.get('foot_x', 640)
            if occluded or vis == 'EDGE' or foot_x < 80 or foot_x > (self.width - 80):
                return True, "VRU occluded/edge-truncated at close range"
        vru_type = enc.get('VRU_TYPE', enc.get('vru_type_code', 1))
        if vru_type in (2, 3, 4):
            return True, "Non-pedestrian: foot point may not match ground"
        return False, ""

    def _navigate_to_suggested_frame(self):
        """Jump to suggested frame for current coding variable.
        Most variables point to the perception frame (S + 200ms) where
        the rider's brain has processed the visual scene.
        VRU_AGE_GROUP points to perception frame.
        """
        if self.state != self.CODING:
            return
        enc = self.encounters[self.selected_enc_idx]
        var_name = self.coding_var_names[self.coding_var_idx]
        var_def = MANUAL_VARIABLES[var_name]
        suggested = var_def.get("suggested_frame")
        if suggested == "perception":
            self.current_frame = enc.get('frame_perception', enc['frame_start'])
        elif suggested == "start":
            self.current_frame = enc['frame_start']
        elif suggested == "mindist":
            self.current_frame = enc['frame_mindist']
        elif suggested == "end":
            self.current_frame = enc['frame_end']

    # ─── Interaction grouping + group-level coding ───

    def _enter_interaction_grouping(self):
        """Enter the interaction grouping state after all encounters are coded/skipped.

        Auto-detects temporal groups: encounters with overlapping frame ranges
        (within 1s tolerance) are grouped. The rider responds to each group as
        a unit, so group-level variables are coded once per group.
        """
        if not GROUP_VARIABLES:
            # No group-level variables → skip grouping, go to trip annotation
            self.state = self.TRIP_ANNOTATION
            self.trip_var_idx = 0
            print("\n  No group-level variables to code. Proceeding to trip annotation.")
            return

        self.interaction_groups = {}
        self.next_group_id = 1

        # Only group coded encounters (skip skipped ones)
        coded_indices = [i for i, e in enumerate(self.encounters)
                         if e.get('status') == 'coded']

        # Auto-detect groups from overlapping frame ranges (1s = 30 frames tolerance)
        # Class constraint: groups cannot mix VRU types (pedestrian vs cyclist vs e-scooter)
        OVERLAP_TOLERANCE = int(self.fps)  # 30 frames = 1 second
        assigned = set()
        for i in coded_indices:
            if i in assigned:
                continue
            enc_i = self.encounters[i]
            group_members = {i}
            # Find all encounters overlapping with this one (transitive, same VRU class)
            changed = True
            while changed:
                changed = False
                for j in coded_indices:
                    if j in group_members:
                        continue
                    enc_j = self.encounters[j]
                    # Class constraint: only group encounters of same VRU type
                    type_i = enc_i.get('codes', {}).get('VRU_TYPE') or enc_i.get('vru_type_code', 9)
                    type_j = enc_j.get('codes', {}).get('VRU_TYPE') or enc_j.get('vru_type_code', 9)
                    if type_i != type_j:
                        continue
                    # Check if j overlaps with any member of the group
                    for m in list(group_members):
                        enc_m = self.encounters[m]
                        if (enc_j['frame_start'] <= enc_m['frame_end'] + OVERLAP_TOLERANCE and
                                enc_j['frame_end'] >= enc_m['frame_start'] - OVERLAP_TOLERANCE):
                            group_members.add(j)
                            changed = True
                            break

            gid = self.next_group_id
            self.interaction_groups[gid] = group_members
            for m in group_members:
                self.encounters[m]['interaction_group'] = gid
                assigned.add(m)
            self.next_group_id += 1

        # Solo encounters get their own group too
        for i in coded_indices:
            if i not in assigned:
                gid = self.next_group_id
                self.interaction_groups[gid] = {i}
                self.encounters[i]['interaction_group'] = gid
                self.next_group_id += 1

        # Initialize group codes
        self.group_codes = {}
        for gid in self.interaction_groups:
            self.group_codes[gid] = OrderedDict(
                [(k, None) for k in GROUP_VARIABLES])

        self.grouping_selected = 0
        self.state = self.INTERACTION_GROUPING
        print("\n  ═══ INTERACTION GROUPING ═══")
        print("  Encounters happening at the same time are auto-grouped.")
        print("  ./, = navigate  |  1-9 = set group  |  0 = clear")
        print("  ENTER = done (code group-level variables)  |  BACK = encounter list")
        self._print_grouping_status()

    def _print_grouping_status(self):
        """Print current interaction group status."""
        vru_type_labels = {1: "ped", 2: "cyc", 3: "esc", 4: "mmv", 5: "mot",
                           6: "anm", 7: "obs", 9: "?"}
        print(f"\n  Interaction groups (class-separated):")
        for gid, members in sorted(self.interaction_groups.items()):
            encs_str = ", ".join(
                f"E{self.encounters[m]['idx']+1:03d}(T{self.encounters[m]['primary_track']})"
                for m in sorted(members))
            # Show VRU type for the group
            first_enc = self.encounters[min(members)]
            vt = first_enc.get('codes', {}).get('VRU_TYPE') or first_enc.get('vru_type_code', 9)
            vt_label = vru_type_labels.get(vt, "?")
            print(f"    Group {gid} [{vt_label}]: {encs_str}")
        sel = self.grouping_selected
        if 0 <= sel < len(self.encounters):
            enc = self.encounters[sel]
            grp = enc.get('interaction_group', '-')
            print(f"\n  >>> E{enc['idx']+1:03d} T{enc['primary_track']} "
                  f"({enc.get('primary_type','?')[:3]}) group={grp}")

    def _handle_grouping_key(self, key):
        """Handle keypress during INTERACTION_GROUPING state.
        Navigation (./, and BACKSPACE) handled by main loop.
        """
        if key == 13:  # ENTER = done, proceed to group coding
            coded_groups = [gid for gid, members in self.interaction_groups.items()
                            if any(self.encounters[m].get('status') == 'coded'
                                   for m in members)]
            if coded_groups:
                self._enter_group_coding(coded_groups)
            else:
                # No groups to code → skip to trip
                self.state = self.TRIP_ANNOTATION
                self.trip_var_idx = 0
            return
        if ord('0') <= key <= ord('9'):
            group_id = key - ord('0')
            sel = self.grouping_selected
            enc = self.encounters[sel]
            old_group = enc.get('interaction_group')
            if group_id == 0:
                # Clear: put in own solo group
                if old_group and old_group in self.interaction_groups:
                    self.interaction_groups[old_group].discard(sel)
                    if not self.interaction_groups[old_group]:
                        del self.interaction_groups[old_group]
                        if old_group in self.group_codes:
                            del self.group_codes[old_group]
                new_gid = self.next_group_id
                self.next_group_id += 1
                self.interaction_groups[new_gid] = {sel}
                enc['interaction_group'] = new_gid
                self.group_codes[new_gid] = OrderedDict(
                    [(k, None) for k in GROUP_VARIABLES])
                print(f"    E{enc['idx']+1:03d} → solo group {new_gid}")
            else:
                # Assign to group
                if old_group and old_group in self.interaction_groups:
                    self.interaction_groups[old_group].discard(sel)
                    if not self.interaction_groups[old_group]:
                        del self.interaction_groups[old_group]
                        if old_group in self.group_codes:
                            del self.group_codes[old_group]
                enc['interaction_group'] = group_id
                if group_id not in self.interaction_groups:
                    self.interaction_groups[group_id] = set()
                    self.group_codes[group_id] = OrderedDict(
                        [(k, None) for k in GROUP_VARIABLES])
                self.interaction_groups[group_id].add(sel)
                print(f"    E{enc['idx']+1:03d} → group {group_id}")
            self._print_grouping_status()
            return

    # ─── Group-level coding ───

    def _enter_group_coding(self, group_ids):
        """Enter group coding state to code GROUP_VARIABLES per group."""
        self.groups_to_code = sorted(group_ids)
        self.group_coding_idx = 0
        self.group_var_idx = 0
        self.input_buffer = ""
        self.state = self.GROUP_CODING
        gid = self.groups_to_code[0]
        members = self.interaction_groups.get(gid, set())
        encs_str = ", ".join(f"E{self.encounters[m]['idx']+1:03d}" for m in sorted(members))
        print(f"\n  ═══ GROUP CODING ═══")
        print(f"  Coding group-level variables for {len(self.groups_to_code)} group(s)")
        print(f"  Group {gid}: {encs_str}")
        # Jump to first encounter of the group
        if members:
            first_enc = self.encounters[min(members)]
            self.current_frame = first_enc.get('frame_perception', first_enc['frame_start'])

    def _handle_group_coding_key(self, key):
        """Handle keypress during GROUP_CODING state."""
        gid = self.groups_to_code[self.group_coding_idx]
        var_name = self.group_var_names[self.group_var_idx]
        var_def = GROUP_VARIABLES[var_name]

        if key == 8 or key == 127:  # BACKSPACE = go back
            if self.group_var_idx > 0:
                self.group_var_idx -= 1
            elif self.group_coding_idx > 0:
                self.group_coding_idx -= 1
                self.group_var_idx = len(self.group_var_names) - 1
            else:
                # At start — go back to grouping
                self.state = self.INTERACTION_GROUPING
                print("  Back to interaction grouping")
            return

        if key == 9:  # TAB = skip
            self.group_codes[gid][var_name] = 9
            self._advance_group_coding()
            return

        if var_def["type"] == "categorical":
            if ord('0') <= key <= ord('9'):
                code = key - ord('0')
                if code in var_def["codes"]:
                    self.group_codes[gid][var_name] = code
                    self._advance_group_coding()
                else:
                    print(f"    Invalid code {code} for {var_name}")

    def _advance_group_coding(self):
        """Move to next group variable or next group."""
        self.group_var_idx += 1
        if self.group_var_idx >= len(self.group_var_names):
            # This group is done
            self.group_var_idx = 0
            self.group_coding_idx += 1
            if self.group_coding_idx >= len(self.groups_to_code):
                # All groups coded → trip annotation
                print("  All groups coded. Proceeding to trip annotation.")
                self.state = self.TRIP_ANNOTATION
                self.trip_var_idx = 0
            else:
                # Next group
                gid = self.groups_to_code[self.group_coding_idx]
                members = self.interaction_groups.get(gid, set())
                encs_str = ", ".join(
                    f"E{self.encounters[m]['idx']+1:03d}" for m in sorted(members))
                print(f"\n  Group {gid}: {encs_str}")
                # Jump to first encounter of next group
                if members:
                    first_enc = self.encounters[min(members)]
                    self.current_frame = first_enc.get(
                        'frame_perception', first_enc['frame_start'])

    # ─── Trip annotation logic ───

    def _handle_trip_key(self, key):
        """Handle keypress during TRIP_ANNOTATION state."""
        var_name = self.trip_var_names[self.trip_var_idx]
        var_def = TRIP_VARIABLES[var_name]

        if key == 8 or key == 127:  # BACKSPACE = go back
            if var_def["type"] == "integer" and self.input_buffer:
                self.input_buffer = self.input_buffer[:-1]
                return
            if self.trip_var_idx > 0:
                self._retreat_trip()
            else:
                # At first trip var — go back to encounter list
                print("  [BACK] Returning to encounter list")
                self.state = self.ENCOUNTER_LIST
            return
        if key == 9:  # TAB = skip
            self.trip_codes[var_name] = 9
            self.input_buffer = ""
            self._advance_trip()
            return

        if var_def["type"] == "categorical":
            if key == 13 or key == 10:  # ENTER = accept auto-suggested value
                auto_val = self._get_trip_auto_value(var_name)
                if auto_val is not None and auto_val in var_def["codes"]:
                    self.trip_codes[var_name] = auto_val
                    self._advance_trip()
            elif ord('0') <= key <= ord('9'):
                code = key - ord('0')
                if code in var_def["codes"]:
                    self.trip_codes[var_name] = code
                    self._advance_trip()
                else:
                    print(f"    Invalid code {code} for {var_name}")
        elif var_def["type"] == "integer":
            if key == 13 or key == 10:  # ENTER = confirm
                if self.input_buffer:
                    try:
                        val = int(self.input_buffer)
                        lo = var_def.get("min", 0)
                        hi = var_def.get("max", 9999)
                        if lo <= val <= hi:
                            self.trip_codes[var_name] = val
                            self.input_buffer = ""
                            self._advance_trip()
                        else:
                            print(f"    Value {val} out of range [{lo}-{hi}]")
                    except ValueError:
                        print(f"    Invalid number: {self.input_buffer}")
                else:
                    # ENTER with empty buffer = accept auto value
                    auto_val = self._get_trip_auto_value(var_name)
                    if auto_val is not None:
                        self.trip_codes[var_name] = auto_val
                        self._advance_trip()
            elif ord('0') <= key <= ord('9'):
                self.input_buffer += chr(key)

    def _advance_trip(self):
        """Move to next trip variable."""
        self.trip_var_idx += 1
        if self.trip_var_idx >= len(self.trip_var_names):
            # All trip vars coded -> save trip -> DONE
            self._save_trip_annotation()
            self.state = self.DONE
            print("    Trip annotation complete -> DONE")

    def _get_trip_auto_value(self, var_name):
        """Get auto-computed value for trip-level variable.

        Auto-suggests values based on available metadata:
        - LIGHTING: inferred from clip timestamp (hour 7-19 = Daylight)
        - ZONE_TYPE: from pre-filled zone data (if loaded from zones CSV)
        """
        if var_name == "LIGHTING" and self.clip_start_ms is not None:
            try:
                dt = datetime.fromtimestamp(self.clip_start_ms / 1000.0)
                hour = dt.hour
                if 7 <= hour <= 19:
                    print(f"    [AUTO] LIGHTING=1 (Daylight) suggested from timestamp "
                          f"{dt.strftime('%H:%M')}")
                    return 1
                elif hour in (6, 20, 21):
                    print(f"    [AUTO] LIGHTING=2 (Dawn/Dusk) suggested from timestamp "
                          f"{dt.strftime('%H:%M')}")
                    return 2
                else:
                    print(f"    [AUTO] LIGHTING=3 (Dark+lit) suggested from timestamp "
                          f"{dt.strftime('%H:%M')}")
                    return 3
            except (ValueError, OSError):
                pass
        if var_name == "ZONE_TYPE":
            prefilled = self.trip_codes.get('ZONE_TYPE')
            if prefilled is not None and prefilled != '' and prefilled != 9:
                print(f"    [AUTO] ZONE_TYPE={prefilled} pre-filled from zone data")
                return prefilled
        return None

    def _retreat_trip(self):
        """Go back to previous trip variable."""
        if self.trip_var_idx > 0:
            self.trip_var_idx -= 1

    # ─── Rider segmentation (acceleration + steering) ───

    def _auto_suggest_boundaries(self, signal_key, threshold,
                                   min_gap_s=1.0):
        """Auto-suggest segment boundaries from IMU signal zero-crossings.

        Returns list of frame numbers where the signal crosses the threshold.
        Enforces minimum gap of min_gap_s seconds between boundaries
        (rule of thumb: max 1 phase change per second).
        """
        min_gap_frames = int(min_gap_s * self.fps)
        raw_boundaries = []
        prev_sign = None
        for f in range(self.total_frames):
            sensor = self._sensor_by_frame.get(f, {})
            val = sensor.get(signal_key, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            sign = 1 if val > threshold else (-1 if val < -threshold else 0)
            if prev_sign is not None and sign != prev_sign and sign != 0:
                raw_boundaries.append(f)
            if sign != 0:
                prev_sign = sign

        # Filter: enforce minimum gap between boundaries
        if not raw_boundaries:
            return raw_boundaries
        filtered = [raw_boundaries[0]]
        for b in raw_boundaries[1:]:
            if b - filtered[-1] >= min_gap_frames:
                filtered.append(b)
        return filtered

    def _merge_nearby_segments(self, boundaries, min_gap_s=5.0):
        """Merge steering boundaries separated by less than min_gap_s seconds.

        Prevents over-segmentation from noisy IMU zero-crossings.
        """
        if len(boundaries) < 2:
            return boundaries
        min_gap_frames = int(min_gap_s * self.fps)
        merged = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged[-1] < min_gap_frames:
                merged[-1] = b  # extend previous to current
            else:
                merged.append(b)
        return merged

    def _auto_strong_steering_bounds(self, signal_key, strong_threshold):
        """Find onset/offset frames for strong steering episodes (|GyrZ| > threshold).

        Uses strong_threshold directly for onset (no ramp-up capture for boundaries).
        Episodes separated by <0.5s are merged. Only keeps episodes reaching strong_threshold.
        """
        # Build per-frame signal values
        vals = {}
        for f in range(self.total_frames):
            sensor = self._sensor_by_frame.get(f, {})
            val = sensor.get(signal_key, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            vals[f] = abs(val)

        # Find frames above threshold
        onset_frames = [f for f in range(self.total_frames) if vals[f] > strong_threshold]
        if not onset_frames:
            return []

        # Group into episodes (merge gaps < 0.5s = 15 frames)
        episodes = []
        ep_start = onset_frames[0]
        ep_end = onset_frames[0]
        for f in onset_frames[1:]:
            if f - ep_end <= 15:
                ep_end = f
            else:
                episodes.append((ep_start, ep_end))
                ep_start = f
                ep_end = f
        episodes.append((ep_start, ep_end))

        # Return onset/offset as boundaries (with 5-frame padding)
        bounds = []
        for onset, offset in episodes:
            bounds.append(max(0, onset - 5))
            bounds.append(min(self.total_frames - 1, offset + 5))
        return bounds

    def _enter_rider_segmentation(self):
        """Enter rider segment annotation: steering pass only.

        Braking/acceleration is NOT manually coded — it is computed
        automatically in post-processing from AccX (no added value
        from manual annotation with video).
        """
        self.rider_pass = 'steer'
        self.rider_accel_segments = []  # empty — computed in post-processing
        self.rider_steer_segments = []
        self.rider_boundaries = [0]
        self.show_imu_overlay = True
        self.state = self.RIDER_SEGMENT
        self.current_frame = 0
        self.playing = False

        # Auto-suggest steering boundaries disabled (IMU sync unreliable)
        # peak_bounds = self._auto_strong_steering_bounds('yaw_rate_dps', 30.0)

        print("\n" + "=" * 60)
        print("  RIDER SEGMENTATION — STEERING")
        print("  Mark phases where the ego-rider is actively steering.")
        print("  Camera rotation during steering degrades VRU trajectories.")
        print("  IMU signals shown on video (GyrZ = yaw rate).")
        print("  Auto-suggested boundaries shown (from GyrZ zero-crossings).")
        print("  r = add/adjust boundary | BACKSPACE = undo last")
        print("  ENTER = finalize + code each segment")
        print("  ESC = skip rider segmentation")
        print("=" * 60)

    def _handle_rider_segment_key(self, key):
        """Handle keypress in RIDER_SEGMENT state (boundary marking)."""
        if key == ord('r'):
            f = self.current_frame
            if f not in self.rider_boundaries:
                self.rider_boundaries.append(f)
                self.rider_boundaries.sort()
                n = len(self.rider_boundaries) - 1
                label = "ACCEL" if self.rider_pass == 'accel' else "STEER"
                print(f"    [{label}] Boundary at F{f} "
                      f"({f / self.fps:.2f}s) — {n} segment(s)")
        elif key == 13 or key == 10:  # ENTER = finalize
            self._finalize_rider_boundaries()
        elif key == 27:  # ESC = skip
            if self.rider_pass == 'accel' and not self.rider_accel_segments:
                print("    [RIDER] Skipping rider segmentation.")
                self.show_imu_overlay = False
                self.state = self.DONE
            elif self.rider_pass == 'steer':
                # Skip steering pass, keep accel results
                print("    [RIDER] Skipping steering pass.")
                self._save_rider_segments()
                self.show_imu_overlay = False
                self.state = self.DONE
        elif key == 8 or key == 127:  # BACKSPACE = remove last boundary
            if len(self.rider_boundaries) > 1:
                removed = self.rider_boundaries.pop()
                print(f"    [RIDER] Removed boundary at F{removed}")
        elif key == ord('c'):  # Clear auto-suggestions, start fresh
            self.rider_boundaries = [0]
            print("    [RIDER] Cleared all boundaries.")

    def _finalize_rider_boundaries(self):
        """Convert boundaries to segments and start coding them."""
        last_frame = self.total_frames - 1
        if last_frame not in self.rider_boundaries:
            self.rider_boundaries.append(last_frame)
        self.rider_boundaries = sorted(set(self.rider_boundaries))

        segments = []
        for i in range(len(self.rider_boundaries) - 1):
            seg = {
                'frame_start': self.rider_boundaries[i],
                'frame_end': self.rider_boundaries[i + 1],
                'time_start': round(self.rider_boundaries[i] / self.fps, 2),
                'time_end': round(self.rider_boundaries[i + 1] / self.fps, 2),
                'code': None,
            }
            segments.append(seg)

        if not segments:
            segments = [{
                'frame_start': 0, 'frame_end': last_frame,
                'time_start': 0.0,
                'time_end': round(last_frame / self.fps, 2),
                'code': None,
            }]

        # Auto-fill codes from IMU data
        codes = RIDER_ACCEL_CODES if self.rider_pass == 'accel' else RIDER_STEER_CODES
        signal_key = 'acc_x_g' if self.rider_pass == 'accel' else 'yaw_rate_dps'
        for seg in segments:
            vals = []
            for f in range(seg['frame_start'], seg['frame_end'] + 1):
                sensor = self._sensor_by_frame.get(f, {})
                v = sensor.get(signal_key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            if vals:
                mean_v = np.mean(vals)
                if self.rider_pass == 'accel':
                    if mean_v > 0.03:
                        seg['code'] = 1  # Accelerating
                    elif mean_v < -0.03:
                        seg['code'] = 2  # Decelerating
                    else:
                        seg['code'] = 3  # Constant
                else:
                    if mean_v > 3.0:
                        seg['code'] = 1  # Steering left
                    elif mean_v < -3.0:
                        seg['code'] = 2  # Steering right
                    else:
                        seg['code'] = 3  # Straight

        n = len(segments)
        label = "ACCELERATION" if self.rider_pass == 'accel' else "STEERING"
        print(f"\n    [{label}] {n} segment(s) to code:")
        for i, seg in enumerate(segments):
            auto_label = ""
            if seg['code'] is not None:
                auto_label = f" (auto: {codes.get(seg['code'], '?')})"
            print(f"      S{i+1}: F{seg['frame_start']}-F{seg['frame_end']} "
                  f"({seg['time_start']:.1f}s-{seg['time_end']:.1f}s){auto_label}")

        if self.rider_pass == 'accel':
            self.rider_accel_segments = segments
        else:
            self.rider_steer_segments = segments

        self.rider_seg_idx = 0
        self.state = self.RIDER_SEGMENT_CODING
        self.current_frame = segments[0]['frame_start']

    def _handle_rider_segment_coding_key(self, key):
        """Handle keypress in RIDER_SEGMENT_CODING state."""
        segments = (self.rider_accel_segments if self.rider_pass == 'accel'
                    else self.rider_steer_segments)
        codes = RIDER_ACCEL_CODES if self.rider_pass == 'accel' else RIDER_STEER_CODES
        seg = segments[self.rider_seg_idx]

        if key == 8 or key == 127:  # BACKSPACE
            if self.rider_seg_idx > 0:
                self.rider_seg_idx -= 1
                self.current_frame = segments[self.rider_seg_idx]['frame_start']
            return

        if key == 13 or key == 10:  # ENTER = accept auto-fill
            if seg['code'] is not None:
                label = codes.get(seg['code'], '?')
                print(f"    S{self.rider_seg_idx+1} = {seg['code']} ({label}) [accepted]")
                self._advance_rider_seg()
                return

        if key == 9:  # TAB = skip (unknown)
            seg['code'] = 9
            self._advance_rider_seg()
            return

        if ord('0') <= key <= ord('9'):
            code = key - ord('0')
            if code in codes:
                seg['code'] = code
                print(f"    S{self.rider_seg_idx+1} = {code} ({codes[code]})")
                self._advance_rider_seg()

    def _advance_rider_seg(self):
        """Move to next segment or finish."""
        segments = (self.rider_accel_segments if self.rider_pass == 'accel'
                    else self.rider_steer_segments)
        self.rider_seg_idx += 1
        if self.rider_seg_idx >= len(segments):
            if self.rider_pass == 'accel':
                # Legacy path (acceleration pass) — if somehow entered, go to steer
                print("\n    Acceleration segmentation complete.")
                self.rider_pass = 'steer'
                self.rider_boundaries = [0]
                # Auto-suggest steering boundaries disabled (IMU sync unreliable)
                # peak_bounds = self._auto_strong_steering_bounds('yaw_rate_dps', 30.0)
                self.state = self.RIDER_SEGMENT
                self.current_frame = 0
                self.playing = False
                print("\n" + "=" * 60)
                print("  RIDER SEGMENTATION — STEERING")
                print("  r = add boundary | ENTER = finalize | ESC = skip")
                print("=" * 60)
            else:
                # Done with steering
                self._save_rider_segments()
                self.show_imu_overlay = False
                if self.pre_encounter_phase:
                    # Pre-encounter flow: steering done -> obstacle marking
                    self._enter_obstacle_marking()
                else:
                    # Post-trip flow (legacy): steering done -> DONE
                    self.state = self.DONE
                    print("    Rider segmentation complete -> DONE")
        else:
            seg = segments[self.rider_seg_idx]
            self.current_frame = seg['frame_start']

    def _save_rider_segments(self):
        """Save both acceleration and steering segments to CSV."""
        base = str(Path(self.output_path).parent /
                   Path(self.output_path).stem)

        # Acceleration segments
        if self.rider_accel_segments:
            accel_path = f"{base}_rider_accel.csv"
            rows = []
            for i, seg in enumerate(self.rider_accel_segments):
                row = OrderedDict()
                row['SEGMENT_ID'] = f"A{i+1:03d}"
                row['TRIP_ID'] = self.trip_id
                row['RIDER_ID'] = self.rider_id
                row['RATER_ID'] = self.rater_id
                row['FRAME_START'] = seg['frame_start']
                row['FRAME_END'] = seg['frame_end']
                row['TIME_START'] = seg['time_start']
                row['TIME_END'] = seg['time_end']
                row['DURATION_S'] = round(seg['time_end'] - seg['time_start'], 2)
                row['ACCEL_CODE'] = seg.get('code', '')
                row['ACCEL_LABEL'] = RIDER_ACCEL_CODES.get(seg.get('code'), '')
                rows.append(row)
            with open(accel_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Acceleration segments saved: {accel_path} ({len(rows)})")

        # Steering segments
        if self.rider_steer_segments:
            steer_path = f"{base}_rider_steer.csv"
            rows = []
            for i, seg in enumerate(self.rider_steer_segments):
                row = OrderedDict()
                row['SEGMENT_ID'] = f"S{i+1:03d}"
                row['TRIP_ID'] = self.trip_id
                row['RIDER_ID'] = self.rider_id
                row['RATER_ID'] = self.rater_id
                row['FRAME_START'] = seg['frame_start']
                row['FRAME_END'] = seg['frame_end']
                row['TIME_START'] = seg['time_start']
                row['TIME_END'] = seg['time_end']
                row['DURATION_S'] = round(seg['time_end'] - seg['time_start'], 2)
                row['STEER_CODE'] = seg.get('code', '')
                row['STEER_LABEL'] = RIDER_STEER_CODES.get(seg.get('code'), '')
                rows.append(row)
            with open(steer_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Steering segments saved: {steer_path} ({len(rows)})")

    # ─── Pre-encounter phase: obstacles + calibration ───

    def _enter_obstacle_marking(self):
        """Enter clip-level obstacle marking phase."""
        self.state = self.OBSTACLE_MARKING
        self.clip_obstacle_zones = []
        self.clip_obstacle_open = None
        self.clip_obstacle_open_type = None
        self._waiting_zone_type = False
        self._zone_type_frame = None
        self.current_frame = 0
        self.playing = False
        print("\n" + "=" * 60)
        print("  OBSTACLE & LANE MARKING (clip-level)")
        print("  Mark frame ranges where obstacles affect the rider's path.")
        print("  These zones will flag encounters for potential exclusion.")
        print("  8 = start/end obstacle range (toggle)")
        print("  5 = click obstacle bottom to measure distance")
        print("  l = mark lane edges (4 clicks: 2 left + 2 right)")
        print("      Press 'l' again at a different frame for additional pairs")
        print("  ./,  = navigate frames    SPACE = play/pause")
        print("  ENTER = finish marking -> calibration")
        print("  ESC = skip -> calibration")
        print("  BACKSPACE = undo last zone")
        print("=" * 60)

    def _handle_obstacle_marking_key(self, key):
        """Handle keypress in OBSTACLE_MARKING state."""
        # Zone type selection (after pressing 8 to open a zone)
        if getattr(self, '_waiting_zone_type', False):
            if key == 27:  # ESC = cancel
                self._waiting_zone_type = False
                print("  [ZONE] Cancelled.")
            elif key >= ord('1') and key <= ord('7'):
                zone_type_code = int(chr(key))
                zone_type_name = self.ZONE_TYPE_CODES.get(zone_type_code, 'obstacle')
                self.clip_obstacle_open = self._zone_type_frame
                self.clip_obstacle_open_type = zone_type_name
                self._waiting_zone_type = False
                print(f"  [ZONE] Zone started ({zone_type_name}) at F{self._zone_type_frame}. "
                      f"Press 8 again to close.")
            return  # Consume all keys while waiting for zone type
        if key == ord('l'):
            if self.lane_marking_mode:
                self.lane_marking_mode = False
                self.lane_marking_clicks = []
                print("    [LANE] Lane marking cancelled.")
            else:
                active = self._get_lane_for_frame(self.current_frame)
                has_active = (active is not None
                              and active.get('end_frame') is None)
                self.lane_marking_mode = True
                self.lane_marking_clicks = []
                self._lane_pending_end = has_active  # flag: 0 = end lane
                n_segs = len(self.clip_lane_lines_list)
                if has_active:
                    si = self.clip_lane_lines_list.index(active) + 1
                    print(f"    [LANE] L{si} is active (from F{active['frame']}).")
                    print(f"           Press 0 to END it at F{self.current_frame},")
                    print(f"           or click 4 points for a NEW segment. ESC=cancel.")
                elif n_segs > 0:
                    print(f"    [LANE] {n_segs} segment(s) exist. Adding new segment at F{self.current_frame}.")
                    print(f"           Click 4 points (2 LEFT + 2 RIGHT). Right-click = undo last segment.")
                else:
                    print("    [LANE] Click 2 points for LEFT edge, then 2 points for RIGHT edge (4 clicks total).")
        elif key == ord('0') and self.lane_marking_mode and getattr(self, '_lane_pending_end', False):
            # End the active lane segment at current frame
            active = self._get_lane_for_frame(self.current_frame)
            if active is not None and active.get('end_frame') is None:
                active['end_frame'] = self.current_frame
                si = self.clip_lane_lines_list.index(active) + 1
                print(f"    [LANE] L{si} ended at F{self.current_frame} "
                      f"(was F{active['frame']}-now).")
            self.lane_marking_mode = False
            self.lane_marking_clicks = []
            self._lane_pending_end = False
        elif key == ord('5'):
            self.obs_click_mode = not self.obs_click_mode
            if self.obs_click_mode:
                print(f"\n  [OBS-5] Click obstacle bottom to measure distance.")
                print(f"          Right-click=undo last  |  5=done")
            else:
                n = len(self.clip_obstacle_points)
                print(f"  [OBS-5] Click mode off. {n} obstacle point(s) saved.")
        elif key == ord('8'):
            if self.clip_obstacle_open is not None:
                # Close current obstacle range
                zone_type_name = self.clip_obstacle_open_type or 'obstacle'
                obs = {
                    'frame_start': self.clip_obstacle_open,
                    'frame_end': self.current_frame,
                    'time_start': round(self.clip_obstacle_open / self.fps, 2),
                    'time_end': round(self.current_frame / self.fps, 2),
                    'type': zone_type_name,
                }
                self.clip_obstacle_zones.append(obs)
                dur = obs['time_end'] - obs['time_start']
                print(f"  [ZONE] Zone closed ({zone_type_name}): "
                      f"F{obs['frame_start']}-F{obs['frame_end']} ({dur:.2f}s). "
                      f"Total: {len(self.clip_obstacle_zones)} zone(s)")
                self.clip_obstacle_open = None
                self.clip_obstacle_open_type = None
            else:
                # Set flag — zone type selection handled in main loop
                self._waiting_zone_type = True
                self._zone_type_frame = self.current_frame
                print(f"  [ZONE] Select zone type at F{self.current_frame}:")
                print("    1=Pedestrian area  2=Shared space  3=Non-motorised path")
                print("    4=Crosswalk  5=Park  6=Obstacle  7=Dismounted  ESC=Cancel")
        elif key == 13 or key == 10:  # ENTER = done
            if self.clip_obstacle_open is not None:
                # Auto-close any open range
                zone_type_name = self.clip_obstacle_open_type or 'obstacle'
                obs = {
                    'frame_start': self.clip_obstacle_open,
                    'frame_end': self.current_frame,
                    'time_start': round(self.clip_obstacle_open / self.fps, 2),
                    'time_end': round(self.current_frame / self.fps, 2),
                    'type': zone_type_name,
                }
                self.clip_obstacle_zones.append(obs)
                self.clip_obstacle_open = None
                self.clip_obstacle_open_type = None
            n = len(self.clip_obstacle_zones)
            n_pts = len(self.clip_obstacle_points)
            self.obs_click_mode = False
            print(f"  [OBS] Obstacle marking complete: {n} zone(s), {n_pts} distance point(s)")
            self._enter_calibration_phase()
        elif key == 8 or key == 127:  # BACKSPACE = remove last zone
            if self.clip_obstacle_open is not None:
                self.clip_obstacle_open = None
                self.clip_obstacle_open_type = None
                print("  [ZONE] Cancelled open zone.")
            elif self.clip_obstacle_zones:
                removed = self.clip_obstacle_zones.pop()
                print(f"  [ZONE] Removed last zone ({removed.get('type', '?')}): "
                      f"F{removed['frame_start']}-F{removed['frame_end']}. "
                      f"{len(self.clip_obstacle_zones)} remain.")

    def _enter_calibration_phase(self):
        """Enter optional calibration phase before encounter detection."""
        self.state = self.CALIBRATION_PHASE
        # Jump to middle of clip — likely to have visible reference objects
        self.current_frame = self.total_frames // 2
        self.playing = False
        print("\n" + "=" * 60)
        if self.calibration_from_file:
            print("  CALIBRATION (saved calibration loaded)")
            print(f"  Source: {self.calibration_source}")
            print(f"  Values: h={self.camera_height_m:.2f}m, "
                  f"pitch={self.pitch_deg:.1f} deg, f={self.focal_length_px:.0f}px")
            print("  ENTER = use saved calibration -> detect encounters")
            print("  6 = recalibrate (HEAD+FOOT / Ref / Markings / Multi-ped)")
        else:
            print("  CALIBRATION (no saved calibration found)")
            print("  Calibrate camera before encounter detection for")
            print("  accurate distance estimation.")
            print(f"  Current: h={self.camera_height_m:.2f}m, "
                  f"pitch={self.pitch_deg:.1f} deg, f={self.focal_length_px:.0f}px")
            print("  6 = start calibration (HEAD+FOOT / Ref / Markings / Multi-ped)")
            print("  ENTER = accept current calibration -> detect encounters")
        print("  ESC = skip calibration -> detect encounters")
        print("=" * 60)

    def _finish_pre_encounter_phase(self):
        """Run encounter detection and transition to ENCOUNTER_LIST."""
        self.pre_encounter_phase = False
        try:
            self._run_encounter_detection()
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] Encounter detection failed: {e}")
            traceback.print_exc()
            self.encounters = []

        # Attempt session resume — restore coded encounters from previous session
        resumed = self._load_session_state()

        self.state = self.ENCOUNTER_LIST
        if not resumed:
            self.selected_enc_idx = 0
        if self.encounters:
            self.current_frame = self.encounters[
                self.selected_enc_idx]['frame_mindist']
        else:
            self.current_frame = 0
        n_flagged = sum(1 for e in self.encounters
                        if e.get('in_steering_zone') or e.get('in_obstacle_zone'))
        print(f"\n  [DETECTION] {len(self.encounters)} encounters detected")
        if n_flagged:
            print(f"  [ZONES] {n_flagged} encounters flagged "
                  f"(min-distance in steering/obstacle zone)")
        # Density warning for clips with many encounters
        n_enc = len(self.encounters)
        if n_enc > 50:
            suggested = max(1, n_enc // 3)
            est_time_s = n_enc * 30
            est_min = est_time_s // 60
            print(f"\n  WARNING: {n_enc} encounters detected. "
                  f"Consider using --max_encounters {suggested} to focus on best quality.")
            print(f"  Dense clips take ~{est_min}min to annotate fully.")
            print(f"  Tip: press 'x' during coding to quick-reject (CONFIRM=0 + skip).\n")
        print(f"  Starting encounter annotation.\n")

    def _finalize_obstacle_points(self, multi_pts, frame, type_code, type_name):
        """Build a single obstacle entry from 1-3 accumulated click points.

        Args:
            multi_pts: list of dicts with px, py, distance_m, lateral_m
            frame: frame number where obstacle was marked
            type_code: obstacle type code (1-5/9)
            type_name: obstacle type name string

        Returns:
            dict with keys: frame, type, type_name, points, distance_m, px, py, width_m
        """
        # Build points list as (px, py, distance_m) tuples
        points = [(p['px'], p['py'], p['distance_m']) for p in multi_pts]
        # distance_m = nearest edge (minimum distance)
        distance_m = round(min(p['distance_m'] for p in multi_pts), 2)
        # width_m = lateral spread (0 if single point)
        if len(multi_pts) > 1:
            laterals = [p['lateral_m'] for p in multi_pts]
            width_m = round(abs(max(laterals) - min(laterals)), 2)
        else:
            width_m = 0.0
        # Center point for backward-compatible CSV columns (px, py)
        center_idx = len(multi_pts) // 2
        center = multi_pts[center_idx]
        return {
            'frame': frame,
            'type': type_code,
            'type_name': type_name,
            'points': points,
            'distance_m': distance_m,
            'px': center['px'],
            'py': center['py'],
            'width_m': width_m,
        }

    def _merge_adjacent_zones(self):
        """Merge adjacent zones of the same type separated by less than min_zone_gap_s.

        Sorts zones by FRAME_START, then merges consecutive zones of the same
        type when the gap between them is shorter than self.min_zone_gap_s.
        Modifies self.clip_obstacle_zones in place.
        """
        if self.min_zone_gap_s <= 0 or len(self.clip_obstacle_zones) < 2:
            return
        self.clip_obstacle_zones.sort(key=lambda z: z['frame_start'])
        merged = [self.clip_obstacle_zones[0]]
        for z in self.clip_obstacle_zones[1:]:
            prev = merged[-1]
            same_type = prev.get('type', 'obstacle') == z.get('type', 'obstacle')
            gap_s = z.get('time_start', 0) - prev.get('time_end', 0)
            if same_type and gap_s < self.min_zone_gap_s:
                # Merge: extend prev to cover z
                prev['frame_end'] = z['frame_end']
                prev['time_end'] = z['time_end']
                print(f"  [ZONE] Merged 2 adjacent {prev.get('type', 'obstacle')} zones "
                      f"(gap {gap_s:.2f}s < {self.min_zone_gap_s:.1f}s)")
            else:
                merged.append(z)
        self.clip_obstacle_zones = merged

    def _save_clip_zones(self):
        """Save clip-level obstacle zones to CSV."""
        if not self.clip_obstacle_zones:
            return
        # Merge adjacent same-type zones with small gaps before saving
        self._merge_adjacent_zones()
        base = str(Path(self.output_path).parent / Path(self.output_path).stem)
        obs_path = f"{base}_obstacle_zones.csv"
        rows = []
        for i, z in enumerate(self.clip_obstacle_zones):
            rows.append({
                'ZONE_ID': f"OBS{i+1:03d}",
                'TRIP_ID': self.trip_id,
                'FRAME_START': z['frame_start'],
                'FRAME_END': z['frame_end'],
                'TIME_START': z.get('time_start', ''),
                'TIME_END': z.get('time_end', ''),
                'TYPE': z.get('type', 'obstacle'),
            })
        with open(obs_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Obstacle zones saved: {obs_path} ({len(rows)})")

    def _save_clip_obstacle_points(self):
        """Save clip-level obstacle distance points to CSV."""
        if not self.clip_obstacle_points:
            return
        base = str(Path(self.output_path).parent / Path(self.output_path).stem)
        pts_path = f"{base}_obstacle_points.csv"
        rows = []
        for i, op in enumerate(self.clip_obstacle_points):
            rows.append({
                'POINT_ID': f"OP{i+1:03d}",
                'TRIP_ID': self.trip_id,
                'FRAME': op['frame'],
                'TIME_S': round(op['frame'] / self.fps, 2),
                # V3.6 skeleton column names (sec 13.4)
                'OBSTACLE_X_PX': op['px'],
                'OBSTACLE_Y_PX': op['py'],
                'OBSTACLE_DIST_M': op['distance_m'],
                'OBSTACLE_TYPE': op.get('type', ''),
            })
        with open(pts_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Obstacle points saved: {pts_path} ({len(rows)})")

    # ─── Save logic ───

    def _encounter_to_row(self, enc):
        """Convert encounter dict to structured FINAL spec CSV row.

        Columns are grouped by role:
          Identity (5) → Frame numbers (6) → Timestamps (5)
          → Auto metrics (7) → Severity (1)
          → Surrogate safety (2) → Distance (2) → VRU auto (3)
          → Manual variables → Context → Obstacles
          → Tail (3)
        """
        def fmt_ts(t):
            if t is None:
                return ""
            m, s = divmod(t, 60)
            h, m = divmod(int(m), 60)
            frac = int((t % 1) * 100)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{frac:02d}"

        row = OrderedDict()
        # ── Identity (5 cols) ──
        # Composite EVENT_ID: RIDER_TRIP_ENC (fallback to E0001 if rider/trip not set)
        enc_num = enc['idx'] + 1
        if self.rider_id and self.trip_id:
            row["EVENT_ID"] = f"{self.rider_id}_{self.trip_id}_E{enc_num:04d}"
        else:
            row["EVENT_ID"] = f"E{enc_num:04d}"
        row["TRIP_ID"] = self.trip_id
        row["RIDER_ID"] = self.rider_id
        row["CITY"] = self.city
        row["VEHICLE_TYPE"] = getattr(self, 'vehicle_type', 'bike')
        row["RATER_ID"] = self.rater_id
        # ── CONFIRM (from manual coding — needed for IRR and DCM filtering) ──
        row["CONFIRM"] = enc.get('codes', {}).get('CONFIRM', '')
        # ── Frame numbers (6 cols) — integer, more precise than HH:MM:SS ──
        ffd = enc.get('frame_first_detection')
        row["FRAME_FIRST_DETECTION"] = ffd if ffd is not None else ''
        row["FRAME_START"] = enc['frame_start']
        row["FRAME_PERCEPTION"] = enc.get('frame_perception', '')
        row["FRAME_MINDIST"] = enc['frame_mindist']
        row["FRAME_END"] = enc['frame_end']
        flv = enc.get('frame_last_valid')
        row["FRAME_LAST_VALID"] = flv if flv is not None else ''
        # ── Timestamps (6 cols) ──
        row["TIMESTAMP_START"] = fmt_ts(enc['ts_start'])
        row["TIMESTAMP_PERCEPTION"] = fmt_ts(enc.get('ts_perception'))
        row["TIMESTAMP_MINDIST"] = fmt_ts(enc['ts_mindist'])
        row["TIMESTAMP_END"] = fmt_ts(enc['ts_end'])
        row["DURATION_S"] = enc['duration_s']
        # VRU awareness: binary categorical (1=Yes, 0=No, 9=Unknown, ""=not coded)
        _aware = enc.get('codes', {}).get('AWARE_BEFORE_MINDIST')
        row["AWARE_BEFORE_MINDIST"] = _aware if _aware is not None else ""
        # ── Auto metrics (7 cols) ──
        row["SPEED_GPS_KMH"] = enc['speed_kmh']
        # Speed at minimum distance frame — injury severity proxy (codebook V2.6)
        speed_at_dmin = ''
        mindist_f = enc.get('frame_mindist')
        if mindist_f is not None:
            sensor_dmin = self._sensor_by_frame.get(mindist_f, {})
            spd = sensor_dmin.get('speed_kmh')
            if spd is not None:
                speed_at_dmin = round(spd, 1)
        row["SPEED_AT_DMIN_KMH"] = speed_at_dmin
        row["INTERACTION_ZONE_M"] = enc['interaction_zone_m']
        row["PEAK_DECEL_MS2"] = enc['peak_decel_ms2']
        row["PEAK_LATERAL_ACCEL_MS2"] = enc.get('peak_lateral_accel_ms2', '')
        row["PEAK_YAW_DEG_S"] = enc['peak_yaw_deg_s']
        row["THW_MIN_S"] = enc.get('min_thw_s', '')
        row["THW_PERCEPTION_S"] = enc.get('thw_perception_s', '')
        row["VRU_COUNT"] = enc.get('vru_count', 1)
        # ── Severity (1 col, auto-computed V2.5) ──
        row["SEVERITY"] = suggest_severity(
            enc.get('peak_decel_ms2'), enc.get('min_dist'),
            peak_yaw=enc.get('peak_yaw_deg_s')
        )
        # ── Surrogate safety measures (4 cols) ──
        # NOTE: TTC computed at perception frame (not global minimum across encounter)
        row["TTC_MIN_S"] = enc.get('ttc_approx_s', '')
        # NOTE: DRAC computed at perception frame (not global maximum across encounter)
        row["DRAC_MAX_MS2"] = enc.get('drac_ms2', '')
        # MTTC_S and SEVERITY_DOCTOR: post-processing placeholders (scripts/enhanced_post_processing.py)
        row["MTTC_S"] = enc.get('mttc_s', '')
        row["SEVERITY_DOCTOR"] = enc.get('severity_doctor', '')
        # ── Distance (4 cols) ──
        row["MIN_DIST_M"] = enc['min_dist']
        row["MIN_DIST_CORRECTED_M"] = enc.get('min_dist_corrected', '')
        row["MIN_LATERAL_DIST_M"] = enc.get('min_lateral_m', '')
        row["ESCOOTER_DECK_CORRECTION"] = enc.get('_escooter_deck_correction', '')
        # ── VRU auto-derived (4 cols) ──
        row["AUTO_VRU_CLASS"] = enc.get('auto_vru_class', '')
        row["VRU_SPEED_KMH"] = enc.get('vru_speed_kmh', '')
        row["VRU_MOVEMENT"] = enc.get('vru_movement', '')
        # VRU_GAIT: 1=Stationary, 2=Walking, 3=Running, 9=Unknown
        # Walk/run threshold: 2.0 m/s (Hreljac 1993; Rotstein 2005)
        row["VRU_GAIT"] = enc.get('vru_gait', '')
        # CROSSING_DIR: auto-computed from lateral_m trajectory
        row["VRU_CROSSING_DIR"] = enc.get('crossing_dir', '')
        # ── Interaction-level manual variables (excludes CONFIRM and awareness) ──
        for var_name in MANUAL_VARIABLES:
            if var_name in ("CONFIRM", "AWARE_BEFORE_MINDIST"):
                continue  # already written above
            val = enc['codes'].get(var_name)
            row[var_name] = val if val is not None else ""
        # ── GROUP_FLAG (no longer manually coded — derive in post-processing) ──
        row["GROUP_FLAG"] = ""
        # ── Group-level variables ──
        row["INTERACTION_GROUP"] = enc.get('interaction_group', '')
        gid = enc.get('interaction_group')
        if gid and gid in self.group_codes:
            for var_name in GROUP_VARIABLES:
                val = self.group_codes[gid].get(var_name)
                row[var_name] = val if val is not None else ""
        else:
            for var_name in GROUP_VARIABLES:
                row[var_name] = ""
        # ── Contextual vehicles (motor vehicles present during encounter) ──
        ctx_v = enc.get('contextual_vehicles', [])
        row["CONTEXTUAL_VEHICLES"] = ";".join(
            f"T{v['track_id']}({v['type']},{v['min_dist']:.1f}m)" for v in ctx_v
        ) if ctx_v else ""
        row["N_CONTEXTUAL_VEHICLES"] = len(ctx_v)
        # ── Constrained path zones ──
        row["CONSTRAINED_PATH"] = int(bool(enc.get('constrained_path', False)))
        row["CONSTRAINED_ZONE_TYPE"] = enc.get('constrained_zone_type', '')
        # ── Zone overlap flags ──
        row["STEERING_OVERLAP"] = int(bool(enc.get('steering_overlap', False)))
        row["OBSTACLE_ZONE_OVERLAP"] = int(bool(enc.get('obstacle_zone_overlap', False)))
        # ── Truncation flags ──
        row["TRUNCATED"] = int(bool(enc.get('truncated', False)))
        row["N_TRUNCATED_FRAMES"] = enc.get('n_truncated_frames', '')
        # ── Density flags (V3.6) ──
        row["N_SIMULTANEOUS_VRUS"] = enc.get('n_simultaneous_vrus', '')
        row["DENSITY_SECONDARY"] = int(bool(enc.get('density_secondary', False)))
        row["DENSITY_RANK"] = enc.get('density_rank', '')
        # ── Basket occlusion flag ──
        row["BASKET_OCCLUDED"] = int(bool(enc.get('basket_occluded', False)))
        # ── Obstacles ──
        obs = enc.get('obstacles', [])
        def _fmt_obs(o):
            if o.get('time_end') is not None:
                return f"{o['type']}@{o['time_s']:.2f}s-{o['time_end']:.2f}s"
            return f"{o['type']}@{o['time_s']:.2f}s"
        row["OBSTACLES"] = ";".join(_fmt_obs(o) for o in obs) if obs else ""
        # ── Obstacle points (key 'o' manual measurements) ──
        obs_pts = enc.get('obstacle_points', [])
        row["OBSTACLE_POINTS"] = json.dumps(obs_pts) if obs_pts else ""
        # ── Obstacle width (from 3-point marking: left edge, centre, right edge) ──
        _obs_width = ""
        if len(obs_pts) >= 3:
            try:
                _cx = self.width / 2
                _h = self.camera_height_m
                _f = self.focal_length_px
                _hv = self.height / 2.0 - _f * np.tan(np.radians(getattr(self, 'pitch_deg', 0)))
                _ldv = obs_pts[0]['py'] - _hv
                _rdv = obs_pts[2]['py'] - _hv
                if _ldv > 1 and _rdv > 1:
                    _l_lat = (obs_pts[0]['px'] - _cx) * _h / _ldv
                    _r_lat = (obs_pts[2]['px'] - _cx) * _h / _rdv
                    _obs_width = round(abs(_r_lat - _l_lat), 2)
            except (KeyError, IndexError, TypeError):
                pass
        row["OBSTACLE_WIDTH_M"] = _obs_width
        # ── Lane lines (per-encounter, nearest segment by frame) ──
        enc_lane = self._get_lane_for_frame(enc.get('min_dist_frame', 0))
        row["LANE_LINES"] = json.dumps(enc_lane) if enc_lane else ""
        # ── Lane ground-plane distances (post-processing convenience) ──
        row["LANE_LEFT_DIST_M"] = ""
        row["LANE_RIGHT_DIST_M"] = ""
        row["LANE_WIDTH_M"] = ""
        row["VRU_OFFSET_FROM_LEFT_M"] = ""
        row["VRU_OFFSET_FROM_RIGHT_M"] = ""
        if enc_lane:
            try:
                f_px = self.focal_length_px
                cx = self.width / 2
                cy = self.height / 2
                h = self.camera_height_m
                p_deg = getattr(self, 'pitch_deg', 0)
                # Convert lane edge endpoints to ground-plane X at VRU's distance
                left_pts = enc_lane.get('left', [])
                right_pts = enc_lane.get('right', [])
                if len(left_pts) == 2 and len(right_pts) == 2:
                    # Use the lower (closer) point of each edge for more reliable estimate
                    lp = max(left_pts, key=lambda p: p[1])  # higher v = closer
                    rp = max(right_pts, key=lambda p: p[1])
                    lX, lY = pixel_to_ground(lp[0], lp[1], f_px, cx, cy, h, p_deg)
                    rX, rY = pixel_to_ground(rp[0], rp[1], f_px, cx, cy, h, p_deg)
                    if lY != float('inf') and rY != float('inf'):
                        width_m = abs(rX - lX)
                        row["LANE_LEFT_DIST_M"] = round(lY, 2)
                        row["LANE_RIGHT_DIST_M"] = round(rY, 2)
                        row["LANE_WIDTH_M"] = round(width_m, 2)
                        # VRU lateral offset from lane edges at min-distance frame
                        vru_lat = enc.get('min_lateral_m')
                        if vru_lat is not None and not (isinstance(vru_lat, float) and math.isnan(vru_lat)):
                            vru_lat = float(vru_lat)
                            # lX = left edge lateral, rX = right edge lateral (in ego frame)
                            row["VRU_OFFSET_FROM_LEFT_M"] = round(vru_lat - lX, 2)
                            row["VRU_OFFSET_FROM_RIGHT_M"] = round(rX - vru_lat, 2)
            except Exception:
                pass  # Silently skip on any conversion error
        # ── Tail ──
        row["NOTES"] = enc.get('notes', '')
        # Note timestamps — frame numbers when each note was recorded
        note_ts = enc.get('note_timestamps', [])
        row["NOTE_TIMESTAMPS"] = ";".join(str(t) for t in note_ts) if note_ts else ""
        # Correction log: distance corrections + flags + linked tracks
        correction_parts = []
        if enc.get('min_dist_corrected'):
            correction_parts.append(f"dist_corr={enc['min_dist_corrected']:.2f}m")
        if enc.get('flags'):
            correction_parts.append("flags=" + ",".join(enc['flags']))
        linked = enc.get('linked_tracks', [])
        if linked:
            correction_parts.append("linked=T" + ",T".join(str(t) for t in linked))
        row["CORRECTION_LOG"] = "; ".join(correction_parts)
        # ── Trip context (joined from trip-level coding) ──
        row["WEATHER"] = self.trip_codes.get('WEATHER', '')
        row["LIGHTING"] = self.trip_codes.get('LIGHTING', '')
        row["SURFACE_CONDITION"] = self.trip_codes.get('SURFACE_CONDITION', '')
        row["ZONE_TYPE"] = self.trip_codes.get('ZONE_TYPE', '')
        row["VISUAL_SEGREGATION"] = self.trip_codes.get('VISUAL_SEGREGATION', '')
        row["RIDING_COMPANION"] = self.trip_codes.get('RIDING_COMPANION', '')
        # ── Coding timestamps (annotation effort analysis) ──
        row["CODING_START_TS"] = enc.get('coding_start_ts', '')
        row["CODING_END_TS"] = enc.get('coding_end_ts', '')
        # ── Clip metadata (analysis convenience) ──
        row["CLIP_DURATION_S"] = round(self.duration_total, 2)
        row["CLIP_FPS"] = round(self.fps, 2)
        # ── Calibration parameters (reproducibility) ──
        row["CAL_FOCAL_PX"] = self.focal_length_px
        row["CAL_CAM_HEIGHT_M"] = self.camera_height_m
        row["CAL_PITCH_DEG"] = getattr(self, 'pitch_deg', '')
        return row

    def _encounter_to_debug_row(self, enc):
        """Convert encounter dict to extended debug CSV row (internal use)."""
        def fmt_ts(t):
            if t is None:
                return ""
            m, s = divmod(t, 60)
            h, m = divmod(int(m), 60)
            frac = int((t % 1) * 100)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{frac:02d}"

        row = OrderedDict()
        enc_num = enc['idx'] + 1
        if self.rider_id and self.trip_id:
            row["EVENT_ID"] = f"{self.rider_id}_{self.trip_id}_E{enc_num:04d}"
        else:
            row["EVENT_ID"] = f"E{enc_num:04d}"
        row["TRIP_ID"] = self.trip_id
        row["RIDER_ID"] = self.rider_id
        row["CITY"] = self.city
        row["RATER_ID"] = self.rater_id
        ffd = enc.get('frame_first_detection')
        row["FRAME_FIRST_DETECTION"] = ffd if ffd is not None else ''
        row["FRAME_START"] = enc['frame_start']
        row["FRAME_PERCEPTION"] = enc.get('frame_perception', '')
        row["FRAME_MINDIST"] = enc['frame_mindist']
        row["FRAME_END"] = enc['frame_end']
        flv = enc.get('frame_last_valid')
        row["FRAME_LAST_VALID"] = flv if flv is not None else ''
        row["TIMESTAMP_START"] = fmt_ts(enc['ts_start'])
        row["TIMESTAMP_PERCEPTION"] = fmt_ts(enc.get('ts_perception'))
        row["TIMESTAMP_MINDIST"] = fmt_ts(enc['ts_mindist'])
        row["TIMESTAMP_END"] = fmt_ts(enc['ts_end'])
        row["DURATION_S"] = enc['duration_s']
        # VRU awareness: binary categorical (1=Yes, 0=No, 9=Unknown, ""=not coded)
        _aware2 = enc.get('codes', {}).get('AWARE_BEFORE_MINDIST')
        row["AWARE_BEFORE_MINDIST"] = _aware2 if _aware2 is not None else ""
        row["SPEED_GPS_KMH"] = enc['speed_kmh']
        # Speed at minimum distance frame — injury severity proxy (codebook V2.6)
        speed_at_dmin = ''
        mindist_f = enc.get('frame_mindist')
        if mindist_f is not None:
            sensor_dmin = self._sensor_by_frame.get(mindist_f, {})
            spd = sensor_dmin.get('speed_kmh')
            if spd is not None:
                speed_at_dmin = round(spd, 1)
        row["SPEED_AT_DMIN_KMH"] = speed_at_dmin
        row["INTERACTION_ZONE_M"] = enc['interaction_zone_m']
        row["PEAK_DECEL_MS2"] = enc['peak_decel_ms2']
        row["PEAK_LATERAL_ACCEL_MS2"] = enc.get('peak_lateral_accel_ms2', '')
        row["PEAK_YAW_DEG_S"] = enc['peak_yaw_deg_s']
        row["THW_MIN_S"] = enc.get('min_thw_s', '')
        row["THW_PERCEPTION_S"] = enc.get('thw_perception_s', '')
        row["MIN_DIST_M"] = enc['min_dist']
        row["MIN_DIST_CORRECTED_M"] = enc.get('min_dist_corrected', '')
        row["VRU_COUNT"] = enc['vru_count']
        row["PRIMARY_TRACK_ID"] = enc['primary_track']
        row["VRU_TYPE"] = enc.get('primary_type', '')
        row["VRU_SPEED_KMH"] = enc.get('vru_speed_kmh', '')
        row["VRU_MOVEMENT"] = enc.get('vru_movement', '')
        row["VRU_CROSSING_DIR"] = enc.get('crossing_dir', '')
        row["REACTION_TIME_S"] = enc.get('reaction_time_s', '')
        row["ANTICIPATORY_REDUCTION_KMH"] = enc.get('anticipatory_reduction_kmh', '')
        row["MEAN_JERK_MS3"] = enc.get('mean_jerk_ms3', '')
        for var_name in MANUAL_VARIABLES:
            val = enc['codes'].get(var_name)
            row[var_name] = val if val is not None else ""
        row["TRACK_FLAGS"] = ",".join(enc.get('flags', []))
        row["SWAP_FRAME"] = enc.get('swap_frame', '')
        linked = enc.get('linked_tracks', [])
        row["LINKED_TRACKS"] = ",".join(str(t) for t in linked) if linked else ""
        row["AUTO_VRU_CLASS"] = enc.get('auto_vru_class', '')
        row["TTC_MIN_S"] = enc.get('ttc_approx_s', '')
        row["DRAC_MAX_MS2"] = enc.get('drac_ms2', '')
        row["MIN_DISTANCE_CONFIDENCE"] = enc.get('min_distance_confidence', '')
        row["TRUNCATED"] = int(bool(enc.get('truncated', False)))
        row["N_TRUNCATED_FRAMES"] = enc.get('n_truncated_frames', '')
        row["INTERACTION_GROUP"] = enc.get('interaction_group', '')
        row["IMU_CONFIRMED"] = int(bool(enc.get('imu_confirmed', False)))
        # Contextual vehicles (obstacles, not interaction targets)
        ctx_v = enc.get('contextual_vehicles', [])
        row["CONTEXTUAL_VEHICLES"] = ";".join(
            f"T{v['track_id']}({v['type']},{v['min_dist']:.1f}m)" for v in ctx_v
        ) if ctx_v else ""
        row["N_CONTEXTUAL_VEHICLES"] = len(ctx_v)
        # Constrained path zones
        row["CONSTRAINED_PATH"] = int(bool(enc.get('constrained_path', False)))
        row["CONSTRAINED_ZONE_TYPE"] = enc.get('constrained_zone_type', '')
        # Group-level codes
        gid = enc.get('interaction_group')
        if gid and gid in self.group_codes:
            for var_name in GROUP_VARIABLES:
                val = self.group_codes[gid].get(var_name)
                row[var_name] = val if val is not None else ""
        else:
            for var_name in GROUP_VARIABLES:
                row[var_name] = ""
        # Zone overlap flags
        row["STEERING_OVERLAP"] = int(bool(enc.get('steering_overlap', False)))
        row["OBSTACLE_ZONE_OVERLAP"] = int(bool(enc.get('obstacle_zone_overlap', False)))
        # Density flags (V3.6)
        row["N_SIMULTANEOUS_VRUS"] = enc.get('n_simultaneous_vrus', '')
        row["DENSITY_SECONDARY"] = int(bool(enc.get('density_secondary', False)))
        row["DENSITY_RANK"] = enc.get('density_rank', '')
        # Basket occlusion flag
        row["BASKET_OCCLUDED"] = int(bool(enc.get('basket_occluded', False)))
        # Obstacles
        obs = enc.get('obstacles', [])
        def _fmt_obs(o):
            if o.get('time_end') is not None:
                return f"{o['type']}@{o['time_s']:.2f}s-{o['time_end']:.2f}s"
            return f"{o['type']}@{o['time_s']:.2f}s"
        row["OBSTACLES"] = ";".join(_fmt_obs(o) for o in obs) if obs else ""
        # Obstacle points (key 'o' manual measurements)
        obs_pts = enc.get('obstacle_points', [])
        row["OBSTACLE_POINTS"] = json.dumps(obs_pts) if obs_pts else ""
        row["NOTES"] = enc.get('notes', '')
        return row

    def _make_baseline_row(self):
        """Create a no-interaction baseline CSV row.

        Column order matches _encounter_to_row() exactly so that
        DictWriter produces consistent headers when baseline and
        encounter rows are mixed in the same file.
        """
        row = OrderedDict()
        # ── Identity ──
        row["EVENT_ID"] = "E0000"
        row["TRIP_ID"] = self.trip_id
        row["RIDER_ID"] = self.rider_id
        row["CITY"] = self.city
        row["VEHICLE_TYPE"] = getattr(self, 'vehicle_type', 'bike')
        row["RATER_ID"] = self.rater_id
        row["CONFIRM"] = 0
        # ── Frame numbers ──
        row["FRAME_FIRST_DETECTION"] = ""
        row["FRAME_START"] = ""
        row["FRAME_PERCEPTION"] = ""
        row["FRAME_MINDIST"] = ""
        row["FRAME_END"] = ""
        row["FRAME_LAST_VALID"] = ""
        # ── Timestamps ──
        row["TIMESTAMP_START"] = ""
        row["TIMESTAMP_PERCEPTION"] = ""
        row["TIMESTAMP_MINDIST"] = ""
        row["TIMESTAMP_END"] = ""
        row["DURATION_S"] = ""
        row["AWARE_BEFORE_MINDIST"] = ""
        # ── Auto metrics ──
        row["SPEED_GPS_KMH"] = ""
        row["SPEED_AT_DMIN_KMH"] = ""
        row["INTERACTION_ZONE_M"] = ""
        row["PEAK_DECEL_MS2"] = ""
        row["PEAK_LATERAL_ACCEL_MS2"] = ""
        row["PEAK_YAW_DEG_S"] = ""
        row["THW_MIN_S"] = ""
        row["THW_PERCEPTION_S"] = ""
        row["VRU_COUNT"] = ""
        # ── Severity ──
        row["SEVERITY"] = ""
        # ── Surrogate safety ──
        row["TTC_MIN_S"] = ""
        row["DRAC_MAX_MS2"] = ""
        row["MTTC_S"] = ""
        row["SEVERITY_DOCTOR"] = ""
        # ── Distance ──
        row["MIN_DIST_M"] = ""
        row["MIN_DIST_CORRECTED_M"] = ""
        row["MIN_LATERAL_DIST_M"] = ""
        row["ESCOOTER_DECK_CORRECTION"] = ""
        # ── VRU auto-derived ──
        row["AUTO_VRU_CLASS"] = ""
        row["VRU_SPEED_KMH"] = ""
        row["VRU_MOVEMENT"] = ""
        row["VRU_GAIT"] = ""
        row["VRU_CROSSING_DIR"] = ""
        # ── Manual variables ──
        for var_name in MANUAL_VARIABLES:
            if var_name in ("CONFIRM", "AWARE_BEFORE_MINDIST"):
                continue
            row[var_name] = ""
        row["GROUP_FLAG"] = ""
        # ── Group ──
        row["INTERACTION_GROUP"] = ""
        for var_name in GROUP_VARIABLES:
            row[var_name] = ""
        # ── Context ──
        row["CONTEXTUAL_VEHICLES"] = ""
        row["N_CONTEXTUAL_VEHICLES"] = ""
        row["CONSTRAINED_PATH"] = ""
        row["CONSTRAINED_ZONE_TYPE"] = ""
        # ── Flags ──
        row["STEERING_OVERLAP"] = ""
        row["OBSTACLE_ZONE_OVERLAP"] = ""
        row["TRUNCATED"] = ""
        row["N_TRUNCATED_FRAMES"] = ""
        row["N_SIMULTANEOUS_VRUS"] = ""
        row["DENSITY_SECONDARY"] = ""
        row["DENSITY_RANK"] = ""
        row["BASKET_OCCLUDED"] = ""
        # ── Obstacles ──
        row["OBSTACLES"] = ""
        row["OBSTACLE_POINTS"] = ""
        row["OBSTACLE_WIDTH_M"] = ""
        # ── Lane lines ──
        row["LANE_LINES"] = ""
        row["LANE_LEFT_DIST_M"] = ""
        row["LANE_RIGHT_DIST_M"] = ""
        row["LANE_WIDTH_M"] = ""
        row["VRU_OFFSET_FROM_LEFT_M"] = ""
        row["VRU_OFFSET_FROM_RIGHT_M"] = ""
        # ── Tail ──
        row["NOTES"] = "no_interaction_baseline"
        row["NOTE_TIMESTAMPS"] = ""
        row["CORRECTION_LOG"] = ""
        # ── Trip context (joined from trip-level coding) ──
        row["WEATHER"] = self.trip_codes.get('WEATHER', '')
        row["LIGHTING"] = self.trip_codes.get('LIGHTING', '')
        row["SURFACE_CONDITION"] = self.trip_codes.get('SURFACE_CONDITION', '')
        row["ZONE_TYPE"] = self.trip_codes.get('ZONE_TYPE', '')
        row["VISUAL_SEGREGATION"] = self.trip_codes.get('VISUAL_SEGREGATION', '')
        row["RIDING_COMPANION"] = self.trip_codes.get('RIDING_COMPANION', '')
        # ── Coding timestamps ──
        row["CODING_START_TS"] = ""
        row["CODING_END_TS"] = ""
        # ── Clip metadata ──
        row["CLIP_DURATION_S"] = round(self.duration_total, 2)
        row["CLIP_FPS"] = round(self.fps, 2)
        # ── Calibration parameters (reproducibility) ──
        row["CAL_FOCAL_PX"] = self.focal_length_px
        row["CAL_CAM_HEIGHT_M"] = self.camera_height_m
        row["CAL_PITCH_DEG"] = getattr(self, 'pitch_deg', '')
        return row

    def _finalise_manual_track(self):
        """Finalise a manually created track: add to det_df and create encounter."""
        tid = self.manual_track_id
        points = self.manual_track_points

        # Compute ground-plane distance for each clicked foot point
        horizon_v = self.height / 2.0 - self.focal_length_px * np.tan(
            np.radians(self.pitch_deg))
        h_cam = self.camera_height_m
        f_px = self.focal_length_px

        new_rows = []
        for frame, (fx, fy) in sorted(points.items()):
            dv = fy - horizon_v
            dist_m = f_px * h_cam / dv if dv > 5 else np.nan
            lat_m = (fx - self.width / 2.0) * dist_m / f_px if not np.isnan(dist_m) else np.nan
            # Approximate bbox from foot position (assume ~120px height person)
            est_bbox_h = f_px * h_cam * 1.7 / max(dv, 10)  # visible person height in px
            est_bbox_h = min(est_bbox_h, self.height * 0.8)
            new_rows.append({
                'frame': frame,
                'track_id': tid,
                'user_type': 'pedestrian',
                'distance_m': round(dist_m, 2) if not np.isnan(dist_m) else 0,
                'lateral_m': round(lat_m, 2) if not np.isnan(lat_m) else 0,
                'foot_x': fx,
                'foot_y': fy,
                'bbox_height': round(est_bbox_h, 1),
                'bbox_x1': int(fx - est_bbox_h * 0.25),
                'bbox_y1': int(fy - est_bbox_h),
                'bbox_x2': int(fx + est_bbox_h * 0.25),
                'bbox_y2': int(fy),
                'confidence': 1.0,
                'visibility_status': 'MANUAL',
                'is_occluded': False,
                'is_interpolated': False,
                'is_interacting': True,
                'distance_method': 'manual_ground_plane',
            })

        if not new_rows:
            print("    [MANUAL] No valid points. Cancelled.")
            self.manual_track_mode = False
            return

        new_df = pd.DataFrame(new_rows)

        # Add time_s column
        new_df['time_s'] = new_df['frame'] / self.fps

        # Append to det_df
        self.det_df = pd.concat([self.det_df, new_df], ignore_index=True)
        self.det_df = self.det_df.sort_values(['frame', 'track_id']).reset_index(drop=True)

        # Re-index detections by frame
        self._det_by_frame = {}
        for frame_num, group in self.det_df.groupby('frame'):
            self._det_by_frame[int(frame_num)] = group

        # Create encounter from this manual track
        frames = sorted(points.keys())
        dists = [r['distance_m'] for r in new_rows if r['distance_m'] > 0]
        min_dist = min(dists) if dists else 0.0
        min_dist_frame = frames[0]
        if dists:
            min_dist_idx = dists.index(min_dist)
            min_dist_frame = sorted(points.keys())[min_dist_idx]

        # Speed at min_dist frame
        speed_kmh = 0.0
        sensor = self._sensor_by_frame.get(min_dist_frame, {})
        speed_kmh = sensor.get('speed_kmh', 0.0)

        f_start = min(frames)
        f_end = max(frames)
        enc = {
            'idx': len(self.encounters),
            'primary_track': tid,
            'primary_type': 'pedestrian',
            'frame_start': f_start,
            'frame_end': f_end,
            'frame_mindist': min_dist_frame,
            'frame_perception': f_start,
            'frame_first_detection': f_start,
            'frame_last_valid': f_end,
            'ts_start': f_start / self.fps,
            'ts_mindist': min_dist_frame / self.fps,
            'ts_end': f_end / self.fps,
            'min_dist': min_dist,
            'duration_s': round((f_end - f_start) / self.fps, 2),
            'speed_kmh': speed_kmh,
            'interaction_zone_m': 10.0,
            'peak_decel_ms2': 0.0,
            'peak_yaw_deg_s': 0.0,
            'vru_count': 1,
            'perception_frame': f_start,
            'auto_vru_class': 'pedestrian',
            'class_prob': 1.0,
            'imu_confirmed': False,
            'vru_movement': 'unknown',
            'n_truncated_frames': 0,
            'truncated': False,
            'constrained_path': False,
            'constrained_zone_type': '',
            'constrained_zone_desc': '',
            'status': 'pending',
            'codes': OrderedDict([(k, None) for k in MANUAL_VARIABLES]),
            'notes': '',
            'note_timestamps': [],
            'linked_tracks': [],
            'contextual_vehicles': [],
            'flags': ['MANUAL'],
            'obstacles': [],
        }

        self.encounters.append(enc)
        self.manual_tracks_created.append(tid)

        n_pts = len(points)
        print(f"\n    [MANUAL] Track T{tid} created: {n_pts} frames "
              f"(F{min(frames)}-F{max(frames)}), min_dist={min_dist:.2f}m")
        print(f"    [MANUAL] Encounter E{enc['idx']+1:03d} added. "
              f"Total encounters: {len(self.encounters)}")

        # Reset state
        self.manual_track_mode = False
        self.manual_track_id = None
        self.manual_track_points = {}

        # Select the new encounter
        self.selected_enc_idx = len(self.encounters) - 1
        self.state = self.ENCOUNTER_LIST

    def _get_lane_for_frame(self, frame):
        """Return the lane segment active at the given frame, or None.

        A segment is active if frame >= segment['frame'] and
        (no end_frame set, or frame <= end_frame).
        If multiple match, picks the one with the latest start frame.
        Falls back to earliest segment if all are after the frame.
        Returns None if the frame falls after an ended segment with
        no successor.
        """
        if not self.clip_lane_lines_list:
            return self.clip_lane_lines  # legacy fallback
        # Find segments active at this frame
        candidates = []
        for s in self.clip_lane_lines_list:
            if s['frame'] <= frame:
                end = s.get('end_frame')
                if end is None or frame <= end:
                    candidates.append(s)
        if candidates:
            return max(candidates, key=lambda s: s['frame'])
        # Check if all segments are after this frame — use earliest
        future = [s for s in self.clip_lane_lines_list if s['frame'] > frame]
        if len(future) == len(self.clip_lane_lines_list):
            return min(self.clip_lane_lines_list, key=lambda s: s['frame'])
        # Frame is past an ended segment with no successor
        return None

    def _save_encounters(self):
        """Save all coded encounters to CSV."""
        # Apply distance corrections to encounters before saving
        for enc in self.encounters:
            tid = enc['primary_track']
            corrections = [(f, d) for (f, t), d in self.dist_corrections.items() if t == tid]
            if corrections:
                min_corrected = min(d for _, d in corrections)
                enc['min_dist_corrected'] = round(min_corrected, 2)

        # Include both coded (CONFIRM=1) and skipped (CONFIRM=0) encounters
        # Both are valid rater decisions that must be saved for IRR
        # Check that CONFIRM_INTERACTION was actually set (not just present as None)
        annotated = [e for e in self.encounters
                     if e['status'] in ('coded', 'skipped')
                     and e.get('codes', {}).get('CONFIRM') is not None]

        # If no encounters at all, save a baseline row for DCM modeling
        if len(self.encounters) == 0:
            rows = [self._make_baseline_row()]
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Saved no-interaction baseline row to {self.output_path}")
            return

        if not annotated:
            print("  No annotated encounters to save.")
            return

        rows = [self._encounter_to_row(e) for e in annotated]
        file_exists = os.path.exists(self.output_path)

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"  Saved {len(rows)} encounters ({len(rows[0])} cols) to {self.output_path}")

        # Also save extended debug encounter CSV with all fields
        debug_enc_path = str(Path(self.output_path).with_suffix('')) + "_debug_encounters.csv"
        debug_rows = [self._encounter_to_debug_row(e) for e in annotated]
        with open(debug_enc_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=debug_rows[0].keys())
            writer.writeheader()
            writer.writerows(debug_rows)
        print(f"  Debug encounter CSV: {debug_enc_path}")

        # Also save debug file with ALL encounters (including skipped) + per-frame track data
        self._save_debug_data()

        # Save clip-level obstacle zones and points if any
        self._save_clip_zones()
        self._save_clip_obstacle_points()

        # Run sanity checks after all saves
        self._run_sanity_checks(annotated)

    def _run_sanity_checks(self, annotated):
        """Run post-annotation sanity checks and print summary.

        Checks:
        1. Encounters with CONFIRM=1 but missing VRU_TYPE or INTERACTION_TYPE
        2. Encounters with min_dist > 15m (should not exist after filter)
        3. Encounters with negative or zero duration
        4. Trip variables that are all default/unknown (code 9)
        5. Encounters with SPEED_AT_DMIN_KMH = 0 (GPS dropout?)
        6. Distance corrections (key 5) that increase distance by >50%
        7. Summary: X/Y encounters fully coded, Z issues found
        """
        print("\n  === SANITY CHECKS ===")
        issues = []
        n_confirmed = 0
        n_fully_coded = 0

        for enc in annotated:
            eid = f"E{enc['idx']+1:04d}"
            codes = enc.get('codes', {})
            confirm = codes.get('CONFIRM')

            if confirm == 1:
                n_confirmed += 1

                # Check 1: CONFIRM=1 but missing VRU_TYPE or INTERACTION_TYPE
                vru_type = codes.get('VRU_TYPE')
                interaction_type = codes.get('INTERACTION_TYPE')
                missing = []
                if vru_type is None or vru_type == '':
                    missing.append('VRU_TYPE')
                if interaction_type is None or interaction_type == '':
                    missing.append('INTERACTION_TYPE')
                if missing:
                    issues.append(f"  [1] {eid}: CONFIRM=1 but missing "
                                  f"{', '.join(missing)}")
                else:
                    n_fully_coded += 1

            # Check 2: min_dist > 15m
            min_dist = enc.get('min_dist')
            if min_dist is not None and min_dist > 15.0:
                issues.append(f"  [2] {eid}: min_dist={min_dist:.1f}m > 15m "
                              f"(outside interaction zone)")

            # Check 3: negative or zero duration
            duration = enc.get('duration_s')
            if duration is not None and duration <= 0:
                issues.append(f"  [3] {eid}: duration={duration:.2f}s "
                              f"(<= 0)")

            # Check 5: SPEED_AT_DMIN_KMH = 0 (GPS dropout)
            mindist_f = enc.get('frame_mindist')
            if mindist_f is not None:
                sensor_dmin = self._sensor_by_frame.get(mindist_f, {})
                spd = sensor_dmin.get('speed_kmh')
                if spd is not None and spd == 0.0:
                    issues.append(f"  [5] {eid}: SPEED_AT_DMIN=0 km/h "
                                  f"at F{mindist_f} (GPS dropout?)")

            # Check 6: distance correction increases distance by >50%
            min_corrected = enc.get('min_dist_corrected')
            if (min_corrected is not None and min_dist is not None
                    and min_dist > 0):
                ratio = min_corrected / min_dist
                if ratio > 1.5:
                    issues.append(
                        f"  [6] {eid}: distance correction "
                        f"{min_dist:.1f}m -> {min_corrected:.1f}m "
                        f"(+{(ratio-1)*100:.0f}%)")

        # Check 4: trip variables all default/unknown
        trip_all_unknown = True
        for var_name in TRIP_VARIABLES:
            val = self.trip_codes.get(var_name)
            if val is not None and val != '' and val != 9:
                trip_all_unknown = False
                break
        if trip_all_unknown and self.trip_codes:
            issues.append("  [4] Trip variables are ALL unknown/default "
                          "(code 9 or empty)")

        # Print summary
        n_total = len(annotated)
        n_issues = len(issues)
        if issues:
            for issue in issues:
                print(issue)
        summary_line = (f"Summary: {n_fully_coded}/{n_confirmed} confirmed "
                        f"encounters fully coded, "
                        f"{n_total} total annotated, "
                        f"{n_issues} issue(s) found")
        print(f"\n  {summary_line}")
        if n_issues == 0:
            print("  All checks passed.")
        print("  === END SANITY CHECKS ===\n")

        # Save sanity check warnings to log file
        if issues:
            log_path = Path(self.output_path).with_name(
                Path(self.output_path).stem.replace(
                    '_encounters', '_sanity_check') + '.log')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"Sanity check: {datetime.now().isoformat()}\n")
                f.write(f"Clip: {self.trip_id}\n")
                f.write(f"Rater: {self.rater_id}\n\n")
                for w in issues:
                    f.write(w + '\n')
                f.write(f"\n{summary_line}\n")
            print(f"  Sanity check log saved: {log_path}")

    def _save_debug_data(self):
        """Save detailed auto-detection data for debugging distance jumps, speed issues, etc."""
        debug_path = str(Path(self.output_path).parent /
                         f"{self.video_stem}_debug_autodetect.csv")

        rows = []
        for enc in self.encounters:
            tid = enc['primary_track']
            track_df = self.det_df[self.det_df['track_id'] == tid]

            # Pre-compute per-frame TTC and DRAC via 3-frame central difference
            track_sorted = track_df.sort_values('frame')
            sorted_frames = track_sorted['frame'].values.astype(int)
            sorted_dists = track_sorted['distance_m'].values
            frame_ttc = {}
            frame_drac = {}
            fps = self.fps
            for i in range(1, len(sorted_frames) - 1):
                d_before = sorted_dists[i - 1]
                d_now = sorted_dists[i]
                d_after = sorted_dists[i + 1]
                if (d_before is None or d_now is None or d_after is None
                        or pd.isna(d_before) or pd.isna(d_now) or pd.isna(d_after)):
                    continue
                dt = (sorted_frames[i + 1] - sorted_frames[i - 1]) / fps
                if dt <= 0:
                    continue
                d_dot = (float(d_after) - float(d_before)) / dt
                f_key = int(sorted_frames[i])
                dist_val = float(d_now)
                if d_dot < -0.1:  # closing approach
                    v_closing = abs(d_dot)
                    ttc_val = dist_val / v_closing
                    frame_ttc[f_key] = round(min(ttc_val, 30.0), 2)
                    if dist_val > 0.01:
                        drac_val = v_closing ** 2 / (2.0 * dist_val)
                        frame_drac[f_key] = round(min(drac_val, 20.0), 2)

            for _, det in track_df.iterrows():
                frame = int(det['frame'])
                sensor = self._sensor_by_frame.get(frame, {})
                row = OrderedDict()
                row['encounter_idx'] = enc['idx']
                row['encounter_status'] = enc['status']
                row['track_id'] = tid
                row['frame'] = frame
                row['time_s'] = round(frame / self.fps, 3)
                row['distance_smooth_m'] = det.get('distance_m', '')
                row['distance_raw_m'] = det.get('distance_raw_m', det.get('distance_m', ''))
                row['user_type_yolo'] = det.get('user_type', '')
                row['foot_x'] = det.get('foot_x', '')
                row['foot_y'] = det.get('foot_y', '')
                row['bbox_height'] = det.get('bbox_height', '')
                row['lateral_m'] = det.get('lateral_m', '')
                row['is_occluded'] = det.get('is_occluded', '')
                row['is_interpolated'] = det.get('is_interpolated', '')
                row['speed_kmh_corrected'] = sensor.get('speed_kmh', '')
                row['acc_x_g'] = det.get('acc_x_g', '')
                row['yaw_rate_dps'] = det.get('yaw_rate_dps', '')
                # Per-frame SSMs and quality (V3.6 skeleton sec 13.2)
                dist_m = det.get('distance_m')
                spd_kmh = sensor.get('speed_kmh')
                if dist_m and spd_kmh and spd_kmh > 0.5:
                    spd_ms = spd_kmh / 3.6
                    row['thw_s'] = round(float(dist_m) / spd_ms, 2) if spd_ms > 0.1 else ''
                else:
                    row['thw_s'] = ''
                row['ttc_s'] = frame_ttc.get(frame, '')
                row['drac_ms2'] = frame_drac.get(frame, '')
                # Distance quality: sigmoid(bbox_height, k=0.3, x0=30)
                bh = det.get('bbox_height')
                if bh and pd.notna(bh):
                    bh_f = float(bh)
                    row['distance_quality'] = round(1.0 / (1.0 + np.exp(-0.3 * (bh_f - 30))), 3)
                else:
                    row['distance_quality'] = ''
                rows.append(row)

        if rows:
            with open(debug_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Debug data saved: {debug_path} ({len(rows)} rows)")

    def _save_trip_annotation(self):
        """Save trip-level annotation to trip_annotations.csv.

        Uses read-filter-write to deduplicate: if this (TRIP_ID, CLIP_FILE, RATER_ID)
        already exists, the old row is replaced with the new one.
        """
        n_coded = sum(1 for e in self.encounters if e['status'] == 'coded')
        row = OrderedDict()
        row["TRIP_ID"] = self.trip_id
        row["RIDER_ID"] = self.rider_id
        row["CLIP_FILE"] = os.path.basename(self.video_path)
        row["RATER_ID"] = self.rater_id
        for var_name in TRIP_VARIABLES:
            val = self.trip_codes.get(var_name)
            row[var_name] = val if val is not None else ""
        row["N_ENCOUNTERS"] = n_coded

        # Read existing rows, filter out duplicates for this (trip, clip, rater)
        existing_rows = []
        if os.path.exists(self.trip_output_path):
            try:
                with open(self.trip_output_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for existing in reader:
                        if (existing.get('TRIP_ID') == str(self.trip_id) and
                                existing.get('CLIP_FILE') == os.path.basename(self.video_path) and
                                str(existing.get('RATER_ID', '')) == str(self.rater_id)):
                            continue  # skip old duplicate
                        existing_rows.append(existing)
            except Exception:
                pass  # if file is corrupted, start fresh

        # Write all rows (existing non-duplicates + new row)
        all_rows = existing_rows + [row]
        with open(self.trip_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            for r in all_rows:
                # Ensure all keys exist (old rows may have different columns)
                safe_row = {k: r.get(k, '') for k in row.keys()}
                writer.writerow(safe_row)

        print(f"  Trip annotation saved to {self.trip_output_path}")

    # ─── Main loop ───

    def run(self):
        """Main annotation loop."""
        _tag = os.environ.get('NEWMOB_WINDOW_TAG', '')
        window_name = f"{_tag}NewMob v3 - {Path(self.video_path).stem}"
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(self.width, 1600), min(self.height, 900))
        cv2.setMouseCallback(window_name, self._mouse_callback)

        # Video recorder
        self._recorder = None
        if self.record_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            rec_w = min(self.width, 1600)
            rec_h = min(self.height, 900)
            self._recorder = cv2.VideoWriter(self.record_path, fourcc, 15.0, (rec_w, rec_h))
            print(f"  Recording to: {self.record_path} ({rec_w}x{rec_h} @15fps)")

        # Start with pre-encounter phase: obstacles -> calibration -> detect
        # (Steering segmentation removed — harsh steering episodes are flagged
        # automatically via IMU yaw_rate in encounter list as [STEER] tags.
        # No manual steering phase needed.)
        if self.pre_encounter_phase:
            print(f"\n  Starting annotation tool v3.")
            print(f"  Pre-encounter phase: Obstacles -> Calibration -> Detect")
            print(f"  Camera: h={self.camera_height_m:.2f}m f={self.focal_length_px:.0f}px")
            self._enter_obstacle_marking()
        else:
            if not self.encounters:
                print("\n  No encounters detected. Nothing to annotate.")
                print("  Press ESC to quit or 6 to calibrate and re-detect.")
            print(f"\n  Starting annotation tool v3.")
            print(f"  {len(self.encounters)} encounters auto-detected.")
            print(f"  Controls: ./,=navigate ENTER=open 3=YOLO 6=calibrate ESC=quit\n")
            if self.encounters:
                self.current_frame = self.encounters[0]['frame_mindist']

        last_frame_time = time.time()

        while True:
            # Auto-advance if playing
            now = time.time()
            if self.playing and (now - last_frame_time) >= 1.0 / self.annotation_fps:
                self.current_frame = min(self.current_frame + self.frame_step, self.total_frames - 1)
                last_frame_time = now

            self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))

            # Get primary track for current encounter
            primary_track = None
            current_enc = None
            if self.encounters and self.selected_enc_idx < len(self.encounters):
                current_enc = self.encounters[self.selected_enc_idx]
                primary_track = current_enc.get('primary_track')

            # Draw
            img = self._get_frame_image(self.current_frame)
            img = self._draw_yolo_overlay(img, self.current_frame, primary_track)

            # ── Horizon line + pitch readout ──
            # Visual verification that dynamic pitch correction is working
            if self.show_yolo:
                cx = self.width / 2.0
                cy = self.height / 2.0
                # Use per-frame pitch if IMU data available
                _eff_pitch = self.pitch_deg
                _pitch_src = "static"
                if hasattr(self, 'det_df') and self.det_df is not None and 'pitch_deg' in self.det_df.columns:
                    frame_rows = self._det_by_frame.get(self.current_frame)
                    if frame_rows is not None and len(frame_rows) > 0:
                        _imu_p = frame_rows.iloc[0].get('pitch_deg')
                        if pd.notna(_imu_p):
                            _imu_baseline = getattr(self, '_imu_pitch_baseline', None)
                            if _imu_baseline is not None:
                                _delta = max(-3.0, min(3.0, float(_imu_p) - _imu_baseline))
                                _eff_pitch = self.pitch_deg + _delta
                                _pitch_src = "IMU"
                _hv = int(cy - self.focal_length_px * np.tan(np.radians(_eff_pitch)))
                # Draw dashed horizon line
                dash_len = 20
                for x0 in range(0, self.width, dash_len * 2):
                    x1 = min(x0 + dash_len, self.width)
                    cv2.line(img, (x0, _hv), (x1, _hv), (0, 255, 255), 1)
                # Pitch readout in top-left area (below status bar)
                cv2.putText(img, f"pitch={_eff_pitch:.1f} ({_pitch_src})",
                            (self.width - 220, self.height - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # ── Basket/handlebar mask overlay ──
            if self.basket_mask:
                bx1, by1, bx2, by2 = self.basket_mask
                overlay = img.copy()
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
                img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 180), 1)
                cv2.putText(img, "BASKET", (bx1 + 5, by1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

            # ── Lane marking: show saved lane lines on video + in-progress clicks ──
            # Draw saved lane segments (persistent — always visible)
            if self.clip_lane_lines_list:
                active_lane = self._get_lane_for_frame(self.current_frame)
                for si, seg in enumerate(self.clip_lane_lines_list):
                    is_active = (seg is active_lane)
                    alpha = 1.0 if is_active else 0.5
                    for side, base_color in [('left', (0, 255, 0)), ('right', (0, 200, 255))]:
                        pts = seg.get(side, [])
                        if len(pts) >= 2:
                            color = tuple(int(c * alpha) for c in base_color)
                            thickness = 2 if is_active else 1
                            cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, thickness)
                    if is_active:
                        cv2.putText(img, f"LANE seg{si+1}", (10, self.height - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw in-progress clicks while in lane_marking_mode
            if self.lane_marking_mode and self.lane_marking_clicks:
                for i, (cx, cy) in enumerate(self.lane_marking_clicks):
                    color = (0, 255, 0) if i < 2 else (0, 200, 255)
                    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 15, 2)
                    label = f"L{i+1}" if i < 2 else f"R{i-1}"
                    cv2.putText(img, label, (cx + 10, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # Draw partial lines
                if len(self.lane_marking_clicks) >= 2:
                    p1, p2 = self.lane_marking_clicks[0], self.lane_marking_clicks[1]
                    cv2.line(img, p1, p2, (0, 255, 0), 2)
                if len(self.lane_marking_clicks) >= 4:
                    p3, p4 = self.lane_marking_clicks[2], self.lane_marking_clicks[3]
                    cv2.line(img, p3, p4, (0, 200, 255), 2)

            # Trajectory overlay: full VRU foot path for primary track
            if (getattr(self, 'show_trajectory', False)
                    and primary_track is not None
                    and self.det_df is not None
                    and self.state in (self.ENCOUNTER_VIEW, self.CODING)):
                trk = self.det_df[self.det_df['track_id'] == primary_track].sort_values('frame')
                pts = []
                for _, r in trk.iterrows():
                    fx = r.get('foot_x')
                    fy = r.get('foot_y') if 'foot_y' in r else r.get('bbox_y2')
                    if pd.notna(fx) and pd.notna(fy):
                        pts.append((int(fx), int(fy), int(r['frame'])))
                if len(pts) >= 2:
                    for i in range(len(pts) - 1):
                        # Color fades from blue (past) to green (future) relative to current frame
                        f = pts[i][2]
                        if f <= self.current_frame:
                            alpha = max(0.2, 1.0 - (self.current_frame - f) / max(1, len(pts)))
                            c = (int(180 * alpha), int(120 * alpha), 0)  # Teal-ish past
                        else:
                            c = (0, 200, 100)  # Green future
                        cv2.line(img, (pts[i][0], pts[i][1]),
                                 (pts[i+1][0], pts[i+1][1]), c, 2)
                    # Mark current-frame position with a larger dot
                    for px, py, pf in pts:
                        if pf == self.current_frame:
                            cv2.circle(img, (px, py), 6, (0, 255, 255), -1)
                            break
            img = self._draw_timeline(img, current_enc)
            if self.state in (self.ENCOUNTER_VIEW, self.CODING):
                img = self._draw_signal_charts(img, current_enc)
            img = self._draw_encounter_list(img)
            img = self._draw_coding_overlay(img)
            img = self._draw_review_overlay(img)
            img = self._draw_grouping_overlay(img)
            img = self._draw_calibration(img)
            # IMU signal overlay (always on during rider segmentation, toggle with 'i')
            if self.show_imu_overlay or self.state in (self.RIDER_SEGMENT,
                                                        self.RIDER_SEGMENT_CODING):
                img = self._draw_imu_overlay(img)
            # Rider segment boundary markers
            if self.state in (self.RIDER_SEGMENT, self.RIDER_SEGMENT_CODING):
                img = self._draw_rider_segment_overlay(img)
            # Bird's-eye view minimap (toggle with 'b')
            if self.show_bev_minimap:
                img = self._draw_bev_minimap(img)
            # GPS trajectory minimap (toggle with 'g')
            if self.show_gps_minimap:
                img = self._draw_gps_minimap(img)
            # Manual track creation mode indicator
            if self.manual_track_mode:
                n_pts = len(self.manual_track_points)
                has_current = self.current_frame in self.manual_track_points
                msg = (f"MANUAL TRACK T{self.manual_track_id}: "
                       f"{n_pts} pts | "
                       f"{'[*]' if has_current else 'click foot'} | "
                       f"ENTER=done ESC=cancel")
                cv2.putText(img, msg, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                # Draw all clicked points as green markers
                for f, (px, py) in self.manual_track_points.items():
                    # Past frames: dim green dots
                    if f != self.current_frame:
                        age = abs(self.current_frame - f)
                        brightness = max(80, 220 - age * 8)
                        cv2.circle(img, (px, py), 3, (0, brightness, 0), -1)
                    else:
                        # Current frame: bright green cross
                        cv2.drawMarker(img, (px, py), (0, 255, 0),
                                       cv2.MARKER_CROSS, 15, 2)
                        # Show distance
                        horizon_v = self.height / 2.0 - self.focal_length_px * np.tan(
                            np.radians(self.pitch_deg))
                        dv = py - horizon_v
                        if dv > 5:
                            d = self.focal_length_px * self.camera_height_m / dv
                            cv2.putText(img, f"{d:.1f}m", (px + 10, py - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw trail connecting points in frame order
                sorted_pts = sorted(self.manual_track_points.items())
                if len(sorted_pts) >= 2:
                    for i in range(len(sorted_pts) - 1):
                        p1 = sorted_pts[i][1]
                        p2 = sorted_pts[i + 1][1]
                        cv2.line(img, p1, p2, (0, 200, 0), 1)

            # Distance correction mode indicator
            if self.dist_correction_mode:
                n_pts = len(self.dist_correction_points)
                quick_foot = getattr(self, 'dist_correction_quick_foot', False)
                mode_tag = " [QUICK-FOOT]" if quick_foot else ""
                if self.dist_correction_pending_click:
                    msg = "Press 1=Head 2=Shoulder 3=Hip 4=Knee 5=Foot"
                elif quick_foot and n_pts == 0:
                    msg = f"DIST{mode_tag}: click VRU feet to measure. f=toggle  ESC=done"
                elif n_pts == 0:
                    msg = f"DIST{mode_tag}: click body parts (top-to-bottom). ENTER when done. f=quick-foot"
                else:
                    msg = f"DIST{mode_tag}: {n_pts} pts. Click more or ENTER to compute."
                cv2.putText(img, msg, (self.width // 2 - 250, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
                # Draw nearest previous corrected frame's points as ghost markers
                prev_corr_frame = None
                for pf in range(self.current_frame - 1, max(0, self.current_frame - 30), -1):
                    if pf in self.dist_correction_history:
                        prev_corr_frame = pf
                        break
                if prev_corr_frame is not None:
                    prev_pts = self.dist_correction_history[prev_corr_frame]
                    ghost_color = (0, 200, 255)  # Orange — more visible
                    for px, py, pid, pname in prev_pts:
                        cv2.drawMarker(img, (px, py), ghost_color,
                                       cv2.MARKER_CROSS, 14, 2)
                        cv2.putText(img, pname[:3].lower(), (px + 10, py - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ghost_color, 1)
                    if len(prev_pts) >= 2:
                        sp = sorted(prev_pts, key=lambda p: p[1])
                        for i in range(len(sp) - 1):
                            cv2.line(img, (sp[i][0], sp[i][1]),
                                     (sp[i+1][0], sp[i+1][1]), ghost_color, 1)
                    cv2.putText(img, f"prev F{prev_corr_frame}", (self.width // 2 - 250, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ghost_color, 1)
                # Draw track trail from detection data (last 10 frames)
                if primary_track is not None and self.det_df is not None:
                    for trail_f in range(max(0, self.current_frame - 10), self.current_frame):
                        mask = ((self.det_df['track_id'] == primary_track) &
                                (self.det_df['frame'] == trail_f))
                        rows = self.det_df[mask]
                        for _, r in rows.iterrows():
                            fx = int(r['foot_x']) if 'foot_x' in r and pd.notna(r['foot_x']) else None
                            fy = int(r['bbox_y2']) if 'bbox_y2' in r and pd.notna(r.get('bbox_y2')) else None
                            if fy is None and 'foot_y' in r and pd.notna(r.get('foot_y')):
                                fy = int(r['foot_y'])
                            if fx is not None and fy is not None:
                                age = self.current_frame - trail_f
                                brightness = max(80, 220 - age * 14)
                                cv2.circle(img, (fx, fy), 3, (brightness, brightness // 2, 0), -1)
                                if age == 1:
                                    cv2.putText(img, "prev", (fx + 6, fy - 3),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 120, 0), 1)
                # Draw labeled points with names
                for px, py, pid, pname in self.dist_correction_points:
                    cv2.drawMarker(img, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
                    cv2.putText(img, pname[:3].upper(), (px + 8, py - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
                # Draw line connecting all points top to bottom
                if n_pts >= 2:
                    sorted_pts = sorted(self.dist_correction_points, key=lambda p: p[1])
                    for i in range(len(sorted_pts) - 1):
                        cv2.line(img, (sorted_pts[i][0], sorted_pts[i][1]),
                                 (sorted_pts[i+1][0], sorted_pts[i+1][1]), (255, 0, 255), 1)
                # Draw pending unlabeled click
                if self.dist_correction_pending_click:
                    cx, cy = self.dist_correction_pending_click
                    cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_DIAMOND, 15, 2)
            # Show distance correction result ONLY on the exact corrected frame
            if (not self.dist_correction_mode and self.dist_correction_last_result
                    and self.state in (self.ENCOUNTER_VIEW, self.CODING)):
                res_frame, res_dist, res_x, res_y = self.dist_correction_last_result
                if self.current_frame == res_frame:
                    label = f"{res_dist:.2f}m"
                    cv2.putText(img, label, (res_x + 10, res_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.drawMarker(img, (res_x, res_y), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

            # Obstacle point marking mode indicator (supports 1-3 points per obstacle)
            if self.obstacle_point_mode:
                staged = getattr(self, '_obs_pt_staged', None)
                enc = self.encounters[self.selected_enc_idx] if self.encounters else {}
                n_saved = len(enc.get('obstacle_points', []))
                n_multi = len(self._obs_pt_multi)
                count_str = f"[{n_saved} saved] " if n_saved > 0 else ""
                if n_multi > 0:
                    msg = (f"OBS-PT {count_str}{n_multi}/3 pt marked. "
                           f"Click more or press type (1-5/9) / ENTER to confirm")
                else:
                    msg = f"OBS-PT {count_str}Click ground contact (1-3 pts per obstacle). ESC=done"
                cv2.putText(img, msg, (self.width // 2 - 380, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2)
                # Draw all accumulated multi-points for current obstacle
                for pi, mpt in enumerate(self._obs_pt_multi):
                    cv2.drawMarker(img, (mpt['px'], mpt['py']),
                                   (0, 200, 255), cv2.MARKER_DIAMOND, 15, 2)
                    cv2.putText(img, f"{mpt['distance_m']:.2f}m",
                                (mpt['px'] + 10, mpt['py'] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                # Draw lines connecting accumulated multi-points
                if len(self._obs_pt_multi) > 1:
                    for pi in range(len(self._obs_pt_multi) - 1):
                        p1 = (self._obs_pt_multi[pi]['px'], self._obs_pt_multi[pi]['py'])
                        p2 = (self._obs_pt_multi[pi + 1]['px'], self._obs_pt_multi[pi + 1]['py'])
                        cv2.line(img, p1, p2, (0, 200, 255), 2)
                # Draw ALL previously saved obstacle points (persistent markers)
                for i, opt in enumerate(enc.get('obstacle_points', [])):
                    color = (0, 180, 255)  # orange
                    # Draw footprint lines for multi-point obstacles
                    pts = opt.get('points', [(opt['px'], opt['py'], opt['distance_m'])])
                    for pi in range(len(pts)):
                        cv2.drawMarker(img, (pts[pi][0], pts[pi][1]),
                                       color, cv2.MARKER_TILTED_CROSS, 12, 2)
                    for pi in range(len(pts) - 1):
                        cv2.line(img, (pts[pi][0], pts[pi][1]),
                                 (pts[pi + 1][0], pts[pi + 1][1]), color, 1)
                    width_str = f" w={opt.get('width_m', 0):.1f}m" if opt.get('width_m', 0) > 0 else ""
                    label = f"#{i+1} {opt.get('type_name', '?')[:4]} {opt['distance_m']:.1f}m{width_str}"
                    cv2.putText(img, label, (opt['px'] + 10, opt['py'] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Show obstacle point result ONLY on the exact marked frame
            if (not self.obstacle_point_mode and self.obstacle_point_last_result
                    and self.state in (self.ENCOUNTER_VIEW, self.CODING)):
                oframe, odist, ox, oy, otype = self.obstacle_point_last_result
                if self.current_frame == oframe:
                    olabel = f"OBS:{self.OBSTACLE_TYPE_CODES.get(otype, '?')[:4]} {odist:.2f}m"
                    cv2.putText(img, olabel, (ox + 10, oy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                    cv2.drawMarker(img, (ox, oy), (0, 200, 255),
                                   cv2.MARKER_TILTED_CROSS, 15, 2)

            # Show all saved obstacle points for this encounter (on ALL frames, dimmer if not exact frame)
            if (self.state in (self.ENCOUNTER_VIEW, self.CODING) and self.encounters
                    and not self.obstacle_point_mode):
                enc = self.encounters[self.selected_enc_idx]
                for i, opt in enumerate(enc.get('obstacle_points', [])):
                    is_exact = (opt['frame'] == self.current_frame)
                    color = (0, 200, 255) if is_exact else (0, 120, 180)
                    thickness = 2 if is_exact else 1
                    # Draw all points and footprint lines for multi-point obstacles
                    pts = opt.get('points', [(opt['px'], opt['py'], opt['distance_m'])])
                    for pi in range(len(pts)):
                        cv2.drawMarker(img, (pts[pi][0], pts[pi][1]),
                                       color, cv2.MARKER_TILTED_CROSS, 12, thickness)
                    for pi in range(len(pts) - 1):
                        cv2.line(img, (pts[pi][0], pts[pi][1]),
                                 (pts[pi + 1][0], pts[pi + 1][1]), color, thickness)
                    width_str = f" w={opt.get('width_m', 0):.1f}m" if opt.get('width_m', 0) > 0 else ""
                    label = f"#{i+1} {opt.get('type_name', '?')[:4]} {opt['distance_m']:.1f}m{width_str}"
                    cv2.putText(img, label, (opt['px'] + 10, opt['py'] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Lane marking mode indicator
            if self.lane_marking_mode:
                n_clicks = len(self.lane_marking_clicks)
                if n_clicks < 2:
                    lane_msg = f"[LANE MARKING: click L{n_clicks+1} for left edge ({n_clicks}/4)]"
                elif n_clicks < 4:
                    lane_msg = f"[LANE MARKING: click R{n_clicks-1} for right edge ({n_clicks}/4)]"
                else:
                    lane_msg = "[LANE MARKING: complete!]"
                cv2.putText(img, lane_msg, (self.width // 2 - 250, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Live distance preview at cursor (obstacle/dist correction modes)
            if (self._hover_pos is not None
                    and (self.obstacle_point_mode or self.dist_correction_mode)):
                hx, hy = self._hover_pos
                cy_val = self.height / 2.0
                horizon_v = cy_val - self.focal_length_px * np.tan(
                    np.radians(self.pitch_deg))
                dv = hy - horizon_v
                if dv > 1:
                    hover_dist = self.focal_length_px * self.camera_height_m / dv
                    cv2.putText(img, f"{hover_dist:.1f}m",
                                (hx + 15, hy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    # Horizontal guide line at cursor y
                    cv2.line(img, (0, hy), (self.width, hy), (80, 80, 80), 1)

            img = self._draw_status_bar(img)

            # Display FPS indicator (top-left, yellow)
            if self.frame_step > 1 and self.fps_display is not None:
                cv2.putText(img, f"DISPLAY: {self.fps_display}fps",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if self._recorder is not None:
                rec_h, rec_w = img.shape[:2]
                target_w = min(self.width, 1600)
                target_h = min(self.height, 900)
                if (rec_w, rec_h) != (target_w, target_h):
                    rec_img = cv2.resize(img, (target_w, target_h))
                else:
                    rec_img = img
                self._recorder.write(rec_img)

            # ── Apply zoom crop if zoomed in ──
            if self.zoom_level > 1.0:
                vw = int(self.width / self.zoom_level)
                vh = int(self.height / self.zoom_level)
                x0 = max(0, min(self.zoom_cx - vw // 2, self.width - vw))
                y0 = max(0, min(self.zoom_cy - vh // 2, self.height - vh))
                crop = img[y0:y0+vh, x0:x0+vw]
                img_show = cv2.resize(crop, (self.width, self.height),
                                      interpolation=cv2.INTER_LINEAR)
                # Zoom indicator
                cv2.putText(img_show, f"ZOOM {self.zoom_level:.1f}x (scroll to adjust)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                img_show = img

            # Timer-based auto-save (every 60 seconds)
            if time.time() - self._last_autosave_time > 60:
                self._save_session_state(silent=True)
                self._last_autosave_time = time.time()
                self._autosave_flash_until = time.time() + 2.0  # Show indicator for 2s

            # Magnifier loupe overlay (precision-click modes only)
            self._draw_magnifier(img_show, img)

            cv2.imshow(window_name, img_show)

            wait_ms = 1 if self.playing else 30
            raw_key = cv2.waitKeyEx(wait_ms)
            key = raw_key & 0xFF

            if raw_key == -1:
                continue

            # ─── Distance correction: label pending click (1-5) or ENTER to compute ───
            if self.dist_correction_mode:
                # Toggle quick-foot mode with 'f' key
                if key == ord('f') and self.dist_correction_pending_click is None:
                    self.dist_correction_quick_foot = not getattr(self, 'dist_correction_quick_foot', False)
                    if self.dist_correction_quick_foot:
                        print("    [DIST] Quick-foot mode ON: clicks auto-label as foot. Press 'f' to toggle off.")
                    else:
                        print("    [DIST] Quick-foot mode OFF: clicks require 1-5 label.")
                    continue
                # Label a pending click with body part key 1-5
                if self.dist_correction_pending_click is not None:
                    part_key = key - ord('0')  # convert key to int 1-5
                    if part_key in self.BODY_PART_POS:
                        px, py = self.dist_correction_pending_click
                        part_name, part_pos = self.BODY_PART_POS[part_key]
                        self.dist_correction_points.append((px, py, part_key, part_name))
                        self.dist_correction_pending_click = None
                        n = len(self.dist_correction_points)
                        print(f"    [DIST] {part_name} at ({px}, {py}). "
                              f"{n} point(s) labeled. Click more or ENTER to compute.")
                        continue
                    elif key == 27:
                        pass  # fall through to ESC
                    else:
                        continue  # ignore, keep waiting for 1-5

                # ENTER: compute distance from all labeled points (need >= 2)
                elif key == 13 or key == 10:
                    pts = self.dist_correction_points
                    if len(pts) < 2:
                        print(f"    [DIST] Need at least 2 labeled points. Click more body parts.")
                        continue

                    # Sort by y position (top to bottom)
                    pts_sorted = sorted(pts, key=lambda p: p[1])

                    # Ground-plane method: extrapolate foot_y from body proportions,
                    # then use d = f * h_cam / (foot_y - horizon).
                    # No assumption on pedestrian height needed — only calibrated
                    # camera height and anthropometric ratios.
                    #
                    # If foot was directly clicked, use it as-is.
                    # Otherwise, extrapolate from 2+ body part clicks.

                    has_foot_click = any(p[2] == 5 for p in pts)

                    if has_foot_click:
                        # Direct foot click — use its y position
                        foot_clicks = [p for p in pts if p[2] == 5]
                        extrapolated_foot_y = float(np.median([p[1] for p in foot_clicks]))
                        print(f"      Foot clicked directly at y={extrapolated_foot_y:.0f}")
                    else:
                        # Extrapolate foot_y from body part pairs
                        foot_y_estimates = []
                        for i in range(len(pts_sorted)):
                            for j in range(i + 1, len(pts_sorted)):
                                yi = pts_sorted[i][1]
                                yj = pts_sorted[j][1]
                                pos_i = self.BODY_PART_POS[pts_sorted[i][2]][1]
                                pos_j = self.BODY_PART_POS[pts_sorted[j][2]][1]
                                segment_ratio = pos_j - pos_i
                                if segment_ratio < 0.05:
                                    continue
                                segment_px = yj - yi
                                if segment_px < 5:
                                    continue
                                full_h_px = segment_px / segment_ratio
                                # Extrapolate foot: top_point_y + (1.0 - top_ratio) * full_h
                                est_foot_y = yi + (1.0 - pos_i) * full_h_px
                                foot_y_estimates.append(est_foot_y)
                                print(f"      {pts_sorted[i][3]}->{pts_sorted[j][3]}: "
                                      f"{segment_px}px / {segment_ratio:.0%} "
                                      f"→ foot_y={est_foot_y:.0f}px")
                        if not foot_y_estimates:
                            print(f"    [DIST] No valid segment pairs. Try again.")
                            continue
                        extrapolated_foot_y = float(np.median(foot_y_estimates))
                        print(f"      Extrapolated foot_y={extrapolated_foot_y:.0f}px "
                              f"(median of {len(foot_y_estimates)} estimates)")

                    # Ground-plane distance: d = f * h_cam / (foot_y - horizon)
                    cy = self.height / 2.0
                    horizon_v = cy - self.focal_length_px * np.tan(
                        np.radians(self.pitch_deg))
                    dv = extrapolated_foot_y - horizon_v
                    if dv <= 1:
                        print(f"    [DIST] Foot at/above horizon — can't compute distance.")
                        continue
                    corrected_dist = self.focal_length_px * self.camera_height_m / dv

                    enc = self.encounters[self.selected_enc_idx]
                    tid = enc['primary_track']
                    frame = self.current_frame
                    self.dist_corrections[(frame, tid)] = round(corrected_dist, 2)
                    self.dist_correction_history[frame] = list(self.dist_correction_points)
                    foot_pt = sorted(self.dist_correction_points, key=lambda p: p[1])[-1]
                    self.dist_correction_last_result = (
                        frame, round(corrected_dist, 2), foot_pt[0], foot_pt[1])
                    # Update encounter min_dist_corrected in real-time
                    all_corr = [d for (f, t), d in self.dist_corrections.items() if t == tid]
                    enc['min_dist_corrected'] = round(min(all_corr), 2)
                    method = "direct foot" if has_foot_click else "extrapolated foot"
                    print(f"    [DIST] {method}: foot_y={extrapolated_foot_y:.0f}px "
                          f"→ d = f×h_cam/dv = {self.focal_length_px:.0f}×"
                          f"{self.camera_height_m:.2f}/{dv:.0f} "
                          f"= {corrected_dist:.2f}m  (F{frame} T{tid})")
                    self.dist_correction_mode = False
                    self.dist_correction_points = []
                    self.dist_correction_pending_click = None
                    continue

                elif key == 27:
                    pass  # fall through to ESC handler
                else:
                    continue  # in correction mode, ignore other keys

            # ─── Obstacle point marking: handle type label keys for staged point ───
            if self.obstacle_point_mode:
                staged = getattr(self, '_obs_pt_staged', None)
                multi = self._obs_pt_multi
                if staged and multi and key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('9')):
                    # Apply type label and finalize multi-point obstacle
                    type_code = int(chr(key))
                    type_name = self.OBSTACLE_TYPE_CODES.get(type_code, 'Other')
                    obs_entry = self._finalize_obstacle_points(
                        multi, staged['frame'], type_code, type_name)
                    enc = self.encounters[self.selected_enc_idx]
                    obs_pts = enc.setdefault('obstacle_points', [])
                    obs_pts.append(obs_entry)
                    center = obs_entry['points'][len(obs_entry['points']) // 2]
                    self.obstacle_point_last_result = (
                        obs_entry['frame'], obs_entry['distance_m'],
                        center[0], center[1], type_code)
                    n_total = len(obs_pts)
                    n_pts = len(obs_entry['points'])
                    width_str = f", width={obs_entry['width_m']:.2f}m" if obs_entry['width_m'] > 0 else ""
                    print(f"    [OBS-PT] #{n_total} saved: {type_name} ({n_pts} pt) at "
                          f"{obs_entry['distance_m']:.2f}m{width_str} (F{obs_entry['frame']}). "
                          f"Click next obstacle or ESC to finish.")
                    self._obs_pt_staged = None
                    self._obs_pt_multi = []
                    self.obstacle_point_pending_click = None
                    continue
                elif staged and multi and (key == 13 or key == 10):
                    # ENTER = accept with default type (Other/9)
                    obs_entry = self._finalize_obstacle_points(
                        multi, staged['frame'], staged['type'], staged['type_name'])
                    enc = self.encounters[self.selected_enc_idx]
                    obs_pts = enc.setdefault('obstacle_points', [])
                    obs_pts.append(obs_entry)
                    center = obs_entry['points'][len(obs_entry['points']) // 2]
                    self.obstacle_point_last_result = (
                        obs_entry['frame'], obs_entry['distance_m'],
                        center[0], center[1], obs_entry['type'])
                    n_total = len(obs_pts)
                    n_pts = len(obs_entry['points'])
                    width_str = f", width={obs_entry['width_m']:.2f}m" if obs_entry['width_m'] > 0 else ""
                    print(f"    [OBS-PT] #{n_total} saved: {obs_entry['type_name']} ({n_pts} pt) at "
                          f"{obs_entry['distance_m']:.2f}m{width_str} (F{obs_entry['frame']}). "
                          f"Click next obstacle or ESC to finish.")
                    self._obs_pt_staged = None
                    self._obs_pt_multi = []
                    self.obstacle_point_pending_click = None
                    continue
                elif key == 27:
                    pass  # fall through to ESC handler below
                # else: let other keys (playback, navigation) work normally

            # ─── ESC: quit / cancel calibration / cancel distance correction ───
            if key == 27:
                if self.lane_marking_mode:
                    self.lane_marking_mode = False
                    self.lane_marking_clicks = []
                    self._lane_pending_end = False
                    print("    [LANE] Lane marking cancelled.")
                    continue
                if self.obstacle_point_mode:
                    self.obstacle_point_mode = False
                    self.obstacle_point_pending_click = None
                    self._obs_pt_staged = None
                    self._obs_pt_multi = []
                    print("    [OBS-PT] Cancelled.")
                    continue
                if self.dist_correction_mode:
                    self.dist_correction_mode = False
                    self.dist_correction_points = []
                    self.dist_correction_pending_click = None
                    print("    [DIST] Cancelled.")
                    continue
                if self.cal_state is not None:
                    self.cal_state = None
                    self.cal_head_xy = None
                    self.cal_ref_p1 = None
                    self.cal_ref_input = ""
                    self.cal_marking_pairs = []
                    self.cal_marking_p1 = None
                    self.cal_height_input = ""
                    self.cal_ped_pairs = []
                    print("    [CAL] Cancelled.")
                    continue
                # RIDER_SEGMENT: ESC = skip to obstacle marking (pre-encounter) or DONE
                if self.state == self.RIDER_SEGMENT:
                    if self.rider_pass == 'accel' and not self.rider_accel_segments:
                        print("    [RIDER] Skipping rider segmentation.")
                        self.show_imu_overlay = False
                    elif self.rider_pass == 'steer':
                        print("    [RIDER] Skipping steering pass.")
                        self._save_rider_segments()
                        self.show_imu_overlay = False
                    if self.pre_encounter_phase:
                        self._enter_obstacle_marking()
                    else:
                        self.state = self.DONE
                    continue
                # OBSTACLE_MARKING: ESC = skip to calibration
                if self.state == self.OBSTACLE_MARKING:
                    self.clip_obstacle_open = None
                    self.clip_obstacle_open_type = None
                    self.obs_click_mode = False
                    self._obs_pt_multi = []
                    self._obs_pt_staged = None
                    print("  [ZONE] Skipped zone marking")
                    self._enter_calibration_phase()
                    continue
                # CALIBRATION_PHASE: ESC = skip to detection
                if self.state == self.CALIBRATION_PHASE:
                    self._finish_pre_encounter_phase()
                    continue
                # If in TRIP_ANNOTATION, INTERACTION_GROUPING, or GROUP_CODING, ESC goes back to list
                if self.state in (self.TRIP_ANNOTATION, self.INTERACTION_GROUPING, self.GROUP_CODING):
                    print("  [ESC] Back to encounter list")
                    self.state = self.ENCOUNTER_LIST
                    continue
                # First ESC: set pending flag; second ESC within 2s: actually quit
                now = time.time()
                if not hasattr(self, '_esc_pending') or now - self._esc_pending > 2.0:
                    self._esc_pending = now
                    print("  [ESC] Press ESC again within 2s to save & quit")
                    continue
                # Second ESC confirmed — save and go to trip annotation
                del self._esc_pending
                self._save_encounters()
                self._save_session_state()
                if self.state != self.DONE and self.trip_var_idx < len(self.trip_var_names):
                    print("\n  Trip-level annotation (weather, lighting, segregation, companion, etc.):")
                    self.state = self.TRIP_ANNOTATION
                    self.trip_var_idx = 0
                    continue
                break

            # ─── Calibration ref_input state: capture typed length ───
            if self.cal_state == 'ref_input':
                if key == 13 or key == 10:  # ENTER
                    try:
                        ref_length = float(self.cal_ref_input)
                        if ref_length > 0:
                            self._solve_single_reference(ref_length)
                    except ValueError:
                        print(f"    [CAL] Invalid number.")
                    self.cal_state = None
                    self.cal_ref_input = ""
                    continue
                elif key == 8 or key == 127:  # BACKSPACE
                    self.cal_ref_input = self.cal_ref_input[:-1]
                    continue
                elif ord('0') <= key <= ord('9') or key == ord('.'):
                    self.cal_ref_input += chr(key)
                    continue
                continue

            # ─── Calibration head state: mode switching via 2/3/4, height input via digits ───
            if self.cal_state == 'head':
                if key == ord('2'):
                    self.cal_state = 'ref_p1'
                    print(f"    [CAL] Reference mode. Click first endpoint of known-length object.")
                    continue
                elif key == ord('3'):
                    self.cal_state = 'marking_p1'
                    self.cal_marking_pairs = []
                    self.cal_marking_p1 = None
                    print(f"    [CAL] Lane markings mode. Click pairs of endpoints on road markings.")
                    print(f"           Click endpoint 1, then endpoint 2. Repeat for 2+ markings.")
                    print(f"           Press ENTER when done clicking, then select marking type.")
                    continue
                elif key == ord('4'):
                    self.cal_state = 'multi_head'
                    self.cal_ped_pairs = []
                    self.cal_height_input = ""
                    print(f"    [CAL] Multi-ped mode. Click HEAD of person #1 [h={self.ped_height_m:.2f}m].")
                    print(f"           Type digits to set height per person. ENTER when 2+ pairs done.")
                    continue
                elif ord('0') <= key <= ord('9') or key == ord('.'):
                    self.cal_height_input += chr(key)
                    h_str = self.cal_height_input
                    print(f"    [CAL] Height: {h_str}m")
                    continue
                elif key == 8 or key == 127:  # BACKSPACE in height input
                    if self.cal_height_input:
                        self.cal_height_input = self.cal_height_input[:-1]
                        h_str = self.cal_height_input or f"{self.ped_height_m:.2f}"
                        print(f"    [CAL] Height: {h_str}m")
                        continue

            # ─── Calibration marking: ENTER to finish clicking ───
            if self.cal_state == 'marking_p1' and (key == 13 or key == 10):
                if len(self.cal_marking_pairs) < 2:
                    print(f"    [CAL] Need at least 2 marking pairs (have {len(self.cal_marking_pairs)}). Keep clicking.")
                    continue
                self.cal_state = 'marking_type'
                print(f"    [CAL] {len(self.cal_marking_pairs)} pairs collected. Select marking type:")
                print(f"           1=Dash(3.0m) 2=Crossing(0.5m) 3=Lane(3.0m) 4=Custom")
                continue

            # ─── Calibration marking states ───
            if self.cal_state == 'marking_type':
                # Select marking type: 1=Dash 2=Crossing 3=Lane 4=Custom
                if key in (ord('1'), ord('2'), ord('3')):
                    code = key - ord('0')
                    target_len, desc = MARKING_LENGTHS[code]
                    print(f"    [CAL] Marking type: {desc}")
                    self._solve_marking_calibration(target_len)
                    continue
                elif key == ord('4'):
                    self.cal_state = 'marking_custom'
                    self.cal_ref_input = ""
                    print(f"    [CAL] Enter custom length in meters:")
                    continue
                continue

            if self.cal_state == 'marking_custom':
                if key == 13 or key == 10:
                    try:
                        target_len = float(self.cal_ref_input)
                        if target_len > 0:
                            print(f"    [CAL] Custom length: {target_len:.2f}m")
                            self._solve_marking_calibration(target_len)
                        else:
                            print(f"    [CAL] Invalid length.")
                            self.cal_state = None
                    except ValueError:
                        print(f"    [CAL] Invalid number.")
                        self.cal_state = None
                    self.cal_ref_input = ""
                    continue
                elif key == 8 or key == 127:
                    self.cal_ref_input = self.cal_ref_input[:-1]
                    continue
                elif ord('0') <= key <= ord('9') or key == ord('.'):
                    self.cal_ref_input += chr(key)
                    continue
                continue

            # ─── Multi-ped calibration: ENTER to solve, digits for height ───
            if self.cal_state in ('multi_head', 'multi_foot'):
                if key == 13 or key == 10:  # ENTER
                    if len(self.cal_ped_pairs) >= 2:
                        self._solve_multi_ped()
                    else:
                        print(f"    [CAL] Need at least 2 pairs (have {len(self.cal_ped_pairs)}). Keep clicking.")
                    continue
                elif self.cal_state == 'multi_head' and (ord('0') <= key <= ord('9') or key == ord('.')):
                    self.cal_height_input += chr(key)
                    print(f"    [CAL] Height for next person: {self.cal_height_input}m")
                    continue
                elif self.cal_state == 'multi_head' and (key == 8 or key == 127):
                    if self.cal_height_input:
                        self.cal_height_input = self.cal_height_input[:-1]
                        h_str = self.cal_height_input or f"{self.ped_height_m:.2f}"
                        print(f"    [CAL] Height for next person: {h_str}m")
                    continue
                continue

            # ─── Global: SPACE = play/pause (skip when editing notes) ───
            if key == 32 and not self.notes_editing:
                self.playing = not self.playing
                continue

            # ─── . = next frame / next encounter (in list/grouping) ───
            if key == ord('.') or key == ord('>'):
                self.playing = False
                if self.state == self.ENCOUNTER_LIST and self.encounters:
                    if self.selected_enc_idx < len(self.encounters) - 1:
                        self.selected_enc_idx += 1
                    enc = self.encounters[self.selected_enc_idx]
                    self.current_frame = enc['frame_mindist']
                elif self.state == self.SAME_USER_GROUPING:
                    self.grouping_selected = min(self.grouping_selected + 1, len(self.encounters) - 1)
                    enc = self.encounters[self.grouping_selected]
                    self.current_frame = enc['frame_mindist']
                    self._print_grouping_status()
                else:
                    self.current_frame += self.frame_step
                continue

            # ─── , = prev frame / prev encounter (in list/grouping) ───
            if key == ord(',') or key == ord('<'):
                self.playing = False
                if self.state == self.ENCOUNTER_LIST and self.encounters:
                    if self.selected_enc_idx > 0:
                        self.selected_enc_idx -= 1
                    enc = self.encounters[self.selected_enc_idx]
                    self.current_frame = enc['frame_mindist']
                elif self.state == self.SAME_USER_GROUPING:
                    self.grouping_selected = max(self.grouping_selected - 1, 0)
                    enc = self.encounters[self.grouping_selected]
                    self.current_frame = enc['frame_mindist']
                    self._print_grouping_status()
                else:
                    self.current_frame -= self.frame_step
                continue

            # ─── Fast forward/rewind: RIGHT=+30f  LEFT=-30f  UP=+1s  DOWN=-1s ───
            # Arrow keys: macOS=63232-63235, Linux/X11=65362/65364/65361/65363
            if raw_key in (63235, 65363):  # RIGHT arrow = +1 second
                self.playing = False
                self.current_frame = min(self.current_frame + int(self.fps),
                                         self.total_frames - 1)
                continue
            if raw_key in (63234, 65361):  # LEFT arrow = -1 second
                self.playing = False
                self.current_frame = max(self.current_frame - int(self.fps), 0)
                continue
            if raw_key in (63232, 65362):  # UP arrow = +5 seconds
                self.playing = False
                self.current_frame = min(self.current_frame + int(self.fps * 5),
                                         self.total_frames - 1)
                continue
            if raw_key in (63233, 65364):  # DOWN arrow = -5 seconds
                self.playing = False
                self.current_frame = max(self.current_frame - int(self.fps * 5), 0)
                continue

            # ─── Global: z = reset zoom ───
            if key == ord('z') and self.zoom_level > 1.0:
                self.zoom_level = 1.0
                print("  Zoom reset to 1.0x")
                continue

            # ─── Global: 3 = toggle YOLO ───
            if key == ord('3') and self.state in (self.ENCOUNTER_LIST, self.ENCOUNTER_VIEW):
                self.show_yolo = not self.show_yolo
                print(f"  YOLO overlay: {'ON' if self.show_yolo else 'OFF'}")
                continue

            # ─── Global: 5 = distance correction (ENCOUNTER_VIEW or CODING) ───
            if key == ord('5') and self.state in (self.ENCOUNTER_VIEW, self.CODING) and self.encounters:
                # Clear any conflicting click modes before entering distance correction
                self.obstacle_point_mode = False
                self.obs_click_mode = False
                self.lane_marking_mode = False
                self.lane_marking_clicks = []
                self._lane_pending_end = False
                self.manual_track_mode = False
                enc = self.encounters[self.selected_enc_idx]
                vru_code = enc.get('codes', {}).get('VRU_TYPE') or enc.get('vru_type_code', 1)
                self.BODY_PART_POS = self.BODY_PART_POS_BY_TYPE.get(vru_code, self.BODY_PART_POS_BY_TYPE[1])
                type_name = {1: "pedestrian", 2: "cyclist", 3: "e-scooterist",
                             4: "other MMV", 5: "motorised", 6: "animal",
                             7: "stationary"}.get(vru_code, "pedestrian")
                self.dist_correction_mode = True
                self.dist_correction_quick_foot = False
                self.dist_correction_points = []
                self.dist_correction_pending_click = None
                # Auto-navigate to min-distance frame only on first correction
                # (no prior corrections for this track). After that, the user
                # is navigating manually and wants to correct the current frame.
                tid = enc['primary_track']
                has_prior = any(t == tid for (f, t) in self.dist_corrections)
                if not has_prior:
                    mindist_frame = enc.get('frame_mindist')
                    if mindist_frame is not None:
                        self.current_frame = mindist_frame
                n_prior = sum(1 for (f, t) in self.dist_corrections if t == tid)
                print(f"\n  ── DISTANCE CORRECTION ({type_name} body model) F{self.current_frame} T{tid} ──")
                if n_prior:
                    print(f"  ({n_prior} prior correction(s) for this track)")
                print("  Click visible body parts. After each click, label with:")
                print("    1=Head  2=Shoulder  3=Hip  4=Knee  5=Foot")
                print("  ENTER=compute distance  Right-click=undo  ESC=cancel")
                print("  TIP: Press 'f' for quick-foot mode (clicks auto-label as foot)")
                continue

            # ─── Global: o = obstacle point marking (ENCOUNTER_VIEW or CODING) ───
            if key == ord('o') and self.state in (self.ENCOUNTER_VIEW, self.CODING) and self.encounters:
                self.obstacle_point_mode = True
                self.obstacle_point_pending_click = None
                self._obs_pt_multi = []
                enc = self.encounters[self.selected_enc_idx]
                n_existing = len(enc.get('obstacle_points', []))
                print(f"\n  ── OBSTACLE POINT MARKING ({n_existing} saved) ──")
                print("  Click 1-3 ground contact points per obstacle (wide = 3 pts).")
                print("  Then press 1-5/9 to label type, or ENTER for Other.")
                print("  Right-click = undo last  |  ESC = done")
                continue

            # ─── Global: l = lane marking / lane end ───
            if key == ord('l') and self.state in (self.ENCOUNTER_LIST,
                                                   self.ENCOUNTER_VIEW, self.CODING):
                if self.lane_marking_mode:
                    # Already in lane mode: check for end key (0) — handled below
                    self.lane_marking_mode = False
                    self.lane_marking_clicks = []
                    self._lane_pending_end = False
                    print("    [LANE] Lane marking cancelled.")
                else:
                    active = self._get_lane_for_frame(self.current_frame)
                    has_active = (active is not None
                                  and active.get('end_frame') is None)
                    self.lane_marking_mode = True
                    self.lane_marking_clicks = []
                    self._lane_pending_end = has_active
                    n_segs = len(self.clip_lane_lines_list)
                    if has_active:
                        si = self.clip_lane_lines_list.index(active) + 1
                        print(f"    [LANE] L{si} is active (from F{active['frame']}).")
                        print(f"           Press 0 to END it at F{self.current_frame},")
                        print(f"           or click 4 points for a NEW segment. ESC=cancel.")
                    elif n_segs > 0:
                        print(f"    [LANE] {n_segs} segment(s) exist. Adding new at F{self.current_frame}.")
                        print(f"           Click 4 points (2 LEFT + 2 RIGHT).")
                    else:
                        print("    [LANE] Click 2 LEFT edge points, then 2 RIGHT edge points.")
                continue

            # ─── Global: 0 in lane mode = end active lane ───
            if (key == ord('0') and self.lane_marking_mode
                    and getattr(self, '_lane_pending_end', False)):
                active = self._get_lane_for_frame(self.current_frame)
                if active is not None and active.get('end_frame') is None:
                    active['end_frame'] = self.current_frame
                    si = self.clip_lane_lines_list.index(active) + 1
                    print(f"    [LANE] L{si} ended at F{self.current_frame} "
                          f"(span F{active['frame']}-F{self.current_frame}).")
                self.lane_marking_mode = False
                self.lane_marking_clicks = []
                self._lane_pending_end = False
                continue

            # ─── Global: t = toggle trajectory ───
            if key == ord('t') and self.state in (self.ENCOUNTER_VIEW, self.CODING):
                self.show_trajectory = not self.show_trajectory
                print(f"  Trajectory overlay: {'ON' if self.show_trajectory else 'OFF'}")
                continue

            # ─── Global: s = toggle signal charts ───
            if key == ord('s') and self.state in (self.ENCOUNTER_VIEW, self.CODING):
                self.show_signals = not self.show_signals
                print(f"  Signal charts: {'ON' if self.show_signals else 'OFF'}")
                continue

            # ─── Global: i = toggle IMU overlay ───
            if key == ord('i') and self.state in (self.ENCOUNTER_LIST, self.ENCOUNTER_VIEW,
                                                   self.CODING, self.RIDER_SEGMENT,
                                                   self.RIDER_SEGMENT_CODING,
                                                   self.OBSTACLE_MARKING,
                                                   self.CALIBRATION_PHASE):
                self.show_imu_overlay = not self.show_imu_overlay
                print(f"  IMU overlay: {'ON' if self.show_imu_overlay else 'OFF'}")
                continue

            # ─── Global: d = toggle density secondary encounters visibility ───
            if key == ord('d') and self.state == self.ENCOUNTER_LIST:
                self.show_density_secondary = not self.show_density_secondary
                print(f"  Density secondary encounters: "
                      f"{'SHOWN' if self.show_density_secondary else 'HIDDEN'}")
                continue

            # ─── Global: D = batch-skip all density_secondary encounters ───
            if key == ord('D') and self.state == self.ENCOUNTER_LIST:
                n_skipped = 0
                for enc in self.encounters:
                    if enc.get('density_secondary') and enc['status'] == 'pending':
                        enc['status'] = 'skipped'
                        n_skipped += 1
                if n_skipped > 0:
                    print(f"  [DENSE] Batch-skipped {n_skipped} density_secondary encounters")
                    self._save_session_state()
                else:
                    print(f"  [DENSE] No pending density_secondary encounters to skip")
                continue

            # ─── Global: b = toggle BEV minimap ───
            if key == ord('b') and self.state in (self.ENCOUNTER_LIST, self.ENCOUNTER_VIEW,
                                                   self.CODING, self.RIDER_SEGMENT,
                                                   self.RIDER_SEGMENT_CODING,
                                                   self.OBSTACLE_MARKING,
                                                   self.CALIBRATION_PHASE):
                self.show_bev_minimap = not self.show_bev_minimap
                print(f"  BEV minimap: {'ON' if self.show_bev_minimap else 'OFF'}")
                continue

            # ─── Global: g = toggle GPS minimap ───
            if key == ord('g') and self.state in (self.ENCOUNTER_LIST, self.ENCOUNTER_VIEW,
                                                   self.CODING, self.RIDER_SEGMENT,
                                                   self.RIDER_SEGMENT_CODING,
                                                   self.OBSTACLE_MARKING,
                                                   self.CALIBRATION_PHASE):
                if self._gps_trajectory:
                    self.show_gps_minimap = not self.show_gps_minimap
                    print(f"  GPS minimap: {'ON' if self.show_gps_minimap else 'OFF'}")
                else:
                    print(f"  [GPS] No GPS data (need gps_lat/gps_lon in detection CSV)")
                continue

            # ─── Global: 6 = calibrate ───
            if key == ord('6') and self.cal_state is None and self.state in (
                    self.ENCOUNTER_LIST, self.ENCOUNTER_VIEW, self.CALIBRATION_PHASE):
                self.cal_state = 'head'
                self.cal_height_input = ""
                # Remember current encounter track for restoring after calibration
                if self.encounters and self.selected_enc_idx < len(self.encounters):
                    self._pre_cal_track = self.encounters[self.selected_enc_idx]['primary_track']
                h_str = self.cal_height_input or f"{self.ped_height_m:.2f}"
                print(f"    [CAL] HEAD+FOOT mode [h={h_str}m]. Click HEAD of person.")
                print(f"           Type digits to set height | 2=Ref 3=Markings 4=Multi-ped | ESC=cancel")
                continue

            # ─── BACKSPACE (skip when editing notes — handled in REVIEW block) ───
            if (key == 8 or key == 127) and not self.notes_editing:
                if self.state == self.CODING:
                    if self.input_buffer:
                        self.input_buffer = self.input_buffer[:-1]
                    else:
                        self._retreat_coding()
                elif self.state == self.REVIEW:
                    self.state = self.CODING
                    self.coding_var_idx = len(self.coding_var_names) - 1
                    print("  Back to coding")
                elif self.state == self.ENCOUNTER_VIEW:
                    self.state = self.ENCOUNTER_LIST
                    print("  Back to encounter list")
                elif self.state == self.INTERACTION_GROUPING:
                    self.state = self.ENCOUNTER_LIST
                    print("  Back to encounter list")
                elif self.state == self.GROUP_CODING:
                    self._handle_group_coding_key(key)
                elif self.state == self.TRIP_ANNOTATION:
                    self._retreat_trip()
                elif self.state == self.ENCOUNTER_LIST:
                    self.playing = False
                    self.current_frame = max(0, self.current_frame - int(self.fps))
                continue

            # ═══ ENCOUNTER_LIST state ═══
            if self.state == self.ENCOUNTER_LIST:
                if not self.encounters:
                    continue

                if key == 13 or key == 10:  # ENTER = open encounter
                    enc = self.encounters[self.selected_enc_idx]
                    # Dense scene secondary: warn but still open normally
                    if enc.get('density_secondary'):
                        print(f"  [DENSE] E{enc['idx']+1:03d} is density_secondary "
                              f"(rank={enc.get('density_rank', '?')}) — TAB to skip if unneeded")
                    self.state = self.ENCOUNTER_VIEW
                    # V3.0: Show first detection frame for quick CONFIRM triage
                    self.current_frame = enc.get(
                        'onset_frame', enc.get('frame_start', enc['frame_mindist']))
                    self.playing = False
                    print(f"  Opened E{enc['idx']+1:03d}")

                elif key == 9:  # TAB = skip encounter
                    enc = self.encounters[self.selected_enc_idx]
                    if enc['status'] != 'coded':
                        enc['status'] = 'skipped'
                        enc['coding_end_ts'] = datetime.now().isoformat()
                        self._save_session_state()
                        print(f"  Skipped E{enc['idx']+1:03d}")
                        # Move to next
                        if self.selected_enc_idx < len(self.encounters) - 1:
                            self.selected_enc_idx += 1
                            self.current_frame = self.encounters[self.selected_enc_idx]['frame_mindist']

                elif key == ord('j'):  # j = jump to next uncoded/pending encounter
                    found = False
                    start = self.selected_enc_idx + 1
                    for i in list(range(start, len(self.encounters))) + list(range(0, start)):
                        if self.encounters[i]['status'] in ('pending', 'review_later'):
                            self.selected_enc_idx = i
                            self.current_frame = self.encounters[i]['frame_mindist']
                            found = True
                            print(f"  [JUMP] -> E{self.encounters[i]['idx']+1:03d} (pending)")
                            break
                    if not found:
                        print(f"  [JUMP] No pending encounters remaining")

                elif key == ord('p'):  # p = jump to previous uncoded encounter
                    found = False
                    start = self.selected_enc_idx - 1
                    for i in list(range(start, -1, -1)) + list(range(len(self.encounters) - 1, start, -1)):
                        if self.encounters[i]['status'] in ('pending', 'review_later'):
                            self.selected_enc_idx = i
                            self.current_frame = self.encounters[i]['frame_mindist']
                            found = True
                            print(f"  [PREV] -> E{self.encounters[i]['idx']+1:03d} (pending)")
                            break
                    if not found:
                        print(f"  [PREV] No pending encounters remaining")

                elif key == ord('r'):  # r = replay current encounter from onset
                    if self.encounters:
                        enc = self.encounters[self.selected_enc_idx]
                        self.current_frame = enc.get('onset_frame',
                                                     enc.get('frame_start', enc['frame_mindist']))
                        self.playing = True
                        print(f"  Replaying E{enc['idx']+1:03d} from F{self.current_frame}")

                # ./,  navigate encounters in list mode (handled above)

            # ═══ Manual track creation mode (any state) ═══
            if self.manual_track_mode:
                if key == 13 or key == 10:  # ENTER = finalise manual track
                    if len(self.manual_track_points) < 1:
                        print("    [MANUAL] Need at least 1 point. Click foot positions first.")
                        continue
                    self._finalise_manual_track()
                    continue
                elif key == 27:  # ESC = cancel
                    print(f"    [MANUAL] Cancelled. Discarded {len(self.manual_track_points)} points.")
                    self.manual_track_mode = False
                    self.manual_track_id = None
                    self.manual_track_points = {}
                    continue
                # In manual mode, . and , advance frames (handled above)
                # Left-click adds points (handled in mouse callback)
                continue

            # ═══ Key 'n' = New manual track (from ENCOUNTER_LIST or ENCOUNTER_VIEW) ═══
            if key == ord('n') and self.state in (self.ENCOUNTER_LIST, self.ENCOUNTER_VIEW):
                # Assign next track ID (max existing + 100 to avoid collisions)
                max_tid = 0
                if self.det_df is not None and len(self.det_df) > 0:
                    max_tid = int(self.det_df['track_id'].max())
                self.manual_track_id = max(max_tid + 1, 900)  # start at 900+
                while self.manual_track_id in self.manual_tracks_created:
                    self.manual_track_id += 1
                self.manual_track_mode = True
                self.manual_track_points = {}
                print(f"\n  ── MANUAL TRACK CREATION (T{self.manual_track_id}) ──")
                print("  Click the pedestrian's FOOT on each frame.")
                print("  Advance frames with . (next) and , (prev)")
                print("  Right-click = undo current frame's point")
                print("  ENTER = finalise track → creates encounter")
                print("  ESC = cancel")
                continue

            # ═══ ENCOUNTER_VIEW state ═══
            elif self.state == self.ENCOUNTER_VIEW:
                if key == 13 or key == 10:  # ENTER = start coding
                    enc = self.encounters[self.selected_enc_idx]
                    enc['status'] = 'coding'
                    enc['coding_start_ts'] = datetime.now().isoformat()
                    self.state = self.CODING
                    self.coding_var_idx = 0
                    self.input_buffer = ""
                    self._undo_stack = []  # Clear undo stack for new encounter
                    # Apply VLM suggestions as defaults (if loaded)
                    tid = enc.get('primary_track')
                    vlm = self._vlm_suggestions.get(tid, {})
                    if vlm:
                        for vn, vc in vlm.items():
                            if vn in enc['codes'] and enc['codes'][vn] is None:
                                enc['codes'][vn] = vc
                        vlm_vars = ", ".join(f"{k}={v}" for k, v in vlm.items())
                        print(f"    [VLM] Pre-filled: {vlm_vars}")
                    # Apply carry-forward
                    for var_name, val in self.carry_forward.items():
                        if var_name in enc['codes'] and enc['codes'][var_name] is None:
                            enc['codes'][var_name] = val
                    self._navigate_to_suggested_frame()
                    print(f"  Coding E{enc['idx']+1:03d}")

                elif key == ord('L'):  # Mark last valid frame (Shift+L)
                    enc = self.encounters[self.selected_enc_idx]
                    enc['frame_last_valid'] = self.current_frame
                    # Recompute frame_end if needed
                    if self.current_frame < enc['frame_end']:
                        enc['frame_end'] = self.current_frame
                        enc['duration_s'] = round((enc['frame_end'] - enc['frame_start']) / self.fps, 1)
                    print(f"  [LAST] Last valid frame set to F{self.current_frame} for T{enc['primary_track']}")

                elif key == ord('7'):  # Mark general timestamp at current frame
                    enc = self.encounters[self.selected_enc_idx]
                    ts = round(self.current_frame / self.fps, 2)
                    enc['_aware_frame'] = self.current_frame
                    enc['ts_vru_awareness'] = ts
                    print(f"  [MARK] Frame timestamp F{self.current_frame} ({ts:.2f}s) "
                          f"E{enc['idx']+1:03d}")

                elif key == ord('8'):  # Mark obstacle start/end range
                    enc = self.encounters[self.selected_enc_idx]
                    obstacles = enc.setdefault('obstacles', [])
                    # Check if there's an open obstacle (has start but no end)
                    open_obs = [o for o in obstacles
                                if o['type'] == 'obstacle' and o.get('frame_end') is None]
                    if open_obs:
                        # Close the most recent open obstacle
                        o = open_obs[-1]
                        o['frame_end'] = self.current_frame
                        o['time_end'] = round(self.current_frame / self.fps, 2)
                        dur = o['time_end'] - o['time_s']
                        print(f"  [OBS] Obstacle END at F{self.current_frame} "
                              f"(F{o['frame']}→F{self.current_frame}, {dur:.2f}s)")
                    else:
                        # Start a new obstacle range
                        obstacles.append({
                            'frame': self.current_frame,
                            'time_s': round(self.current_frame / self.fps, 2),
                            'frame_end': None,
                            'time_end': None,
                            'type': 'obstacle',
                        })
                        print(f"  [OBS] Obstacle START at F{self.current_frame} "
                              f"— press 8 again to mark end")
                    # Hint: use key 'o' for obstacle point distance marking
                    if not self.obstacle_point_mode:
                        print(f"  [OBS] Tip: press 'o' to measure obstacle distance.")

                elif key == ord('9'):  # Mark note at current frame
                    enc = self.encounters[self.selected_enc_idx]
                    obstacles = enc.setdefault('obstacles', [])
                    obstacles.append({
                        'frame': self.current_frame,
                        'time_s': round(self.current_frame / self.fps, 2),
                        'type': 'note',
                    })
                    print(f"  [NOTE] Note marker at F{self.current_frame} "
                          f"({len(obstacles)} total)")

                elif key == ord('x'):  # Quick-reject: CONFIRM=0, skip all vars, next
                    self._quick_reject_encounter()

            # ═══ CODING state ═══
            elif self.state == self.CODING:
                if key == ord('x'):  # Quick-reject during coding
                    self._quick_reject_encounter()
                else:
                    self._handle_coding_key(key)

            # ═══ REVIEW state ═══
            elif self.state == self.REVIEW:
                if self.notes_editing:
                    # In-window notes editing: capture characters via waitKey
                    if key == 13 or key == 10:  # ENTER = confirm notes
                        enc = self.encounters[self.selected_enc_idx]
                        if self.notes_buffer.strip():
                            # Append new note with frame timestamp (don't overwrite)
                            frame_ts = f"[F{self.current_frame}] "
                            new_note = frame_ts + self.notes_buffer.strip()
                            existing = enc.get('notes', '') or ''
                            if existing:
                                enc['notes'] = existing + " | " + new_note
                            else:
                                enc['notes'] = new_note
                            enc.setdefault('note_timestamps', []).append(self.current_frame)
                            print(f"  Notes saved: {enc['notes'][-60:]}")
                        else:
                            print("  Notes unchanged.")
                        self.notes_editing = False
                        self.notes_buffer = ""
                    elif key == 27:  # ESC = cancel notes
                        print("  Notes cancelled.")
                        self.notes_editing = False
                        self.notes_buffer = ""
                    elif key == 8 or key == 127:  # BACKSPACE = delete char
                        if self.notes_buffer:
                            self.notes_buffer = self.notes_buffer[:-1]
                    elif 32 <= key < 127:  # Printable ASCII
                        self.notes_buffer += chr(key)
                    continue

                if key == 9:  # TAB = start notes editing in-window
                    enc = self.encounters[self.selected_enc_idx]
                    self.notes_editing = True
                    self.notes_buffer = ""  # blank buffer — new notes append to existing
                    existing = enc.get('notes', '') or ''
                    if existing:
                        print(f"  Adding note for E{enc['idx']+1:03d} (existing: {existing[-50:]})")
                    else:
                        print(f"  Adding note for E{enc['idx']+1:03d} (type in window, ENTER=save, ESC=cancel)")
                elif key == 13 or key == 10:  # ENTER = save and return to list
                    enc = self.encounters[self.selected_enc_idx]
                    enc['status'] = 'coded'
                    enc['coding_end_ts'] = datetime.now().isoformat()
                    self.saved_encounters.append(enc)
                    self._coding_timestamps.append((enc['idx'], time.time()))
                    # Compute coding speed for this encounter
                    if enc.get('coding_start_ts'):
                        try:
                            start_dt = datetime.fromisoformat(enc['coding_start_ts'])
                            dur_s = (datetime.now() - start_dt).total_seconds()
                            enc['_coding_duration_s'] = round(dur_s, 1)
                            print(f"  E{enc['idx']+1:03d} coded! ({dur_s:.1f}s)")
                        except (ValueError, TypeError):
                            print(f"  E{enc['idx']+1:03d} coded!")
                    else:
                        print(f"  E{enc['idx']+1:03d} coded!")
                    try:
                        self._save_encounters()
                        self._save_session_state()
                    except Exception as save_err:
                        print(f"  WARNING: Save error: {save_err}")
                        print(f"  Encounter is marked coded in memory — will retry on next save.")
                    # Auto-save safety: periodic backup
                    self.coded_since_last_autosave += 1
                    if self.coded_since_last_autosave >= self.auto_save_interval:
                        self.coded_since_last_autosave = 0
                        print(f"  [AUTO-SAVE] Checkpoint ({self.auto_save_interval} coded)")

                    # Check if all encounters are coded or skipped
                    all_done = all(e['status'] in ('coded', 'skipped')
                                   for e in self.encounters)
                    if all_done:
                        self._enter_interaction_grouping()
                    else:
                        # Move to next pending/review_later encounter
                        self.state = self.ENCOUNTER_LIST
                        found_next = False
                        for i in range(len(self.encounters)):
                            if self.encounters[i]['status'] in ('pending', 'review_later'):
                                self.selected_enc_idx = i
                                self.current_frame = self.encounters[i]['frame_mindist']
                                found_next = True
                                break
                        if not found_next:
                            self._enter_interaction_grouping()

            # ═══ INTERACTION_GROUPING state ═══
            elif self.state == self.INTERACTION_GROUPING:
                self._handle_grouping_key(key)

            # ═══ GROUP_CODING state ═══
            elif self.state == self.GROUP_CODING:
                self._handle_group_coding_key(key)

            # ═══ TRIP_ANNOTATION state ═══
            elif self.state == self.TRIP_ANNOTATION:
                self._handle_trip_key(key)

            # ═══ RIDER_SEGMENT state (boundary marking) ═══
            elif self.state == self.RIDER_SEGMENT:
                self._handle_rider_segment_key(key)

            # ═══ RIDER_SEGMENT_CODING state ═══
            elif self.state == self.RIDER_SEGMENT_CODING:
                self._handle_rider_segment_coding_key(key)

            # ═══ OBSTACLE_MARKING state ═══
            elif self.state == self.OBSTACLE_MARKING:
                self._handle_obstacle_marking_key(key)

            # ═══ CALIBRATION_PHASE state ═══
            elif self.state == self.CALIBRATION_PHASE:
                if key == ord('6') and self.cal_state is None:
                    self.cal_state = 'head'
                    self.cal_height_input = ""
                    print(f"    [CAL] HEAD+FOOT mode. Click HEAD of person.")
                    print(f"           2=Ref 3=Markings 4=Multi-ped | ESC=cancel")
                elif (key == 13 or key == 10) and self.cal_state is None:
                    # ENTER = accept calibration, proceed to encounter detection
                    self._finish_pre_encounter_phase()
                elif key == 27 and self.cal_state is None:
                    # ESC = skip calibration, proceed to encounter detection
                    self._finish_pre_encounter_phase()

            # ═══ DONE state ═══
            elif self.state == self.DONE:
                if key == 27:
                    break

        # Safety-net: ensure all annotated encounters (coded or skipped) are persisted
        if any(e['status'] in ('coded', 'skipped')
               and e.get('codes', {}).get('CONFIRM') is not None
               for e in self.encounters):
            self._save_encounters()
            self._save_session_state()

        cv2.destroyAllWindows()
        self.cap.release()
        if self._recorder is not None:
            self._recorder.release()
            print(f"  Recording saved: {self.record_path}")
        n_coded = sum(1 for e in self.encounters if e['status'] == 'coded')
        print(f"\n  Done. {n_coded}/{len(self.encounters)} encounters coded.")

    def _update_selected_from_frame(self):
        """In ENCOUNTER_LIST, update selected encounter based on current frame position."""
        if not self.encounters:
            return
        # Find nearest encounter to current frame
        best_idx = 0
        best_dist = float('inf')
        for i, enc in enumerate(self.encounters):
            # Distance from current frame to encounter window
            if enc['frame_start'] <= self.current_frame <= enc['frame_end']:
                best_idx = i
                best_dist = 0
                break
            d = min(abs(self.current_frame - enc['frame_start']),
                    abs(self.current_frame - enc['frame_end']))
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.selected_enc_idx = best_idx


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def auto_discover_detections(video_path):
    """Try to find a matching detections CSV for the video."""
    stem = Path(video_path).stem
    det_name = f"{stem}_detections.csv"

    search_roots = set()
    for start in [Path(video_path).resolve().parent, Path.cwd().resolve()]:
        p = start
        for _ in range(5):
            search_roots.add(p)
            if (p / 'my_analysis').is_dir():
                search_roots.add(p / 'my_analysis')
            p = p.parent

    for root in search_roots:
        candidate = root / det_name
        if candidate.exists():
            return str(candidate)
        # Rank output dirs: prefer "final" in name, then by number of CSVs (completeness)
        output_dirs = [d for d in root.glob("output*") if d.is_dir()]
        output_dirs.sort(key=lambda d: (
            'final' in d.name,            # prefer dirs with "final"
            len(list(d.glob("*_detections.csv"))),  # prefer more complete sets
            d.name,                        # alphabetical tiebreak
        ), reverse=True)
        for d in output_dirs:
            candidate = d / det_name
            if candidate.exists():
                return str(candidate)
    return None


def auto_discover_zones():
    """Try to find pedestrian_zone_segments.csv."""
    search_paths = [
        Path.cwd() / "my_analysis" / "NewMobKML" / "pedestrian_zone_segments.csv",
        Path.cwd() / "pedestrian_zone_segments.csv",
        Path.cwd().parent / "my_analysis" / "NewMobKML" / "pedestrian_zone_segments.csv",
    ]
    for p in search_paths:
        if p.exists():
            return str(p)
    return None


def main():
    p = argparse.ArgumentParser(
        description="NewMob Encounter Annotation Tool v2 — Auto-Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With explicit detections CSV
  python annotate_encounters.py --video segment.mp4 --detections segment_detections.csv

  # Auto-discover detections (searches output* dirs)
  python annotate_encounters.py --video segment.mp4

  # With zone lookup and trip info
  python annotate_encounters.py --video segment.mp4 --detections det.csv \\
      --zones pedestrian_zone_segments.csv --trip_id P012_VAE_20230315_T001
""")
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--detections", default=None,
                   help="Path to detections CSV (auto-discovered if not provided)")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: <video>_encounters.csv)")
    p.add_argument("--zones", default=None,
                   help="Path to pedestrian_zone_segments.csv")
    p.add_argument("--trip_id", default="", help="Trip ID")
    p.add_argument("--city", default="", help="City name")
    p.add_argument("--rater", "--rater_id", type=int, default=1,
                   help="Rater ID (1 or 2) — used for double-coding; appended to output filename")
    # Vehicle type (auto-detects camera defaults)
    p.add_argument("--vehicle_type", default="auto",
                   choices=["auto", "bike", "escooter"],
                   help="Vehicle type for camera defaults (default: auto-detect from filename)")
    # Camera calibration (None = use vehicle-type default)
    p.add_argument("--camera_height", type=float, default=None,
                   help="Camera height in meters (auto-set by vehicle type if not specified)")
    p.add_argument("--focal_length", type=float, default=None,
                   help="Focal length in pixels (auto: 1445@1080p, 963@720p)")
    p.add_argument("--pitch", type=float, default=None,
                   help="Camera pitch in degrees (auto-set by vehicle type if not specified)")
    p.add_argument("--height", type=float, default=1.70,
                   help="Assumed pedestrian height in meters (default: 1.70)")
    p.add_argument("--calibration", default=None,
                   help="Calibration JSON file")
    p.add_argument("--speed_offset", type=float, default=None,
                   help=f"GPS/IMU speed offset in seconds (default: {DEFAULT_SPEED_OFFSET_S}s). "
                        "Negative = GPS lags video (GPS reports position after it occurred).")
    p.add_argument("--no_recompute_distances", action="store_true",
                   help="Skip recomputing distances from foot_y (use raw CSV distances)")
    p.add_argument("--d_threshold", type=float, default=15.0,
                   help="Fixed distance threshold (m) for encounter detection. "
                        "Default: 15.0m (interaction = d < 15m). "
                        "Use 0 for speed-adaptive 8-20m.")
    p.add_argument("--max_lateral", type=float, default=None,
                   help="Max lateral distance (m) for encounter detection. "
                        "VRUs beyond this lateral distance are excluded (e.g., sidewalk pedestrians). "
                        "Default: None (no lateral filter)")
    p.add_argument("--constrained_zones", default=None,
                   help="CSV with constrained path zones (start_frame,end_frame,zone_type,description). "
                        "Encounters within these zones are flagged as constrained_path=True.")
    p.add_argument("--thw_threshold", "--thw-threshold", type=float, default=5.0,
                   help="THW threshold in seconds (default: 5.0). "
                        "Tracks with min(THW) > threshold are excluded. "
                        "Disabled by default (--no_filter is now default). "
                        "Use --thw_filter to enable.")
    p.add_argument("--max_distance", type=float, default=15.0,
                   help="Max distance in meters for interaction (default 15.0). "
                        "Encounters with min_dist > this are excluded.")
    p.add_argument("--min_ego_speed", type=float, default=0.0,
                   help="Min rider speed in km/h for interaction (default 0 = no filter). "
                        "Encounters where ego speed < threshold are excluded.")
    p.add_argument("--no_filter", action="store_true", default=True,
                   help="Disable THW pre-filter (present all tracks). "
                        "This is now the default — encounters defined by distance < 15m only.")
    p.add_argument("--thw_filter", action="store_true", default=False,
                   help="Enable THW pre-filter (re-enable if needed for focused annotation).")
    p.add_argument("--no_smooth", action="store_true",
                   help="Skip RTS smoothing in GUI (use raw pipeline distances)")
    p.add_argument("--rider_id", type=str, default=None,
                   help="Rider identifier (e.g., R01). Auto-derived from filename if omitted.")
    p.add_argument("--record", default=None,
                   help="Record GUI to video file (e.g., --record demo.mp4). "
                        "Saves every displayed frame including overlays and status bar.")
    p.add_argument("--no_resume", action="store_true", default=False,
                   help="Ignore existing session state file and start fresh. "
                        "By default, the tool auto-resumes from a previous session.")
    p.add_argument("--annotation_fps", type=float, default=10.0,
                   help="GUI frame rate for stepping (default 10, i.e. every 3rd frame). "
                        "Use 1 for 1 Hz review (every 30th frame = data reduction). "
                        "Use 30 for frame-by-frame. Use 5 for fast scanning.")
    p.add_argument("--fps_display", type=int, default=None,
                   help="Display framerate for annotation comfort (default: same as annotation_fps). "
                        "Reduces visual fatigue from camera vibration by showing fewer frames. "
                        "All frames are still processed internally. Example: --fps_display 10")
    p.add_argument("--min_zone_gap", type=float, default=2.0,
                   help="Minimum gap in seconds between same-type zones before auto-merging. "
                        "Adjacent zones of the same type separated by less than this are merged. "
                        "Default: 2.0s. Set to 0 to disable merging.")
    p.add_argument("--basket_mask", type=str, default=None,
                   help="Mask region x1,y1,x2,y2 for basket/handlebar occlusion "
                        "(e.g., '0,500,200,720'). Detections with foot_x,foot_y within "
                        "the mask (+20px margin) are flagged as basket_occluded.")
    p.add_argument("--dense_scene_k", type=int, default=5,
                   help="Dense scene threshold: number of simultaneous VRUs to trigger "
                        "density filtering (default 5). When >K VRUs visible, only nearest N "
                        "are presented for manual coding.")
    p.add_argument("--dense_scene_n", type=int, default=3,
                   help="Number of nearest VRUs to keep for manual coding in dense scenes "
                        "(default 3). The rest are auto-filled with DENSITY_SECONDARY=1.")
    p.add_argument("--max_encounters", type=int, default=None,
                   help="Max encounters to show (best quality first). Default: all. "
                        "When set, encounters are ranked by quality score (distance, duration, "
                        "bbox height, confidence) and only the top N are presented.")
    p.add_argument("--suggestions", default=None,
                   help="Path to VLM pre-annotation suggestions CSV "
                        "(generated by scripts/vlm_pre_annotate.py). "
                        "Columns: track_id, VRU_TYPE, INTERACTION_TYPE, VRU_AGE_GROUP, VRU_GAIT. "
                        "Values pre-fill the coding form; rater presses ENTER to accept or digit to override.")

    args = p.parse_args()

    # ── Vehicle type detection + camera defaults ──
    vehicle_type = args.vehicle_type
    if vehicle_type == 'auto':
        vehicle_type = detect_vehicle_type(args.video)
    veh_defaults = VEHICLE_CAMERA_DEFAULTS.get(vehicle_type, VEHICLE_CAMERA_DEFAULTS['bike'])

    camera_height = args.camera_height if args.camera_height is not None else veh_defaults['camera_height']
    focal_length = args.focal_length  # None = auto-detect from video resolution in __init__
    pitch = args.pitch if args.pitch is not None else veh_defaults['pitch']

    print(f"  Vehicle type: {vehicle_type} "
          f"(h={camera_height:.2f}m, pitch={pitch:.1f}°, f={'auto' if focal_length is None else f'{focal_length:.0f}px'})")

    # d_threshold: 0 means speed-adaptive, any positive value is fixed threshold
    d_threshold = args.d_threshold if args.d_threshold and args.d_threshold > 0 else None
    if d_threshold:
        print(f"  Distance threshold: fixed {d_threshold:.1f}m")
    else:
        print(f"  Distance threshold: speed-adaptive (8-20m)")

    # Auto-discover detections if not provided
    det_path = args.detections
    if not det_path:
        det_path = auto_discover_detections(args.video)
        if det_path:
            print(f"  Auto-discovered detections: {det_path}")
        else:
            print("Error: No detections CSV found. Pass --detections <path>.")
            sys.exit(1)

    # Auto-discover zones
    zones_path = args.zones
    if not zones_path:
        zones_path = auto_discover_zones()
        if zones_path:
            print(f"  Auto-discovered zones: {zones_path}")

    # THW filter: disabled by default (encounters = distance < 15m only).
    # Use --thw_filter to re-enable if focused annotation is needed.
    thw_threshold = args.thw_threshold if args.thw_filter else None
    if thw_threshold is not None:
        print(f"  THW filter: {thw_threshold:.1f}s (tracks with min(THW) > threshold excluded)")
    else:
        print(f"  THW filter: disabled")

    # Parse basket mask region
    basket_mask = None
    if args.basket_mask:
        try:
            bm_parts = [int(v.strip()) for v in args.basket_mask.split(',')]
            if len(bm_parts) == 4:
                basket_mask = tuple(bm_parts)
                print(f"  Basket mask: ({bm_parts[0]},{bm_parts[1]})-({bm_parts[2]},{bm_parts[3]})")
            else:
                print(f"  Warning: --basket_mask requires exactly 4 values (x1,y1,x2,y2), got {len(bm_parts)}")
        except ValueError:
            print(f"  Warning: --basket_mask must be integers (x1,y1,x2,y2), ignoring '{args.basket_mask}'")

    annotator = EncounterAnnotator(
        video_path=args.video,
        detections_path=det_path,
        output_path=args.output,
        zones_path=zones_path,
        trip_id=args.trip_id,
        city=args.city,
        rater_id=args.rater,
        rider_id=args.rider_id,
        camera_height=camera_height,
        focal_length=focal_length,
        pitch=pitch,
        ped_height=args.height,
        calibration_path=args.calibration,
        speed_offset_s=args.speed_offset,
        recompute_distances=not args.no_recompute_distances,
        d_threshold=d_threshold,
        max_lateral_m=args.max_lateral,
        constrained_zones_path=args.constrained_zones,
        record_path=args.record,
        no_smooth=args.no_smooth,
        thw_threshold=thw_threshold,
        max_distance=args.max_distance,
        min_ego_speed_kmh=args.min_ego_speed,
        no_resume=args.no_resume,
        annotation_fps=args.annotation_fps,
        fps_display=args.fps_display,
        min_zone_gap_s=args.min_zone_gap,
        basket_mask=basket_mask,
        dense_scene_k=args.dense_scene_k,
        dense_scene_n=args.dense_scene_n,
        max_encounters=args.max_encounters,
    )

    # Load VLM pre-annotation suggestions if provided
    if args.suggestions and os.path.exists(args.suggestions):
        try:
            sug_df = pd.read_csv(args.suggestions)
            n_loaded = 0
            var_cols = [c for c in sug_df.columns
                        if c in ('VRU_TYPE', 'INTERACTION_TYPE', 'VRU_AGE_GROUP', 'VRU_GAIT')]
            for _, row in sug_df.iterrows():
                tid = int(row.get('track_id', -1))
                if tid < 0:
                    continue
                suggestions = {}
                for vc in var_cols:
                    val = row.get(vc)
                    if pd.notna(val):
                        suggestions[vc] = int(val)
                if suggestions:
                    annotator._vlm_suggestions[tid] = suggestions
                    n_loaded += 1
            print(f"  [VLM] Loaded {n_loaded} suggestions from {args.suggestions} "
                  f"({', '.join(var_cols)})")
        except Exception as e:
            print(f"  [VLM] Failed to load suggestions: {e}")
    elif args.suggestions:
        print(f"  [VLM] Suggestions file not found: {args.suggestions}")

    # Crash recovery wrapper: save session state on unexpected errors
    try:
        annotator.run()
    except Exception as e:
        print(f"\n[CRASH RECOVERY] Saving session after error: {e}")
        try:
            annotator._save_session_state()
            print("[CRASH RECOVERY] Session saved successfully!")
        except:
            print("[CRASH RECOVERY] Failed to save session!")
        raise  # Re-raise so user sees the error


if __name__ == "__main__":
    main()
