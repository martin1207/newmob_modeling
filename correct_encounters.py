#!/usr/bin/env python3
"""
correct_encounters.py — Outil interactif minimaliste pour corriger les
annotations d'encounters d'un clip escooter (inspiré d'annotate_encounters.py
mais limité aux corrections).

Permet de :
  - Ajouter un encounter (clic sur une bbox d'autodetect → range de frames → 4 codes)
  - Supprimer un encounter existant
  - Ajouter / supprimer une zone obstacle (range de frames)

Inputs (par clip <CLIP>) :
  <ESCOOTER_DIR>/<CLIP>.mp4
  <CODEBOOK>/<CLIP>_debug_autodetect.csv
  <CODEBOOK>/<CLIP>_rater{R}_encounters_debug_encounters.csv
  <CODEBOOK>/<CLIP>_rater{R}_encounters_obstacle_zones.csv
  <CODEBOOK>/<CLIP>_rater{R}_session.json   (optionnel, mis à jour si présent)

Lecture seule :
  <ESCOOTER_DIR>/<CLIP>_corrected_with_offset.csv  (pour vitesse GPS)

Sortie :  modifs en place avec backup *.bak du fichier modifié.

Touches :
  ,  .            frame précédente / suivante
  ←  →            saut ±1 seconde
  ↑  ↓            saut ±5 secondes
  SPACE           play / pause
  G               saute à un # de frame (saisie clavier dans le terminal)

  A               add encounter — clic sur une bbox d'autodetect (range = tout
                  le track) → saisie directe des 4 codes (1–9)
  D               delete encounter → numéro 1-9 à l'écran
  E               edit encounter → numéro 1-9, puis traverse les 4 codes
                  (ENTER = garde la valeur actuelle, 1-9 = change)
  U               unlink → numéro 1-9 ; vide le champ LINKED_TRACKS d'un
                  encounter (les liens sont auto-générés par annotate_encounters
                  et souvent pollués par des piétons étrangers).
  J               join : recolle deux track_id quand un track casse d'une
                  frame à l'autre (re-tracking par YOLO sous un nouveau ID).
                  • 1er clic : track qui RESTE (avant le break)
                  • navigue jusqu'à après le break
                  • 2e clic : track ABSORBÉ (après le break)
                  Toutes les frames du 2e track sont remappées sur le 1er,
                  les encounters et LINKED_TRACKS sont remappés aussi.
  M               manual track : on dessine soi-même la bbox à plusieurs
                  keyframes, le reste est interpolé linéairement.
                  • drag souris bouton GAUCHE = pose une keyframe ici
                  • bouton DROIT = supprime la keyframe ici
                  • , . nav, SPACE play/pause, ENTER finalise, ESC annule
                  • à la fin → saisie codes (= un nouveau track + encounter)
  O               add obstacle zone (nav → ENTER pour start, nav → ENTER pour end)
  X               delete obstacle zone   → numéro 1-9 à l'écran
  ENTER           valider l'étape courante (code, frame d'obstacle…)
  ESC             annuler la saisie en cours / quitter (avec confirmation)
  S               sauvegarder (sans quitter)
  Q               sauvegarder + quitter

Usage :
  python correct_encounters.py --clip "389t_2023-05-23 18_01_48_389t_7_32_8_28"
  python correct_encounters.py --clip <CLIP> --rater 2
"""

import os
import sys
import csv
import json
import glob
import shutil
import bisect
import argparse
import datetime
from collections import defaultdict

import cv2

from overlay_annotations import (
    CODE_LABELS, label_of, _open_dictreader,
    COLORS, DEFAULT_COLOR, FONT, BLACK,
    draw_speed, draw_frame_number, draw_obstacle_banner,
    load_speed_by_frame,
)

ROOT         = "/Volumes/My Passport/NEWMOB"
ESCOOTER_DIR = f"{ROOT}/escooter"
CODEBOOK     = f"{ROOT}/codebookescooter"

WIN = "correct_encounters"


# ── Modes / steps de la state machine ────────────────────────────────────────
M_BROWSE   = "BROWSE"
M_ADD_PICK = "ADD: clique une bbox"
M_ADD_F0   = "ADD: nav -> frame_start, ENTER"
M_ADD_F1   = "ADD: nav -> frame_end, ENTER"
M_ADD_CODE = "ADD: saisie codes"          # sub-step driven by code_step
M_DEL      = "DELETE encounter: tape #"
M_EDIT_PICK = "EDIT: tape #"
M_EDIT_CODE = "EDIT: saisie codes"
M_UNLINK    = "UNLINK LINKED_TRACKS: tape #"
M_OBS_F0   = "OBSTACLE: ENTER pour fixer start"
M_OBS_F1   = "OBSTACLE: nav -> end, ENTER"
M_OBS_DEL  = "DELETE obstacle: tape #"
M_MANUAL   = "MANUAL: drag = keyframe, ENTER finalise"
M_MERGE1   = "MERGE: clic 1er track (KEEP)"
M_MERGE2   = "MERGE: clic 2e track (ABSORB)"


CODE_STEPS = [
    ("VRU_TYPE",         "1=Pedestrian 2=Cyclist 3=E-scooter 4=OtherMMV 5=Motor 6=Animal 7=Stationary 9=Unknown"),
    ("VRU_GAIT",         "1=Standing 2=Walking 3=Running 9=Unknown"),
    ("VRU_AGE_GROUP",    "1=Child 2=Adult 3=Elderly 9=Unknown"),
    ("INTERACTION_TYPE", "1=Same-direction 2=Opposite-direction 3=Crossing 4=Stationary 9=Unknown"),
]


# ── I/O helpers ──────────────────────────────────────────────────────────────
def detect_rater(clip):
    """Trouve un rater {1,2} qui a un encounters CSV pour ce clip."""
    for r in (2, 1):
        if os.path.exists(f"{CODEBOOK}/{clip}_rater{r}_encounters_debug_encounters.csv"):
            return r
    return None


def derive_paths(clip, rater):
    return {
        "video":   f"{ESCOOTER_DIR}/{clip}.mp4",
        "speed":   f"{ESCOOTER_DIR}/{clip}_corrected_with_offset.csv",
        "auto":    f"{CODEBOOK}/{clip}_debug_autodetect.csv",
        "enc":     f"{CODEBOOK}/{clip}_rater{rater}_encounters_debug_encounters.csv",
        "obs":     f"{CODEBOOK}/{clip}_rater{rater}_encounters_obstacle_zones.csv",
        "session": f"{CODEBOOK}/{clip}_rater{rater}_session.json",
    }


def load_csv_as_dicts(path):
    """Renvoie (header_list, rows_as_dicts, delimiter). Header None si fichier absent."""
    if not os.path.exists(path):
        return None, [], ","
    with open(path, newline="") as g:
        sample = g.readline()
    delim = ";" if sample.count(";") > sample.count(",") else ","
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return header, rows, delim


def write_csv(path, header, rows, delim):
    if header is None:
        return
    backup_once(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter=delim,
                           extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            full = {k: r.get(k, "") for k in header}
            w.writerow(full)


def backup_once(path):
    if os.path.exists(path) and not os.path.exists(path + ".bak"):
        shutil.copy2(path, path + ".bak")


def load_autodetect_bbox(path):
    """Charge l'autodetect.
    Renvoie (by_track, by_frame, header, delim, yolo_rows, manual_track_ids).
      - by_track : track_id -> {frame: (foot_x, foot_y, bbox_height, user_type_yolo)}
      - by_frame : frame -> [(track_id, x1,y1,x2,y2, vtype_yolo), ...]
      - yolo_rows : toutes les lignes brutes hors manual (pour réécrire plus tard)
      - manual_track_ids : set des track_id avec user_type_yolo=='manual'
    """
    by_track = defaultdict(dict)
    by_frame = defaultdict(list)
    header = None
    delim  = ","
    yolo_rows = []
    manual_ids = set()
    if not os.path.exists(path):
        return by_track, by_frame, header, delim, yolo_rows, manual_ids
    with open(path, newline="") as g:
        sample = g.readline()
    delim = ";" if sample.count(";") > sample.count(",") else ","
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        header = list(reader.fieldnames or [])
        for row in reader:
            tid_raw = row.get("track_id", "") or ""
            tid = tid_raw.strip() if isinstance(tid_raw, str) else ""
            vtype_yolo = (row.get("user_type_yolo") or "").strip()
            is_manual = (vtype_yolo == "manual")
            if is_manual and tid:
                manual_ids.add(tid)
            else:
                yolo_rows.append(row)
            try:
                fr = int(float(row["frame"]))
                fx = float(row["foot_x"])
                fy = float(row["foot_y"])
                bh = float(row["bbox_height"])
            except (ValueError, KeyError, TypeError):
                continue
            by_track[tid][fr] = (fx, fy, bh, vtype_yolo)
            bw = bh * 0.45
            x1 = int(fx - bw / 2); y1 = int(fy - bh)
            x2 = int(fx + bw / 2); y2 = int(fy)
            by_frame[fr].append((tid, x1, y1, x2, y2, vtype_yolo))
    return by_track, by_frame, header, delim, yolo_rows, manual_ids


def load_obstacles(path):
    if not os.path.exists(path):
        return None, [], ","
    return load_csv_as_dicts(path)


# ── Geometric / lookup utils ─────────────────────────────────────────────────
def _bbox_from_foot(fx, fy, bh):
    bw = bh * 0.45
    return (int(round(fx - bw / 2)), int(round(fy - bh)),
            int(round(fx + bw / 2)), int(round(fy)))


def interpolate_bbox_for_track(tid, target_frame, auto_by_track):
    """Renvoie (x1, y1, x2, y2, kind) où kind ∈ {'exact','interp','extrapol'}.
    None si le track n'a aucune frame du tout."""
    frames = sorted(auto_by_track.get(tid, {}).keys())
    if not frames:
        return None
    fx_at = lambda f: auto_by_track[tid][f]   # (foot_x, foot_y, bh, vtype)
    if target_frame in auto_by_track[tid]:
        fx, fy, bh, _ = fx_at(target_frame)
        return (*_bbox_from_foot(fx, fy, bh), "exact")
    pos = bisect.bisect_right(frames, target_frame)
    left  = frames[pos - 1] if pos > 0 else None
    right = frames[pos]     if pos < len(frames) else None
    if left is None or right is None:
        # extrapolation : on prend la bbox du voisin le + proche, sans bouger
        anchor = right if left is None else left
        fx, fy, bh, _ = fx_at(anchor)
        return (*_bbox_from_foot(fx, fy, bh), "extrapol")
    fxL, fyL, bhL, _ = fx_at(left)
    fxR, fyR, bhR, _ = fx_at(right)
    t = (target_frame - left) / max(1, (right - left))
    fx = fxL + t * (fxR - fxL)
    fy = fyL + t * (fyR - fyL)
    bh = bhL + t * (bhR - bhL)
    return (*_bbox_from_foot(fx, fy, bh), "interp")


def draw_dashed_rect(frame, p1, p2, color, thickness=2, dash=8, gap=6):
    x1, y1 = p1; x2, y2 = p2
    for x in range(x1, x2, dash + gap):
        x_end = min(x + dash, x2)
        cv2.line(frame, (x, y1), (x_end, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y2), (x_end, y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash + gap):
        y_end = min(y + dash, y2)
        cv2.line(frame, (x1, y), (x1, y_end), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y), (x2, y_end), color, thickness, cv2.LINE_AA)


def short_event_id(eid):
    """`<rater>_<clip>_E0042` → `E0042`. Renvoie eid sans changement si pas de `_E`."""
    if not eid:
        return ""
    if "_E" in eid:
        tail = eid.rsplit("_E", 1)[-1]
        return f"E{tail}"
    return eid


def encounter_at_frame(encounters, frame):
    """Liste des encounters CONFIRMED qui couvrent cette frame, avec leur idx ds la liste."""
    out = []
    for i, enc in enumerate(encounters):
        if enc.get("CONFIRM", "").strip() != "1":
            continue
        try:
            f0 = int(enc["FRAME_START"]); f1 = int(enc["FRAME_END"])
        except (ValueError, KeyError):
            continue
        if f0 <= frame <= f1:
            out.append((i, enc))
    return out


def obstacle_at_frame(zones, frame):
    out = []
    for i, z in enumerate(zones):
        try:
            f0 = int(z["FRAME_START"]); f1 = int(z["FRAME_END"])
        except (ValueError, KeyError):
            continue
        if f0 <= frame <= f1:
            out.append((i, z))
    return out


def pick_track_at(x, y, frame, auto_by_frame):
    """Trouve la bbox la plus petite contenant (x,y). Renvoie track_id ou None."""
    best = None
    best_area = None
    for tid, x1, y1, x2, y2, _ in auto_by_frame.get(frame, []):
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            if best_area is None or area < best_area:
                best_area = area
                best = tid
    return best


# ── Rendu d'une frame ────────────────────────────────────────────────────────
def draw_overlay(frame, fr_idx, total, speed, encounters, zones,
                 auto_by_frame, auto_by_track,
                 mode, status, hud_lines, hint_extra=""):
    h, w = frame.shape[:2]

    # 1) bbox de tous les tracks autodetect (gris dim)
    for tid, x1, y1, x2, y2, _ in auto_by_frame.get(fr_idx, []):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)
        cv2.putText(frame, f"t{tid}", (x1 + 2, max(y1 - 4, 12)),
                    FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

    # 2) bbox des encounters confirmés à cette frame.
    # On NE colore QUE le PRIMARY_TRACK_ID — pas les LINKED_TRACKS, qui sont
    # auto-générés par detect_same_user_links() (gap 3s, 150px) et souvent
    # pollués par des tracks de piétons étrangers. Donc 1 encounter ↔ 1 bbox
    # colorée, conforme à la sémantique "1 piéton = 1 encounter".
    enc_here = encounter_at_frame(encounters, fr_idx)
    for n, (idx, enc) in enumerate(enc_here, start=1):
        tid = (enc.get("PRIMARY_TRACK_ID") or "").strip()
        if not tid:
            continue

        # 1° essai : bbox exacte
        bbox = None
        for ttid, x1, y1, x2, y2, _ in auto_by_frame.get(fr_idx, []):
            if ttid == tid:
                bbox = (x1, y1, x2, y2, "exact"); break
        # 2° essai : interpolation entre 2 voisins du même track
        if bbox is None:
            interp = interpolate_bbox_for_track(tid, fr_idx, auto_by_track)
            if interp is not None and interp[4] == "interp":
                bbox = interp

        vtype_lbl = label_of("VRU_TYPE", enc.get("VRU_TYPE"))
        color = COLORS.get(vtype_lbl.lower(), DEFAULT_COLOR)
        eid = short_event_id(enc.get("EVENT_ID", "")) or f"#{n}"

        if bbox:
            x1, y1, x2, y2, kind = bbox
            tag = "" if kind == "exact" else " (interp)"
            label = (f"{eid} {vtype_lbl} | "
                     f"{label_of('INTERACTION_TYPE', enc.get('INTERACTION_TYPE'))}{tag}")
            if kind == "exact":
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            else:
                draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness=2)
            cv2.rectangle(frame, (x1, max(y1 - 22, 0)),
                          (x1 + 8 + 7*len(label), y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, max(y1 - 6, 14)),
                        FONT, 0.5, BLACK, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{eid} {vtype_lbl} (no bbox)",
                        (20, 200 + 24*n), FONT, 0.6, color, 2, cv2.LINE_AA)

    # 3) bannière obstacle si zone active
    obs_here = obstacle_at_frame(zones, fr_idx)
    for j, (zi, z) in enumerate(obs_here):
        zid = z.get("ZONE_ID") or f"OBS{zi+1:03d}"
        draw_obstacle_banner(frame, zid)
        # un petit numéro à droite pour le delete
        cv2.putText(frame, f"obs#{j+1}", (w - 200, 100 + 24*j),
                    FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # 4) hud frame + speed
    draw_frame_number(frame, fr_idx, total)
    if speed is not None:
        draw_speed(frame, speed)

    # 5) bandeau mode + instructions
    bar_h = 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, f"[{mode}]  {status}", (20, h - bar_h + 28),
                FONT, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (20, h - bar_h + 56 + 22*i),
                    FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    if hint_extra:
        cv2.putText(frame, hint_extra, (20, h - 14),
                    FONT, 0.55, (200, 240, 200), 1, cv2.LINE_AA)


# ── Encounter / obstacle factories ───────────────────────────────────────────
def next_event_id(clip, rater, encounters):
    """Crée un EVENT_ID unique : <rater>_<clip>_E<NNNN>"""
    n = 1
    pat = f"_{clip}_E"
    used = []
    for e in encounters:
        eid = e.get("EVENT_ID", "")
        if pat in eid:
            try:
                used.append(int(eid.split("_E")[-1]))
            except ValueError:
                pass
    if used:
        n = max(used) + 1
    return f"{rater}_{clip}_E{n:04d}"


def make_encounter_row(header, clip, rater, track_id, f0, f1, codes, fps):
    """Construit une row complète, champs inconnus = ''."""
    row = {k: "" for k in header}
    row["EVENT_ID"]         = ""   # rempli par le caller via next_event_id
    row["TRIP_ID"]          = clip
    row["RATER_ID"]         = str(rater)
    row["FRAME_START"]      = str(f0)
    row["FRAME_END"]        = str(f1)
    row["FRAME_FIRST_DETECTION"] = str(f0)
    row["FRAME_LAST_VALID"]      = str(f1)
    row["DURATION_S"]       = f"{(f1 - f0 + 1) / max(fps, 1e-6):.3f}"
    row["PRIMARY_TRACK_ID"] = str(track_id)
    row["CONFIRM"]          = "1"
    for k, v in codes.items():
        row[k] = str(v)
    return row


def next_obstacle_id(zones):
    n = 1
    for z in zones:
        zid = (z.get("ZONE_ID") or "").strip()
        if zid.startswith("OBS"):
            try:
                n = max(n, int(zid[3:]) + 1)
            except ValueError:
                pass
    return f"OBS{n:03d}"


def make_obstacle_row(header, clip, f0, f1, fps):
    row = {k: "" for k in (header or
                           ["ZONE_ID", "TRIP_ID", "FRAME_START", "FRAME_END",
                            "TIME_START", "TIME_END", "TYPE"])}
    row["TRIP_ID"]     = clip
    row["FRAME_START"] = str(f0)
    row["FRAME_END"]   = str(f1)
    row["TIME_START"]  = f"{f0 / max(fps, 1e-6):.2f}"
    row["TIME_END"]    = f"{f1 / max(fps, 1e-6):.2f}"
    row["TYPE"]        = "obstacle"
    return row


# ── Session JSON sync ────────────────────────────────────────────────────────
def update_session_json(path, encounters, zones):
    if not os.path.exists(path):
        return
    backup_once(path)
    with open(path) as f:
        data = json.load(f)

    # Reconstruit la liste encounters (idx, primary_track, status, codes, ...)
    new_list = []
    for i, e in enumerate(encounters):
        codes = {}
        for k in ("CONFIRM", "VRU_TYPE", "INTERACTION_TYPE", "VRU_GAIT", "VRU_AGE_GROUP"):
            v = (e.get(k) or "").strip()
            if v != "":
                try:
                    codes[k] = int(v)
                except ValueError:
                    codes[k] = v
        new_list.append({
            "idx": i,
            "primary_track": e.get("PRIMARY_TRACK_ID", ""),
            "status": "coded" if codes.get("CONFIRM") == 1 else "skipped",
            "codes": codes,
            "notes": e.get("NOTES", ""),
            "note_timestamps": [],
            "coding_start_ts": e.get("CODING_START_TS", ""),
            "coding_end_ts":   e.get("CODING_END_TS",   ""),
        })
    data["encounters"]          = new_list
    data["clip_obstacle_zones"] = zones
    data["timestamp"]           = datetime.datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Application principale ───────────────────────────────────────────────────
class Corrector:
    def __init__(self, clip, rater):
        self.clip = clip
        self.rater = rater
        self.paths = derive_paths(clip, rater)

        # Vidéo
        self.cap = cv2.VideoCapture(self.paths["video"])
        if not self.cap.isOpened():
            raise SystemExit(f"Vidéo illisible: {self.paths['video']}")
        self.fps   = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w     = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h     = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Données
        self.enc_header, self.encounters, self.enc_delim = load_csv_as_dicts(self.paths["enc"])
        self.obs_header, self.zones,      self.obs_delim = load_obstacles(self.paths["obs"])
        if self.obs_header is None:
            self.obs_header = ["ZONE_ID", "TRIP_ID", "FRAME_START", "FRAME_END",
                               "TIME_START", "TIME_END", "TYPE"]
            self.obs_delim = ","
        (self.auto_by_track, self.auto_by_frame,
         self.auto_header, self.auto_delim,
         self.auto_yolo_rows, auto_manual_ids) = load_autodetect_bbox(self.paths["auto"])
        self.speed_by_frame = load_speed_by_frame(self.paths["speed"])

        # État UI
        self.frame_idx = 0
        self.cur_frame = None
        self.playing = False
        self.dirty = False
        self.mode = M_BROWSE

        # Buffers d'édition
        self.add_track = None
        self.add_f0 = None
        self.add_f1 = None
        self.add_codes = {}
        self.add_step = 0          # 0..3 dans CODE_STEPS
        self.obs_f0 = None

        # Cycle "del" — liste figée des indices au moment où on entre en mode delete
        self.del_targets = []

        # Édition en place
        self.edit_target_idx = None
        self.edit_codes = {}
        self.edit_step  = 0

        # Merge tracks
        self.merge_keep_tid = None

        # Manual track buffers
        # IDs détectés à l'init (user_type_yolo == 'manual' dans l'autodetect)
        self.manual_track_ids = set(auto_manual_ids)
        self.manual_keyframes = []          # [(frame, (x1,y1,x2,y2))] pour le track en cours
        self.manual_track_id = None         # id du track en cours de saisie
        self.drag_start = None
        self.drag_current = None

        # Mouse
        self.last_click = None  # (x, y) en coord image

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, min(1600, self.w), min(900, self.h))
        cv2.setMouseCallback(WIN, self._on_mouse)

        self._read_frame_at(0)

    # — autodetect (avec manual rows mergés) I/O —
    def _autodet_row_for_manual(self, tid, fr, fx, fy, bh):
        """Construit une ligne au format autodetect pour un track manuel."""
        row = {k: "" for k in (self.auto_header or [])}
        row["track_id"]       = tid
        row["frame"]          = str(fr)
        row["foot_x"]         = f"{fx:.1f}"
        row["foot_y"]         = f"{fy:.1f}"
        row["bbox_height"]    = f"{bh:.1f}"
        row["user_type_yolo"] = "manual"
        if "time_s" in row:           row["time_s"] = f"{fr / max(self.fps, 1e-6):.4f}"
        if "is_occluded" in row:      row["is_occluded"] = "False"
        if "is_interpolated" in row:  row["is_interpolated"] = "False"
        if "encounter_status" in row: row["encounter_status"] = "manual"
        return row

    def _save_autodetect(self):
        """Réécrit l'autodetect : lignes YOLO d'origine + lignes manual fraîches."""
        if not self.auto_header:
            return  # pas d'autodetect → rien à écrire
        # Si rien n'a changé côté manual et le fichier ne contenait pas de manual,
        # on évite l'écriture.
        if not self.manual_track_ids and not any(
                (r.get("user_type_yolo") or "").strip() == "manual"
                for r in self.auto_yolo_rows):
            return
        backup_once(self.paths["auto"])
        with open(self.paths["auto"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.auto_header,
                               delimiter=self.auto_delim or ",",
                               extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            for row in self.auto_yolo_rows:
                w.writerow({k: row.get(k, "") for k in self.auto_header})
            for tid in sorted(self.manual_track_ids):
                for fr in sorted(self.auto_by_track.get(tid, {}).keys()):
                    fx, fy, bh, _ = self.auto_by_track[tid][fr]
                    w.writerow(self._autodet_row_for_manual(tid, fr, fx, fy, bh))

    def _next_manual_track_id(self):
        n = 1
        while True:
            tid = f"M{n:03d}"
            if tid not in self.manual_track_ids and tid not in self.auto_by_track:
                return tid
            n += 1

    # — frame management —
    def _read_frame_at(self, idx):
        idx = max(0, min(self.total - 1, idx))
        if idx != self.frame_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if not ok:
            return
        self.frame_idx = idx
        self.cur_frame = fr

    def _step(self, delta):
        self._read_frame_at(self.frame_idx + delta)

    # — mouse —
    def _on_mouse(self, ev, x, y, flags, _):
        if self.mode == M_MANUAL:
            if ev == cv2.EVENT_LBUTTONDOWN:
                self.drag_start = (x, y)
                self.drag_current = (x, y)
            elif ev == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
                self.drag_current = (x, y)
            elif ev == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
                x0, y0 = self.drag_start
                x1, x2 = sorted((x0, x))
                y1, y2 = sorted((y0, y))
                self.drag_start = None
                self.drag_current = None
                if x2 - x1 >= 5 and y2 - y1 >= 5:
                    self.manual_keyframes = [(f, b) for f, b in self.manual_keyframes
                                             if f != self.frame_idx]
                    self.manual_keyframes.append((self.frame_idx, (x1, y1, x2, y2)))
                    self.manual_keyframes.sort(key=lambda kf: kf[0])
                    print(f"  + keyframe @ {self.frame_idx}: ({x1},{y1})-({x2},{y2}) "
                          f"[{len(self.manual_keyframes)} keyframes]")
            elif ev == cv2.EVENT_RBUTTONDOWN:
                before = len(self.manual_keyframes)
                self.manual_keyframes = [(f, b) for f, b in self.manual_keyframes
                                         if f != self.frame_idx]
                if len(self.manual_keyframes) < before:
                    print(f"  - keyframe @ {self.frame_idx} retirée")
            return

        if ev == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)

    # — modes / actions —
    def _enter_mode(self, mode):
        self.mode = mode
        self.last_click = None

    def _reset_add(self):
        self.add_track = None
        self.add_f0 = None
        self.add_f1 = None
        self.add_codes = {}
        self.add_step = 0

    def _apply_add(self):
        codes = self.add_codes
        new = make_encounter_row(self.enc_header, self.clip, self.rater,
                                 self.add_track, self.add_f0, self.add_f1, codes, self.fps)
        new["EVENT_ID"]         = next_event_id(self.clip, self.rater, self.encounters)
        new["CODING_START_TS"]  = datetime.datetime.now().isoformat()
        new["CODING_END_TS"]    = new["CODING_START_TS"]
        self.encounters.append(new)
        self.dirty = True
        print(f"[+] encounter ajouté: track={self.add_track} frames={self.add_f0}-{self.add_f1} "
              f"VRU={codes.get('VRU_TYPE')} INT={codes.get('INTERACTION_TYPE')} "
              f"GAIT={codes.get('VRU_GAIT')} AGE={codes.get('VRU_AGE_GROUP')}")
        self._reset_add()

    def _apply_delete_enc(self, n):
        """n = 1-based index dans self.del_targets (snapshot de la frame courante)."""
        if not (1 <= n <= len(self.del_targets)):
            return
        global_idx = self.del_targets[n - 1]
        removed = self.encounters.pop(global_idx)
        self.dirty = True
        print(f"[-] encounter retiré: {removed.get('EVENT_ID', '?')} "
              f"track={removed.get('PRIMARY_TRACK_ID')} "
              f"frames={removed.get('FRAME_START')}-{removed.get('FRAME_END')}")

    def _consolidate_encounters_for_track(self, tid):
        """Fusionne les encounters confirmés sur `tid` ayant strictement
        les mêmes (VRU_TYPE, INTERACTION_TYPE, VRU_GAIT, VRU_AGE_GROUP) en
        un seul encounter couvrant l'union des frame_ranges.

        Les codes différents → encounters laissés séparés (le coder a
        explicitement codé différemment, on ne fusionne pas en silence)."""
        code_keys = ("VRU_TYPE", "INTERACTION_TYPE", "VRU_GAIT", "VRU_AGE_GROUP")
        groups = {}                # codes_tuple -> [indices dans self.encounters]
        for i, e in enumerate(self.encounters):
            if (e.get("CONFIRM") or "").strip() != "1":
                continue
            if (e.get("PRIMARY_TRACK_ID") or "").strip() != tid:
                continue
            codes = tuple((e.get(k) or "").strip() for k in code_keys)
            groups.setdefault(codes, []).append(i)

        to_remove = []
        n_merged = 0
        for codes, idxs in groups.items():
            if len(idxs) < 2:
                continue
            keeper = self.encounters[idxs[0]]
            try:
                f0 = int(keeper["FRAME_START"]); f1 = int(keeper["FRAME_END"])
            except (KeyError, ValueError):
                continue
            keeper_eid = keeper.get("EVENT_ID", "?")
            absorbed = []
            for i in idxs[1:]:
                other = self.encounters[i]
                try:
                    of0 = int(other["FRAME_START"]); of1 = int(other["FRAME_END"])
                except (KeyError, ValueError):
                    continue
                f0 = min(f0, of0)
                f1 = max(f1, of1)
                absorbed.append(other.get("EVENT_ID", "?"))
                to_remove.append(i)
                n_merged += 1
            keeper["FRAME_START"] = str(f0)
            keeper["FRAME_END"]   = str(f1)
            keeper["CODING_END_TS"] = datetime.datetime.now().isoformat()
            print(f"[merge] consolide encounter {keeper_eid}: "
                  f"absorbe {absorbed} → range {f0}-{f1}")

        for i in sorted(to_remove, reverse=True):
            self.encounters.pop(i)
        return n_merged

    def _apply_merge(self, keep_tid, absorb_tid):
        """absorb_tid → keep_tid partout. En cas de collision sur une frame
        (les 2 ont une bbox au même frame), on garde celle de keep_tid."""
        if not keep_tid or not absorb_tid or keep_tid == absorb_tid:
            print("[!] track_id invalide ou identique"); return False

        # diagnostics avant
        keep_frames_before   = sorted(self.auto_by_track.get(keep_tid, {}).keys())
        absorb_frames_before = sorted(self.auto_by_track.get(absorb_tid, {}).keys())
        print(f"[merge] avant : keep={keep_tid} a {len(keep_frames_before)} frames"
              f"  ({keep_frames_before[0] if keep_frames_before else '-'}.."
              f"{keep_frames_before[-1] if keep_frames_before else '-'}) ; "
              f"absorb={absorb_tid} a {len(absorb_frames_before)} frames"
              f"  ({absorb_frames_before[0] if absorb_frames_before else '-'}.."
              f"{absorb_frames_before[-1] if absorb_frames_before else '-'})")
        if not absorb_frames_before:
            print(f"[merge] ⚠ absorb_tid={absorb_tid!r} introuvable dans auto_by_track. "
                  f"Tracks dispos (10 premiers): "
                  f"{sorted(self.auto_by_track.keys())[:10]}")

        # 1) auto_yolo_rows : remap track_id, drop si collision (frame)
        dst_frames_in_yolo = set()
        for r in self.auto_yolo_rows:
            if (r.get("track_id") or "").strip() == keep_tid:
                try:
                    dst_frames_in_yolo.add(int(float(r.get("frame", ""))))
                except (ValueError, TypeError):
                    pass
        new_rows = []
        n_remapped = n_dropped = 0
        for r in self.auto_yolo_rows:
            tid = (r.get("track_id") or "").strip()
            if tid == absorb_tid:
                try:
                    fr = int(float(r.get("frame", "")))
                except (ValueError, TypeError):
                    fr = None
                if fr is not None and fr in dst_frames_in_yolo:
                    n_dropped += 1
                    continue
                r["track_id"] = keep_tid
                n_remapped += 1
            new_rows.append(r)
        self.auto_yolo_rows = new_rows

        # 2) auto_by_track : merge dict, keep_tid prioritaire en cas de collision
        src = self.auto_by_track.get(absorb_tid, {})
        dst = self.auto_by_track.get(keep_tid, {})
        n_added = 0
        for fr, val in src.items():
            if fr not in dst:
                dst[fr] = val
                n_added += 1
        self.auto_by_track[keep_tid] = dst
        self.auto_by_track.pop(absorb_tid, None)

        # 3) auto_by_frame : remap, dedupe (keep_tid prioritaire)
        for fr in list(self.auto_by_frame.keys()):
            has_keep = any(t[0] == keep_tid for t in self.auto_by_frame[fr])
            new_list = []
            for tup in self.auto_by_frame[fr]:
                if tup[0] == absorb_tid:
                    if has_keep:
                        continue
                    new_list.append((keep_tid,) + tup[1:])
                    has_keep = True
                else:
                    new_list.append(tup)
            self.auto_by_frame[fr] = new_list

        # 4) Manual track ids : si l'absorbé est manuel, on le bascule
        if absorb_tid in self.manual_track_ids:
            self.manual_track_ids.discard(absorb_tid)
            self.manual_track_ids.add(keep_tid)

        # 5) encounters : PRIMARY_TRACK_ID + LINKED_TRACKS
        n_pt = n_lk = 0
        for e in self.encounters:
            if (e.get("PRIMARY_TRACK_ID") or "").strip() == absorb_tid:
                e["PRIMARY_TRACK_ID"] = keep_tid
                n_pt += 1
            lk = (e.get("LINKED_TRACKS") or "").strip()
            if lk:
                tokens = [t.strip() for t in lk.split(",") if t.strip()]
                if absorb_tid in tokens or keep_tid in tokens:
                    new_tokens = [keep_tid if t == absorb_tid else t for t in tokens]
                    # dedupe en gardant l'ordre, et retire l'auto-référence
                    seen = set(); deduped = []
                    pt_here = (e.get("PRIMARY_TRACK_ID") or "").strip()
                    for t in new_tokens:
                        if t in seen or t == pt_here:
                            continue
                        seen.add(t); deduped.append(t)
                    e["LINKED_TRACKS"] = ",".join(deduped)
                    n_lk += 1

        # 6) consolidation des encounters désormais sur le même track
        n_consolidated = self._consolidate_encounters_for_track(keep_tid)
        if n_consolidated:
            print(f"[merge] {n_consolidated} encounter(s) fusionné(s) "
                  f"sur le track {keep_tid}")

        self.dirty = True

        # diagnostics après
        keep_frames_after = sorted(self.auto_by_track.get(keep_tid, {}).keys())
        n_in_byframe = sum(1 for fr in self.auto_by_frame
                           for tup in self.auto_by_frame[fr] if tup[0] == keep_tid)
        leftover = sum(1 for fr in self.auto_by_frame
                       for tup in self.auto_by_frame[fr] if tup[0] == absorb_tid)
        print(f"[~] merge {absorb_tid} → {keep_tid} : "
              f"autodet {n_remapped} rows remappés ({n_dropped} en collision droppés), "
              f"{n_added} frames ajoutées, "
              f"encounters PT={n_pt} LINKED_TRACKS={n_lk}")
        print(f"[merge] après : keep={keep_tid} a {len(keep_frames_after)} frames"
              f"  ({keep_frames_after[0] if keep_frames_after else '-'}.."
              f"{keep_frames_after[-1] if keep_frames_after else '-'})  "
              f"présent dans auto_by_frame sur {n_in_byframe} occurrences ; "
              f"reste-t-il du absorbé : {leftover}")
        if leftover:
            print(f"[merge] ⚠ {leftover} occurrences de {absorb_tid} non remappées — bug ?")
        return True

    def _apply_unlink(self, n):
        if not (1 <= n <= len(self.del_targets)):
            return
        gi = self.del_targets[n - 1]
        e = self.encounters[gi]
        old = (e.get("LINKED_TRACKS") or "").strip()
        if not old:
            print(f"[=] {e.get('EVENT_ID', '?')} : aucun LINKED_TRACKS"); return
        e["LINKED_TRACKS"] = ""
        e["CODING_END_TS"] = datetime.datetime.now().isoformat()
        self.dirty = True
        print(f"[~] {e.get('EVENT_ID', '?')} : LINKED_TRACKS={old!r} → vidé")

    def _apply_edit(self):
        """Applique edit_codes au encounter ciblé."""
        if self.edit_target_idx is None:
            return
        e = self.encounters[self.edit_target_idx]
        changes = []
        for k, v in self.edit_codes.items():
            old = (e.get(k) or "").strip()
            new = str(v)
            if old != new:
                changes.append(f"{k}:{old}->{new}")
                e[k] = new
        e["CODING_END_TS"] = datetime.datetime.now().isoformat()
        self.dirty = True
        if changes:
            print(f"[~] enc {e.get('EVENT_ID', '?')} édité — " + " ".join(changes))
        else:
            print(f"[=] enc {e.get('EVENT_ID', '?')} : aucune modif")
        self.edit_target_idx = None
        self.edit_codes = {}
        self.edit_step = 0

    def _apply_delete_obs(self, n):
        if not (1 <= n <= len(self.del_targets)):
            return
        gi = self.del_targets[n - 1]
        rm = self.zones.pop(gi)
        self.dirty = True
        print(f"[-] obstacle retiré: {rm.get('ZONE_ID')} "
              f"frames={rm.get('FRAME_START')}-{rm.get('FRAME_END')}")

    def _finalize_manual(self):
        """Interpole linéairement entre keyframes et crée le track manuel."""
        if not self.manual_keyframes:
            print("[!] aucune keyframe — annule")
            self._reset_manual()
            self._enter_mode(M_BROWSE)
            return False
        kfs = sorted(self.manual_keyframes)
        tid = self._next_manual_track_id()
        self.manual_track_id = tid
        self.manual_track_ids.add(tid)

        f_min = kfs[0][0]
        f_max = kfs[-1][0]

        # Pour chaque frame [f_min..f_max], interpole entre les 2 KFs encadrantes
        for fr in range(f_min, f_max + 1):
            # find left/right keyframe
            left = right = None
            for k in kfs:
                if k[0] <= fr:
                    left = k
                if k[0] >= fr and right is None:
                    right = k
            if left is None: left = right
            if right is None: right = left
            if left[0] == right[0]:
                bb = left[1]
            else:
                t = (fr - left[0]) / (right[0] - left[0])
                bb = tuple(int(round(left[1][i] + t * (right[1][i] - left[1][i])))
                           for i in range(4))
            x1, y1, x2, y2 = bb
            fx = (x1 + x2) / 2.0
            fy = float(y2)
            bh = float(y2 - y1)
            # bbox d'affichage normalisée (aspect 0.45, comme YOLO via foot/bh)
            # → ce que tu vois après finalize = ce qui sera réaffiché après reload.
            bw_disp = bh * 0.45
            x1d = int(round(fx - bw_disp / 2))
            x2d = int(round(fx + bw_disp / 2))
            self.auto_by_track[tid][fr] = (fx, fy, bh, "manual")
            self.auto_by_frame[fr].append((tid, x1d, y1, x2d, y2, "manual"))

        n_kf = len(kfs)
        n_frames = f_max - f_min + 1
        print(f"  → manual track {tid} créé : {n_kf} keyframes, "
              f"{n_frames} frames interpolées [{f_min}..{f_max}]")

        # Pré-remplit l'encounter pour ce track et passe à la saisie codes
        self.add_track = tid
        self.add_f0 = f_min
        self.add_f1 = f_max
        self.add_step = 0
        self.add_codes = {}
        self._reset_manual()
        self.dirty = True
        self._enter_mode(M_ADD_CODE)
        return True

    def _reset_manual(self):
        self.manual_keyframes = []
        self.manual_track_id = None
        self.drag_start = None
        self.drag_current = None

    def _apply_obs(self):
        zid = next_obstacle_id(self.zones)
        row = make_obstacle_row(self.obs_header, self.clip,
                                self.obs_f0, self.frame_idx, self.fps)
        row["ZONE_ID"] = zid
        self.zones.append(row)
        self.dirty = True
        print(f"[+] obstacle ajouté {zid}: frames={self.obs_f0}-{self.frame_idx}")
        self.obs_f0 = None

    def _trim_encounters_to_tracks(self):
        """Resserre chaque encounter à [min(track_frames), max(track_frames)]
        de son PRIMARY_TRACK_ID. Évite les bords 'no bbox' / extrapolés."""
        n = 0
        for e in self.encounters:
            tid = (e.get("PRIMARY_TRACK_ID") or "").strip()
            if not tid:
                continue
            frames = self.auto_by_track.get(tid)
            if not frames:
                continue
            try:
                f0 = int(e["FRAME_START"]); f1 = int(e["FRAME_END"])
            except (KeyError, ValueError):
                continue
            tmin, tmax = min(frames), max(frames)
            new_f0 = max(f0, tmin)
            new_f1 = min(f1, tmax)
            if new_f0 > new_f1:
                continue   # cas improbable : on laisse l'encounter en l'état
            if new_f0 != f0 or new_f1 != f1:
                e["FRAME_START"] = str(new_f0)
                e["FRAME_END"]   = str(new_f1)
                print(f"  trim {e.get('EVENT_ID', '?')}: "
                      f"{f0}-{f1} → {new_f0}-{new_f1}")
                n += 1
        return n

    # — saving —
    def save(self):
        n_trim = self._trim_encounters_to_tracks()
        write_csv(self.paths["enc"], self.enc_header, self.encounters, self.enc_delim)
        write_csv(self.paths["obs"], self.obs_header, self.zones, self.obs_delim)
        self._save_autodetect()
        update_session_json(self.paths["session"], self.encounters, self.zones)
        self.dirty = False
        bits = [os.path.basename(self.paths['enc']),
                os.path.basename(self.paths['obs'])]
        if self.manual_track_ids:
            bits.append(os.path.basename(self.paths['auto']) + " (+manual)")
        if os.path.exists(self.paths["session"]):
            bits.append(os.path.basename(self.paths['session']))
        print(f"[OK] sauvegardé : {', '.join(bits)}"
              + (f"  ({n_trim} encounter(s) trim)" if n_trim else ""))

    # — manual rendering helper —
    def _draw_manual_overlay(self, frame):
        if self.mode != M_MANUAL:
            return
        # rubber-band en cours de drag
        if self.drag_start is not None and self.drag_current is not None:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_current
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)

        # toutes les keyframes posées : jaune épais pour la frame courante,
        # jaune dim pour les autres, et indication de la distance en frames.
        for fr, (x1, y1, x2, y2) in self.manual_keyframes:
            color = (0, 255, 255) if fr == self.frame_idx else (40, 180, 200)
            thick = 2 if fr == self.frame_idx else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
            d = fr - self.frame_idx
            tag = f"KF f={fr}" + ("" if d == 0 else f" ({d:+d})")
            cv2.putText(frame, tag, (x1 + 3, max(y1 - 5, 12)),
                        FONT, 0.45, color, 1, cv2.LINE_AA)

        # interpolation provisoire (entre 2 KFs encadrantes) à la frame courante
        kfs = sorted(self.manual_keyframes)
        if len(kfs) >= 1 and not any(f == self.frame_idx for f, _ in kfs):
            left = right = None
            for k in kfs:
                if k[0] <= self.frame_idx: left = k
                if k[0] >= self.frame_idx and right is None: right = k
            if left is not None and right is not None:
                if left[0] == right[0]:
                    bb = left[1]
                else:
                    t = (self.frame_idx - left[0]) / (right[0] - left[0])
                    bb = tuple(int(round(left[1][i] + t * (right[1][i] - left[1][i])))
                               for i in range(4))
                x1, y1, x2, y2 = bb
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 255, 180), 1, cv2.LINE_AA)
                cv2.putText(frame, "interp", (x1 + 3, max(y1 - 5, 12)),
                            FONT, 0.4, (180, 255, 180), 1, cv2.LINE_AA)

    # — main loop —
    NAV_MODES = (M_BROWSE, M_ADD_PICK, M_ADD_F0, M_ADD_F1,
                 M_OBS_F0, M_OBS_F1, M_MANUAL, M_MERGE1, M_MERGE2)

    def run(self):
        print(f"=== correct_encounters: {self.clip}  rater{self.rater} "
              f"({self.total}f @ {self.fps:.1f}fps) ===")
        print(f"  encounters chargés: {len(self.encounters)}  "
              f"obstacles: {len(self.zones)}  bbox tracks: {len(self.auto_by_track)}")

        while True:
            # advance for play
            if self.playing and self.mode in self.NAV_MODES:
                self._step(1)

            if self.cur_frame is None:
                self._read_frame_at(self.frame_idx)
                if self.cur_frame is None:
                    print("[!] frame illisible, exit")
                    break
            disp = self.cur_frame.copy()

            status, hud, hint = self._compose_status()
            speed = self.speed_by_frame.get(self.frame_idx)
            draw_overlay(disp, self.frame_idx, self.total, speed,
                         self.encounters, self.zones,
                         self.auto_by_frame, self.auto_by_track,
                         self.mode, status, hud, hint)
            self._draw_manual_overlay(disp)

            cv2.imshow(WIN, disp)
            key = cv2.waitKeyEx(20 if self.playing and self.mode in self.NAV_MODES else 0)

            # — clic en attente (MERGE1 / MERGE2) —
            if self.mode == M_MERGE1 and self.last_click is not None:
                tid = pick_track_at(self.last_click[0], self.last_click[1],
                                    self.frame_idx, self.auto_by_frame)
                self.last_click = None
                if tid is None:
                    print("[!] aucun track sous le clic — ré-essaye")
                else:
                    self.merge_keep_tid = tid
                    self.mode = M_MERGE2
                    print(f"  → 1er track (KEEP) = {tid}. "
                          "Navigue jusqu'au 2e track puis clique-le.")
                continue

            if self.mode == M_MERGE2 and self.last_click is not None:
                tid = pick_track_at(self.last_click[0], self.last_click[1],
                                    self.frame_idx, self.auto_by_frame)
                self.last_click = None
                if tid is None:
                    print("[!] aucun track sous le clic — ré-essaye")
                elif tid == self.merge_keep_tid:
                    print(f"[!] {tid} = même track, pas de merge")
                else:
                    self._apply_merge(self.merge_keep_tid, tid)
                    self.merge_keep_tid = None
                    self._enter_mode(M_BROWSE)
                continue

            # — clic en attente (ADD_PICK) —
            if self.mode == M_ADD_PICK and self.last_click is not None:
                tid = pick_track_at(self.last_click[0], self.last_click[1],
                                    self.frame_idx, self.auto_by_frame)
                self.last_click = None
                if tid is None:
                    print("[!] aucun track sous le clic — ré-essaye")
                else:
                    track_frames = sorted(self.auto_by_track.get(tid, {}).keys())
                    if not track_frames:
                        print(f"[!] track {tid} sans frames dans autodetect")
                    else:
                        self.add_track = tid
                        self.add_f0 = track_frames[0]
                        self.add_f1 = track_frames[-1]
                        self.add_step = 0
                        self.mode = M_ADD_CODE
                        print(f"  → track {tid} sélectionné, "
                              f"range complet {self.add_f0}-{self.add_f1} "
                              f"({len(track_frames)} frames). Saisie des codes…")
                continue

            if key == -1:
                continue
            if not self._handle_key(key):
                break

        cv2.destroyAllWindows()
        self.cap.release()

    def _compose_status(self):
        """Renvoie (status_text, [hud_lines], hint_extra) selon le mode."""
        common = ("  A=add  D=del  E=edit  J=join  M=manual  U=unlink  "
                  "O=obstacle  X=del-obs  S=save  Q=quit  ESC=cancel")
        if self.mode == M_BROWSE:
            enc_here = encounter_at_frame(self.encounters, self.frame_idx)
            obs_here = obstacle_at_frame(self.zones,    self.frame_idx)
            lines = [f"encs ici: {len(enc_here)}   obstacles ici: {len(obs_here)}",
                     "navig: , .  ←→ ±10   Shift+←/→ ±100   SPACE play   G goto"]
            return ("BROWSE" + ("*" if self.dirty else ""), lines, common)

        if self.mode == M_ADD_PICK:
            return ("ADD_ENC", [
                "Clique une bbox (gris). Tout le track sera ajouté → saisie codes.",
                "SPACE play/pause, , . nav, ESC annule."
            ], "")
        if self.mode == M_ADD_F0:
            return ("ADD_ENC f0", [
                f"track={self.add_track}  start sera = frame courante",
                "Navigue à la frame de DÉBUT puis ENTER. ESC pour annuler."
            ], f"start provisoire: {self.add_f0}")
        if self.mode == M_ADD_F1:
            return ("ADD_ENC f1", [
                f"track={self.add_track}  start={self.add_f0}",
                "Navigue à la frame de FIN puis ENTER. ESC pour annuler."
            ], "")
        if self.mode == M_ADD_CODE:
            field, prompt = CODE_STEPS[self.add_step]
            done = "  ".join(f"{k}={v}" for k, v in self.add_codes.items())
            return ("ADD_ENC code", [
                f"{field}: {prompt}",
                f"déjà saisi: {done}"
            ], "ENTER pour passer (= 9 inconnu)  BACKSPACE retour")

        if self.mode == M_DEL:
            lines = ["Encounters confirmés à cette frame :"]
            for n, gi in enumerate(self.del_targets, start=1):
                e = self.encounters[gi]
                lines.append(f"  {n}: {e.get('EVENT_ID','')}  "
                             f"PT={e.get('PRIMARY_TRACK_ID')}  "
                             f"f={e.get('FRAME_START')}-{e.get('FRAME_END')}  "
                             f"VRU={label_of('VRU_TYPE', e.get('VRU_TYPE'))}")
            return ("DELETE", lines, "Tape 1-9 pour supprimer, ESC pour annuler")
        if self.mode == M_EDIT_PICK:
            lines = ["Encounters confirmés à cette frame :"]
            for n, gi in enumerate(self.del_targets, start=1):
                e = self.encounters[gi]
                lines.append(f"  {n}: {e.get('EVENT_ID','')}  "
                             f"PT={e.get('PRIMARY_TRACK_ID')}  "
                             f"VRU={label_of('VRU_TYPE', e.get('VRU_TYPE'))}  "
                             f"INT={label_of('INTERACTION_TYPE', e.get('INTERACTION_TYPE'))}  "
                             f"GAIT={label_of('VRU_GAIT', e.get('VRU_GAIT'))}  "
                             f"AGE={label_of('VRU_AGE_GROUP', e.get('VRU_AGE_GROUP'))}")
            return ("EDIT", lines, "Tape 1-9 pour éditer, ESC pour annuler")
        if self.mode == M_EDIT_CODE:
            field, prompt = CODE_STEPS[self.edit_step]
            e = self.encounters[self.edit_target_idx]
            cur = (e.get(field) or "").strip()
            cur_lbl = label_of(field, cur) or "?"
            new = self.edit_codes.get(field)
            new_lbl = label_of(field, str(new)) if new is not None else "(garde)"
            return ("EDIT codes", [
                f"{field}: actuel = {cur_lbl} ({cur or '?'})  →  nouveau = {new_lbl}",
                prompt,
            ], "ENTER passe (garde l'actuel)  1-9 change  BACKSPACE retour  ESC annule")
        if self.mode == M_UNLINK:
            lines = ["Encounters avec LINKED_TRACKS à cette frame :"]
            for n, gi in enumerate(self.del_targets, start=1):
                e = self.encounters[gi]
                lk = (e.get("LINKED_TRACKS") or "").strip()
                lines.append(f"  {n}: {e.get('EVENT_ID','')}  "
                             f"PT={e.get('PRIMARY_TRACK_ID')}  "
                             f"LINKED={lk}")
            return ("UNLINK", lines, "Tape 1-9 pour vider LINKED_TRACKS, ESC pour annuler")
        if self.mode == M_OBS_F0:
            return ("OBSTACLE start", [
                "Navigue à la frame de DÉBUT obstacle puis ENTER (ou O encore).",
                "ESC pour annuler."
            ], "")
        if self.mode == M_OBS_F1:
            return ("OBSTACLE end", [
                f"start = {self.obs_f0}",
                "Navigue à la frame de FIN puis ENTER (ou O encore). ESC pour annuler."
            ], "")
        if self.mode == M_OBS_DEL:
            lines = ["Zones obstacle à cette frame :"]
            for n, gi in enumerate(self.del_targets, start=1):
                z = self.zones[gi]
                lines.append(f"  {n}: {z.get('ZONE_ID','')}  "
                             f"f={z.get('FRAME_START')}-{z.get('FRAME_END')}")
            return ("DELETE OBS", lines, "Tape 1-9 pour supprimer, ESC pour annuler")
        if self.mode == M_MANUAL:
            kfs = sorted(f for f, _ in self.manual_keyframes)
            kf_str = (", ".join(map(str, kfs[:8])) +
                      ("…" if len(kfs) > 8 else "")) if kfs else "(aucune)"
            return ("MANUAL TRACK", [
                f"keyframes posées : {len(kfs)}  → frames {kf_str}",
                "drag G = nouvelle KF ici (remplace si déjà), drag D = supprime",
                "navigue puis drag à nouveau pour interpoler. ENTER = finalise."
            ], "ESC pour annuler")
        if self.mode == M_MERGE1:
            return ("MERGE 1/2", [
                "Clique le track qui RESTE (avant le break).",
                "Navigation OK. ESC pour annuler."
            ], "")
        if self.mode == M_MERGE2:
            return ("MERGE 2/2", [
                f"Track conservé : {self.merge_keep_tid}",
                "Navigue jusqu'à après le break, clique le track à ABSORBER.",
                "ESC pour annuler."
            ], "")
        return (self.mode, [], "")

    def _handle_key(self, key):
        """Renvoie False si on doit quitter."""
        # Mac OpenCV: arrows = 81/82/83/84 ou 2/3/0/1 selon plateforme. On gère le large set.
        LEFT, RIGHT  = (81, 2),   (83, 3)
        SHIFT_LEFT, SHIFT_RIGHT = (391,), (393,)
        ENTER  = (13, 10)
        ESC    = 27
        SPACE  = 32
        BACK   = (8, 127)

        # ── codes input mode ─────────────────────────────────────────────────
        if self.mode == M_ADD_CODE:
            if key in ENTER:
                # value 9 (unknown) si non rempli
                field, _ = CODE_STEPS[self.add_step]
                if field not in self.add_codes:
                    self.add_codes[field] = 9
                self.add_step += 1
                if self.add_step >= len(CODE_STEPS):
                    self._apply_add()
                    self._enter_mode(M_BROWSE)
                return True
            if key in BACK and self.add_step > 0:
                self.add_step -= 1
                f, _ = CODE_STEPS[self.add_step]
                self.add_codes.pop(f, None)
                return True
            if key == ESC:
                self._reset_add()
                self._enter_mode(M_BROWSE)
                return True
            if 48 <= key <= 57:           # 0-9
                v = key - 48
                if v == 0:
                    return True
                field, _ = CODE_STEPS[self.add_step]
                self.add_codes[field] = v
            return True

        # ── delete pick ──────────────────────────────────────────────────────
        if self.mode in (M_DEL, M_OBS_DEL):
            if key == ESC:
                self._enter_mode(M_BROWSE); return True
            if 49 <= key <= 57:
                n = key - 48
                if self.mode == M_DEL:
                    self._apply_delete_enc(n)
                else:
                    self._apply_delete_obs(n)
                self._enter_mode(M_BROWSE)
            return True

        # ── unlink pick ──────────────────────────────────────────────────────
        if self.mode == M_UNLINK:
            if key == ESC:
                self._enter_mode(M_BROWSE); return True
            if 49 <= key <= 57:
                self._apply_unlink(key - 48)
                self._enter_mode(M_BROWSE)
            return True

        # ── edit : pick d'abord, puis codes ──────────────────────────────────
        if self.mode == M_EDIT_PICK:
            if key == ESC:
                self._enter_mode(M_BROWSE); return True
            if 49 <= key <= 57:
                n = key - 48
                if 1 <= n <= len(self.del_targets):
                    self.edit_target_idx = self.del_targets[n - 1]
                    self.edit_codes = {}
                    self.edit_step = 0
                    self._enter_mode(M_EDIT_CODE)
            return True

        if self.mode == M_EDIT_CODE:
            if key == ESC:
                self.edit_target_idx = None
                self.edit_codes = {}
                self.edit_step = 0
                self._enter_mode(M_BROWSE); return True
            if key in ENTER:
                # passe au champ suivant (sans changement si rien tapé)
                self.edit_step += 1
                if self.edit_step >= len(CODE_STEPS):
                    self._apply_edit()
                    self._enter_mode(M_BROWSE)
                return True
            if key in BACK and self.edit_step > 0:
                f, _ = CODE_STEPS[self.edit_step - 1]
                self.edit_codes.pop(f, None)
                self.edit_step -= 1
                return True
            if 48 <= key <= 57:
                v = key - 48
                if v == 0:
                    return True
                field, _ = CODE_STEPS[self.edit_step]
                self.edit_codes[field] = v
            return True

        # ── ADD_F0 / ADD_F1 / OBS_F0 / OBS_F1 : nav + ENTER ─────────────────
        if self.mode in (M_ADD_F0, M_ADD_F1, M_OBS_F0, M_OBS_F1):
            if key == ESC:
                if self.mode in (M_ADD_F0, M_ADD_F1):
                    self._reset_add()
                else:
                    self.obs_f0 = None
                self._enter_mode(M_BROWSE); return True
            if key in ENTER:
                if self.mode == M_ADD_F0:
                    self.add_f0 = self.frame_idx
                    self._enter_mode(M_ADD_F1)
                elif self.mode == M_ADD_F1:
                    self.add_f1 = self.frame_idx
                    if self.add_f1 < self.add_f0:
                        self.add_f0, self.add_f1 = self.add_f1, self.add_f0
                    self.add_step = 0
                    self._enter_mode(M_ADD_CODE)
                elif self.mode == M_OBS_F0:
                    self.obs_f0 = self.frame_idx
                    self._enter_mode(M_OBS_F1)
                elif self.mode == M_OBS_F1:
                    self._apply_obs()
                    self._enter_mode(M_BROWSE)
                return True
            # nav
            return self._handle_nav(key)

        # ── ADD_PICK : juste ESC + nav (clic est traité dans run()) ─────────
        if self.mode == M_ADD_PICK:
            if key == ESC:
                self._reset_add(); self._enter_mode(M_BROWSE)
            else:
                self._handle_nav(key)
            return True

        # ── MERGE1 / MERGE2 : ESC + nav (clics traités dans run()) ──────────
        if self.mode in (M_MERGE1, M_MERGE2):
            if key == ESC:
                self.merge_keep_tid = None
                self._enter_mode(M_BROWSE)
            else:
                self._handle_nav(key)
            return True

        # ── MANUAL : drag est traité dans le mouse callback ─────────────────
        if self.mode == M_MANUAL:
            if key == ESC:
                self._reset_manual(); self._enter_mode(M_BROWSE)
                return True
            if key in ENTER:
                self._finalize_manual()  # transitionne vers ADD_CODE si OK
                return True
            return self._handle_nav(key)

        # ── BROWSE ──────────────────────────────────────────────────────────
        if key == ESC:
            return self._confirm_quit()

        if key in (ord('Q'), ord('q')):
            self.save()
            return False
        if key in (ord('S'), ord('s')):
            self.save(); return True

        if key in (ord('A'), ord('a')):
            self._reset_add()
            self._enter_mode(M_ADD_PICK); return True

        if key in (ord('M'), ord('m')):
            self._reset_manual()
            self._enter_mode(M_MANUAL)
            print("  manual track : drag pour poser une keyframe, navigue, "
                  "drag à nouveau pour une 2e KF, ENTER finalise.")
            return True

        if key in (ord('J'), ord('j')):
            self.merge_keep_tid = None
            self._enter_mode(M_MERGE1)
            print("  merge : clique le track qui RESTE (avant le break), "
                  "puis navigue et clique le track à ABSORBER.")
            return True

        if key in (ord('D'), ord('d')):
            self.del_targets = [i for i, _ in encounter_at_frame(self.encounters, self.frame_idx)]
            if not self.del_targets:
                print("[!] aucun encounter confirmé à cette frame")
            else:
                self._enter_mode(M_DEL)
            return True

        if key in (ord('E'), ord('e')):
            self.del_targets = [i for i, _ in encounter_at_frame(self.encounters, self.frame_idx)]
            if not self.del_targets:
                print("[!] aucun encounter confirmé à cette frame")
            else:
                self._enter_mode(M_EDIT_PICK)
            return True

        if key in (ord('U'), ord('u')):
            self.del_targets = [
                i for i, _ in encounter_at_frame(self.encounters, self.frame_idx)
                if (self.encounters[i].get("LINKED_TRACKS") or "").strip()
            ]
            if not self.del_targets:
                print("[!] aucun encounter avec LINKED_TRACKS à cette frame")
            else:
                self._enter_mode(M_UNLINK)
            return True

        if key in (ord('O'), ord('o')):
            # si on est déjà au-dessus d'une obstacle zone → propose suppression au lieu d'ajout
            self._enter_mode(M_OBS_F0)
            print(f"  obstacle: start sera la frame courante = {self.frame_idx}. "
                  "Navigue jusqu'à la fin puis ENTER (ou O).")
            return True

        if key in (ord('X'), ord('x')):
            self.del_targets = [i for i, _ in obstacle_at_frame(self.zones, self.frame_idx)]
            if not self.del_targets:
                print("[!] aucune zone obstacle à cette frame")
            else:
                self._enter_mode(M_OBS_DEL)
            return True

        if key in (ord('G'), ord('g')):
            try:
                tgt = int(input("Goto frame # : ").strip())
                self._read_frame_at(tgt)
            except ValueError:
                pass
            return True

        return self._handle_nav(key)

    # Codes "raw" (waitKeyEx) — macOS et Linux/X11
    KEY_LEFT  = (63234, 65361)   # ←  : -1 s
    KEY_RIGHT = (63235, 65363)   # →  : +1 s
    KEY_UP    = (63232, 65362)   # ↑  : +5 s
    KEY_DOWN  = (63233, 65364)   # ↓  : -5 s

    def _handle_nav(self, key):
        if key == 32:  # SPACE → play/pause
            self.playing = not self.playing
            return True
        if key in (ord(','), ord('<')):
            self._step(-1); return True
        if key in (ord('.'), ord('>')):
            self._step(1); return True
        fps = max(1, int(round(self.fps)))
        if key in self.KEY_LEFT:
            self.playing = False; self._step(-fps); return True
        if key in self.KEY_RIGHT:
            self.playing = False; self._step(fps); return True
        if key in self.KEY_UP:
            self.playing = False; self._step(5 * fps); return True
        if key in self.KEY_DOWN:
            self.playing = False; self._step(-5 * fps); return True
        return True

    def _confirm_quit(self):
        if not self.dirty:
            return False
        ans = input("Modifs non sauvegardées. Sauver avant de quitter ? [Y/n] ").strip().lower()
        if ans in ("", "y", "o", "yes", "oui"):
            self.save()
        return False


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True, help="nom du clip (sans extension)")
    p.add_argument("--rater", type=int, default=None,
                   help="rater id 1 ou 2 (auto-détecté si absent)")
    args = p.parse_args()

    rater = args.rater or detect_rater(args.clip)
    if rater is None:
        sys.exit(f"Aucun rater{{1,2}} encounters CSV trouvé pour {args.clip}")

    paths = derive_paths(args.clip, rater)
    for k, v in paths.items():
        if not os.path.exists(v) and k not in ("session", "obs", "speed"):
            sys.exit(f"Manquant: {k} = {v}")

    Corrector(args.clip, rater).run()


if __name__ == "__main__":
    main()
