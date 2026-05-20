"""
Curb Detection Manual Correction Tool
======================================
- Charge les pixels détectés depuis un CSV (format frame/pixels)
- Regroupe les pixels en clusters par composantes connexes
- Tracke les clusters sur TOUTE la vidéo (tracking global stable, union-find)
- UI OpenCV interactive : supprimer / restaurer des tracks entiers
- Sous-échantillonnage : on corrige à ~3 Hz (3 frames/seconde),
  le CSV de sortie est donc à 3 Hz (réglable via --step)
- Entrée CLI = la vidéo ; les CSV sont déduits dans <video>/../lanes :
    détections : <nom_video>_segformer_pixels.csv
    corrigé    : <nom_video>_corrected_lanes.csv
- Reprise automatique : si le CSV corrigé existe déjà il est rechargé en
  priorité (déjà à 3 Hz → pas de ré-échantillonnage)
- Les très petits clusters (< MIN_CLUSTER_SIZE) sont DROP (non sauvegardés)
- Distance sol min de chaque cluster à la caméra, calculée depuis
  <video>/../codebookescooter/<nom_video>_calibration.json (affichée + point
  rouge sur le pixel le plus proche)

Contrôles :
  ← / →       : frame échantillonnée précédente / suivante (~1/3 s)
  Espace      : lecture / pause (défilement rapide des frames 3 Hz)
  [ / ]       : reculer / avancer de 10 frames échantillonnées
  Clic gauche : sélectionner un track et le supprimer/restaurer
                (la suppression s'applique sur TOUTE la vidéo)
  L           : mode "créer une ligne" — clic1 = départ, clic2 = fin
                (clic droit annule le point en attente)
  D           : supprimer TOUS les clusters visibles sur cette frame
  R           : restaurer TOUS les clusters visibles sur cette frame
  A / Z       : supprimer / restaurer un track entier (ID sélectionné)
  S           : sauvegarder le CSV corrigé
  Q / Echap   : quitter
"""

from __future__ import annotations

import cv2
import pandas as pd
import numpy as np
from scipy.ndimage import label
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse
import json
import sys
import os

# ─────────────────────────────────────────────
# PARAMÈTRES PAR DÉFAUT (modifiables en CLI)
# ─────────────────────────────────────────────
DEFAULT_VIDEO  = "/Volumes/My Passport/NEWMOB/escooter/391t_2023-04-11 17_10_50_391t_9_48_10_31.mp4"


VIDEO_DIRNAME = "escooter"


def _video_stem(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def resolve_video(video_path: str) -> str:
    """Renvoie un chemin vidéo existant.

    Les vidéos sont dans 'escooter' (plus dans 'PRIMARY'). Si le chemin
    fourni n'existe pas, on cherche le même fichier dans le dossier
    escooter (frère, ex. .../PRIMARY → .../escooter)."""
    if os.path.exists(video_path):
        return video_path
    base = os.path.basename(video_path)
    root = os.path.dirname(os.path.dirname(video_path))
    candidates = [
        os.path.join(root, VIDEO_DIRNAME, base),
        video_path.replace(f"{os.sep}PRIMARY{os.sep}",
                           f"{os.sep}{VIDEO_DIRNAME}{os.sep}"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return video_path


def lanes_dir_for(video_path: str) -> str:
    """Dossier 'lanes' : frère du dossier de la vidéo (ex. .../escooter)."""
    return os.path.join(os.path.dirname(os.path.dirname(video_path)), "lanes")


def segformer_csv_for(video_path: str, lanes_dir: str | None = None) -> str:
    """CSV de détections déduit : <lanes>/<nom_video>_segformer_pixels.csv"""
    d = lanes_dir or lanes_dir_for(video_path)
    return os.path.join(d, f"{_video_stem(video_path)}_segformer_pixels.csv")


def corrected_csv_for(video_path: str, lanes_dir: str | None = None) -> str:
    """CSV corrigé déduit : <lanes>/<nom_video>_corrected_lanes.csv"""
    d = lanes_dir or lanes_dir_for(video_path)
    return os.path.join(d, f"{_video_stem(video_path)}_corrected_lanes.csv")


CODEBOOK_DIRNAME = "codebookescooter"


def calibration_json_for(video_path: str,
                         codebook_dir: str | None = None) -> str:
    """Calibration déduite : <codebookescooter>/<nom_video>_calibration.json"""
    d = codebook_dir or os.path.join(
        os.path.dirname(os.path.dirname(video_path)), CODEBOOK_DIRNAME)
    return os.path.join(d, f"{_video_stem(video_path)}_calibration.json")


MASK_COLOR_ACTIVE    = (0, 200, 255)   # BGR jaune-orange : cluster actif
MASK_COLOR_DELETED   = (80,  80,  80)  # gris  : supprimé
ALPHA                = 0.45
MIN_CLUSTER_SIZE     = 10              # pixels min pour garder un cluster (les + petits sont DROP)
IOU_THRESHOLD        = 0.05            # seuil IoU pour le tracking inter-frame
MAX_TRACK_GAP        = 5              # frames max de gap autorisé pour continuer un track
PLAY_FPS_FALLBACK    = 30             # cadence de lecture si la vidéo n'expose pas son fps
TARGET_HZ            = 3              # cadence de correction visée (frames/seconde)
MANUAL_LINE_THICKNESS = 3             # épaisseur (px) d'une ligne créée à la main
JUMP_FRAMES          = 10             # nb de frames pour les sauts [ / ]

# Palette de couleurs pour les track IDs (cycle)
PALETTE = [
    (255, 80,  80),   # rouge
    (80,  255, 80),   # vert
    (80,  80,  255),  # bleu
    (255, 255, 80),   # jaune
    (255, 80,  255),  # magenta
    (80,  255, 255),  # cyan
    (255, 160, 80),   # orange
    (160, 80,  255),  # violet
    (80,  200, 160),  # vert-mer
    (200, 160, 80),   # or
]


# ─────────────────────────────────────────────
# LECTURE CSV
# ─────────────────────────────────────────────

def load_csv(csv_path: str) -> dict[int, list[tuple[int,int]]]:
    df = pd.read_csv(csv_path)
    frame_pixels: dict[int, list[tuple[int,int]]] = {}
    for _, row in df.iterrows():
        frame_id  = int(row["frame"])
        pixel_str = str(row["pixels"])
        coords = []
        for p in pixel_str.split(";"):
            p = p.strip()
            if "_" not in p:
                continue
            try:
                x_str, y_str = p.split("_")
                coords.append((int(x_str), int(y_str)))
            except Exception:
                continue
        frame_pixels[frame_id] = coords
    return frame_pixels


# ─────────────────────────────────────────────
# CLUSTERING : composantes connexes
# ─────────────────────────────────────────────

def cluster_frame(pixels: list[tuple[int,int]], height: int, width: int,
                  min_size: int = MIN_CLUSTER_SIZE
                  ) -> list[list[tuple[int,int]]]:
    """Retourne une liste de clusters (chaque cluster = liste de (x,y))."""
    if not pixels:
        return []
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in pixels:
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = 1
    # Composantes STRICTEMENT contiguës : pas de dilatation, donc des
    # pixels non adjacents ne sont jamais regroupés dans le même cluster.
    # Connectivité 8 (un contact diagonal compte comme contigu).
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n = label(mask, structure=structure)
    clusters = []
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        pts = list(zip(xs.tolist(), ys.tolist()))
        if len(pts) >= min_size:
            clusters.append(pts)
    return clusters


def cluster_bbox(pts: list[tuple[int,int]]) -> tuple[int,int,int,int]:
    """Bounding box (x1,y1,x2,y2) d'un cluster."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def iou_bbox(a: tuple, b: tuple) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


# ─────────────────────────────────────────────
# DISTANCE SOL (modèle pinhole + sol plat)
# ─────────────────────────────────────────────

def load_calibration(path: str, width: int, height: int) -> dict | None:
    """Charge calibration.json → dict {f, cx, cy, h, pitch, factor}."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            cal = json.load(fh)
    except Exception as e:
        print(f"⚠ calibration illisible ({e})")
        return None
    return {
        "f":      float(cal.get("focal_length_px", cal.get("f", 1445.0))),
        "cx":     float(cal.get("cx", width / 2.0)),
        "cy":     float(cal.get("cy", height / 2.0)),
        "h":      float(cal.get("camera_height_m", cal.get("h_cam", 1.2))),
        "pitch":  float(cal.get("pitch_deg", 0.0)),
        "factor": float(cal.get("distance_calibration_factor", 1.0)),
    }


def cluster_min_distance(cluster: list[tuple[int,int]], cal: dict
                         ) -> tuple[float, tuple[int,int]] | None:
    """Distance sol minimale (m) du cluster à la caméra + pixel le plus proche.

    Modèle identique à annotate_encounters.pixel_to_ground :
      horizon_v = cy - f*tan(pitch) ; Y = f*h/dv ; X = (u-cx)*h/dv
    """
    if not cluster:
        return None
    pts = np.asarray(cluster, dtype=np.float64)
    u, v = pts[:, 0], pts[:, 1]
    horizon_v = cal["cy"] - cal["f"] * np.tan(np.radians(cal["pitch"]))
    dv = v - horizon_v
    valid = dv > 1e-6                       # sous l'horizon uniquement
    if not np.any(valid):
        return None
    u, v, dv = u[valid], v[valid], dv[valid]
    Y = cal["f"] * cal["h"] / dv
    X = (u - cal["cx"]) * cal["h"] / dv
    d = np.sqrt(X * X + Y * Y) * cal["factor"]
    i = int(np.argmin(d))
    return float(d[i]), (int(u[i]), int(v[i]))


# ─────────────────────────────────────────────
# TRACKING greedy IoU
# ─────────────────────────────────────────────

class Tracker:
    def __init__(self, iou_thr=IOU_THRESHOLD, max_gap=MAX_TRACK_GAP):
        self.iou_thr  = iou_thr
        self.max_gap  = max_gap
        self._next_id = 0
        # track_id -> {"last_frame": int, "last_bbox": tuple}
        self._active: dict[int, dict] = {}

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, frame_idx: int,
               clusters: list[list[tuple[int,int]]]
               ) -> list[int]:
        """
        Associe chaque cluster de la frame à un track_id.
        Retourne la liste des track_ids (même ordre que clusters).
        """
        # purger les tracks trop anciens
        to_del = [tid for tid, info in self._active.items()
                  if frame_idx - info["last_frame"] > self.max_gap]
        for tid in to_del:
            del self._active[tid]

        bboxes = [cluster_bbox(c) for c in clusters]
        assigned = [-1] * len(clusters)
        used_tracks = set()

        # Matrice IoU
        active_ids   = list(self._active.keys())
        active_boxes = [self._active[tid]["last_bbox"] for tid in active_ids]

        if active_ids and bboxes:
            iou_mat = np.zeros((len(bboxes), len(active_ids)))
            for i, bb in enumerate(bboxes):
                for j, ab in enumerate(active_boxes):
                    iou_mat[i, j] = iou_bbox(bb, ab)
            # greedy : meilleure paire en premier
            flat = [(iou_mat[i,j], i, j)
                    for i in range(len(bboxes))
                    for j in range(len(active_ids))
                    if iou_mat[i,j] >= self.iou_thr]
            flat.sort(reverse=True)
            used_clusters = set()
            for score, ci, ti in flat:
                if ci in used_clusters or ti in used_tracks:
                    continue
                tid = active_ids[ti]
                assigned[ci] = tid
                used_clusters.add(ci)
                used_tracks.add(ti)

        # Nouveaux tracks pour clusters non assignés
        for i, tid in enumerate(assigned):
            if tid == -1:
                new_tid = self._new_id()
                assigned[i] = new_tid

        # Mise à jour état
        for i, tid in enumerate(assigned):
            self._active[tid] = {
                "last_frame": frame_idx,
                "last_bbox":  bboxes[i],
            }

        return assigned


# ─────────────────────────────────────────────
# STRUCTURE PRINCIPALE : données par frame
# ─────────────────────────────────────────────

class FrameData:
    """Stocke clusters + track_ids + état deleted pour une frame."""
    __slots__ = ("clusters", "track_ids", "deleted")

    def __init__(self, clusters, track_ids):
        self.clusters  : list[list[tuple[int,int]]] = clusters
        self.track_ids : list[int]                  = track_ids
        self.deleted   : list[bool]                 = [False] * len(clusters)


# ─────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────

class CurbCorrector:
    def __init__(self, video_path, csv_path, output_path, frame_step=None,
                 calib_path=None):
        self.video_path  = video_path
        self.csv_path    = csv_path
        self.output_path = output_path

        print("Chargement CSV …")
        self.frame_pixels = load_csv(csv_path)

        print("Ouverture vidéo …")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            sys.exit("Impossible d'ouvrir la vidéo.")

        self.width      = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height     = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps        = self.cap.get(cv2.CAP_PROP_FPS)
        self.total      = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Calibration pour la distance sol (codebookescooter) ──
        self.calib = load_calibration(calib_path, self.width, self.height)
        # Cache distance : (frame_idx, cluster_idx) -> (d_min, (x,y)) | None
        self._dist_cache: dict[tuple[int,int], tuple | None] = {}
        if self.calib:
            print(f"Calibration : f={self.calib['f']:.0f}px "
                  f"h={self.calib['h']:.2f}m pitch={self.calib['pitch']:.2f}° "
                  f"factor={self.calib['factor']:.3f}")
        else:
            print("⚠ Pas de calibration.json → distances non affichées")

        # ── Sous-échantillonnage : on vise TARGET_HZ frames/seconde ──
        if frame_step is None:
            fps = self.fps if (self.fps and self.fps > 1) else PLAY_FPS_FALLBACK
            step = round(fps / TARGET_HZ)
        else:
            step = frame_step
        self.frame_step = max(1, int(step))

        all_fids = sorted(self.frame_pixels.keys())
        self.sampled_fids: list[int] = all_fids[::self.frame_step]
        # On ne conserve QUE les frames échantillonnées (UI, tracking, save)
        self.frame_pixels = {f: self.frame_pixels[f]
                             for f in self.sampled_fids}
        self.n_samples = len(self.sampled_fids)
        if self.frame_step > 1:
            print(f"Sous-échantillonnage ~{TARGET_HZ} Hz : 1 frame / "
                  f"{self.frame_step} → {self.n_samples} frames conservées "
                  f"(sur {len(all_fids)})")

        # Position dans la liste échantillonnée ; frame_idx = vraie frame vidéo
        self.pos       = 0
        self.frame_idx = self.sampled_fids[0] if self.sampled_fids else 0

        # Cache des frames vidéo décodées (en mémoire limitée)
        self._frame_cache: dict[int, np.ndarray] = {}

        # Données par frame (calculées à la demande)
        self.frame_data: dict[int, FrameData] = {}

        # Tracker
        self.tracker = Tracker()

        # Suivi des suppressions par track_id
        # deleted_tracks[tid] = True => tout le track est supprimé
        self.deleted_tracks: dict[int, bool] = defaultdict(lambda: False)

        # État UI
        self.selected_cluster_idx: int | None = None   # index dans frame courante
        self.selected_track_id:    int | None = None

        # Lecture automatique
        self.playing = False

        # Mode création de ligne ("select" = normal, "line" = 2 clics)
        self.mode: str = "select"
        self.pending_pt: tuple[int, int] | None = None   # 1er clic en attente
        self.mouse_pos: tuple[int, int] = (0, 0)          # pour l'aperçu live

        # Nombre total de tracks globaux (rempli par _build_global_tracks)
        self._n_tracks = 0

        # Pré-calcul de TOUTES les frames pour le tracking cohérent
        print("Calcul des clusters …")
        self._precompute_all()
        print("Tracking global sur toute la vidéo …")
        self._build_global_tracks()
        print(f"Prêt. {self.n_samples} frames (~{TARGET_HZ} Hz), "
              f"{self._n_tracks} tracks détectés.")

    # ── pré-calcul ─────────────────────────────

    def _precompute_all(self):
        sorted_frames = sorted(self.frame_pixels.keys())
        for fid in sorted_frames:
            pixels   = self.frame_pixels[fid]
            clusters = cluster_frame(pixels, self.height, self.width)
            # track_ids assignés ensuite par _build_global_tracks
            self.frame_data[fid] = FrameData(clusters, [-1] * len(clusters))

    def _build_global_tracks(self):
        """
        Tracking GLOBAL via union-find sur les meilleures paires IoU.

        Garantie clé : un track ne peut PAS contenir deux clusters de la
        MÊME frame. Deux morceaux distincts présents simultanément restent
        donc deux tracks séparés (et ne s'affichent plus comme un seul).
        Les paires sont fusionnées par IoU décroissant ; une fusion qui
        mettrait deux clusters d'une même frame ensemble est refusée
        (appariement de fait 1-à-1).
        """
        iou_thr = self.tracker.iou_thr
        max_gap = self.tracker.max_gap

        parent: dict[tuple[int, int], tuple[int, int]] = {}
        comp_frames: dict[tuple[int, int], set] = {}   # racine -> {frames}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def try_union(a, b) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            fa, fb = comp_frames[ra], comp_frames[rb]
            if fa & fb:                       # collision de frame → refusé
                return
            parent[rb] = ra
            fa |= fb
            del comp_frames[rb]

        sorted_fids = sorted(self.frame_data.keys())

        # Init noeuds + cache des bbox
        bbox: dict[tuple[int, int], tuple] = {}
        for fid in sorted_fids:
            fd = self.frame_data[fid]
            for ci, cl in enumerate(fd.clusters):
                node = (fid, ci)
                parent[node] = node
                comp_frames[node] = {fid}
                bbox[node] = cluster_bbox(cl)

        # Toutes les paires candidates (frames distantes de ≤ max_gap)
        candidates: list[tuple[float, tuple, tuple]] = []
        for idx, fid in enumerate(sorted_fids):
            fd = self.frame_data[fid]
            for ci in range(len(fd.clusters)):
                b1 = bbox[(fid, ci)]
                for j in range(idx + 1,
                                min(idx + 1 + max_gap, len(sorted_fids))):
                    fid2 = sorted_fids[j]
                    fd2 = self.frame_data[fid2]
                    for cj in range(len(fd2.clusters)):
                        s = iou_bbox(b1, bbox[(fid2, cj)])
                        if s >= iou_thr:
                            candidates.append((s, (fid, ci), (fid2, cj)))

        # Fusion gloutonne : meilleures paires d'abord
        candidates.sort(key=lambda t: t[0], reverse=True)
        for _, a, b in candidates:
            try_union(a, b)

        # Attribution d'IDs de track contigus
        group_id: dict[tuple[int, int], int] = {}
        next_g = 0
        for fid in sorted_fids:
            fd = self.frame_data[fid]
            new_ids = []
            for ci in range(len(fd.clusters)):
                root = find((fid, ci))
                if root not in group_id:
                    group_id[root] = next_g
                    next_g += 1
                new_ids.append(group_id[root])
            fd.track_ids = new_ids
        self._n_tracks = next_g

    # ── lecture vidéo ──────────────────────────

    def get_frame(self, idx: int) -> np.ndarray | None:
        if idx in self._frame_cache:
            return self._frame_cache[idx].copy()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Limiter le cache à 30 frames
        if len(self._frame_cache) > 30:
            oldest = min(self._frame_cache)
            del self._frame_cache[oldest]
        self._frame_cache[idx] = frame.copy()
        return frame

    # ── rendu ──────────────────────────────────

    def _cluster_distance(self, ci: int, cluster):
        """Distance sol min (m) + pixel le + proche, mis en cache."""
        if not self.calib:
            return None
        key = (self.frame_idx, ci)
        if key not in self._dist_cache:
            self._dist_cache[key] = cluster_min_distance(cluster, self.calib)
        return self._dist_cache[key]

    def render(self) -> np.ndarray:
        frame = self.get_frame(self.frame_idx)
        if frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        overlay = frame.copy()
        fd = self.frame_data.get(self.frame_idx)

        if fd:
            for ci, (cluster, tid) in enumerate(zip(fd.clusters, fd.track_ids)):
                is_deleted = fd.deleted[ci] or self.deleted_tracks[tid]
                color = MASK_COLOR_DELETED if is_deleted else PALETTE[tid % len(PALETTE)]
                for x, y in cluster:
                    if 0 <= x < self.width and 0 <= y < self.height:
                        overlay[y, x] = color

                # Bounding box
                x1,y1,x2,y2 = cluster_bbox(cluster)
                thickness = 2 if ci == self.selected_cluster_idx else 1
                bx_color  = (255, 255, 255) if ci == self.selected_cluster_idx else color
                cv2.rectangle(overlay, (x1,y1), (x2,y2), bx_color, thickness)

                # Distance sol minimale (depuis calibration.json)
                dist = self._cluster_distance(ci, cluster)
                if dist is not None and not is_deleted:
                    _, (px, py) = dist
                    cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)
                    cv2.circle(overlay, (px, py), 6, (255, 255, 255), 1)

                # Label track_id (+ distance min)
                label_txt = f"T{tid}"
                if dist is not None:
                    label_txt += f" {dist[0]:.1f}m"
                if is_deleted:
                    label_txt += " [X]"
                cv2.putText(overlay, label_txt,
                            (x1, max(y1-4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, bx_color, 1, cv2.LINE_AA)

        result = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)
        self._draw_hud(result)

        # Aperçu du mode création de ligne
        if self.mode == "line":
            if self.pending_pt is not None:
                cv2.circle(result, self.pending_pt, 4, (0, 0, 255), -1)
                cv2.line(result, self.pending_pt, self.mouse_pos,
                         (0, 0, 255), MANUAL_LINE_THICKNESS)
            cv2.putText(result,
                        "MODE LIGNE: clic1=depart  clic2=fin  (clic droit=annuler)",
                        (10, self.height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                        cv2.LINE_AA)
        return result

    def _draw_hud(self, frame: np.ndarray):
        fd = self.frame_data.get(self.frame_idx)
        n_clusters = len(fd.clusters) if fd else 0
        n_deleted  = sum(1 for ci, tid in enumerate(fd.track_ids)
                         if fd and (fd.deleted[ci] or self.deleted_tracks[tid])) if fd else 0

        play_txt = "▶ LECTURE" if self.playing else "⏸ pause"
        lines = [
            f"{TARGET_HZ}Hz {self.pos+1}/{self.n_samples}  "
            f"(frame video {self.frame_idx})   {play_txt}",
            f"Clusters: {n_clusters}  Supprimés: {n_deleted}",
            f"Track sélectionné: {self.selected_track_id if self.selected_track_id is not None else '-'}",
            "← → nav  Espace play  [ ] saut10  Clic select",
            "L: ligne (2 clics)  D/R frame  A/Z track  S save  Q quit",
        ]
        y = 20
        for line in lines:
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            y += 20

    # ── création manuelle de ligne ─────────────

    def _ensure_frame_data(self, fid: int) -> FrameData:
        fd = self.frame_data.get(fid)
        if fd is None:
            fd = FrameData([], [])
            self.frame_data[fid] = fd
        return fd

    def _rasterize_line(self, p0: tuple[int,int], p1: tuple[int,int],
                        thickness: int) -> list[tuple[int,int]]:
        def clamp(p):
            return (int(min(self.width  - 1, max(0, p[0]))),
                    int(min(self.height - 1, max(0, p[1]))))
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.line(mask, clamp(p0), clamp(p1), 1, max(1, thickness))
        ys, xs = np.where(mask == 1)
        return list(zip(xs.tolist(), ys.tolist()))

    def toggle_line_mode(self):
        if self.mode == "line":
            self.mode = "select"
            self.pending_pt = None
            print("Mode sélection")
        else:
            self.mode = "line"
            self.pending_pt = None
            print("Mode LIGNE : clic1 = départ, clic2 = fin")

    def add_manual_line(self, p0: tuple[int,int], p1: tuple[int,int]):
        """Crée un nouveau cluster (= track) le long du segment p0→p1
        sur la frame courante."""
        pts = self._rasterize_line(p0, p1, MANUAL_LINE_THICKNESS)
        if not pts:
            return
        fd = self._ensure_frame_data(self.frame_idx)
        tid = self._n_tracks
        self._n_tracks += 1
        fd.clusters.append(pts)
        fd.track_ids.append(tid)
        fd.deleted.append(False)
        self.selected_cluster_idx = len(fd.clusters) - 1
        self.selected_track_id = tid
        print(f"Ligne ajoutée (frame vidéo {self.frame_idx}, track {tid}, "
              f"{len(pts)} px)")

    # ── interactions ───────────────────────────

    def click(self, x: int, y: int):
        # Mode création de ligne : on collecte 2 points
        if self.mode == "line":
            if self.pending_pt is None:
                self.pending_pt = (x, y)
            else:
                self.add_manual_line(self.pending_pt, (x, y))
                self.pending_pt = None
            return

        fd = self.frame_data.get(self.frame_idx)
        if fd is None:
            return
        for ci, cluster in enumerate(fd.clusters):
            x1,y1,x2,y2 = cluster_bbox(cluster)
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_cluster_idx = ci
                tid = fd.track_ids[ci]
                self.selected_track_id = tid
                # toggle sur TOUT le track (toutes les frames)
                if self.deleted_tracks[tid]:
                    self.restore_track(tid)
                else:
                    self.delete_track(tid)
                return
        # clic dans le vide : désélection
        self.selected_cluster_idx = None
        self.selected_track_id    = None

    def delete_all_frame(self):
        fd = self.frame_data.get(self.frame_idx)
        if fd:
            fd.deleted = [True] * len(fd.clusters)

    def restore_all_frame(self):
        fd = self.frame_data.get(self.frame_idx)
        if fd:
            fd.deleted = [False] * len(fd.clusters)

    def delete_track(self, tid: int):
        self.deleted_tracks[tid] = True
        print(f"Track {tid} → entièrement supprimé")

    def restore_track(self, tid: int):
        self.deleted_tracks[tid] = False
        # restaurer aussi les suppressions locales pour ce track
        for fd in self.frame_data.values():
            for ci, t in enumerate(fd.track_ids):
                if t == tid:
                    fd.deleted[ci] = False
        print(f"Track {tid} → restauré")

    # ── sauvegarde ─────────────────────────────

    def save(self):
        rows = []
        for fid, fd in sorted(self.frame_data.items()):
            kept_set = set()
            for ci, (cluster, tid) in enumerate(zip(fd.clusters, fd.track_ids)):
                if not fd.deleted[ci] and not self.deleted_tracks[tid]:
                    kept_set.update(cluster)
            # Les très petits clusters (< MIN_CLUSTER_SIZE) ne sont PAS
            # reclusterisés : ils sont DROP, on ne les réinjecte plus.
            if kept_set:
                pixel_str = ";".join(f"{x}_{y}" for x,y in sorted(kept_set))
                rows.append({"frame": fid, "pixels": pixel_str})
        df_out = pd.DataFrame(rows)
        df_out.to_csv(self.output_path, index=False)
        print(f"✅ Sauvegardé : {self.output_path}  ({len(rows)} frames, "
              f"très petits clusters droppés)")

    # ── navigation (sur les frames échantillonnées) ──

    def _goto(self, pos: int):
        if not self.sampled_fids:
            return
        self.pos = max(0, min(self.n_samples - 1, pos))
        self.frame_idx = self.sampled_fids[self.pos]
        self.selected_cluster_idx = None

    # ── boucle principale ──────────────────────

    def run(self):
        WIN = "Curb Corrector"
        # Échelle d'affichage maîtrisée : la fenêtre fait exactement la
        # taille de l'image affichée → les coords souris == coords image
        # (à l'échelle près). Indispensable pour le clic-clic de ligne.
        self.disp_scale = min(1.0, 1280.0 / self.width, 720.0 / self.height)
        disp_w = max(1, int(self.width  * self.disp_scale))
        disp_h = max(1, int(self.height * self.disp_scale))
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

        def on_mouse(event, x, y, flags, param):
            # Conversion coords fenêtre → coords image pleine résolution
            ix = int(min(self.width  - 1, max(0, x / self.disp_scale)))
            iy = int(min(self.height - 1, max(0, y / self.disp_scale)))
            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_pos = (ix, iy)
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.click(ix, iy)
            elif event == cv2.EVENT_RBUTTONDOWN:
                # annule le point de ligne en attente
                self.pending_pt = None

        cv2.setMouseCallback(WIN, on_mouse)

        # Lecture auto : défilement rapide des frames 1 fps (~10/s)
        play_delay = 100

        while True:
            img = self.render()
            if self.disp_scale != 1.0:
                img = cv2.resize(img, (disp_w, disp_h),
                                 interpolation=cv2.INTER_AREA)
            cv2.imshow(WIN, img)

            key = cv2.waitKey(play_delay if self.playing else 20) & 0xFF

            if self.playing:
                if self.pos >= self.n_samples - 1:
                    self.playing = False           # fin de vidéo → pause
                else:
                    self._goto(self.pos + 1)

            if key in (ord('q'), 27):          # quitter
                break
            elif key == 32:                    # Espace : play / pause
                self.playing = not self.playing
            elif key == 81 or key == 2:        # ← (Linux/Mac)
                self.playing = False
                self._goto(self.pos - 1)
            elif key == 83 or key == 3:        # → (Linux/Mac)
                self.playing = False
                self._goto(self.pos + 1)
            elif key == ord('['):              # saut arrière rapide
                self.playing = False
                self._goto(self.pos - JUMP_FRAMES)
            elif key == ord(']'):              # saut avant rapide
                self.playing = False
                self._goto(self.pos + JUMP_FRAMES)
            elif key == ord('l'):              # bascule mode création ligne
                self.toggle_line_mode()
            elif key == ord('d'):              # delete frame
                self.delete_all_frame()
            elif key == ord('r'):              # restore frame
                self.restore_all_frame()
            elif key == ord('a'):              # delete track
                if self.selected_track_id is not None:
                    self.delete_track(self.selected_track_id)
            elif key == ord('z'):              # restore track
                if self.selected_track_id is not None:
                    self.restore_track(self.selected_track_id)
            elif key == ord('s'):              # save
                self.save()

        cv2.destroyAllWindows()
        self.cap.release()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curb Detection Manual Corrector")
    parser.add_argument("video", nargs="?", default=DEFAULT_VIDEO,
                        help="Chemin de la vidéo (.mp4). Les CSV sont déduits.")
    parser.add_argument("--lanes-dir", default=None,
                        help="Dossier des CSV lanes (défaut: <video>/../lanes)")
    parser.add_argument("--csv", default=None,
                        help="Forcer le CSV d'entrée "
                             "(défaut: déduit, reprise prioritaire sur corrected)")
    parser.add_argument("--output", default=None,
                        help="Forcer le CSV corrigé en sortie "
                             "(défaut: <video>_corrected_lanes.csv)")
    parser.add_argument("--step",   type=int, default=None,
                        help="Garder 1 frame sur N (défaut: ~fps/3 => 3 Hz). "
                             "1 = toutes les frames.")
    parser.add_argument("--codebook-dir", default=None,
                        help="Dossier codebookescooter pour calibration.json "
                             "(défaut: <video>/../codebookescooter)")
    args = parser.parse_args()

    # La vidéo est dans 'escooter' : on corrige un éventuel chemin PRIMARY
    args.video = resolve_video(args.video)

    # Chemins déduits du nom de la vidéo
    lanes_dir     = args.lanes_dir or lanes_dir_for(args.video)
    segformer_csv = segformer_csv_for(args.video, lanes_dir)
    corrected_csv = args.output or corrected_csv_for(args.video, lanes_dir)
    calib_path    = calibration_json_for(args.video, args.codebook_dir)

    frame_step = args.step
    if args.csv:                              # entrée forcée en CLI
        csv_path = args.csv
    elif os.path.exists(corrected_csv):       # reprise prioritaire
        print(f"↻ Reprise depuis {corrected_csv} (corrections précédentes)")
        csv_path = corrected_csv
        # Le CSV corrigé est déjà à 3 Hz : on ne ré-échantillonne pas.
        if frame_step is None:
            frame_step = 1
    else:                                     # 1ʳᵉ passe : détections brutes
        csv_path = segformer_csv

    print(f"Vidéo   : {args.video}")
    print(f"Entrée  : {csv_path}")
    print(f"Sortie  : {corrected_csv}")
    print(f"Calib   : {calib_path}")

    app = CurbCorrector(args.video, csv_path, corrected_csv, frame_step,
                        calib_path)
    app.run()
