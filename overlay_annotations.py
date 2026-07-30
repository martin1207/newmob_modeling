"""
Overlay des annotations VRU sur toutes les vidéos escooter.
Affiche uniquement les bbox CODÉES (encounter avec CONFIRM == 1) avec :
  - vru_type
  - gait
  - age_group
  - encounter_type (= INTERACTION_TYPE)

Inputs (par clip) :
  <ESCOOTER_DIR>/<clip>.mp4                         (image brute, pas Canny)
  <ESCOOTER_DIR>/<clip>_corrected_with_offset_gpsfixed.csv   (IMU + GPS recalé)
  <CODEBOOK>/<clip>_debug_autodetect.csv
  <CODEBOOK>/<clip>_rater<N>_encounters_debug_encounters.csv (N quelconque)
Output :
  <OUT_DIR>/<clip>_annotated_codes_raw.mp4

La vitesse affichée est la vitesse GPS Kalman (cf. kalman_speed.py), identique à
la colonne speed_kmh_kalman du dataset.
"""

import os
import re
import cv2
import csv
import glob

import numpy as np

import kalman_speed   # vitesse GPS Kalman partagée avec build_clean_dataset.py

ROOT          = "/Volumes/My Passport/NEWMOB"
ESCOOTER_DIR  = f"{ROOT}/escooter"
CODEBOOK      = f"{ROOT}/codebookescooter"
OUT_DIR       = f"{ROOT}/e_scooter_video_annotated"
LANES_DIR     = f"{ROOT}/lanes"

# Suffixes codebook : le numéro de rater est découvert dynamiquement
# (cf. find_codebook_file), donc plus de _rater2 codé en dur — les clips
# annotés par rater1 uniquement sont désormais pris en charge.
ENC_KIND       = "debug_encounters"       # <clip>_rater<N>_encounters_debug_encounters.csv
OBS_KIND       = "obstacle_zones"         # <clip>_rater<N>_encounters_obstacle_zones.csv
AUTODET_SUFFIX = "_debug_autodetect.csv"
# IMU + GPS recalé (lag GPS estimé par clip déjà appliqué). La vitesse Kalman
# est calculée à partir de Lat/Long (cf. load_speed_by_frame).
SPEED_SUFFIX   = "_corrected_with_offset_gpsfixed.csv"
# Vidéo source : image BRUTE (non-canny). Le variant Canny est "_canny.mp4".
VIDEO_SUFFIX   = ".mp4"
# Nom de sortie distinct de celui des rendus Canny (_annotated_codes.mp4), pour
# ne pas écraser les vidéos déjà produites à partir des sources Canny.
OUT_SUFFIX     = "_annotated_codes_raw.mp4"
LANES_SUFFIX   = "_corrected_lanes.csv"         # courbes curb gauche/droite
CALIB_SUFFIX   = "_calibration.json"            # calibration caméra (sténopé)

# Rendu curb : on réutilise STRICTEMENT la logique de render_corrected_lanes.py
# (clustering, interpolation polynomiale, lissage temporel, polylines+blend).
from curb_corrector import load_csv as load_lanes_csv, cluster_frame, MIN_CLUSTER_SIZE
from render_corrected_lanes import (
    SideTrack, biggest_left_right, curve_keypoints,
    COLOR_LEFT, COLOR_RIGHT, LINE_THICK,
)
CURB_DEGREE     = 2
CURB_MIN_SIZE   = MIN_CLUSTER_SIZE
CURB_SMOOTH_WIN = 3

OVERWRITE = False   # True pour recalculer même si l'output existe déjà

# ── Camembert de direction (choix par seconde, basé sur Δyaw = ∫GyrZ dt) ────
STEERING_PIE_WINDOW_S      = 1.0   # fenêtre d'évaluation du choix de direction (s)
STEERING_PIE_THRESHOLD_DEG = 6.0  # |Δyaw| sur la fenêtre < seuil ⇒ "tout droit"
                                   # sinon : 'gauche' si Δyaw > 0, sinon 'droite'
GYRZ_LEFT_POSITIVE         = True  # convention : True si GyrZ > 0 ⇒ virage à gauche
                                   # (à inverser si l'overlay annote l'inverse)

# ── Freinage (déclenchement par seconde, basé sur le ratio v(t+1)/v(t)) ─────
# Approche RELATIVE : on compare la vitesse moyenne d'une fenêtre à celle de
# la fenêtre suivante. Identique à `escooter_biogeme_avoidance.ipynb`.
BRAKE_WINDOW_S       = 1.0    # fenêtre d'évaluation (s)
BRAKE_RATIO_LOW      = 0.95   # ratio < seuil ⇒ 'brake' (chute > 5%)
ACCEL_RATIO_HIGH     = 1.05   # ratio > seuil ⇒ 'accel' (montée > 5%)
BRAKE_SPEED_MIN_KMH  = 2.0    # sous ce v(t), on neutralise (bruit GPS)

# ── Modèle de distance (bbox → mètres) ───────────────────────────────────────
# Modèle LONGITUDINAL (distance le long de l'axe). Calibré 2026-07, entraîné
# avec h SEUL (+ augmentation warp roll/pitch et bruit h) : h est naturellement
# robuste au tilt caméra, contrairement à cy/bottom_y.
DISTANCE_MODEL_PATH = (
    "/Users/martin.dejaeghere/PhD/NEWMOB-main/model/saved_models/"
    "distance_model_long_h_latest.joblib"
)
# Modèle LATÉRAL (décalage latéral du VRU, m).
LATERAL_MODEL_PATH = (
    "/Users/martin.dejaeghere/PhD/NEWMOB-main/model/saved_models/"
    "distance_model_lat_nw_latest.joblib"
)
# Ordre des features par défaut (cf. .meta.json) ; sera écrasé par la valeur
# stockée dans le bundle joblib si présente. Les bundles no-w n'utilisent que
# cx, cy, h, bottom_y, inv_h — predict() ne lit que bundle["features"].
DEFAULT_DISTANCE_FEATURES = ["cx", "cy", "w", "h", "bottom_y", "aspect", "area", "inv_h"]
_distance_bundle = None  # cache : (estimator, features) ou False si échec
_lateral_bundle = None   # cache modèle latéral


def get_distance_model():
    """Charge le bundle joblib une seule fois.

    Le fichier produit par le notebook est un dict :
        {"model": estimator, "features": [...], "metrics": {...}, ...}
    On accepte aussi le cas où l'objet sauvé serait directement un estimator.
    Renvoie (estimator, features) ou None si indisponible.
    """
    global _distance_bundle
    if _distance_bundle is None:
        try:
            import joblib  # import paresseux
            obj = joblib.load(DISTANCE_MODEL_PATH)
            if isinstance(obj, dict):
                est = obj.get("model") or obj.get("estimator") or obj.get("pipeline")
                feats = obj.get("features", DEFAULT_DISTANCE_FEATURES)
            else:
                est, feats = obj, DEFAULT_DISTANCE_FEATURES
            if est is None or not hasattr(est, "predict"):
                raise ValueError(
                    f"Aucun estimator avec .predict trouvé dans le bundle "
                    f"(clés : {list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__})"
                )
            _distance_bundle = (est, list(feats))
            print(f"[INFO] Modèle de distance chargé : "
                  f"{os.path.basename(DISTANCE_MODEL_PATH)}  "
                  f"features={_distance_bundle[1]}")
        except Exception as e:
            print(f"[WARN] Impossible de charger le modèle de distance "
                  f"({DISTANCE_MODEL_PATH}) : {e}")
            _distance_bundle = False  # marqueur : ne pas réessayer
    return _distance_bundle if _distance_bundle else None


def _bbox_feature_dict(x1, y1, x2, y2):
    """Toutes les features dérivées d'une bbox, indexées par nom."""
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return {
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "bottom_y": cy + h / 2.0,  # == y2 ; cf. notebook d'entraînement
        "aspect":   w / h,
        "area":     w * h,
        "inv_h":    1.0 / h,
    }


def predict_distances(bboxes):
    """bboxes = liste de (x1,y1,x2,y2) → liste de distances (m) ou None.

    Une seule passe `model.predict` sur tout le batch (rapide). L'ordre des
    colonnes suit `bundle["features"]` pour rester aligné avec l'entraînement.
    """
    bundle = get_distance_model()
    if bundle is None or not bboxes:
        return [None] * len(bboxes)
    estimator, feature_names = bundle
    try:
        rows = [_bbox_feature_dict(*b) for b in bboxes]
        X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
        y = estimator.predict(X)
        return [float(v) for v in y]
    except Exception as e:
        print(f"[WARN] Échec de prédiction de distance : {e}")
        return [None] * len(bboxes)


def get_lateral_model():
    """Charge le bundle joblib du modèle LATÉRAL une seule fois.
    Renvoie (estimator, features) ou None si indisponible."""
    global _lateral_bundle
    if _lateral_bundle is None:
        try:
            import joblib
            obj = joblib.load(LATERAL_MODEL_PATH)
            if isinstance(obj, dict):
                est = obj.get("model") or obj.get("estimator") or obj.get("pipeline")
                feats = obj.get("features", DEFAULT_DISTANCE_FEATURES)
            else:
                est, feats = obj, DEFAULT_DISTANCE_FEATURES
            if est is None or not hasattr(est, "predict"):
                raise ValueError("Aucun estimator avec .predict dans le bundle latéral")
            _lateral_bundle = (est, list(feats))
            print(f"[INFO] Modèle latéral chargé : "
                  f"{os.path.basename(LATERAL_MODEL_PATH)}  features={_lateral_bundle[1]}")
        except Exception as e:
            print(f"[WARN] Impossible de charger le modèle latéral "
                  f"({LATERAL_MODEL_PATH}) : {e}")
            _lateral_bundle = False
    return _lateral_bundle if _lateral_bundle else None


def predict_lateral(bboxes):
    """bboxes = liste de (x1,y1,x2,y2) → liste de décalages latéraux (m) ou None."""
    bundle = get_lateral_model()
    if bundle is None or not bboxes:
        return [None] * len(bboxes)
    estimator, feature_names = bundle
    try:
        rows = [_bbox_feature_dict(*b) for b in bboxes]
        X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
        return [float(v) for v in estimator.predict(X)]
    except Exception as e:
        print(f"[WARN] Échec de prédiction latérale : {e}")
        return [None] * len(bboxes)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Couleurs / police (constants partagés) ───────────────────────────────────
FONT  = cv2.FONT_HERSHEY_DUPLEX
BLACK = (  0,   0,   0)
COLORS = {
    "pedestrian": ( 80,  80, 255),   # rose-rouge (BGR)
    "cyclist":    (255, 210,   0),   # cyan
    "e-scooter":  (  0, 200, 255),   # orange
    "other mmv":  (180,  80, 255),   # magenta
    "motor":      ( 60,  60, 200),   # rouge sombre
    "animal":     ( 80, 255,  80),   # vert
    "stationary": (180, 180, 180),   # gris
    "unknown":    (160, 160, 160),
}
DEFAULT_COLOR = (200, 200, 200)

# ── Code → label (cf. build_clean_dataset.py) ────────────────────────────────
CODE_LABELS = {
    "VRU_TYPE":         {"1": "Pedestrian", "2": "Cyclist", "3": "E-scooter",
                         "4": "Other MMV",  "5": "Motor",   "6": "Animal",
                         "7": "Stationary", "9": "Unknown"},
    "INTERACTION_TYPE": {"1": "Same-direction", "2": "Opposite-direction",
                         "3": "Crossing",       "4": "Stationary", "9": "Unknown"},
    "VRU_AGE_GROUP":    {"1": "Child", "2": "Adult", "3": "Elderly", "9": "Unknown"},
    "VRU_GAIT":         {"1": "Standing", "2": "Walking", "3": "Running", "9": "Unknown"},
}


def label_of(field, raw):
    """Code numérique → label. Si déjà du texte, retourné tel quel."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:  
        return ""
    return CODE_LABELS.get(field, {}).get(s, s)


def _open_dictreader(path):
    """csv.DictReader avec détection auto du séparateur (',' ou ';')."""
    f = open(path, newline="")
    sample = f.readline()
    f.seek(0)
    delim = ";" if sample.count(";") > sample.count(",") else ","
    return f, csv.DictReader(f, delimiter=delim)


def find_codebook_file(clip_name, kind):
    """Chemin du CSV codebook `<clip>_rater<N>_encounters_<kind>.csv`, QUEL QUE
    SOIT le numéro de rater. En cas de plusieurs raters, on préfère le numéro le
    plus élevé (rater2 > rater1), comme build_clean_dataset.py. Tolère aussi une
    variante sans numéro de rater. Retourne '' si aucun fichier n'existe."""
    cands = glob.glob(os.path.join(CODEBOOK, f"{clip_name}_rater*_encounters_{kind}.csv"))
    cands += glob.glob(os.path.join(CODEBOOK, f"{clip_name}_encounters_{kind}.csv"))
    cands = [p for p in cands if not os.path.basename(p).startswith("._")]
    if not cands:
        return ""

    def _rater_num(p):
        m = re.search(r"_rater(\d+)_", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return sorted(cands, key=_rater_num)[-1]


def load_enc_meta(enc_csv):
    """{(track_id, frame): meta} pour les encounters CONFIRM == 1."""
    enc_meta = {}
    f, reader = _open_dictreader(enc_csv)
    with f:
        for row in reader:
            if row.get("CONFIRM", "").strip() != "1":
                continue
            try:
                f0 = int(row["FRAME_START"])
                f1 = int(row["FRAME_END"])
            except (ValueError, KeyError):
                continue
            meta = {
                "itype": label_of("INTERACTION_TYPE", row.get("INTERACTION_TYPE")),
                "gait":  label_of("VRU_GAIT",         row.get("VRU_GAIT")),
                "age":   label_of("VRU_AGE_GROUP",    row.get("VRU_AGE_GROUP")),
                "vtype": label_of("VRU_TYPE",         row.get("VRU_TYPE")),
            }
            track_ids = [row["PRIMARY_TRACK_ID"].strip()]
            linked = row.get("LINKED_TRACKS", "").strip()
            if linked:
                track_ids += [t.strip() for t in linked.split(",") if t.strip()]
            for fr in range(f0, f1 + 1):
                for tid in track_ids:
                    enc_meta[(tid, fr)] = meta
    return enc_meta


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_speed_by_frame(speed_csv):
    """{frame: vitesse Kalman (km/h)} — vitesse GPS lissée (kalman_speed.gps_speed_kmh)
    calculée à partir de Lat/Long, puis moyennée par frame vidéo.

    C'est STRICTEMENT la même vitesse que la colonne speed_kmh_kalman du dataset
    (build_clean_dataset.py) : la vitesse n'est stockée dans aucun CSV, elle est
    recalculée ici depuis les positions GPS recalées du fichier gpsfixed.
    """
    speed = {}
    if not os.path.exists(speed_csv):
        return speed
    frames, ts, lat, lon = [], [], [], []
    f, reader = _open_dictreader(speed_csv)
    with f:
        for row in reader:
            try:
                fr = int(float(row["frame"]))
                t  = float(row["TimeStamp"])
            except (ValueError, KeyError, TypeError):
                continue                       # ligne inexploitable (pas de frame/horodatage)
            frames.append(fr); ts.append(t)
            lat.append(_to_float(row.get("Lat")))
            lon.append(_to_float(row.get("Long")))
    if not frames:
        return speed
    v = kalman_speed.gps_speed_kmh(ts, lat, lon)     # km/h, aligné ligne à ligne (NaN si pas de GPS)
    agg = {}
    for fr, vi in zip(frames, v):
        if np.isfinite(vi):
            agg.setdefault(fr, []).append(vi)
    return {fr: float(np.mean(vs)) for fr, vs in agg.items()}


def load_gyrz_by_frame(speed_csv):
    """{frame: GyrZ moyen (deg/s)} — moyenne des échantillons IMU rattachés à la frame.

    Le CSV IMU peut contenir plusieurs lignes par frame vidéo : on agrège par
    moyenne pour avoir un signal aligné au framerate vidéo avant filtrage.
    """
    raw = {}
    if not os.path.exists(speed_csv):
        return {}
    f, reader = _open_dictreader(speed_csv)
    with f:
        for row in reader:
            try:
                fr = int(float(row["frame"]))
                g  = float(row["GyrZ(deg/s)"])
            except (ValueError, KeyError, TypeError):
                continue
            raw.setdefault(fr, []).append(g)
    return {fr: float(np.mean(vs)) for fr, vs in raw.items()}


def compute_steering_choices_per_second(gyrz_by_frame, fps,
                                        window_s=STEERING_PIE_WINDOW_S,
                                        threshold_deg=STEERING_PIE_THRESHOLD_DEG,
                                        left_positive=GYRZ_LEFT_POSITIVE):
    """{frame: ('straight'|'left'|'right', delta_yaw_deg)} — choix par fenêtre.

    On découpe le signal GyrZ en fenêtres consécutives de `window_s` secondes,
    et pour chaque fenêtre on calcule Δyaw = ∫ (GyrZ − biais) dt sur la fenêtre.
    Classification :
      • |Δyaw| < `threshold_deg`           → 'straight'
      • Δyaw > 0  (avec gauche = positif)  → 'left'
      • Δyaw < 0                           → 'right'

    Le choix est ensuite "tenu" sur toutes les frames de la fenêtre — le
    camembert affiche donc le même secteur pendant ~1 s puis change.

    Débiaisage : on retire la médiane globale du signal (robuste : la médiane
    n'est pas tirée par les virages, contrairement à la moyenne).
    """
    if not gyrz_by_frame:
        return {}
    frames = sorted(gyrz_by_frame)
    fmin, fmax = frames[0], frames[-1]
    n = fmax - fmin + 1
    arr = np.full(n, np.nan)
    for fr, g in gyrz_by_frame.items():
        arr[fr - fmin] = g
    # Interpolation linéaire des frames manquantes
    if np.isnan(arr).any():
        idx = np.arange(n)
        ok  = ~np.isnan(arr)
        if ok.sum() >= 2:
            arr = np.interp(idx, idx[ok], arr[ok])
        else:
            arr = np.nan_to_num(arr)
    # Débiaisage par la médiane + signe (gauche = positif)
    arr = arr - float(np.median(arr))
    if not left_positive:
        arr = -arr

    # Découpage en fenêtres de window_s ≈ frames_per_window frames
    frames_per_window = max(1, int(round(window_s * fps)))
    dt = 1.0 / max(fps, 1e-6)

    choices = {}
    for i0 in range(0, n, frames_per_window):
        i1 = min(i0 + frames_per_window, n)
        delta_yaw = float(np.sum(arr[i0:i1]) * dt)   # intégration trapézoïdale ≈
        if abs(delta_yaw) < threshold_deg:
            label = 'straight'
        elif delta_yaw > 0:
            label = 'left'
        else:
            label = 'right'
        for i in range(i0, i1):
            choices[fmin + i] = (label, delta_yaw)
    return choices


def count_steering_maneuvers(choices_by_frame):
    """(n_left, n_right) — nombre de fenêtres distinctes classées 'left' / 'right'."""
    n_left = n_right = 0
    last_label = None
    for fr in sorted(choices_by_frame):
        label = choices_by_frame[fr][0]
        if label != last_label:
            if   label == 'left':  n_left  += 1
            elif label == 'right': n_right += 1
        last_label = label
    return n_left, n_right


def compute_brake_choices_per_second(speed_by_frame, fps,
                                     window_s=BRAKE_WINDOW_S,
                                     ratio_low=BRAKE_RATIO_LOW,
                                     ratio_high=ACCEL_RATIO_HIGH,
                                     vmin_kmh=BRAKE_SPEED_MIN_KMH):
    """{frame: (label, ratio)} — choix de freinage par fenêtre de `window_s`.

    On découpe le signal de vitesse en fenêtres consécutives de `window_s` s,
    et on compare la vitesse moyenne d'une fenêtre à celle de la suivante :
        ratio = v(window_{t+1}) / v(window_t)
      • ratio < `ratio_low`  → 'brake'   (chute > 5% par défaut)
      • ratio > `ratio_high` → 'accel'   (montée > 5%)
      • sinon                → 'cruise'

    Si v(window_t) < `vmin_kmh` (≈ à l'arrêt, où le bruit GPS domine), la
    fenêtre est forcée à 'cruise' avec ratio = 1.0.
    """
    if not speed_by_frame:
        return {}
    frames = sorted(speed_by_frame)
    fmin, fmax = frames[0], frames[-1]
    n = fmax - fmin + 1
    arr = np.full(n, np.nan)
    for fr, v in speed_by_frame.items():
        arr[fr - fmin] = v
    if np.isnan(arr).any():
        idx = np.arange(n)
        ok  = ~np.isnan(arr)
        if ok.sum() >= 2:
            arr = np.interp(idx, idx[ok], arr[ok])
        else:
            arr = np.nan_to_num(arr)

    frames_per_window = max(1, int(round(window_s * fps)))
    # Vitesse moyenne de chaque fenêtre
    window_means = []
    window_ranges = []
    for i0 in range(0, n, frames_per_window):
        i1 = min(i0 + frames_per_window, n)
        window_means.append(float(np.mean(arr[i0:i1])))
        window_ranges.append((i0, i1))

    choices = {}
    for w, (i0, i1) in enumerate(window_ranges):
        v_t = window_means[w]
        v_next = window_means[w + 1] if w + 1 < len(window_means) else v_t
        if v_t < vmin_kmh:
            label, ratio = 'cruise', 1.0
        else:
            ratio = v_next / v_t
            if ratio < ratio_low:
                label = 'brake'
            elif ratio > ratio_high:
                label = 'accel'
            else:
                label = 'cruise'
        for i in range(i0, i1):
            choices[fmin + i] = (label, ratio)
    return choices


def count_brake_events(choices_by_frame):
    """(n_brake, n_accel) — nombre de fenêtres distinctes classées 'brake' / 'accel'."""
    n_brake = n_accel = 0
    last_label = None
    for fr in sorted(choices_by_frame):
        label = choices_by_frame[fr][0]
        if label != last_label:
            if   label == 'brake': n_brake += 1
            elif label == 'accel': n_accel += 1
        last_label = label
    return n_brake, n_accel


def build_curb_tracks(lanes_csv, H, W,
                      degree=CURB_DEGREE, min_size=CURB_MIN_SIZE,
                      smooth_win=CURB_SMOOTH_WIN):
    """(left_tr, right_tr) prêts à interroger via .at(frame_idx), ou (None, None).

    Réplique exacte du pipeline de render_corrected_lanes.render_clip :
    clusters par frame échantillonnée → plus gros cluster gauche/droite
    → courbe polynomiale x=f(y) → lissage temporel par SideTrack.
    """
    if not os.path.exists(lanes_csv):
        return None, None
    frame_pixels = load_lanes_csv(lanes_csv)
    sampled = sorted(frame_pixels.keys())
    if not sampled:
        return None, None
    left_tr, right_tr = SideTrack(), SideTrack()
    for fid in sampled:
        clusters = cluster_frame(frame_pixels[fid], H, W, min_size)
        left, right = biggest_left_right(clusters, W)
        left_tr.add(fid,  curve_keypoints(left,  degree) if left  else None)
        right_tr.add(fid, curve_keypoints(right, degree) if right else None)
    left_tr.finalize(smooth_win)
    right_tr.finalize(smooth_win)
    if not left_tr.any and not right_tr.any:
        return None, None
    return left_tr, right_tr


def draw_curbs(frame, frame_idx, left_tr, right_tr):
    """Trace les curbs gauche/droite à la frame `frame_idx` (mêmes couleurs,
    épaisseur et blend 0.65/0.35 que render_corrected_lanes.py)."""
    if left_tr is None and right_tr is None:
        return frame
    overlay = frame.copy()
    drawn = False
    for tr, color in ((left_tr, COLOR_LEFT), (right_tr, COLOR_RIGHT)):
        if tr is None:
            continue
        curve = tr.at(frame_idx)
        if curve is not None:
            cv2.polylines(overlay, [curve], False, color,
                          LINE_THICK, cv2.LINE_AA)
            drawn = True
    if not drawn:
        return frame
    return cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)


def load_obstacle_zones(obs_csv):
    """Liste de (frame_start, frame_end, zone_id) ou [] si fichier absent."""
    zones = []
    if not os.path.exists(obs_csv):
        return zones
    f, reader = _open_dictreader(obs_csv)
    with f:
        for row in reader:
            try:
                f0 = int(row["FRAME_START"])
                f1 = int(row["FRAME_END"])
            except (ValueError, KeyError):
                continue
            zones.append((f0, f1, row.get("ZONE_ID", "").strip()))
    return zones


def obstacle_id_at(frame_idx, zones):
    for f0, f1, zid in zones:
        if f0 <= frame_idx <= f1:
            return zid or "OBS"
    return None


def load_vrus_by_frame(autodet_csv, enc_meta):
    """{frame: [(x1,y1,x2,y2,meta,dist), ...]} pour les bbox codées uniquement.

    `dist` est la distance (m) prédite par le modèle joblib, ou None si
    le modèle n'est pas chargeable / la prédiction échoue.
    """
    # 1) Collecte des bbox + meta + (frame, index) pour pouvoir réinjecter
    #    les distances après une prédiction batchée.
    raw_entries = []  # liste de (frame, x1, y1, x2, y2, meta)
    f, reader = _open_dictreader(autodet_csv)
    with f:
        for row in reader:
            try:
                fr  = int(float(row["frame"]))
                tid = row["track_id"].strip()
            except (ValueError, KeyError):
                continue
            meta = enc_meta.get((tid, fr))
            if meta is None:
                continue
            try:
                fx = float(row["foot_x"])
                fy = float(row["foot_y"])
                bh = float(row["bbox_height"])
            except (ValueError, KeyError):
                continue
            bw = bh * 0.45
            x1 = int(fx - bw / 2)
            y1 = int(fy - bh)
            x2 = int(fx + bw / 2)
            y2 = int(fy)
            raw_entries.append((fr, x1, y1, x2, y2, meta))

    # 2) Prédiction batchée des distances (1 seul appel model.predict)
    bboxes = [(e[1], e[2], e[3], e[4]) for e in raw_entries]
    distances = predict_distances(bboxes)

    # 3) Regroupement par frame avec la distance attachée
    vrus_by_frame = {}
    for (fr, x1, y1, x2, y2, meta), dist in zip(raw_entries, distances):
        vrus_by_frame.setdefault(fr, []).append((x1, y1, x2, y2, meta, dist))
    return vrus_by_frame


def draw_box(frame, x1, y1, x2, y2, meta, dist=None):
    color = COLORS.get(meta["vtype"].lower(), DEFAULT_COLOR)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    dist_str = f"{dist:.1f} m" if dist is not None else "-"
    lines = [
        f"vru_type: {meta['vtype'] or '-'}",
        f"gait: {meta['gait'] or '-'}",
        f"age: {meta['age'] or '-'}",
        f"enc: {meta['itype'] or '-'}",
        f"dist: {dist_str}",
    ]
    scale, thick, pad = 0.50, 1, 3
    sizes  = [cv2.getTextSize(l, FONT, scale, thick) for l in lines]
    box_w  = max(s[0][0] for s in sizes) + pad * 2
    line_h = sizes[0][0][1]
    box_h  = (line_h + pad) * len(lines) + pad

    lx = x1
    ly = max(y1 - box_h - 4, 0)
    cv2.rectangle(frame, (lx, ly), (lx + box_w, ly + box_h), color, -1)
    for i, (line, ((_, lh), _)) in enumerate(zip(lines, sizes)):
        ty = ly + pad + lh + i * (lh + pad)
        cv2.putText(frame, line, (lx + pad, ty), FONT, scale, BLACK, thick, cv2.LINE_AA)


def draw_frame_number(frame, frame_idx, total):
    text = f"frame {frame_idx}/{total}"
    scale, thick, pad = 0.7, 2, 8
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = 20, 20 + th
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), BLACK, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_speed(frame, speed_kmh):
    text = f"{speed_kmh:.1f} km/h"
    scale, thick, pad = 1.0, 2, 10
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = 20, 80 + th
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), BLACK, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, (0, 220, 0), thick, cv2.LINE_AA)


# Couleurs BGR du badge de freinage
BRAKE_BADGE_COLORS = {
    'brake':  ( 80,  80, 255),   # rouge
    'cruise': (180, 180, 180),   # gris clair
    'accel':  (  0, 220,   0),   # vert
}


def draw_brake_indicator(frame, choice):
    """Badge de freinage juste sous la vitesse.

    `choice` est soit None, soit (label, ratio) avec label ∈
    {'brake','cruise','accel'} et ratio = v(t+1)/v(t) sur la fenêtre.
    """
    if choice is None:
        return
    label, ratio = choice
    label_up = label.upper()
    color    = BRAKE_BADGE_COLORS.get(label, (200, 200, 200))
    # Sous-titre : pourcentage de variation
    pct = (ratio - 1.0) * 100
    sub = f"v(t+1)/v(t) = {ratio:.2f}  ({pct:+.0f}%)"
    scale_a, scale_b, thick, pad = 0.7, 0.45, 2, 8
    (tw_a, th_a), _ = cv2.getTextSize(label_up, FONT, scale_a, thick)
    (tw_b, th_b), _ = cv2.getTextSize(sub, FONT, scale_b, 1)
    tw = max(tw_a, tw_b)
    x  = 20
    y1 = 80 + 30 + th_a + pad           # juste sous draw_speed
    y2 = y1 + th_b + pad
    cv2.rectangle(frame, (x - pad, y1 - th_a - pad),
                  (x + tw + pad, y2 + pad), BLACK, -1)
    cv2.putText(frame, label_up, (x, y1), FONT, scale_a, color, thick, cv2.LINE_AA)
    cv2.putText(frame, sub,      (x, y2), FONT, scale_b,
                (220, 220, 220), 1, cv2.LINE_AA)


def draw_steering_pie(frame, choice):
    """Quart de cadran à 3 secteurs (apex en bas, éventail ouvert vers le haut).

    `choice` est soit None, soit ('straight'|'left'|'right', delta_yaw_deg).
    Éventail de 90° centré sur la verticale "up" (secteur OpenCV 225° → 315°),
    découpé en 3 secteurs de 30° :
      • LEFT     (225° → 255°)
      • STRAIGHT (255° → 285°)
      • RIGHT    (285° → 315°)
    Le secteur "choisi" est rempli de couleur vive ; les autres restent grisés.
    """
    w  = frame.shape[1]
    r  = 90
    cx = w // 2
    cy = r + 22                  # apex sous la courbe → l'éventail tient en haut

    # En OpenCV : angle 0° = +x (droite), 90° = +y (bas), 270° = haut
    sectors = [
        ('left',     225, 255),
        ('straight', 255, 285),
        ('right',    285, 315),
    ]
    COLOR_ON = {
        'straight': ( 80, 220,  80),   # vert
        'left':     (255, 200,   0),   # cyan-bleu (BGR)
        'right':    (  0, 165, 255),   # orange
    }
    COLOR_OFF = (55, 55, 55)

    label = None if choice is None else choice[0]

    # Halo noir (lisibilité sur n'importe quel fond)
    cv2.ellipse(frame, (cx, cy), (r + 5, r + 5), 0, 225, 315, BLACK,
                -1, cv2.LINE_AA)

    # Remplissage des 3 secteurs
    for name, a0, a1 in sectors:
        color = COLOR_ON[name] if name == label else COLOR_OFF
        cv2.ellipse(frame, (cx, cy), (r, r), 0, a0, a1, color, -1, cv2.LINE_AA)

    # Séparateurs (rayons depuis l'apex) + arc extérieur
    for a in (225, 255, 285, 315):
        rad = np.deg2rad(a)
        x2 = cx + int(r * np.cos(rad))
        y2 = cy + int(r * np.sin(rad))
        cv2.line(frame, (cx, cy), (x2, y2), (220, 220, 220), 2, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 225, 315, (220, 220, 220),
                2, cv2.LINE_AA)

    # Petite flèche dans chaque secteur, au centre angulaire (rayon ~0.62 r)
    arrow_color = (240, 240, 240)
    rr  = int(r * 0.62)
    arm = 16
    # LEFT : flèche vers la gauche, centrée sur l'angle 240°
    px = cx + int(rr * np.cos(np.deg2rad(240)))
    py = cy + int(rr * np.sin(np.deg2rad(240)))
    cv2.arrowedLine(frame, (px + arm, py), (px - arm, py),
                    arrow_color, 3, cv2.LINE_AA, tipLength=0.5)
    # STRAIGHT : flèche vers le haut, centrée sur l'angle 270°
    px = cx
    py = cy + int(rr * np.sin(np.deg2rad(270)))   # cy - rr
    cv2.arrowedLine(frame, (px, py + arm), (px, py - arm),
                    arrow_color, 3, cv2.LINE_AA, tipLength=0.5)
    # RIGHT : flèche vers la droite, centrée sur l'angle 300°
    px = cx + int(rr * np.cos(np.deg2rad(300)))
    py = cy + int(rr * np.sin(np.deg2rad(300)))
    cv2.arrowedLine(frame, (px - arm, py), (px + arm, py),
                    arrow_color, 3, cv2.LINE_AA, tipLength=0.5)

    # Bandeau texte sous l'éventail : choix courant + Δyaw
    if choice is not None:
        name, dyaw = choice
        txt = f"{name.upper()}  ({dyaw:+.0f} deg / {STEERING_PIE_WINDOW_S:.0f}s)"
    else:
        txt = "-"
    (tw, th), _ = cv2.getTextSize(txt, FONT, 0.55, 1)
    bx, by = cx - tw // 2, cy + 22
    cv2.rectangle(frame, (bx - 6, by - th - 4), (bx + tw + 6, by + 4),
                  BLACK, -1)
    cv2.putText(frame, txt, (bx, by), FONT, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


# ── Mini-carte « bird-eye » (vue de dessus) ──────────────────────────────────
# Projette curbs + piétons dans un repère sol (X = latéral, Z = profondeur)
# via le modèle sténopé + sol plan de la calibration caméra du clip.
MINIMAP_W           = 260      # largeur du panneau (px)
MINIMAP_H           = 300      # hauteur du panneau (px)
MINIMAP_MARGIN      = 20       # marge au coin haut-droit (px)
MINIMAP_RANGE_M     = 25.0     # profondeur affichée vers l'avant (m)
MINIMAP_HALFWIDTH_M = 12.0     # demi-largeur latérale affichée (m)
MINIMAP_RINGS_M     = (5.0, 10.0, 15.0, 20.0)   # anneaux de distance


def load_calibration(path):
    """Charge le JSON de calibration caméra → dict normalisé, ou None.

    Champs utilisés : focale f (px), point principal (cx, cy), hauteur
    caméra h_cam (m) et pitch (deg). Modèle sténopé + hypothèse sol plan.
    """
    if not os.path.exists(path):
        return None
    try:
        import json  # import paresseux
        with open(path) as fh:
            c = json.load(fh)
        return {
            "f":     float(c.get("f") or c.get("focal_length_px")),
            "cx":    float(c.get("cx", 0.0)),
            "cy":    float(c.get("cy", 0.0)),
            "h_cam": float(c.get("h_cam") or c.get("camera_height_m")),
            "pitch": np.deg2rad(float(c.get("pitch_deg", 0.0))),
        }
    except Exception as e:
        print(f"[WARN] Calibration illisible ({os.path.basename(path)}) : {e}")
        return None


def ground_point(u, v, calib):
    """Pixel (u, v) supposé au sol → (X, Z) en mètres, ou None.

    X = position latérale (droite positive), Z = profondeur vers l'avant.
    Renvoie None si le rayon vise l'horizon ou le ciel (pas d'intersection
    avec le sol). Modèle : caméra sténopé à hauteur h_cam, pitch autour de
    l'axe latéral, sol plan Y = 0.
    """
    xc = (u - calib["cx"]) / calib["f"]
    yc = (v - calib["cy"]) / calib["f"]
    p  = calib["pitch"]
    denom = yc * np.cos(p) - np.sin(p)
    if denom <= 1e-4:                       # rayon vers l'horizon / le ciel
        return None
    Z = calib["h_cam"] * (yc * np.sin(p) + np.cos(p)) / denom
    X = calib["h_cam"] * xc / denom
    return (X, Z)


def draw_minimap(frame, calib, vrus, left_tr, right_tr, frame_idx):
    """Fenêtre vue de dessus en haut à droite.

    - Curbs gauche/droite projetés au sol via la calibration.
    - Piétons placés par distance (modèle joblib) et azimut latéral
      (calculé depuis le centre cx de la box et la focale).
    - Marqueur ego (trottinette), anneaux de distance et cône de champ.
    """
    if calib is None:
        return
    H, W = frame.shape[:2]
    x0 = W - MINIMAP_MARGIN - MINIMAP_W
    y0 = MINIMAP_MARGIN
    if x0 < 0 or y0 + MINIMAP_H > H:        # frame trop petite : on s'abstient
        return

    # Fond semi-transparent + cadre + bandeau titre
    sub = frame[y0:y0 + MINIMAP_H, x0:x0 + MINIMAP_W]
    cv2.addWeighted(np.zeros_like(sub), 0.55, sub, 0.45, 0, sub)
    cv2.rectangle(frame, (x0, y0), (x0 + MINIMAP_W, y0 + MINIMAP_H),
                  (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + MINIMAP_W, y0 + 20), (60, 60, 60), -1)
    cv2.putText(frame, "BIRD-EYE VIEW", (x0 + 8, y0 + 15),
                FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Repère écran : ego en bas-centre, Z (profondeur) vers le haut
    map_cx = x0 + MINIMAP_W // 2
    ego_y  = y0 + MINIMAP_H - 16
    top_y  = y0 + 30
    sz = (ego_y - top_y) / MINIMAP_RANGE_M               # px / m (profondeur)
    sx = (MINIMAP_W / 2 - 10) / MINIMAP_HALFWIDTH_M      # px / m (latéral)

    def w2m(X, Z):
        return (int(round(map_cx + X * sx)), int(round(ego_y - Z * sz)))

    # Anneaux de distance + étiquettes
    for r in MINIMAP_RINGS_M:
        if r > MINIMAP_RANGE_M:
            continue
        ry = int(round(ego_y - r * sz))
        cv2.line(frame, (x0 + 2, ry), (x0 + MINIMAP_W - 2, ry),
                 (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(r)}m", (x0 + 4, ry - 3),
                    FONT, 0.32, (130, 130, 130), 1, cv2.LINE_AA)
    cv2.line(frame, (map_cx, top_y), (map_cx, ego_y),
             (70, 70, 70), 1, cv2.LINE_AA)               # axe central

    # Cône de champ de vision (HFOV déduit de la focale)
    hfov = 2.0 * np.arctan(W / (2.0 * calib["f"]))
    for s in (-1, 1):
        fx = MINIMAP_RANGE_M * np.tan(s * hfov / 2.0)
        cv2.line(frame, (map_cx, ego_y), w2m(fx, MINIMAP_RANGE_M),
                 (90, 90, 110), 1, cv2.LINE_AA)

    # Curbs projetés au sol (mêmes couleurs que l'overlay caméra)
    for tr, color in ((left_tr, COLOR_LEFT), (right_tr, COLOR_RIGHT)):
        if tr is None:
            continue
        curve = tr.at(frame_idx)
        if curve is None:
            continue
        pts = []
        for u, v in curve:
            g = ground_point(float(u), float(v), calib)
            if g is None:
                continue
            X, Z = g
            if not (0.0 < Z <= MINIMAP_RANGE_M):
                continue
            px, py = w2m(X, Z)
            px = min(max(px, x0 + 2), x0 + MINIMAP_W - 2)
            pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, np.int32)], False,
                          color, 2, cv2.LINE_AA)

    # Piétons : rayon = distance (modèle), azimut = centre cx de la box
    for (bx1, by1, bx2, by2, meta, dist) in vrus:
        if dist is None:
            continue
        bcx   = (bx1 + bx2) / 2.0
        theta = np.arctan2(bcx - calib["cx"], calib["f"])   # + = vers la droite
        X = dist * np.sin(theta)
        Z = min(dist * np.cos(theta), MINIMAP_RANGE_M)
        px, py = w2m(X, Z)
        px = min(max(px, x0 + 4), x0 + MINIMAP_W - 4)
        py = min(max(py, top_y),  ego_y)
        color = COLORS.get(meta["vtype"].lower(), DEFAULT_COLOR)
        cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, BLACK, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{dist:.0f}m", (px + 7, py + 4),
                    FONT, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    # Marqueur ego (trottinette) — triangle pointant vers l'avant
    tri = np.array([(map_cx, ego_y - 11), (map_cx - 8, ego_y + 6),
                    (map_cx + 8, ego_y + 6)], np.int32)
    cv2.fillPoly(frame, [tri], (0, 200, 255), lineType=cv2.LINE_AA)
    cv2.polylines(frame, [tri], True, BLACK, 1, cv2.LINE_AA)


def draw_obstacle_banner(frame, zone_id):
    h, w = frame.shape[:2]
    text = f"OBSTACLE [{zone_id}]"
    scale, thick, pad = 1.1, 3, 12
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x = w - tw - pad * 2 - 20
    y = 20 + th
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), (0, 0, 255), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def process_clip(clip_name):
    video_in    = f"{ESCOOTER_DIR}/{clip_name}{VIDEO_SUFFIX}"
    autodet_csv = f"{CODEBOOK}/{clip_name}{AUTODET_SUFFIX}"
    enc_csv     = find_codebook_file(clip_name, ENC_KIND)   # rater découvert dynamiquement
    obs_csv     = find_codebook_file(clip_name, OBS_KIND)
    speed_csv   = f"{ESCOOTER_DIR}/{clip_name}{SPEED_SUFFIX}"
    lanes_csv   = f"{LANES_DIR}/{clip_name}{LANES_SUFFIX}"
    video_out   = f"{OUT_DIR}/{clip_name}{OUT_SUFFIX}"

    missing = [p for p in (video_in, autodet_csv, enc_csv) if not os.path.exists(p)]
    if missing:
        print(f"[SKIP] {clip_name} — manquants: {[os.path.basename(m) for m in missing]}")
        return

    if os.path.exists(video_out) and not OVERWRITE:
        print(f"[SKIP] {clip_name} — déjà annoté")
        return

    enc_meta       = load_enc_meta(enc_csv)
    vrus_by_frame  = load_vrus_by_frame(autodet_csv, enc_meta)
    obstacles      = load_obstacle_zones(obs_csv)
    speed_by_frame = load_speed_by_frame(speed_csv)
    gyrz_by_frame  = load_gyrz_by_frame(speed_csv)
    n_boxes = sum(len(v) for v in vrus_by_frame.values())
    if n_boxes == 0 and not obstacles:
        print(f"[SKIP] {clip_name} — aucun encounter confirmé ni obstacle")
        return

    cap    = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"[ERR ] {clip_name} — vidéo illisible")
        return
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Choix de direction par fenêtre de 1 s (straight / left / right) à partir
    # du yaw intégré sur la fenêtre.
    choices_by_frame = compute_steering_choices_per_second(gyrz_by_frame, fps)
    n_left, n_right  = count_steering_maneuvers(choices_by_frame)

    # Déclenchement de freinage par fenêtre de 1 s à partir du ratio v(t+1)/v(t).
    brake_choices_by_frame = compute_brake_choices_per_second(speed_by_frame, fps)
    n_brake, n_accel       = count_brake_events(brake_choices_by_frame)

    # Curb tracks (seulement si le CSV corrected_lanes existe pour ce clip)
    left_curb, right_curb = build_curb_tracks(lanes_csv, height, width)
    curb_status = "on" if (left_curb is not None or right_curb is not None) else "off"

    # Calibration caméra → mini-carte vue de dessus (bird-eye)
    calib = load_calibration(f"{CODEBOOK}/{clip_name}{CALIB_SUFFIX}")
    minimap_status = "on" if calib is not None else "off"

    print(f"[RUN ] {clip_name}  {width}×{height} @ {fps:.1f}fps  {total}f  "
          f"bbox={n_boxes}  obs_zones={len(obstacles)}  "
          f"speed_pts={len(speed_by_frame)}  steer L/R={n_left}/{n_right}  "
          f"brake/accel={n_brake}/{n_accel}  "
          f"curb={curb_status}  minimap={minimap_status}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Curbs en premier : blend transparent sur la frame brute, ensuite
        # les autres annotations (bbox, bandeau, camembert…) sont dessinées
        # par-dessus, donc à pleine opacité.
        frame = draw_curbs(frame, frame_idx, left_curb, right_curb)
        frame_vrus = vrus_by_frame.get(frame_idx, [])
        for (x1, y1, x2, y2, meta, dist) in frame_vrus:
            draw_box(frame, x1, y1, x2, y2, meta, dist)
        # Mini-carte vue de dessus (avant le bandeau obstacle, qui reste
        # prioritaire à l'affichage en cas de chevauchement au coin droit).
        draw_minimap(frame, calib, frame_vrus, left_curb, right_curb, frame_idx)
        zid = obstacle_id_at(frame_idx, obstacles)
        if zid is not None:
            draw_obstacle_banner(frame, zid)
        # Camembert de direction (toujours dessiné, même si pas de signal IMU)
        draw_steering_pie(frame, choices_by_frame.get(frame_idx))
        draw_frame_number(frame, frame_idx, total)
        v = speed_by_frame.get(frame_idx)
        if v is not None:
            draw_speed(frame, v)
        # Badge de freinage juste sous la vitesse
        draw_brake_indicator(frame, brake_choices_by_frame.get(frame_idx))
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] {clip_name} → {os.path.basename(video_out)}")


def discover_clips():
    """Clips ayant à la fois encounters + autodetect + vidéo Canny.

    Découverte indépendante du numéro de rater : on capture
    `<clip>_rater<N>_encounters_debug_encounters.csv` pour tout N (et la variante
    sans rater), puis on déduplique par nom de clip."""
    enc_files = (
        glob.glob(f"{CODEBOOK}/*_rater*_encounters_{ENC_KIND}.csv")
        + glob.glob(f"{CODEBOOK}/*_encounters_{ENC_KIND}.csv")
    )
    clips = set()
    for p in enc_files:
        base = os.path.basename(p)
        if base.startswith("._"):
            continue
        name = re.sub(rf"_rater\d+_encounters_{ENC_KIND}\.csv$", "", base)
        name = re.sub(rf"_encounters_{ENC_KIND}\.csv$", "", name)
        if (os.path.exists(f"{CODEBOOK}/{name}{AUTODET_SUFFIX}")
                and os.path.exists(f"{ESCOOTER_DIR}/{name}{VIDEO_SUFFIX}")):
            clips.add(name)
    return sorted(clips)


if __name__ == "__main__":
    clips = discover_clips()
    print(f"{len(clips)} clip(s) à traiter\n")
    for i, clip in enumerate(clips, 1):
        print(f"--- [{i}/{len(clips)}] ---")
        try:
            process_clip(clip)
        except Exception as e:
            print(f"[ERR ] {clip} — {e}")
    print(f"\nFini → {OUT_DIR}")
