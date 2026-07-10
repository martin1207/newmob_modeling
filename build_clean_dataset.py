"""
build_clean_dataset.py
─────────────────────────────────────────────────────────────────────────────
Génère un dataset propre à partir des fichiers debug_encounters + IMU
(e-scooter), enrichi de :
  - la largeur de route perpendiculaire au trajet (road.gpkg)
  - les infos rider (genre, age, experience, nb trajets, distance)
    issues du fichier participants Excel
  - les variables temporelles issues du timestamp IMU / nom de fichier

Source principale :
  - debug_encounters uniquement
    → expansion frame-level à partir des intervalles [FRAME_START, FRAME_END]

Filtres appliqués :
  1. CONFIRM == 1 dans debug_encounters
  2. Exclusion des frames chevauchant des zones d'obstacle
  3. (Plus d'offset GPS fixe : le décalage GPS est désormais estimé par clip
     et déjà appliqué en amont dans *_corrected_with_offset_gpsfixed.csv.)

Vitesse :
  - speed_kmh        : VitGPS(km/h) du téléphone (sous-estime ~20 %)
  - speed_kmh_kalman : vitesse GPS lissée par filtre de Kalman position
    (CV 2D) + lisseur RTS, cf. speed/compare_speed_scooter_gps.ipynb §2
    (référence validée contre le compteur du scooter, r≈0.86)

Virages (turn_label) :
  Les virages ne sont PLUS exclus via le GyrZ. Ils sont détectés à partir du
  GPS (changement de direction des segments de route map-matchés / simplifiés)
  et étiquetés dans la colonne `turn_label` :
    none | before_left_turn | after_left_turn | before_right_turn | after_right_turn
  Un virage = angle entre deux segments consécutifs > +80° (droite) ou < −80°
  (gauche) ; les points à moins de 30 m (le long de la trace) sont étiquetés.

Variables frame-level ajoutées depuis debug_encounters :
  - n_vru_total
  - proportions de VRU_TYPE
  - proportions de INTERACTION_TYPE
  - labels contextuels (gait, age group, weather, etc.)

Variables de direction ajoutées depuis l'IMU (cf. overlay_annotations.py) :
  - steering_choice : 'straight' / 'left' / 'right' (Δyaw intégré sur 1 s)
  - delta_yaw_deg   : Δyaw signé (deg) sur la fenêtre (gauche = positif)

Variables temporelles ajoutées (par trajet) :
  timestamp_dt, date, hour, day_of_week, day_name, is_weekend,
  time_of_day, month, season

Export :
  clean_dataset.csv
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import glob
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

# ── Modèle de distance (réutilise overlay_annotations.py) ─────────────────────
import sys as _sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)
try:
    # predict_distances(bboxes) -> distances longitudinales (m) ou None.
    # predict_lateral(bboxes)   -> décalages latéraux (m) ou None.
    # Chargent les modèles joblib paresseusement (cf. overlay_annotations.py).
    from overlay_annotations import predict_distances as _oa_predict_distances
    try:
        from overlay_annotations import predict_lateral as _oa_predict_lateral
    except Exception:
        _oa_predict_lateral = None  # ancien overlay_annotations sans modèle latéral
except Exception as _e:
    print(f"⚠  overlay_annotations indisponible ({_e}) — distances non prédites")
    _oa_predict_distances = None
    _oa_predict_lateral = None

try:
    # Choix de direction par fenêtre de 1 s (Δyaw = ∫GyrZ dt), mêmes
    # constantes et même classification que l'overlay vidéo.
    from overlay_annotations import (
        compute_steering_choices_per_second as _oa_compute_steering,
        count_steering_maneuvers          as _oa_count_steering,
        STEERING_PIE_WINDOW_S,
        STEERING_PIE_THRESHOLD_DEG,
        GYRZ_LEFT_POSITIVE,
    )
except Exception as _e:
    print(f"⚠  overlay_annotations.steering indisponible ({_e}) — steering non calculé")
    _oa_compute_steering = None
    _oa_count_steering   = None
    STEERING_PIE_WINDOW_S      = 1.0
    STEERING_PIE_THRESHOLD_DEG = 10.0
    GYRZ_LEFT_POSITIVE         = True


def _selftest_distance_model():
    """Force un appel sur une bbox factice pour valider tôt que le modèle
    est utilisable. Affiche un diagnostic clair sinon."""
    if _oa_predict_distances is None:
        print("⚠  Modèle de distance NON disponible (import overlay_annotations échoué).")
        return False
    try:
        out = _oa_predict_distances([(100, 100, 200, 300)])
    except Exception as e:
        print(f"⚠  Self-test distance : exception levée : {e}")
        return False
    if not out or out[0] is None:
        print("⚠  Self-test distance : prédiction = None (joblib/sklearn manquants ?")
        print("    → installer joblib + sklearn dans le Python qui exécute ce script,")
        print(f"    → ou vérifier le bundle dans overlay_annotations.DISTANCE_MODEL_PATH.")
        return False
    print(f"✔  Modèle de distance opérationnel (test : {out[0]:.2f} m sur bbox factice)")
    return True


_DIST_MODEL_OK = _selftest_distance_model()


# ── Géométrie caméra (azimut VRU) ────────────────────────────────────────────
# FOV horizontale supposée constante : pas de calibration par clip chargée
# (pour rester aligné sur la demande). Ajuster IMAGE_WIDTH_PX / HFOV_DEG si
# les vidéos sources ont une résolution/optique différente.
IMAGE_WIDTH_PX = 1920
HFOV_DEG       = 90.0
# Focale équivalente (px) déduite du modèle sténopé : f = (W/2) / tan(HFOV/2)
_FOCAL_PX = (IMAGE_WIDTH_PX / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)


# ── Mapping codes ────────────────────────────────────────────────────────────
CODE_LABELS = {
    "CONFIRM":           {"0": "Reject", "1": "Accept", "2": "Review"},
    "VRU_TYPE":          {"1": "Pedestrian", "2": "Cyclist", "3": "E-scooter",
                          "4": "Other MMV",  "5": "Motor",   "6": "Animal",
                          "7": "Stationary", "9": "Unknown"},
    "INTERACTION_TYPE":  {"1": "Same-direction", "2": "Opposite-direction",
                          "3": "Crossing",       "4": "Stationary", "9": "Unknown"},
    "VRU_AGE_GROUP":     {"1": "Child", "2": "Adult", "3": "Elderly", "9": "Unknown"},
    "VRU_GROUP_SIZE":    {"1": "Solo",  "2": "Pair",  "3": "Group (3+)", "9": "Unknown"},
    "VRU_GAIT":          {"1": "Standing", "2": "Walking", "3": "Running", "9": "Unknown"},
    "WEATHER":           {"1": "No adverse", "2": "Adverse", "9": "Unknown"},
    "LIGHTING":          {"1": "Daylight",   "2": "Dawn/Dusk", "9": "Unknown"},
    "SURFACE_CONDITION": {"1": "Dry", "2": "Wet", "3": "Gravel", "4": "Uneven", "9": "Unknown"},
}


# ── Géographie : villes → vague ─────────────────────────────────────────────
CITIES = {
    "marseille": [
        (43.2965, 5.3698),
        (43.6402, 5.0977),
        (43.5297, 5.4474),
    ],
    "lyon":  [(45.7640, 4.8357)],
    "paris": [(48.8566, 2.3522)],
}

VAGUE_MAP = {
    "marseille": "vague1",
    "lyon":      "vague2",
    "paris":     "vague3",
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def closest_city(lat, lon):
    best_city, best_dist = None, float("inf")
    for city, coords_list in CITIES.items():
        for (clat, clon) in coords_list:
            dist = haversine_km(lat, lon, clat, clon)
            if dist < best_dist:
                best_dist = dist
                best_city = city
    return best_city


def vague_from_coords(lat_mean, lon_mean):
    city = closest_city(lat_mean, lon_mean)
    return VAGUE_MAP.get(city, "unknown")


def normalize_label(label: str) -> str:
    return (
        str(label).lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


# ── 1. Paramètres ────────────────────────────────────────────────────────────
# Le décalage GPS→vidéo était auparavant corrigé par un offset fixe de 60 frames
# (≈2 s à 30 fps). Il est désormais estimé PAR CLIP (gps_lag_applied_s) et déjà
# appliqué aux colonnes Lat/Long/VitGPS des fichiers *_corrected_with_offset_
# gpsfixed.csv. On n'applique donc plus aucun offset ici (0 = pas de décalage).
GPS_OFFSET_FRAMES    = 0
TURN_THRESHOLD_DEG_S = 20      # (déprécié) ancien seuil GyrZ d'exclusion des virages
PERP_HALF_LENGTH     = 100
DIRECTION_WINDOW     = 5

# ── Détection des virages par le GPS (route map-matchée) ─────────────────────
# Les virages ne sont plus EXCLUS via le GyrZ : ils sont LABELLISÉS à partir du
# changement de direction de la trace GPS. On simplifie d'abord la trace
# (Douglas-Peucker) pour obtenir des « segments de route » à l'échelle de la
# rue, puis on mesure l'angle signé entre deux segments consécutifs :
#   angle > +80°  → virage à droite     (rotation horaire)
#   angle < -80°  → virage à gauche      (rotation anti-horaire)
# Les points situés à moins de 30 m (le long de la trace) d'un virage sont
# étiquetés before/after left/right turn.
TURN_ANGLE_THRESHOLD_DEG = 80.0    # seuil d'angle entre segments consécutifs
TURN_LABEL_RADIUS_M      = 30.0    # rayon (m, arc) autour d'un virage à étiqueter
TURN_SIMPLIFY_TOL_M      = 8.0     # tolérance Douglas-Peucker (m) pour les segments
# Framerate vidéo utilisé pour intégrer le yaw sur la fenêtre de steering.
# Pas de vidéo ouverte ici → on reprend le fallback de correct_encounters.py
# (cap.get(CAP_PROP_FPS) or 30.0).
VIDEO_FPS            = 30.0

# Tranches horaires : (label, heure_debut_incluse, heure_fin_exclue)
TIME_SLOTS = [
    ("Night",     0,  6),   # 00:00 – 05:59
    ("Morning",   6, 12),   # 06:00 – 11:59
    ("Afternoon", 12, 18),  # 12:00 – 17:59
    ("Evening",   18, 24),  # 18:00 – 23:59
]

CODEBOOK_DIR         = '/Volumes/My Passport/NEWMOB/codebookescooter/'
IMU_DIR              = '/Volumes/My Passport/NEWMOB/escooter/'
ROAD_GPKG            = '/Volumes/My Passport/NEWMOB/road_with_slope.geojson'
PARTICIPANTS_XLS     = '/Volumes/My Passport/NEWMOB/participants_NewMob_Electromob_VAE_TE.xlsx'
OUTPUT_FILE          = '/Volumes/My Passport/NEWMOB/clean_dataset.csv'
DETECTIONS_FILE      = '/Volumes/My Passport/NEWMOB/clean_dataset_vru_detections.csv'
INTERSECTIONS_CSV    = '/Volumes/My Passport/NEWMOB/clips_intersections/recap_intersections.csv'


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — Sélection du fichier IMU
# ═════════════════════════════════════════════════════════════════════════════

def find_imu_candidates(prefix):
    """Fichiers IMU du trajet, en PRÉFÉRANT la variante GPS-recalée
    (*_corrected_with_offset_gpsfixed.csv : lag GPS estimé par clip déjà
    appliqué aux colonnes Lat/Long/VitGPS), puis *_corrected_with_offset.csv,
    puis n'importe quel CSV du trajet."""
    return (
        glob.glob(os.path.join(IMU_DIR, f'{prefix}*corrected_with_offset_gpsfixed*.csv'))
        or glob.glob(os.path.join(IMU_DIR, f'{prefix}*corrected_with_offset*.csv'))
        or glob.glob(os.path.join(IMU_DIR, f'{prefix}*.csv'))
    )


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — Vitesse GPS par filtre de Kalman (position CV 2D + lisseur RTS)
# Implémentation partagée dans kalman_speed.py (numpy pur), pour que la vitesse
# du dataset soit STRICTEMENT identique à celle affichée sur les vidéos par
# overlay_annotations.py. Réf. validée contre le compteur (r≈0.86 ; VitGPS
# sous-estime ~20 %). N'utilise que Lat/Long.
# ═════════════════════════════════════════════════════════════════════════════
import kalman_speed


def kalman_gps_speed_kmh(imu, q=2.0, r=9.0):
    """Vitesse GPS Kalman (km/h) alignée sur l'index de `imu`. NaN là où le
    trajet n'a pas de position GPS valide. Voir kalman_speed.gps_speed_kmh."""
    if not {"TimeStamp", "Lat", "Long"}.issubset(imu.columns):
        return pd.Series(np.nan, index=imu.index, dtype=float)
    v = kalman_speed.gps_speed_kmh(
        imu["TimeStamp"].values, imu["Lat"].values, imu["Long"].values, q=q, r=r
    )
    return pd.Series(v, index=imu.index, dtype=float)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — Variables temporelles
# ═════════════════════════════════════════════════════════════════════════════

def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Autumn"


def time_of_day_from_hour(hour: int) -> str:
    for label, start, end in TIME_SLOTS:
        if start <= hour < end:
            return label
    return "Unknown"


def extract_timestamp_from_imu(imu_df: pd.DataFrame):
    """
    Cherche la première valeur timestamp valide dans le DataFrame IMU.
    """
    exact_priority = ["TimeStamp_dt", "timestamp_dt"]
    for col in exact_priority:
        if col in imu_df.columns:
            series = imu_df[col].dropna()
            if not series.empty:
                try:
                    ts = pd.to_datetime(series.iloc[0])
                    if pd.notna(ts):
                        return ts
                except Exception:
                    pass

    fallback_patterns = ["timestamp_dt", "datetime", "date_time", "timestamp", "time", "date"]
    cols_lower = {c.lower(): c for c in imu_df.columns}
    for pat in fallback_patterns:
        for col_low, col_orig in cols_lower.items():
            if pat in col_low:
                series = imu_df[col_orig].dropna()
                if series.empty:
                    continue
                try:
                    ts = pd.to_datetime(series.iloc[0])
                    if pd.notna(ts):
                        return ts
                except Exception:
                    continue
    return None


def extract_timestamp_from_filename(prefix: str):
    """
    Extrait une date (et éventuellement une heure) depuis le préfixe du fichier.
    """
    m = re.search(r'(\d{4}-\d{2}-\d{2})[_T](\d{2})[-:](\d{2})[-:](\d{2})', prefix)
    if m:
        try:
            return pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}")
        except Exception:
            pass

    m = re.search(r'(\d{4}-\d{2}-\d{2})', prefix)
    if m:
        try:
            return pd.Timestamp(m.group(1))
        except Exception:
            pass

    return None


def build_temporal_features(ts) -> dict:
    if ts is None or pd.isna(ts):
        return {
            "timestamp_dt": pd.NaT,
            "date":         None,
            "hour":         np.nan,
            "day_of_week":  np.nan,
            "day_name":     None,
            "is_weekend":   None,
            "time_of_day":  None,
            "month":        np.nan,
            "season":       None,
        }
    return {
        "timestamp_dt": ts,
        "date":         str(ts.date()),
        "hour":         int(ts.hour),
        "day_of_week":  int(ts.dayofweek),
        "day_name":     ts.day_name(),
        "is_weekend":   ts.dayofweek >= 5,
        "time_of_day":  time_of_day_from_hour(ts.hour),
        "month":        int(ts.month),
        "season":       season_from_month(ts.month),
    }


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — Codebook annoté
# ═════════════════════════════════════════════════════════════════════════════

SESSION_CONTEXT_COLS = [
    "WEATHER", "LIGHTING", "SURFACE_CONDITION",
    "ZONE_TYPE", "VISUAL_SEGREGATION", "RIDING_COMPANION",
]

def load_session_context(prefix: str, codebook_dir: str) -> dict:
    """Charge les attributs de session (météo, éclairage, surface…) depuis le
    fichier encounters normal (pas debug). Retourne un dict col -> label string.

    Itère sur tous les raters disponibles pour le prefix et prend la première
    valeur non-vide pour chaque colonne (certains raters laissent ces champs vides).
    """
    candidates = sorted([
        p for p in glob.glob(os.path.join(codebook_dir, f'{prefix}*_encounters*.csv'))
        if 'debug_encounters' not in os.path.basename(p).lower()
        and 'obstacle_zones'  not in os.path.basename(p).lower()
        and not os.path.basename(p).startswith('.')
    ])
    if not candidates:
        return {}

    # Accumuler les valeurs non-vides par colonne sur tous les raters
    ctx = {}
    for path in candidates:
        try:
            with open(path) as _f:
                _first = _f.readline()
            sep = ';' if _first.count(';') > _first.count(',') else ','
            df = pd.read_csv(path, sep=sep)
            df.columns = df.columns.str.strip()
        except Exception:
            continue
        for col in SESSION_CONTEXT_COLS:
            if col in ctx:          # déjà trouvé pour cette colonne
                continue
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) > 0:
                val = series.iloc[0]
                ctx[f"{col}_LABEL"] = CODE_LABELS.get(col, {}).get(str(int(val)), "Unknown")
    return ctx


def load_codebook_annotated(prefix: str, codebook_dir: str):
    """
    Charge le fichier debug_encounters annoté pour un préfixe donné.
    Garde uniquement CONFIRM == 1.
    """
    candidates = glob.glob(os.path.join(codebook_dir, f'{prefix}*_encounters_debug_encounters*.csv'))
    if not candidates:
        return None
    # Préférer rater2 sur rater1 si les deux sont disponibles
    candidates = sorted(candidates, key=lambda p: (0 if re.search(r'rater2', p, re.IGNORECASE) else 1))
    try:
        with open(candidates[0]) as _f:
            _first = _f.readline()
        sep = ';' if _first.count(';') > _first.count(',') else ','
        df = pd.read_csv(candidates[0], sep=sep)
        df.columns = df.columns.str.strip()

        if "CONFIRM" in df.columns:
            df = df[pd.to_numeric(df["CONFIRM"], errors="coerce") == 1]

        print(f"   Codebook  : {os.path.basename(candidates[0])} — {len(df)} lignes confirmées")
        return df
    except Exception as e:
        print(f"   ⚠  Codebook non chargé : {e}")
        return None

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS — Détections VRU au niveau bbox (autodetect.csv) + distance
#           Réplique la logique de overlay_annotations.py
# ═════════════════════════════════════════════════════════════════════════════

def find_enc_csv_path(prefix: str, codebook_dir: str):
    """Chemin du debug_encounters préférentiel (rater2 > rater1)."""
    candidates = glob.glob(
        os.path.join(codebook_dir, f'{prefix}*_encounters_debug_encounters*.csv')
    )
    if not candidates:
        return None
    candidates = sorted(
        candidates,
        key=lambda p: (0 if re.search(r'rater2', p, re.IGNORECASE) else 1),
    )
    return candidates[0]


def find_autodet_csv_path(prefix: str, codebook_dir: str):
    """Chemin du fichier _debug_autodetect.csv (per-frame bbox)."""
    candidates = glob.glob(
        os.path.join(codebook_dir, f'{prefix}*_debug_autodetect.csv')
    )
    return candidates[0] if candidates else None


def _normalize_tid(x):
    """Track-id → string normalisé ('5' vs '5.0' vs 5 → '5'). None si NaN."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        s = str(x).strip()
        return s if s and s.lower() != 'nan' else None


def load_enc_meta_full(enc_csv: str) -> dict:
    """{(track_id_str, frame): meta} pour CONFIRM == 1.

    Comme `overlay_annotations.load_enc_meta`, plus VRU_GROUP_SIZE et la
    normalisation explicite des track_id en strings entiers.
    Couvre PRIMARY_TRACK_ID + LINKED_TRACKS sur [FRAME_START, FRAME_END].
    """
    enc_meta = {}
    if not enc_csv or not os.path.exists(enc_csv):
        return enc_meta

    with open(enc_csv) as f:
        first = f.readline()
    sep = ';' if first.count(';') > first.count(',') else ','

    df = pd.read_csv(enc_csv, sep=sep)
    df.columns = df.columns.str.strip()
    if 'CONFIRM' not in df.columns:
        return enc_meta
    df = df[pd.to_numeric(df['CONFIRM'], errors='coerce') == 1]
    if df.empty:
        return enc_meta

    def _label(field, raw):
        if pd.isna(raw):
            return ''
        try:
            code = str(int(float(raw)))
        except (ValueError, TypeError):
            return str(raw).strip()
        return CODE_LABELS.get(field, {}).get(code, code)

    for _, row in df.iterrows():
        try:
            f0 = int(row['FRAME_START'])
            f1 = int(row['FRAME_END'])
        except (ValueError, KeyError, TypeError):
            continue
        if f1 < f0:
            continue

        meta = {
            'vtype':      _label('VRU_TYPE',         row.get('VRU_TYPE')),
            'itype':      _label('INTERACTION_TYPE', row.get('INTERACTION_TYPE')),
            'age':        _label('VRU_AGE_GROUP',    row.get('VRU_AGE_GROUP')),
            'gait':       _label('VRU_GAIT',         row.get('VRU_GAIT')),
            'group_size': _label('VRU_GROUP_SIZE',   row.get('VRU_GROUP_SIZE')),
        }

        primary_tid = _normalize_tid(row.get('PRIMARY_TRACK_ID'))
        if primary_tid is None:
            continue
        track_ids = [primary_tid]

        linked_raw = row.get('LINKED_TRACKS', '')
        if not pd.isna(linked_raw):
            linked_str = str(linked_raw).strip()
            if linked_str and linked_str.lower() != 'nan':
                for t in linked_str.split(','):
                    tid = _normalize_tid(t)
                    if tid:
                        track_ids.append(tid)

        for fr in range(f0, f1 + 1):
            for tid in track_ids:
                enc_meta[(tid, fr)] = meta

    return enc_meta


def load_vrus_with_distance(autodet_csv: str, enc_meta: dict) -> pd.DataFrame:
    """Une ligne par détection autodetect dont (track_id, frame) ∈ enc_meta.

    Bbox reconstruite comme dans overlay_annotations.py : w = 0.45 * h.
    Distance prédite par le modèle joblib (batch). Colonnes :
        frame, track_id, x1, y1, x2, y2, foot_x, foot_y, bbox_height, distance_m,
        cx_px, azimuth_deg,
        VRU_TYPE_LABEL, INTERACTION_LABEL,
        VRU_AGE_GROUP_LABEL, VRU_GAIT_LABEL, VRU_GROUP_SIZE_LABEL

    azimuth_deg : angle horizontal du VRU par rapport à l'axe avant de
    l'e-scooter (modèle sténopé, FOV constante). Positif = à droite,
    négatif = à gauche. Cf. constantes IMAGE_WIDTH_PX / HFOV_DEG.
    """
    cols = [
        'frame', 'track_id', 'x1', 'y1', 'x2', 'y2',
        'foot_x', 'foot_y', 'bbox_height', 'distance_m', 'lateral_m',
        'cx_px', 'azimuth_deg',
        'VRU_TYPE_LABEL', 'INTERACTION_LABEL',
        'VRU_AGE_GROUP_LABEL', 'VRU_GAIT_LABEL', 'VRU_GROUP_SIZE_LABEL',
    ]
    if not autodet_csv or not os.path.exists(autodet_csv) or not enc_meta:
        return pd.DataFrame(columns=cols)

    with open(autodet_csv) as f:
        first = f.readline()
    sep = ';' if first.count(';') > first.count(',') else ','

    df = pd.read_csv(autodet_csv, sep=sep)
    df.columns = df.columns.str.strip()

    required = {'frame', 'track_id', 'foot_x', 'foot_y', 'bbox_height'}
    missing = required - set(df.columns)
    if missing:
        print(f"   ⚠  Colonnes autodetect manquantes : {missing}")
        return pd.DataFrame(columns=cols)

    df = df.copy()
    df['frame']       = pd.to_numeric(df['frame'],       errors='coerce')
    df['foot_x']      = pd.to_numeric(df['foot_x'],      errors='coerce')
    df['foot_y']      = pd.to_numeric(df['foot_y'],      errors='coerce')
    df['bbox_height'] = pd.to_numeric(df['bbox_height'], errors='coerce')
    df = df.dropna(subset=['frame', 'foot_x', 'foot_y', 'bbox_height'])
    df['frame']    = df['frame'].astype(int)
    # Normaliser le track_id en place (évite la colonne dupliquée)
    df['track_id'] = df['track_id'].apply(_normalize_tid)
    df = df[df['track_id'].notna()]

    # Filtrage par appartenance à un encounter confirmé
    metas = [enc_meta.get((tid, fr)) for tid, fr in zip(df['track_id'], df['frame'])]
    df['_meta'] = metas
    df = df[df['_meta'].notna()].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=cols)

    # Bbox reconstruite (identique à overlay_annotations.py)
    bw = df['bbox_height'] * 0.45
    df['x1'] = (df['foot_x'] - bw / 2).astype(int)
    df['y1'] = (df['foot_y'] - df['bbox_height']).astype(int)
    df['x2'] = (df['foot_x'] + bw / 2).astype(int)
    df['y2'] = df['foot_y'].astype(int)

    # Prédiction batchée
    bboxes = list(zip(df['x1'], df['y1'], df['x2'], df['y2']))
    if _oa_predict_distances is not None and bboxes:
        try:
            distances = _oa_predict_distances(bboxes)
        except Exception as e:
            print(f"   ⚠  Échec predict_distances : {e}")
            distances = [None] * len(bboxes)
    else:
        distances = [None] * len(bboxes)

    df['distance_m']           = distances

    # Décalage latéral (m) — second modèle bbox→mètres (n'utilise pas w)
    if _oa_predict_lateral is not None and bboxes:
        try:
            df['lateral_m'] = _oa_predict_lateral(bboxes)
        except Exception as e:
            print(f"   ⚠  Échec predict_lateral : {e}")
            df['lateral_m'] = [None] * len(bboxes)
    else:
        df['lateral_m'] = [None] * len(bboxes)

    # Azimut horizontal (deg) — modèle sténopé, point principal = centre image
    df['cx_px']       = (df['x1'] + df['x2']) / 2.0
    df['azimuth_deg'] = np.degrees(np.arctan2(
        df['cx_px'] - IMAGE_WIDTH_PX / 2.0, _FOCAL_PX))

    df['VRU_TYPE_LABEL']       = [m.get('vtype')      or 'Unknown' for m in df['_meta']]
    df['INTERACTION_LABEL']    = [m.get('itype')      or 'Unknown' for m in df['_meta']]
    df['VRU_AGE_GROUP_LABEL']  = [m.get('age')        or 'Unknown' for m in df['_meta']]
    df['VRU_GAIT_LABEL']       = [m.get('gait')       or 'Unknown' for m in df['_meta']]
    df['VRU_GROUP_SIZE_LABEL'] = [m.get('group_size') or 'Unknown' for m in df['_meta']]

    return df[cols].reset_index(drop=True)


def build_frame_level_from_detections(det_df: pd.DataFrame) -> pd.DataFrame:
    """Agrège un DataFrame "une ligne par bbox confirmée" en frame-level.

    Mêmes colonnes que `build_frame_level_from_debug_encounters`, plus :
        n_vru_visible, min_distance_m, mean_distance_m, max_distance_m
    Différence sémantique : on compte les **vraies détections** par frame, pas
    les events actifs sur cette frame (donc pas de comptage des frames d'un
    encounter où le VRU n'est pas détecté).
    """
    base_cols = ['frame']
    if det_df is None or det_df.empty:
        return pd.DataFrame(columns=base_cols)

    df = det_df.copy()

    # Flags de comptage (mêmes définitions que la version "encounters")
    vt = df['VRU_TYPE_LABEL']
    it = df['INTERACTION_LABEL']
    ag = df['VRU_AGE_GROUP_LABEL']
    ga = df['VRU_GAIT_LABEL']
    gs = df['VRU_GROUP_SIZE_LABEL']

    df['_is_pedestrian']         = (vt == 'Pedestrian').astype(int)
    df['_is_elderly']            = (ag == 'Elderly').astype(int)
    df['_is_child']              = (ag == 'Child').astype(int)
    df['_is_standing']           = ((ga == 'Standing') | (it == 'Stationary')).astype(int)
    df['_is_running']            = ((ga == 'Running') & (it != 'Stationary')).astype(int)
    df['_is_group']              = (gs == 'Group (3+)').astype(int)
    df['_is_crossing']           = (it == 'Crossing').astype(int)
    df['_is_ped_crossing']       = ((vt == 'Pedestrian') & (it == 'Crossing')).astype(int)
    df['_is_ped_opposite']       = ((vt == 'Pedestrian') & (it == 'Opposite-direction')).astype(int)
    df['_is_ped_same_direction'] = ((vt == 'Pedestrian') & (it == 'Same-direction')).astype(int)
    df['_is_ped_stationary']     = ((vt == 'Pedestrian') & (it == 'Stationary')).astype(int)
    df['_is_cyclist_crossing']   = ((vt == 'Cyclist') & (it == 'Crossing')).astype(int)

    # ── Comptages par frame ───────────────────────────────────────────────────
    total_per_frame = df.groupby('frame').size().rename('n_vru_total')
    n_visible       = df.groupby('frame').size().rename('n_vru_visible')
    ped_per_frame   = df[vt == 'Pedestrian'].groupby('frame').size().rename('n_pedestrians')
    cyc_per_frame   = df[vt == 'Cyclist'].groupby('frame').size().rename('n_cyclists')
    esc_per_frame   = df[vt == 'E-scooter'].groupby('frame').size().rename('n_escooters')

    # ── Proportions ───────────────────────────────────────────────────────────
    vru_counts = df.groupby(['frame', 'VRU_TYPE_LABEL']).size().unstack(fill_value=0)
    vru_props  = vru_counts.div(vru_counts.sum(axis=1), axis=0).rename(
        columns=lambda c: f'prop_vru_{normalize_label(c)}'
    )
    inter_counts = df.groupby(['frame', 'INTERACTION_LABEL']).size().unstack(fill_value=0)
    inter_props  = inter_counts.div(inter_counts.sum(axis=1), axis=0).rename(
        columns=lambda c: f'prop_interaction_{normalize_label(c)}'
    )

    # ── Comptages par flag ────────────────────────────────────────────────────
    flag_cols = {
        '_is_elderly':            'n_elderly',
        '_is_child':              'n_children',
        '_is_standing':           'n_standing',
        '_is_running':            'n_running',
        '_is_group':              'n_groups',
        '_is_crossing':           'n_crossing',
        '_is_ped_crossing':       'n_pedestrians_crossing',
        '_is_ped_opposite':       'n_pedestrians_opposite',
        '_is_ped_same_direction': 'n_pedestrians_same_direction',
        '_is_ped_stationary':     'n_pedestrians_stationary',
        '_is_cyclist_crossing':   'n_cyclists_crossing',
    }
    count_series = {
        name: df.groupby('frame')[flag].sum().rename(name)
        for flag, name in flag_cols.items() if flag in df.columns
    }

    # ── Distance aggregates (par frame) ───────────────────────────────────────
    # groupby ignore les NaN pour min/mean/max → si toutes les distances sont NaN
    # on obtient une Series de NaN avec le bon index 'frame' (pas un Series vide
    # anonyme qui casserait le reset_index() en aval).
    g_dist = df.groupby('frame')['distance_m']
    min_dist  = g_dist.min().rename('min_distance_m')
    mean_dist = g_dist.mean().rename('mean_distance_m')
    max_dist  = g_dist.max().rename('max_distance_m')

    out = pd.concat(
        [
            total_per_frame, n_visible,
            ped_per_frame, cyc_per_frame, esc_per_frame,
            vru_props, inter_props,
            *count_series.values(),
            min_dist, mean_dist, max_dist,
        ],
        axis=1,
    ).reset_index()

    # Casting / défauts
    int_cols = ['n_vru_total', 'n_vru_visible', 'n_pedestrians', 'n_cyclists',
                'n_escooters'] + list(flag_cols.values())
    for col in int_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)

    expected_vru_cols = [
        'prop_vru_pedestrian', 'prop_vru_cyclist', 'prop_vru_e_scooter',
        'prop_vru_other_mmv', 'prop_vru_motor', 'prop_vru_animal',
        'prop_vru_stationary', 'prop_vru_unknown',
    ]
    expected_inter_cols = [
        'prop_interaction_same_direction', 'prop_interaction_opposite_direction',
        'prop_interaction_crossing', 'prop_interaction_stationary',
        'prop_interaction_unknown',
    ]
    for col in expected_vru_cols + expected_inter_cols:
        if col not in out.columns:
            out[col] = 0.0

    # ── Labels contextuels (première valeur disponible par frame) ─────────────
    context_cols = ['VRU_AGE_GROUP_LABEL', 'VRU_GAIT_LABEL', 'VRU_GROUP_SIZE_LABEL']
    for col in context_cols:
        if col in df.columns:
            ctx = df.groupby('frame')[col].agg(
                lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
            )
            out = out.merge(ctx.rename(col), on='frame', how='left')

    # ── start_crossing : première frame où chaque track ped-crossing apparaît ─
    if 'track_id' in df.columns:
        ped_cross = df[(vt == 'Pedestrian') & (it == 'Crossing')]
        if not ped_cross.empty:
            first_per_track = ped_cross.groupby('track_id')['frame'].min()
            start_pairs = set(zip(first_per_track.index.astype(str),
                                  first_per_track.values.astype(int)))
            out_sorted = out.sort_values('frame').reset_index(drop=True)
            # Une frame est "start_crossing" s'il existe un track ped-crossing
            # qui commence sur cette frame.
            ped_cross_frames = set(first_per_track.values.astype(int))
            out_sorted['start_crossing'] = out_sorted['frame'].isin(ped_cross_frames).astype(int)
            out = out_sorted
        else:
            out['start_crossing'] = 0
    else:
        out['start_crossing'] = 0

    return out


def build_frame_level_from_debug_encounters(df_codebook: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un dataset frame-level à partir de debug_encounters uniquement.

    Chaque event confirmé est actif sur toutes les frames de FRAME_START à FRAME_END.
    Pour chaque frame, on calcule :
      - n_vru_total
      - proportions de VRU_TYPE
      - proportions de INTERACTION_TYPE
      - labels contextuels dominants / première valeur disponible
      - start_crossing = 1 sur la première frame d'un crossing piéton
    """
    if df_codebook is None or df_codebook.empty:
        return pd.DataFrame(columns=["frame"])

    df = df_codebook.copy()
    df.columns = df.columns.str.strip()

    required = {"FRAME_START", "FRAME_END", "VRU_TYPE", "INTERACTION_TYPE"}
    missing = required - set(df.columns)
    if missing:
        print(f"   ⚠ Colonnes manquantes dans debug_encounters : {missing}")
        return pd.DataFrame(columns=["frame"])

    numeric_cols = [
        "FRAME_START", "FRAME_END", "VRU_TYPE", "INTERACTION_TYPE",
        "VRU_AGE_GROUP", "VRU_GAIT", "VRU_GROUP_SIZE",
        "WEATHER", "LIGHTING", "SURFACE_CONDITION"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["FRAME_START", "FRAME_END", "VRU_TYPE", "INTERACTION_TYPE"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["frame"])

    records = []

    # WEATHER/LIGHTING/SURFACE_CONDITION/ZONE_TYPE/VISUAL_SEGREGATION/RIDING_COMPANION
    # ne sont PAS dans les debug_encounters — ils sont injectés après depuis le
    # fichier encounters normal via load_session_context().
    optional_maps = {
        "VRU_AGE_GROUP":  "VRU_AGE_GROUP",
        "VRU_GAIT":       "VRU_GAIT",
        "VRU_GROUP_SIZE": "VRU_GROUP_SIZE",
    }

    for _, row in df.iterrows():
        start = int(row["FRAME_START"])
        end = int(row["FRAME_END"])

        if end < start:
            continue

        vru_label = CODE_LABELS["VRU_TYPE"].get(str(int(row["VRU_TYPE"])), "Unknown")
        interaction_label = CODE_LABELS["INTERACTION_TYPE"].get(str(int(row["INTERACTION_TYPE"])), "Unknown")

        age_label = CODE_LABELS["VRU_AGE_GROUP"].get(
            str(int(row["VRU_AGE_GROUP"])) if "VRU_AGE_GROUP" in df.columns and pd.notna(row.get("VRU_AGE_GROUP")) else "9",
            "Unknown"
        )
        gait_label = CODE_LABELS["VRU_GAIT"].get(
            str(int(row["VRU_GAIT"])) if "VRU_GAIT" in df.columns and pd.notna(row.get("VRU_GAIT")) else "9",
            "Unknown"
        )
        group_size_label = CODE_LABELS["VRU_GROUP_SIZE"].get(
            str(int(row["VRU_GROUP_SIZE"])) if "VRU_GROUP_SIZE" in df.columns and pd.notna(row.get("VRU_GROUP_SIZE")) else "9",
            "Unknown"
        )

        rec = {
            "VRU_TYPE_LABEL":        vru_label,
            "INTERACTION_LABEL":     interaction_label,
            "VRU_AGE_GROUP_LABEL":   age_label,
            "VRU_GAIT_LABEL":        gait_label,
            "VRU_GROUP_SIZE_LABEL":  group_size_label,
            # flags binaires pour comptage
            "_is_pedestrian":        int(vru_label == "Pedestrian"),
            "_is_elderly":           int(age_label == "Elderly"),
            "_is_child":             int(age_label == "Child"),
            "_is_standing":          int(gait_label == "Standing" or interaction_label == "Stationary"),
            "_is_running":           int(gait_label == "Running" and interaction_label != "Stationary"),
            "_is_group":             int(group_size_label == "Group (3+)"),
            "_is_crossing":          int(interaction_label == "Crossing"),
            "_is_ped_crossing":          int(vru_label == "Pedestrian" and interaction_label == "Crossing"),
            "_is_ped_opposite":          int(vru_label == "Pedestrian" and interaction_label == "Opposite-direction" and interaction_label != "Stationary"),
            "_is_ped_same_direction":    int(vru_label == "Pedestrian" and interaction_label == "Same-direction"),
            "_is_ped_stationary":        int(vru_label == "Pedestrian" and interaction_label == "Stationary"),
            "_is_cyclist_crossing":      int(vru_label == "Cyclist" and interaction_label == "Crossing"),
        }

        for col, mapping_key in optional_maps.items():
            if col in df.columns and pd.notna(row.get(col)):
                rec[f"{col}_LABEL"] = CODE_LABELS[mapping_key].get(str(int(row[col])), "Unknown")
            else:
                rec[f"{col}_LABEL"] = np.nan

        # Nouvelle variable : start_crossing
        start_crossing_flag = int(vru_label == "Pedestrian" and interaction_label == "Crossing")

        for frame in range(start, end + 1):
            records.append({
                "frame": frame,
                "start_crossing": 1 if (frame == start and start_crossing_flag == 1) else 0,
                **rec
            })

    if not records:
        return pd.DataFrame(columns=["frame"])

    expanded = pd.DataFrame(records)

    # Total VRU annotés par frame
    total_per_frame = expanded.groupby("frame").size().rename("n_vru_total")

    # Nombre de piétons par frame
    ped_per_frame = (
        expanded[expanded["VRU_TYPE_LABEL"] == "Pedestrian"]
        .groupby("frame")
        .size()
        .rename("n_pedestrians")
    )

    cyc_per_frame = (
        expanded[expanded["VRU_TYPE_LABEL"] == "Cyclist"]
        .groupby("frame")
        .size()
        .rename("n_cyclists")
    )

    escooter_per_frame = (
        expanded[expanded["VRU_TYPE_LABEL"] == "E-scooter"]
        .groupby("frame")
        .size()
        .rename("n_escooters")
    )

    # Proportions VRU_TYPE
    vru_counts = expanded.groupby(["frame", "VRU_TYPE_LABEL"]).size().unstack(fill_value=0)
    vru_props = vru_counts.div(vru_counts.sum(axis=1), axis=0)
    vru_props = vru_props.rename(columns=lambda c: f"prop_vru_{normalize_label(c)}")

    # Proportions INTERACTION_TYPE
    inter_counts = expanded.groupby(["frame", "INTERACTION_LABEL"]).size().unstack(fill_value=0)
    inter_props = inter_counts.div(inter_counts.sum(axis=1), axis=0)
    inter_props = inter_props.rename(columns=lambda c: f"prop_interaction_{normalize_label(c)}")

    # start_crossing par frame
    start_crossing_per_frame = (
        expanded.groupby("frame")["start_crossing"]
        .max()
        .rename("start_crossing")
    )

    # ── Nouvelles variables de comptage par frame ─────────────────────────────
    flag_cols = {
        "_is_elderly":          "n_elderly",
        "_is_child":            "n_children",
        "_is_standing":         "n_standing",
        "_is_running":          "n_running",
        "_is_group":            "n_groups",
        "_is_crossing":         "n_crossing",
        "_is_ped_crossing":         "n_pedestrians_crossing",
        "_is_ped_opposite":         "n_pedestrians_opposite",
        "_is_ped_same_direction":   "n_pedestrians_same_direction",
        "_is_ped_stationary":       "n_pedestrians_stationary",
        "_is_cyclist_crossing":     "n_cyclists_crossing",
    }
    count_frames = {}
    for flag, name in flag_cols.items():
        if flag in expanded.columns:
            count_frames[name] = (
                expanded.groupby("frame")[flag].sum().rename(name)
            )

    out = pd.concat(
        [
            total_per_frame,
            ped_per_frame,
            cyc_per_frame,
            escooter_per_frame,
            vru_props,
            inter_props,
            start_crossing_per_frame,
            *count_frames.values(),
        ],
        axis=1
    ).reset_index()

    for col in ["n_pedestrians", "n_cyclists", "n_escooters"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)

    if "start_crossing" not in out.columns:
        out["start_crossing"] = 0
    out["start_crossing"] = out["start_crossing"].fillna(0).astype(int)

    expected_vru_cols = [
        "prop_vru_pedestrian",
        "prop_vru_cyclist",
        "prop_vru_e_scooter",
        "prop_vru_other_mmv",
        "prop_vru_motor",
        "prop_vru_animal",
        "prop_vru_stationary",
        "prop_vru_unknown",
    ]
    expected_inter_cols = [
        "prop_interaction_same_direction",
        "prop_interaction_opposite_direction",
        "prop_interaction_crossing",
        "prop_interaction_stationary",
        "prop_interaction_unknown",
    ]

    for col in expected_vru_cols + expected_inter_cols:
        if col not in out.columns:
            out[col] = 0.0

    # Variables contextuelles par frame : première valeur disponible
    context_cols = [
        "VRU_AGE_GROUP_LABEL",
        "VRU_GAIT_LABEL",
        "VRU_GROUP_SIZE_LABEL",
        "WEATHER_LABEL",
        "LIGHTING_LABEL",
        "SURFACE_CONDITION_LABEL"
    ]
    for col in context_cols:
        if col in expanded.columns:
            ctx = expanded.groupby("frame")[col].agg(
                lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
            )
            out = out.merge(ctx.rename(col), on="frame", how="left")

    return out

def compute_encounter_summary_from_codebook(df_codebook: pd.DataFrame) -> dict:
    """
    Résumé par trajet à partir des events confirmés de debug_encounters.
    """
    if df_codebook is None or df_codebook.empty:
        return {
            "n_encounters_annotated": 0,
            "pct_same_direction": np.nan,
            "pct_opposite_direction": np.nan,
            "pct_crossing": np.nan,
            "pct_stationary": np.nan,
            "pct_unknown_interaction": np.nan,
            "pct_gait_standing": np.nan,
            "pct_gait_walking": np.nan,
            "pct_gait_running": np.nan,
            "pct_gait_unknown": np.nan,
            "pct_age_child": np.nan,
            "pct_age_adult": np.nan,
            "pct_age_elderly": np.nan,
            "pct_age_unknown": np.nan,
        }

    df = df_codebook.copy()
    df.columns = df.columns.str.strip()

    n = len(df)
    out = {"n_encounters_annotated": n}

    def pct_from_code(col_name, mapping, out_prefix):
        counts = (
            pd.to_numeric(df[col_name], errors="coerce")
            .astype("Int64")
            .astype(str)
            .map(mapping)
            .value_counts(dropna=True)
        )
        for label in mapping.values():
            clean = normalize_label(label)
            out[f"{out_prefix}_{clean}"] = round(100 * counts.get(label, 0) / n, 1) if n > 0 else np.nan

    if "INTERACTION_TYPE" in df.columns:
        pct_from_code("INTERACTION_TYPE", CODE_LABELS["INTERACTION_TYPE"], "pct")

    if "VRU_GAIT" in df.columns:
        pct_from_code("VRU_GAIT", CODE_LABELS["VRU_GAIT"], "pct_gait")

    if "VRU_AGE_GROUP" in df.columns:
        pct_from_code("VRU_AGE_GROUP", CODE_LABELS["VRU_AGE_GROUP"], "pct_age")

    if "VRU_GROUP_SIZE" in df.columns:
        pct_from_code("VRU_GROUP_SIZE", CODE_LABELS["VRU_GROUP_SIZE"], "pct_group_size")

    if "WEATHER" in df.columns:
        pct_from_code("WEATHER", CODE_LABELS["WEATHER"], "pct_weather")

    if "LIGHTING" in df.columns:
        pct_from_code("LIGHTING", CODE_LABELS["LIGHTING"], "pct_lighting")

    if "SURFACE_CONDITION" in df.columns:
        pct_from_code("SURFACE_CONDITION", CODE_LABELS["SURFACE_CONDITION"], "pct_surface")

    if "ZONE_TYPE" in df.columns:
        pct_from_code("ZONE_TYPE", CODE_LABELS["ZONE_TYPE"], "pct_zone")

    if "RIDING_COMPANION" in df.columns:
        pct_from_code("RIDING_COMPANION", CODE_LABELS["RIDING_COMPANION"], "pct_companion")

    # Métriques kinematics moyennes par trajet
    for col, out_key in [
        ("PEAK_DECEL_MS2",        "mean_peak_decel_ms2"),
        ("TTC_MIN_S",             "mean_ttc_min_s"),
        ("DRAC_MAX_MS2",          "mean_drac_max_ms2"),
        ("REACTION_TIME_S",       "mean_reaction_time_s"),
        ("DURATION_S",            "mean_encounter_duration_s"),
        ("N_SIMULTANEOUS_VRUS",   "mean_simultaneous_vrus"),
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[out_key] = round(vals.mean(), 3) if vals.notna().any() else np.nan

    # Assurer la présence des colonnes attendues
    expected = [
        "pct_same_direction",
        "pct_opposite_direction",
        "pct_crossing",
        "pct_stationary",
        "pct_unknown",
        "pct_gait_standing",
        "pct_gait_walking",
        "pct_gait_running",
        "pct_gait_unknown",
        "pct_age_child",
        "pct_age_adult",
        "pct_age_elderly",
        "pct_age_unknown",
    ]
    for col in expected:
        if col not in out:
            out[col] = np.nan

    # Renommage pour cohérence
    out["pct_unknown_interaction"] = out.pop("pct_unknown", np.nan)

    return out


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE 0 — Chargement du fichier participants
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ÉTAPE 0 — Participants (riders info)")
print("=" * 65)

participants = pd.read_excel(PARTICIPANTS_XLS)
participants.columns = [c.strip() for c in participants.columns]

col_rename = {}
for c in participants.columns:
    cl = c.lower()
    if "nb" in cl and "trajet" in cl:
        col_rename[c] = "nb_trajets"
    elif "distance" in cl:
        col_rename[c] = "distance_km"
    elif "experience" in cl:
        col_rename[c] = "experience"
    elif cl == "genre":
        col_rename[c] = "genre"
    elif cl == "age":
        col_rename[c] = "age"
    elif cl == "vague":
        col_rename[c] = "vague"
    elif cl == "vehicle":
        col_rename[c] = "vehicle"
    elif cl == "device":
        col_rename[c] = "device"

participants = participants.rename(columns=col_rename)
participants["device_key"] = participants["device"].astype(str).str.strip().str.lower()
participants["vague_key"]  = participants["vague"].astype(str).str.strip().str.lower()

print(f"  {len(participants)} participants chargés")
print(f"  Colonnes : {list(participants.columns)}")

if {"device", "vague", "device_key", "vague_key"}.issubset(participants.columns):
    print("\n  ── Valeurs brutes Excel (device / vague / device_key / vague_key) ──")
    print(participants[["device", "vague", "device_key", "vague_key"]].to_string(index=False))
    print()


def device_from_prefix(prefix):
    raw = prefix.split("_")[0]
    cleaned = re.sub(r'\(.*?\)', '', raw).strip().lower()
    if cleaned.endswith('t'):
        cleaned = cleaned[:-1] + 't'
    return cleaned


print("  ── Clés participants (device_key / vague_key) ──")
print(participants[["device_key", "vague_key"]].drop_duplicates().to_string(index=False))
print()


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE A — Largeur de route
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ÉTAPE A — Largeur de route (road.gpkg)")
print("=" * 65)
print("Chargement road.gpkg...")

road   = gpd.read_file(ROAD_GPKG)
road_m = road.to_crs(epsg=2154).copy()
sindex = road_m.sindex

print(f"  {len(road_m)} polygones — CRS : {road_m.crs}\n")


def find_polygon_for_point(point, road_gdf, sindex):
    """Retourne (geometry, attrs) du polygone le plus proche du point.

    attrs : dict avec 'env_type', 'slope_pct_max', 'slope_grad_x_pct',
            'slope_grad_y_pct', 'elev_mean_m' (issu de road_with_slope.geojson
            — NaN si la colonne n'existe pas).
    """
    candidate_idx = list(sindex.intersection(point.bounds))
    candidates = road_gdf.iloc[candidate_idx]

    containing = candidates[candidates.contains(point)]
    if len(containing) > 0:
        row = containing.iloc[0]
    elif len(candidates) > 0:
        distances = candidates.geometry.distance(point)
        row = candidates.loc[distances.idxmin()]
    else:
        distances = road_gdf.geometry.distance(point)
        row = road_gdf.loc[distances.idxmin()]

    attrs = {
        "env_type":         row.get("type",              np.nan),
        "slope_pct_max":    row.get("slope_pct_max",     np.nan),
        "slope_grad_x_pct": row.get("slope_grad_x_pct",  np.nan),
        "slope_grad_y_pct": row.get("slope_grad_y_pct",  np.nan),
        "elev_mean_m":      row.get("elev_mean_m",       np.nan),
    }
    return row.geometry, attrs


def polygon_width_at_point(point, polygon, dx, dy, half_length=PERP_HALF_LENGTH):
    norm = np.hypot(dx, dy)
    if norm == 0:
        return np.nan

    px, py = -dy / norm, dx / norm
    p1 = Point(point.x - px * half_length, point.y - py * half_length)
    p2 = Point(point.x + px * half_length, point.y + py * half_length)
    perp = LineString([p1, p2])
    inter = perp.intersection(polygon)

    return np.nan if inter.is_empty else inter.length


def compute_turn_labels(
    xs,
    ys,
    angle_threshold_deg=TURN_ANGLE_THRESHOLD_DEG,
    radius_m=TURN_LABEL_RADIUS_M,
    simplify_tol_m=TURN_SIMPLIFY_TOL_M,
):
    """Étiquette les points GPS situés à proximité d'un virage.

    Méthodologie (à partir du GPS, pas du GyrZ) :
      1. La trace (coords projetées, en m, ordonnées le long du trajet) est
         simplifiée par Douglas-Peucker → « segments de route » à l'échelle de
         la rue, qui approchent la route map-matchée.
      2. À chaque sommet intérieur, on mesure l'angle signé entre le segment
         entrant et le segment sortant, exprimé en degrés de courbure dans
         [-180°, 180°] (cap boussole : positif = rotation horaire) :
             angle > +angle_threshold  → virage à DROITE
             angle < -angle_threshold  → virage à GAUCHE
      3. Les points situés à moins de `radius_m` (distance le long de la trace)
         d'un virage sont étiquetés before/after left/right turn.

    Renvoie une liste de labels (un par point) parmi :
      'none', 'before_left_turn', 'after_left_turn',
      'before_right_turn', 'after_right_turn'.
    Si plusieurs virages se chevauchent, le virage le plus proche l'emporte.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    labels = ['none'] * n
    if n < 3:
        return labels

    # Abscisse curviligne cumulée (m) le long de la trace brute.
    seg_len = np.hypot(np.diff(xs), np.diff(ys))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])

    # Simplification → sommets significatifs (sous-ensemble des points d'origine).
    simp = LineString(np.column_stack([xs, ys])).simplify(simplify_tol_m)
    sc = np.asarray(simp.coords, dtype=float)
    if len(sc) < 3:
        return labels

    # Index du point d'origine le plus proche de chaque sommet simplifié.
    def _nearest_orig(px, py):
        return int(np.argmin((xs - px) ** 2 + (ys - py) ** 2))

    # Pour chaque virage retenu : (index d'origine, abscisse, signe droite/gauche).
    turns = []
    for j in range(1, len(sc) - 1):
        vin_x, vin_y = sc[j] - sc[j - 1]
        vout_x, vout_y = sc[j + 1] - sc[j]
        if (vin_x == 0 and vin_y == 0) or (vout_x == 0 and vout_y == 0):
            continue
        # Cap boussole atan2(est, nord) : tourner à droite augmente le cap.
        bearing_in = math.degrees(math.atan2(vin_x, vin_y))
        bearing_out = math.degrees(math.atan2(vout_x, vout_y))
        deflection = (bearing_out - bearing_in + 180.0) % 360.0 - 180.0
        if deflection > angle_threshold_deg:
            side = 'right'
        elif deflection < -angle_threshold_deg:
            side = 'left'
        else:
            continue
        idx = _nearest_orig(sc[j][0], sc[j][1])
        turns.append((idx, arc[idx], side))

    if not turns:
        return labels

    # Affecter chaque point au virage le plus proche (en abscisse) dans le rayon.
    best_dist = np.full(n, np.inf)
    for idx, s_turn, side in turns:
        d = np.abs(arc - s_turn)
        within = (d <= radius_m) & (d < best_dist)
        for i in np.nonzero(within)[0]:
            phase = 'before' if arc[i] < s_turn else 'after'
            labels[i] = f'{phase}_{side}_turn'
            best_dist[i] = d[i]

    return labels


all_enc_files = glob.glob(os.path.join(CODEBOOK_DIR, '*_encounters_debug_encounters*.csv'))

def extract_prefix(filepath):
    fname = os.path.basename(filepath)
    fname = re.sub(r'\.csv$', '', fname, flags=re.IGNORECASE)

    # enlever le suffixe debug_encounters
    fname = re.sub(r'_encounters_debug_encounters.*$', '', fname, flags=re.IGNORECASE)

    # enlever rater1 / rater2
    fname = re.sub(r'_rater\d+$', '', fname, flags=re.IGNORECASE)

    return fname

def is_duplicate(filepath):
    return bool(re.search(r'_encounters_debug_encounters\s+\d+', os.path.basename(filepath), flags=re.IGNORECASE))

encounter_files = [f for f in all_enc_files if not is_duplicate(f)]
duplicates      = [f for f in all_enc_files if is_duplicate(f)]

print(f"Fichiers debug_encounters  : {len(all_enc_files)}")
print(f"  dont doublons ignorés    : {len(duplicates)}")
print(f"  dont trajets uniques     : {len(encounter_files)}\n")


all_gps = []
source_meta = {}          # prefix → infos de source
encounter_summary_map = {}  # prefix → résumé events
seen_prefixes = set()

for enc_path in sorted(encounter_files):
    prefix = extract_prefix(enc_path)
    if prefix in seen_prefixes:
        print(f"⚠ Prefix déjà traité, ignoré : {prefix}")
        continue
    seen_prefixes.add(prefix)

    imu_candidates = find_imu_candidates(prefix)
    if not imu_candidates:
        print(f"⚠ Aucun IMU trouvé pour prefix = {prefix}")
        print(f"   Recherché : {os.path.join(IMU_DIR, f'{prefix}*_corrected_with_offset_gpsfixed.csv')} (et variantes)")
        continue

    try:
        imu = pd.read_csv(imu_candidates[0], usecols=lambda c: True)

        imu_gps = imu[["Lat", "Long", "frame"]].copy() if all(
            c in imu.columns for c in ["Lat", "Long", "frame"]
        ) else None

        if imu_gps is None:
            continue

        imu_gps = imu_gps.dropna(subset=["Lat", "Long"])
        imu_gps = imu_gps[(imu_gps["Lat"] != 0) & (imu_gps["Long"] != 0)].sort_values("frame")
        imu_gps["source"] = prefix
        all_gps.append(imu_gps)

        lat_mean   = imu_gps["Lat"].mean()
        lon_mean   = imu_gps["Long"].mean()
        vague_key  = vague_from_coords(lat_mean, lon_mean)
        device_key = device_from_prefix(prefix)

        ts = extract_timestamp_from_imu(imu)
        ts_source = "IMU column"
        if ts is None:
            ts = extract_timestamp_from_filename(prefix)
            ts_source = "filename"

        temporal_feats = build_temporal_features(ts)

        source_meta[prefix] = {
            "vague_key":  vague_key,
            "device_key": device_key,
            **temporal_feats,
        }

        # Résumé encounter par trajet
        df_codebook_tmp = load_codebook_annotated(prefix, CODEBOOK_DIR)
        encounter_summary_map[prefix] = compute_encounter_summary_from_codebook(df_codebook_tmp)

        ts_str = str(ts) if ts is not None else "N/A"
        print(f"  {prefix[-30:]:30s}  ville={closest_city(lat_mean, lon_mean):10s}  "
              f"vague={vague_key}  ts={ts_str} [{ts_source}]")

    except Exception as e:
        print(f"  ⚠  GPS non chargé pour {prefix} : {e}")

if not all_gps:
    raise ValueError("Aucun fichier GPS valide — vérifier IMU_DIR.")

gps_all = pd.concat(all_gps, ignore_index=True)

gdf = gpd.GeoDataFrame(
    gps_all,
    geometry=gpd.points_from_xy(gps_all["Long"], gps_all["Lat"]),
    crs="EPSG:4326",
).to_crs(epsg=2154)


road_width_records = []

for src, group in gdf.groupby("source"):
    group = group.sort_values("frame").reset_index(drop=True)
    widths = []
    print(f"  Calcul largeur : {src[-40:]}  ({len(group)} points)")

    env_types          = []
    slope_signed_pcts  = []   # pente signée selon le sens du trajet (%)
    slope_max_pcts     = []   # pente max du polygone (% — absolue, sans signe)
    elev_means         = []   # altitude moyenne du polygone (m)
    for i in range(len(group)):
        point          = group.geometry.iloc[i]
        polygon, attrs = find_polygon_for_point(point, road_m, sindex)
        env_types.append(attrs["env_type"])
        slope_max_pcts.append(attrs["slope_pct_max"])
        elev_means.append(attrs["elev_mean_m"])

        i_prev = max(0, i - DIRECTION_WINDOW)
        i_next = min(len(group) - 1, i + DIRECTION_WINDOW)

        if i_prev == i_next:
            widths.append(np.nan)
            slope_signed_pcts.append(np.nan)
            continue

        dx = group.geometry.iloc[i_next].x - group.geometry.iloc[i_prev].x
        dy = group.geometry.iloc[i_next].y - group.geometry.iloc[i_prev].y
        widths.append(polygon_width_at_point(point, polygon, dx, dy))

        # Pente signée : projection du gradient (% par m) sur le vecteur
        # unitaire de déplacement → positif = montée dans le sens de circulation.
        norm = math.hypot(dx, dy)
        gx, gy = attrs["slope_grad_x_pct"], attrs["slope_grad_y_pct"]
        if norm > 0 and pd.notna(gx) and pd.notna(gy):
            slope_signed_pcts.append(gx * dx / norm + gy * dy / norm)
        else:
            slope_signed_pcts.append(np.nan)

    group = group.copy()
    group["road_width_perp_m"] = widths
    group["road_width_perp_m"] = group["road_width_perp_m"].interpolate(
        method="linear", limit_direction="both"
    )
    group["environment_type"] = env_types
    group["slope_signed_pct"] = slope_signed_pcts
    group["slope_pct_max"]    = slope_max_pcts
    group["elev_mean_m"]      = elev_means
    group["slope_signed_pct"] = group["slope_signed_pct"].interpolate(
        method="linear", limit_direction="both"
    )

    # Label de virage à partir du GPS (cf. compute_turn_labels) : on n'exclut
    # plus les virages, on les étiquette before/after left/right turn.
    group["turn_label"] = compute_turn_labels(
        group.geometry.x.to_numpy(), group.geometry.y.to_numpy()
    )
    n_turn_pts = int((group["turn_label"] != "none").sum())
    print(f"     Virages GPS : {n_turn_pts} points étiquetés "
          f"({sorted(set(group['turn_label']) - {'none'})})")

    cols_keep_iter = ["frame", "road_width_perp_m", "environment_type",
                      "slope_signed_pct", "slope_pct_max", "elev_mean_m",
                      "turn_label"]
    for _, row in group[cols_keep_iter].iterrows():
        road_width_records.append({
            "source":            src,
            "frame_imu_raw":     row["frame"],
            "road_width_perp_m": row["road_width_perp_m"],
            "environment_type":  row["environment_type"],
            "slope_signed_pct":  row["slope_signed_pct"],
            "slope_pct_max":     row["slope_pct_max"],
            "elev_mean_m":       row["elev_mean_m"],
            "turn_label":        row["turn_label"],
        })

road_width_df = pd.DataFrame(road_width_records)
road_width_df["frame"] = road_width_df["frame_imu_raw"] - GPS_OFFSET_FRAMES
road_width_df = road_width_df[[
    "source", "frame", "road_width_perp_m", "environment_type",
    "slope_signed_pct", "slope_pct_max", "elev_mean_m", "turn_label",
]]

print(f"\n  ✔  road_width_df : {len(road_width_df)} lignes\n")


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE B — Dataset frame-level depuis debug_encounters uniquement
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ÉTAPE B — Dataset frame-level depuis debug_encounters")
print("=" * 65)


def load_obstacle_intervals(df_obstacles):
    if df_obstacles is None or df_obstacles.empty:
        return []
    required = {"FRAME_START", "FRAME_END"}
    if not required.issubset(df_obstacles.columns):
        return []
    df = df_obstacles[["FRAME_START", "FRAME_END"]].apply(pd.to_numeric, errors="coerce").dropna()
    return list(zip(df["FRAME_START"], df["FRAME_END"]))


def frame_in_obstacle(frame, intervals):
    return any(s <= frame <= e for s, e in intervals)


# ── Intersections ─────────────────────────────────────────────────────────────
def load_intersection_intervals_for_prefix(prefix: str, intersections_csv: str) -> list:
    """Retourne la liste des intervalles [frame_start, frame_end] pour les passages
    en intersection du trajet `prefix`, depuis recap_intersections.csv.

    Le fichier identifie les trajets via la colonne `video` (ex: '335t_2023-…_.mp4').
    Le prefix correspond au nom de la vidéo sans extension.
    """
    try:
        df = pd.read_csv(intersections_csv)
        df.columns = df.columns.str.strip()
        # Normaliser : retirer .mp4 et espaces
        df['_prefix'] = df['video'].str.replace(r'\.mp4$', '', regex=True).str.strip()
        mask = df['_prefix'] == prefix
        sub = df.loc[mask, ['frame_start', 'frame_end']].apply(pd.to_numeric, errors='coerce').dropna()
        return list(zip(sub['frame_start'].astype(int), sub['frame_end'].astype(int)))
    except Exception:
        return []


all_rows = []
all_detections = []          # bbox-level (une ligne par détection confirmée)
seen_prefixes = set()
for enc_path in sorted(encounter_files):
    prefix = extract_prefix(enc_path)
    if prefix in seen_prefixes:
        print(f"⚠ Prefix déjà traité, ignoré : {prefix}")
        continue
    seen_prefixes.add(prefix)

    imu_candidates = find_imu_candidates(prefix)
    if not imu_candidates:
        print(f"⚠  [{prefix}] Pas de fichier IMU — trajet ignoré")
        continue

    imu_path = imu_candidates[0]

    obs_candidates     = glob.glob(os.path.join(CODEBOOK_DIR, f'{prefix}*obstacle_zones*.csv'))
    df_obstacles       = pd.read_csv(obs_candidates[0]) if obs_candidates else None
    obs_intervals      = load_obstacle_intervals(df_obstacles)
    inter_intervals    = load_intersection_intervals_for_prefix(prefix, INTERSECTIONS_CSV)

    # Détections autodetect + filtrage par encounters confirmés (cf. overlay_annotations.py)
    enc_csv_path     = find_enc_csv_path(prefix, CODEBOOK_DIR)
    autodet_csv_path = find_autodet_csv_path(prefix, CODEBOOK_DIR)
    enc_meta_full    = load_enc_meta_full(enc_csv_path) if enc_csv_path else {}
    det_df           = load_vrus_with_distance(autodet_csv_path, enc_meta_full)

    print(f"\n▶  {prefix}")
    print(f"   IMU           : {os.path.basename(imu_path)}")
    print(f"   Obstacles     : {os.path.basename(obs_candidates[0]) if obs_candidates else 'aucun'}")
    print(f"   Intersections : {len(inter_intervals)} intervalle(s)")
    print(f"   Autodetect    : {os.path.basename(autodet_csv_path) if autodet_csv_path else 'aucun'}")
    n_with_dist = int(det_df['distance_m'].notna().sum()) if not det_df.empty else 0
    print(f"   Détections    : {len(det_df)} bbox confirmées  (distance prédite : {n_with_dist})")

    # df_codebook : utilisé pour load_session_context, déjà via load_codebook_annotated
    # ci-dessous. On garde l'appel pour le log "Codebook  : … lignes confirmées".
    df_codebook    = load_codebook_annotated(prefix, CODEBOOK_DIR)
    session_ctx    = load_session_context(prefix, CODEBOOK_DIR)

    try:
        imu = pd.read_csv(imu_path)

        required_imu = {"frame", "VitGPS(km/h)", "GyrZ(deg/s)"}
        missing_imu = required_imu - set(imu.columns)
        if missing_imu:
            print(f"   ⚠  Colonnes IMU manquantes : {missing_imu} — ignoré")
            continue

        imu = imu.sort_values("frame").copy()

        # Interpolation linéaire de VitGPS(km/h) entre valeurs voisines valides
        # (même source garantie — pas d'extrapolation aux bords)
        n_nan_speed = imu["VitGPS(km/h)"].isna().sum()
        imu["VitGPS(km/h)"] = imu["VitGPS(km/h)"].interpolate(method="linear")
        n_interpolated = n_nan_speed - imu["VitGPS(km/h)"].isna().sum()
        if n_nan_speed > 0:
            print(f"   VitGPS NaN : {n_nan_speed} → {n_interpolated} interpolées")

        # Vitesse GPS lissée par Kalman position + RTS (cf. speed/§2), calculée
        # sur les fixes GPS du clip puis interpolée par ligne IMU. Se joint
        # ensuite comme speed_kmh (colonne speed_kmh_kalman).
        imu["speed_kmh_kalman"] = kalman_gps_speed_kmh(imu)
        n_kal_ok = int(imu["speed_kmh_kalman"].notna().sum())
        print(f"   Kalman GPS : {n_kal_ok}/{len(imu)} lignes avec vitesse "
              f"(moy {imu['speed_kmh_kalman'].mean():.1f} km/h)"
              if n_kal_ok else "   Kalman GPS : aucune position exploitable")

        imu["frame_corrected"] = imu["frame"] - GPS_OFFSET_FRAMES

        # ── Steering (cf. overlay_annotations.py) ─────────────────────────────
        # GyrZ moyen par frame brute (le CSV IMU a plusieurs lignes/frame),
        # exactement comme overlay_annotations.load_gyrz_by_frame, puis choix
        # de direction (straight/left/right) par fenêtre de 1 s sur Δyaw.
        # Le résultat est ramené dans l'espace frame_corrected (− offset GPS)
        # pour s'aligner sur frame_df, comme la jointure IMU plus bas.
        steer_df = pd.DataFrame(columns=["frame", "steering_choice", "delta_yaw_deg"])
        n_steer_left = n_steer_right = 0
        if _oa_compute_steering is not None:
            gyrz_by_frame = (
                imu.dropna(subset=["GyrZ(deg/s)"])
                   .groupby("frame")["GyrZ(deg/s)"]
                   .mean()
                   .to_dict()
            )
            steering_choices = _oa_compute_steering(gyrz_by_frame, VIDEO_FPS)
            if steering_choices:
                steer_df = pd.DataFrame(
                    [(fr - GPS_OFFSET_FRAMES, lbl, dy)
                     for fr, (lbl, dy) in steering_choices.items()],
                    columns=["frame", "steering_choice", "delta_yaw_deg"],
                )
                if _oa_count_steering is not None:
                    n_steer_left, n_steer_right = _oa_count_steering(steering_choices)
        print(f"   Steering  : L/R = {n_steer_left}/{n_steer_right}  "
              f"(fenêtre {STEERING_PIE_WINDOW_S:.0f}s, "
              f"seuil {STEERING_PIE_THRESHOLD_DEG:.0f}°)")

        # Les virages ne sont plus exclus ici : ils sont étiquetés à partir du
        # GPS (colonne turn_label, jointe via road_width_df). On conserve donc
        # toutes les frames.

        # Frame-level construit à partir des VRAIES détections autodetect
        # (et non des intervalles d'encounters), comme overlay_annotations.py.
        frame_df = build_frame_level_from_detections(det_df)

        # Injecter les attributs de session depuis le fichier encounters normal
        for col_label, value in session_ctx.items():
            frame_df[col_label] = value

        # Flag at_intersection : 1 si la frame est dans un intervalle d'intersection
        if inter_intervals:
            frame_df['at_intersection'] = frame_df['frame'].apply(
                lambda f: int(any(s <= f <= e for s, e in inter_intervals))
            )
        else:
            frame_df['at_intersection'] = 0

        if frame_df.empty:
            print("   ℹ  Aucune détection autodetect confirmée (frame_df vide)")
            continue

        # speed_kmh_t1 : vitesse IMU à la frame suivante, même source uniquement.
        # Calculé avant filtrage pour conserver la valeur même si t+1 est filtré.
        speed_map = imu.drop_duplicates("frame_corrected").set_index("frame_corrected")["VitGPS(km/h)"].to_dict()
        frame_df["speed_kmh_t1"] = frame_df["frame"].add(1).map(speed_map)
        kalman_map = imu.drop_duplicates("frame_corrected").set_index("frame_corrected")["speed_kmh_kalman"].to_dict()
        frame_df["speed_kmh_kalman_t1"] = frame_df["frame"].add(1).map(kalman_map)

        # (Virages : plus d'exclusion — labellisés via turn_label depuis le GPS.)

        n1 = len(frame_df)
        if obs_intervals:
            mask_obs  = ~frame_df["frame"].apply(lambda f: frame_in_obstacle(f, obs_intervals))
            frame_df  = frame_df.loc[mask_obs].copy()
            if not det_df.empty:
                det_mask = ~det_df["frame"].apply(lambda f: frame_in_obstacle(f, obs_intervals))
                det_df   = det_df.loc[det_mask].copy()
        print(f"   Obstacles : {n1 - len(frame_df):>4} lignes retirées → {len(frame_df)} restantes")

        if frame_df.empty:
            print("   ℹ  Plus aucune frame après filtrage")
            continue

        # Persister les détections (bbox-level) une fois filtrées
        if not det_df.empty:
            det_df = det_df.copy()
            det_df.insert(0, "source", prefix)
            all_detections.append(det_df)

        # Forcer NaN sur la dernière frame restante par source (pas de t+1 garanti
        # dans la même source après concaténation).
        last_frame = frame_df["frame"].max()
        frame_df.loc[frame_df["frame"] == last_frame, "speed_kmh_t1"] = np.nan
        frame_df.loc[frame_df["frame"] == last_frame, "speed_kmh_kalman_t1"] = np.nan

        # Jointure IMU (vitesse GPS brute, vitesse Kalman, gyro)
        imu_sub = (
            imu[["frame_corrected", "VitGPS(km/h)", "speed_kmh_kalman", "GyrZ(deg/s)"]]
            .rename(columns={
                "frame_corrected": "frame",
                "VitGPS(km/h)":    "speed_kmh",
                "GyrZ(deg/s)":     "gyrz_deg_s",
            })
        )
        frame_df = frame_df.merge(imu_sub, on="frame", how="left")

        # Jointure steering (choix de direction par fenêtre de 1 s)
        frame_df = frame_df.merge(steer_df, on="frame", how="left")

        frame_df.insert(0, "source", prefix)

        keep = [
    "source", "frame",
    "speed_kmh", "speed_kmh_t1",
    # Vitesse GPS lissée par Kalman position + RTS (réf. compteur, cf. speed/§2)
    "speed_kmh_kalman", "speed_kmh_kalman_t1",
    "gyrz_deg_s",
    # Steering (Δyaw intégré sur 1 s, cf. overlay_annotations.py)
    "steering_choice", "delta_yaw_deg",
    "n_vru_total",
    "n_vru_visible",
    "n_pedestrians",
    "n_cyclists",
    "n_escooters",
    # Distance (modèle bbox → m)
    "min_distance_m",
    "mean_distance_m",
    "max_distance_m",
    "prop_vru_pedestrian",
    "prop_vru_cyclist",
    "prop_vru_e_scooter",
    "prop_vru_other_mmv",
    "prop_vru_motor",
    "prop_vru_animal",
    "prop_vru_stationary",
    "prop_vru_unknown",
    "prop_interaction_same_direction",
    "prop_interaction_opposite_direction",
    "prop_interaction_crossing",
    "prop_interaction_stationary",
    "prop_interaction_unknown",
    "start_crossing",
    # Comptages annotés par frame
    "n_elderly",
    "n_children",
    "n_standing",
    "n_running",
    "n_groups",
    "n_crossing",
    "n_pedestrians_crossing",
    "n_pedestrians_opposite",
    "n_pedestrians_same_direction",
    "n_pedestrians_stationary",
    "n_cyclists_crossing",
    # Labels contextuels
    "VRU_AGE_GROUP_LABEL",
    "VRU_GAIT_LABEL",
    "VRU_GROUP_SIZE_LABEL",
    "WEATHER_LABEL",
    "LIGHTING_LABEL",
    "SURFACE_CONDITION_LABEL",
    # Intersection
    "at_intersection",
    # Environnement
    "environment_type",
    # Pente / élévation (issues de road_with_slope.geojson)
    "slope_signed_pct",     # signé : positif = montée dans le sens du trajet
    "slope_pct_max",        # magnitude de pente du polygone (toujours >= 0)
    "elev_mean_m",
]
        
        frame_df = frame_df[[c for c in keep if c in frame_df.columns]].copy()

        all_rows.append(frame_df)
        print(f"   ✔  {len(frame_df)} lignes ajoutées")

    except Exception as exc:
        import traceback
        print(f"   ❌ Erreur : {exc}")
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# ÉTAPE C — Jointure largeur de route + participants + variables temporelles
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ÉTAPE C — Jointure road_width + participants + variables temporelles")
print("=" * 65)

if not all_rows:
    print("Aucune donnée valide — dataset non généré.")
else:
    dataset = pd.concat(all_rows, ignore_index=True)

    # Jointure road_width
    dataset = dataset.merge(road_width_df, on=["source", "frame"], how="left")

    # turn_label : 'none' par défaut (frames hors GPS ou non proches d'un virage)
    if "turn_label" in dataset.columns:
        dataset["turn_label"] = dataset["turn_label"].fillna("none")

    n_missing_rw = dataset["road_width_perp_m"].isna().sum()
    if n_missing_rw > 0:
        print(f"  ⚠  {n_missing_rw} lignes sans largeur de route — remplissage par valeur la plus proche (ffill/bfill)")
        dataset = dataset.sort_values(["source", "frame"])
        for col in ["road_width_perp_m", "environment_type",
                    "slope_signed_pct", "slope_pct_max", "elev_mean_m"]:
            if col in dataset.columns:
                dataset[col] = (
                    dataset.groupby("source")[col]
                    .transform(lambda s: s.ffill().bfill())
                )
        n_still_missing = dataset["road_width_perp_m"].isna().sum()
        if n_still_missing > 0:
            print(f"  ⚠  {n_still_missing} lignes encore NaN (source entièrement sans GPS)")

    print("\n" + "=" * 65)
    print("ÉTAPE D — Jointure riders + variables temporelles")
    print("=" * 65)

    temporal_cols = [
        "timestamp_dt", "date", "hour", "day_of_week", "day_name",
        "is_weekend", "time_of_day", "month", "season",
    ]

    source_meta_df = pd.DataFrame([
        {
            "source":        src,
            "_device_key":   m["device_key"],
            "_vague_key":    m["vague_key"],
            **{c: m.get(c) for c in temporal_cols},
        }
        for src, m in source_meta.items()
    ])
    dataset = dataset.merge(source_meta_df, on="source", how="left")

    rider_cols_available = [c for c in
        ["device_key", "vague_key", "genre", "age", "experience", "nb_trajets", "distance_km"]
        if c in participants.columns]

    participants_sub = participants[rider_cols_available].drop_duplicates(
        subset=["device_key", "vague_key"]
    )

    print("  Clés source extraites :")
    unique_keys = dataset[["_device_key", "_vague_key"]].drop_duplicates()
    for _, row in unique_keys.iterrows():
        dk, vk = row["_device_key"], row["_vague_key"]
        match = participants_sub[
            (participants_sub["device_key"] == dk) &
            (participants_sub["vague_key"] == vk)
        ]
        status = f"✔  {len(match)} correspondance(s)" if len(match) > 0 else "⚠  AUCUNE correspondance"
        print(f"    device={dk!r:15s}  vague={vk!r:8s}  →  {status}")

    dataset = dataset.merge(
        participants_sub.rename(columns={
            "device_key": "_device_key",
            "vague_key":  "_vague_key",
        }),
        on=["_device_key", "_vague_key"],
        how="left",
    )

    dataset["rider_id"] = dataset["_device_key"] + "_" + dataset["_vague_key"]
    dataset = dataset.drop(columns=["_device_key", "_vague_key"])

    n_missing_rider = dataset["genre"].isna().sum() if "genre" in dataset.columns else 0
    if n_missing_rider > 0:
        print(f"\n  ⚠  {n_missing_rider} lignes sans correspondance rider")
    else:
        print(f"\n  ✔  Toutes les lignes ont un rider correspondant")

    # ── Jointure résumé encounters par trajet ─────────────────────────────
    enc_summary_df = pd.DataFrame([
        {"source": src, **vals} for src, vals in encounter_summary_map.items()
    ])
    if not enc_summary_df.empty:
        dataset = dataset.merge(enc_summary_df, on="source", how="left")

    # ── Résumé couverture temporelle ──────────────────────────────────────
    n_with_ts = dataset["timestamp_dt"].notna().sum()
    n_total   = len(dataset)
    print(f"\n  ── Couverture variables temporelles ──")
    print(f"  Lignes avec timestamp    : {n_with_ts} / {n_total} "
          f"({100*n_with_ts/n_total:.1f} %)")

    if "time_of_day" in dataset.columns:
        tod_counts = dataset["time_of_day"].value_counts(dropna=False)
        print("  Distribution time_of_day :")
        for label, count in tod_counts.items():
            print(f"    {str(label):12s} : {count}")

    if "is_weekend" in dataset.columns:
        wd = dataset["is_weekend"].value_counts(dropna=False)
        for val, count in wd.items():
            print(f"  is_weekend={val} : {count}")

    if "day_name" in dataset.columns:
        dn = dataset["day_name"].value_counts(dropna=False)
        print("  Jours de la semaine :")
        for val, count in dn.items():
            print(f"    {str(val):12s} : {count}")

    # ── Résumé couverture proportions ─────────────────────────────────────
    print(f"\n  ── Couverture proportions frame-level ──")
    for col in [
        "prop_vru_pedestrian",
        "prop_vru_cyclist",
        "prop_vru_e_scooter",
        "prop_interaction_same_direction",
        "prop_interaction_opposite_direction",
        "prop_interaction_crossing",
    ]:
        if col in dataset.columns:
            print(f"  {col:35s} : {dataset[col].notna().sum()} lignes non nulles")

    # Tri logique
    sort_cols = ["source", "frame"]
    dataset = dataset.sort_values(sort_cols).reset_index(drop=True)

    dataset.to_csv(OUTPUT_FILE, index=False)

    # ── Export bbox-level (une ligne par détection confirmée + distance) ──
    if all_detections:
        detections = pd.concat(all_detections, ignore_index=True)
        detections = detections.sort_values(["source", "frame", "track_id"]).reset_index(drop=True)
        detections.to_csv(DETECTIONS_FILE, index=False)
        n_det      = len(detections)
        n_with_dist = int(detections["distance_m"].notna().sum())
        print(f"\nDétections (bbox-level) exportées : {DETECTIONS_FILE}")
        print(f"  Lignes              : {n_det}")
        print(f"  Distances prédites  : {n_with_dist} ({100*n_with_dist/max(n_det,1):.1f} %)")
        print(f"  Trajets             : {detections['source'].nunique()}")
    else:
        print("\n⚠  Aucune détection bbox-level — fichier détections non exporté")

    print(f"\nDataset exporté  : {OUTPUT_FILE}")
    print(f"  Lignes         : {len(dataset)}")
    print(f"  Trajets        : {dataset['source'].nunique()}")
    print(f"  Colonnes       : {list(dataset.columns)}")

    # ── Résumé par trajet ─────────────────────────────────────────────────
    agg_dict = dict(
        n_lignes                    = ("frame", "count"),
        n_frames_uniq               = ("frame", "nunique"),
        speed_mean                  = ("speed_kmh", "mean"),
        speed_max                   = ("speed_kmh", "max"),
        speed_kalman_mean           = ("speed_kmh_kalman", "mean"),
        speed_kalman_max            = ("speed_kmh_kalman", "max"),
        n_vru_total_mean            = ("n_vru_total", "mean"),
        n_vru_total_max             = ("n_vru_total", "max"),
        road_width_mean             = ("road_width_perp_m", "mean"),
        road_width_min              = ("road_width_perp_m", "min"),
        road_width_max              = ("road_width_perp_m", "max"),
        prop_vru_pedestrian_mean    = ("prop_vru_pedestrian", "mean"),
        prop_vru_cyclist_mean       = ("prop_vru_cyclist", "mean"),
        prop_vru_e_scooter_mean     = ("prop_vru_e_scooter", "mean"),
        prop_interaction_same_direction_mean = ("prop_interaction_same_direction", "mean"),
        prop_interaction_opposite_direction_mean = ("prop_interaction_opposite_direction", "mean"),
        prop_interaction_crossing_mean = ("prop_interaction_crossing", "mean"),
        prop_interaction_stationary_mean = ("prop_interaction_stationary", "mean"),
        # Temporel
        date                        = ("date", "first"),
        hour                        = ("hour", "first"),
        day_name                    = ("day_name", "first"),
        is_weekend                  = ("is_weekend", "first"),
        time_of_day                 = ("time_of_day", "first"),
        season                      = ("season", "first"),
    )

    for col in ["genre", "age", "experience", "nb_trajets", "distance_km"]:
        if col in dataset.columns:
            agg_dict[col] = (col, "first")

    for col in [
        "n_encounters_annotated",
        "pct_same_direction",
        "pct_opposite_direction",
        "pct_crossing",
        "pct_stationary",
        "pct_unknown_interaction",
        "pct_gait_standing",
        "pct_gait_walking",
        "pct_gait_running",
        "pct_gait_unknown",
        "pct_age_child",
        "pct_age_adult",
        "pct_age_elderly",
        "pct_age_unknown",
    ]:
        if col in dataset.columns:
            agg_dict[col] = (col, "first")

    summary = dataset.groupby("source").agg(**agg_dict).round(3)

    print("\nRésumé par trajet :")
    print(summary.to_string())

    print("\n  ── Variables encounter agrégées ──")
    enc_pct_cols = [c for c in summary.columns
                    if c.startswith("pct_") or c == "n_encounters_annotated"]
    if enc_pct_cols:
        print(summary[enc_pct_cols].to_string())