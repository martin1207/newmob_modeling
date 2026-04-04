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
  2. Exclusion des frames en virage (|GyrZ| > TURN_THRESHOLD_DEG_S)
  3. Exclusion des frames chevauchant des zones d'obstacle
  4. Offset GPS de GPS_OFFSET_FRAMES frames

Variables frame-level ajoutées depuis debug_encounters :
  - n_vru_total
  - proportions de VRU_TYPE
  - proportions de INTERACTION_TYPE
  - labels contextuels (gait, age group, weather, etc.)

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
GPS_OFFSET_FRAMES    = 60
TURN_THRESHOLD_DEG_S = 20
PERP_HALF_LENGTH     = 100
DIRECTION_WINDOW     = 5

# Tranches horaires : (label, heure_debut_incluse, heure_fin_exclue)
TIME_SLOTS = [
    ("Night",     0,  6),   # 00:00 – 05:59
    ("Morning",   6, 12),   # 06:00 – 11:59
    ("Afternoon", 12, 18),  # 12:00 – 17:59
    ("Evening",   18, 24),  # 18:00 – 23:59
]

CODEBOOK_DIR         = '/Volumes/My Passport/NEWMOB/codebookescooter/'
IMU_DIR              = '/Volumes/My Passport/NEWMOB/escooter/'
ROAD_GPKG            = '/Volumes/My Passport/NEWMOB/road.gpkg'
PARTICIPANTS_XLS     = '/Volumes/My Passport/NEWMOB/participants_NewMob_Electromob_VAE_TE.xlsx'
OUTPUT_FILE          = '/Volumes/My Passport/NEWMOB/clean_dataset.csv'
INTERSECTIONS_CSV    = '/Volumes/My Passport/NEWMOB/clips_intersections/recap_intersections.csv'


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
            df = pd.read_csv(path)
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
    try:
        df = pd.read_csv(candidates[0])
        df.columns = df.columns.str.strip()

        if "CONFIRM" in df.columns:
            df = df[pd.to_numeric(df["CONFIRM"], errors="coerce") == 1]

        print(f"   Codebook  : {os.path.basename(candidates[0])} — {len(df)} lignes confirmées")
        return df
    except Exception as e:
        print(f"   ⚠  Codebook non chargé : {e}")
        return None

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
            "_is_running":           int(gait_label == "Running"),
            "_is_group":             int(group_size_label == "Group (3+)"),
            "_is_crossing":          int(interaction_label == "Crossing"),
            "_is_ped_crossing":      int(vru_label == "Pedestrian" and interaction_label == "Crossing"),
            "_is_ped_opposite":      int(vru_label == "Pedestrian" and interaction_label == "Opposite-direction"),
            "_is_cyclist_crossing":  int(vru_label == "Cyclist" and interaction_label == "Crossing"),
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
        "_is_running":          "n_running",
        "_is_group":            "n_groups",
        "_is_crossing":         "n_crossing",
        "_is_ped_crossing":     "n_pedestrians_crossing",
        "_is_ped_opposite":     "n_pedestrians_opposite",
        "_is_cyclist_crossing": "n_cyclists_crossing",
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
    candidate_idx = list(sindex.intersection(point.bounds))
    candidates = road_gdf.iloc[candidate_idx]

    containing = candidates[candidates.contains(point)]
    if len(containing) > 0:
        return containing.iloc[0].geometry

    if len(candidates) > 0:
        distances = candidates.geometry.distance(point)
        return candidates.loc[distances.idxmin(), "geometry"]

    distances = road_gdf.geometry.distance(point)
    return road_gdf.loc[distances.idxmin(), "geometry"]


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

    imu_candidates = (
        glob.glob(os.path.join(IMU_DIR, f'{prefix}*corrected_with_offset*.csv'))
        or glob.glob(os.path.join(IMU_DIR, f'{prefix}*.csv'))
    )
    if not imu_candidates:
        print(f"⚠ Aucun IMU trouvé pour prefix = {prefix}")
        print(f"   Pattern 1: {os.path.join(IMU_DIR, f'{prefix}*corrected_with_offset*.csv')}")
        print(f"   Pattern 2: {os.path.join(IMU_DIR, f'{prefix}*.csv')}")
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

    for i in range(len(group)):
        point   = group.geometry.iloc[i]
        polygon = find_polygon_for_point(point, road_m, sindex)

        i_prev = max(0, i - DIRECTION_WINDOW)
        i_next = min(len(group) - 1, i + DIRECTION_WINDOW)

        if i_prev == i_next:
            widths.append(np.nan)
            continue

        dx = group.geometry.iloc[i_next].x - group.geometry.iloc[i_prev].x
        dy = group.geometry.iloc[i_next].y - group.geometry.iloc[i_prev].y
        widths.append(polygon_width_at_point(point, polygon, dx, dy))

    group = group.copy()
    group["road_width_perp_m"] = widths
    group["road_width_perp_m"] = group["road_width_perp_m"].interpolate(
        method="linear", limit_direction="both"
    )

    for _, row in group[["frame", "road_width_perp_m"]].iterrows():
        road_width_records.append({
            "source":            src,
            "frame_imu_raw":     row["frame"],
            "road_width_perp_m": row["road_width_perp_m"],
        })

road_width_df = pd.DataFrame(road_width_records)
road_width_df["frame"] = road_width_df["frame_imu_raw"] - GPS_OFFSET_FRAMES
road_width_df = road_width_df[["source", "frame", "road_width_perp_m"]]

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
seen_prefixes = set()
for enc_path in sorted(encounter_files):
    prefix = extract_prefix(enc_path)
    if prefix in seen_prefixes:
        print(f"⚠ Prefix déjà traité, ignoré : {prefix}")
        continue
    seen_prefixes.add(prefix)

    imu_candidates = (
        glob.glob(os.path.join(IMU_DIR, f'{prefix}*corrected_with_offset*.csv'))
        or glob.glob(os.path.join(IMU_DIR, f'{prefix}*.csv'))
    )
    if not imu_candidates:
        print(f"⚠  [{prefix}] Pas de fichier IMU — trajet ignoré")
        continue

    imu_path = imu_candidates[0]

    obs_candidates     = glob.glob(os.path.join(CODEBOOK_DIR, f'{prefix}*obstacle_zones*.csv'))
    df_obstacles       = pd.read_csv(obs_candidates[0]) if obs_candidates else None
    obs_intervals      = load_obstacle_intervals(df_obstacles)
    inter_intervals    = load_intersection_intervals_for_prefix(prefix, INTERSECTIONS_CSV)

    print(f"\n▶  {prefix}")
    print(f"   IMU           : {os.path.basename(imu_path)}")
    print(f"   Obstacles     : {os.path.basename(obs_candidates[0]) if obs_candidates else 'aucun'}")
    print(f"   Intersections : {len(inter_intervals)} intervalle(s)")

    df_codebook    = load_codebook_annotated(prefix, CODEBOOK_DIR)
    session_ctx    = load_session_context(prefix, CODEBOOK_DIR)

    try:
        imu = pd.read_csv(imu_path)

        required_imu = {"frame", "VitGPS(km/h)", "GyrZ(deg/s)"}
        missing_imu = required_imu - set(imu.columns)
        if missing_imu:
            print(f"   ⚠  Colonnes IMU manquantes : {missing_imu} — ignoré")
            continue

        imu = imu.copy()
        imu["frame_corrected"] = imu["frame"] - GPS_OFFSET_FRAMES

        valid_frames = set(
            imu.loc[
                imu["GyrZ(deg/s)"].isna() | (imu["GyrZ(deg/s)"].abs() <= TURN_THRESHOLD_DEG_S),
                "frame_corrected",
            ]
        )

        frame_df = build_frame_level_from_debug_encounters(df_codebook)

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

        n0 = len(frame_df)
        if frame_df.empty:
            print("   ℹ  Aucune frame issue de debug_encounters")
            continue

        frame_df = frame_df[frame_df["frame"].isin(valid_frames)].copy()
        print(f"   Virages   : {n0 - len(frame_df):>4} lignes retirées → {len(frame_df)} restantes")

        n1 = len(frame_df)
        if obs_intervals:
            frame_raw = frame_df["frame"] + GPS_OFFSET_FRAMES
            mask_obs  = ~frame_raw.apply(lambda f: frame_in_obstacle(f, obs_intervals))
            frame_df  = frame_df.loc[mask_obs].copy()
        print(f"   Obstacles : {n1 - len(frame_df):>4} lignes retirées → {len(frame_df)} restantes")

        if frame_df.empty:
            print("   ℹ  Plus aucune frame après filtrage")
            continue

        # Jointure IMU (vitesse, gyro)
        imu_sub = (
            imu[["frame_corrected", "VitGPS(km/h)", "GyrZ(deg/s)"]]
            .rename(columns={
                "frame_corrected": "frame",
                "VitGPS(km/h)":    "speed_kmh",
                "GyrZ(deg/s)":     "gyrz_deg_s",
            })
        )
        frame_df = frame_df.merge(imu_sub, on="frame", how="left")
        frame_df.insert(0, "source", prefix)

        keep = [
    "source", "frame",
    "speed_kmh", "gyrz_deg_s",
    "n_vru_total",
    "n_pedestrians",
    "n_cyclists",
    "n_escooters",
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
    "n_running",
    "n_groups",
    "n_crossing",
    "n_pedestrians_crossing",
    "n_pedestrians_opposite",
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
]
        
        frame_df = frame_df[[c for c in keep if c in frame_df.columns]].copy()

        all_rows.append(frame_df)
        print(f"   ✔  {len(frame_df)} lignes ajoutées")

    except Exception as exc:
        print(f"   ❌ Erreur : {exc}")


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

    n_missing_rw = dataset["road_width_perp_m"].isna().sum()
    if n_missing_rw > 0:
        print(f"  ⚠  {n_missing_rw} lignes sans largeur de route")

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