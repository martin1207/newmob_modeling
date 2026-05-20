"""
Overlay des annotations VRU sur toutes les vidéos escooter.
Affiche uniquement les bbox CODÉES (encounter avec CONFIRM == 1) avec :
  - vru_type
  - gait
  - age_group
  - encounter_type (= INTERACTION_TYPE)

Inputs (par clip) :
  <ESCOOTER_DIR>/<clip>.mp4
  <CODEBOOK>/<clip>_debug_autodetect.csv
  <CODEBOOK>/<clip>_rater2_encounters_debug_encounters.csv
Output :
  <OUT_DIR>/<clip>_annotated_codes.mp4
"""

import os
import cv2
import csv
import glob

import numpy as np

ROOT          = "/Volumes/My Passport/NEWMOB"
ESCOOTER_DIR  = f"{ROOT}/escooter"
CODEBOOK      = f"{ROOT}/codebookescooter"
OUT_DIR       = f"{ROOT}/e_scooter_video_annotated"

ENC_SUFFIX     = "_rater2_encounters_debug_encounters.csv"
OBS_SUFFIX     = "_rater2_encounters_obstacle_zones.csv"
AUTODET_SUFFIX = "_debug_autodetect.csv"
SPEED_SUFFIX   = "_corrected_with_offset.csv"   # IMU + GPS dans <ESCOOTER_DIR>
VIDEO_SUFFIX   = "_canny.mp4"                   # vidéo source (filtre Canny)
OUT_SUFFIX     = "_annotated_codes.mp4"

OVERWRITE = False   # True pour recalculer même si l'output existe déjà

# ── Camembert de direction (choix par seconde, basé sur Δyaw = ∫GyrZ dt) ────
STEERING_PIE_WINDOW_S      = 1.0   # fenêtre d'évaluation du choix de direction (s)
STEERING_PIE_THRESHOLD_DEG = 10.0  # |Δyaw| sur la fenêtre < seuil ⇒ "tout droit"
                                   # sinon : 'gauche' si Δyaw > 0, sinon 'droite'
GYRZ_LEFT_POSITIVE         = True  # convention : True si GyrZ > 0 ⇒ virage à gauche
                                   # (à inverser si l'overlay annote l'inverse)

# ── Modèle de distance (bbox → mètres) ───────────────────────────────────────
DISTANCE_MODEL_PATH = (
    "/Users/martin.dejaeghere/PhD/NEWMOB-main/model/saved_models/"
    "distance_model_20260506_093716.joblib"
)
# Ordre des features par défaut (cf. .meta.json) ; sera écrasé par la valeur
# stockée dans le bundle joblib si présente.
DEFAULT_DISTANCE_FEATURES = ["cx", "cy", "w", "h", "bottom_y", "aspect", "area", "inv_h"]
_distance_bundle = None  # cache : (estimator, features) ou False si échec


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


def load_speed_by_frame(speed_csv):
    """{frame: VitGPS(km/h)} ; ignore les lignes vides."""
    speed = {}
    if not os.path.exists(speed_csv):
        return speed
    f, reader = _open_dictreader(speed_csv)
    with f:
        for row in reader:
            try:
                fr = int(float(row["frame"]))
                v  = float(row["VitGPS(km/h)"])
            except (ValueError, KeyError):
                continue
            speed[fr] = v
    return speed


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
    enc_csv     = f"{CODEBOOK}/{clip_name}{ENC_SUFFIX}"
    obs_csv     = f"{CODEBOOK}/{clip_name}{OBS_SUFFIX}"
    speed_csv   = f"{ESCOOTER_DIR}/{clip_name}{SPEED_SUFFIX}"
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

    print(f"[RUN ] {clip_name}  {width}×{height} @ {fps:.1f}fps  {total}f  "
          f"bbox={n_boxes}  obs_zones={len(obstacles)}  "
          f"speed_pts={len(speed_by_frame)}  steer L/R={n_left}/{n_right}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for (x1, y1, x2, y2, meta, dist) in vrus_by_frame.get(frame_idx, []):
            draw_box(frame, x1, y1, x2, y2, meta, dist)
        zid = obstacle_id_at(frame_idx, obstacles)
        if zid is not None:
            draw_obstacle_banner(frame, zid)
        # Camembert de direction (toujours dessiné, même si pas de signal IMU)
        draw_steering_pie(frame, choices_by_frame.get(frame_idx))
        draw_frame_number(frame, frame_idx, total)
        v = speed_by_frame.get(frame_idx)
        if v is not None:
            draw_speed(frame, v)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] {clip_name} → {os.path.basename(video_out)}")


def discover_clips():
    """Clips ayant à la fois encounters + autodetect + vidéo Canny."""
    enc_files = glob.glob(f"{CODEBOOK}/*{ENC_SUFFIX}")
    clips = []
    for p in enc_files:
        name = os.path.basename(p)[: -len(ENC_SUFFIX)]
        if (os.path.exists(f"{CODEBOOK}/{name}{AUTODET_SUFFIX}")
                and os.path.exists(f"{ESCOOTER_DIR}/{name}{VIDEO_SUFFIX}")):
            clips.append(name)
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
