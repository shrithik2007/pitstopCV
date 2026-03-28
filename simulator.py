# -*- coding: utf-8 -*-
"""
ORACLE RED BULL F1 PIT WALL SIMULATOR  -  CQ Hacks
====================================================
Controls:
  Pinch (index+thumb < 40px) near car to grab.
  Drag across wall (x=250) to start.
  Hit CP1->CP2->CP3 in order, return to GARAGE for a valid lap.
  Release pinch on track = PENALTY (snap back).
  Press Q to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import math, random, time, os
from collections import deque

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
    pygame.mixer.init()
    _PYGAME_OK = True
except Exception as _e:
    print(f"[WARN] pygame unavailable: {_e}")
    _PYGAME_OK = False

# ── BGR Colors ────────────────────────────────────────────────
NAVY     = (150,   0,   6)
YELLOW   = (  0, 204, 255)
RED      = (  0,   0, 255)
WHITE    = (255, 255, 255)
BLACK    = (  0,   0,   0)
GREEN    = (  0, 255,   0)
ORANGE   = (  0, 140, 255)
DARK_BG  = ( 20,  10,  10)
GRAY     = (120, 120, 120)
LGRAY    = (200, 200, 200)

# ── Constants ─────────────────────────────────────────────────
WALL_X      = 250
GARAGE_X    = 125
GARAGE_Y    = 300
PINCH_DIST  = 40
GRAB_DIST   = 80
CP_RADIUS   = 40
CHECKPOINTS = [(400, 150), (800, 350), (400, 550)]

# ── MediaPipe ─────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing: {MODEL_PATH}\n"
        "Download: python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task','hand_landmarker.task')\""
    )

_opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO,
)
hand_lm = mp_vision.HandLandmarker.create_from_options(_opts)

# ── Camera ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
if not cap.isOpened():
    hand_lm.close()
    raise RuntimeError("Cannot open webcam.")
FW, FH = 1280, 720
print(f"[CAM] Forced to {FW}x{FH}")

# ── Audio ─────────────────────────────────────────────────────
snd_engine = snd_box = None
if _PYGAME_OK:
    for fname, tag in [("engine.mp3","engine"),("box_box.mp3","box")]:
        p = os.path.join(HERE, fname)
        if os.path.exists(p):
            try:
                s = pygame.mixer.Sound(p)
                if tag == "engine": snd_engine = s
                else:               snd_box    = s
            except Exception as ex: print(f"[WARN] {p}: {ex}")
        else: print(f"[WARN] Not found: {p}")

def _play(s, loops=0):
    try: s and s.play(loops=loops)
    except: pass
def _stop(s):
    try: s and s.stop()
    except: pass

# ── Game State ────────────────────────────────────────────────
car_x, car_y   = GARAGE_X, GARAGE_Y
is_dragging    = False
race_state     = "IDLE"          # IDLE | RACING | INVALID
start_time     = 0.0
current_time   = 0.0
cp_cleared     = [False, False, False]
leaderboard    = []
drs_speed      = 300
session_msg    = "AWAITING DEPLOYMENT"
session_color  = YELLOW
flash_until    = 0.0
invalid_until  = 0.0
trail          = deque(maxlen=20)
frame_ts_ms    = 0

def set_msg(text, color):
    global session_msg, session_color
    session_msg, session_color = text, color

def reset_car():
    global car_x, car_y, cp_cleared, current_time
    car_x, car_y = GARAGE_X, GARAGE_Y
    cp_cleared   = [False, False, False]
    current_time = 0.0
    trail.clear()

def reset_lap():
    global cp_cleared, current_time
    cp_cleared   = [False, False, False]
    current_time = 0.0
    trail.clear()

# ── Procedural F1 Car ─────────────────────────────────────────
def draw_f1_car(frame, cx, cy):
    # Fuselage
    fuselage = np.array([
        [cx-10, cy-55],[cx+10, cy-55],
        [cx+18, cy+50],[cx-18, cy+50]], np.int32)
    cv2.fillPoly(frame, [fuselage], NAVY)

    # Front wing
    fw = np.array([
        [cx-50, cy-48],[cx+50, cy-48],
        [cx+42, cy-33],[cx-42, cy-33]], np.int32)
    cv2.fillPoly(frame, [fw], NAVY)
    # Front endplates (Yellow)
    cv2.fillPoly(frame, [np.array([
        [cx-57,cy-52],[cx-45,cy-52],[cx-42,cy-27],[cx-54,cy-27]], np.int32)], YELLOW)
    cv2.fillPoly(frame, [np.array([
        [cx+45,cy-52],[cx+57,cy-52],[cx+54,cy-27],[cx+42,cy-27]], np.int32)], YELLOW)

    # Rear wing
    rw = np.array([
        [cx-44, cy+34],[cx+44, cy+34],
        [cx+38, cy+50],[cx-38, cy+50]], np.int32)
    cv2.fillPoly(frame, [rw], NAVY)
    # Rear endplates (Red)
    cv2.fillPoly(frame, [np.array([
        [cx-48,cy+31],[cx-38,cy+31],[cx-34,cy+55],[cx-46,cy+55]], np.int32)], RED)
    cv2.fillPoly(frame, [np.array([
        [cx+38,cy+31],[cx+48,cy+31],[cx+46,cy+55],[cx+34,cy+55]], np.int32)], RED)

    # Tires (dark gray)
    tc = (28, 28, 28)
    for rect in [
        ((cx-36, cy-50),(cx-19, cy-22)),
        ((cx+19, cy-50),(cx+36, cy-22)),
        ((cx-34, cy+18),(cx-17, cy+50)),
        ((cx+17, cy+18),(cx+34, cy+50)),
    ]:
        cv2.rectangle(frame, rect[0], rect[1], tc, -1)
        cv2.rectangle(frame, rect[0], rect[1], GRAY, 1)

    # Cockpit
    cv2.ellipse(frame, (cx, cy-6), (9,12), 0, 0, 360, WHITE, -1)
    # Center stripe
    cv2.line(frame, (cx, cy-55), (cx, cy+50), YELLOW, 3)
    # Nose tip
    cv2.circle(frame, (cx, cy-55), 5, RED, -1)

# ── Trail ─────────────────────────────────────────────────────
def draw_trail(frame):
    pts = list(trail)
    for i in range(1, len(pts)):
        ratio = i / len(pts)
        cv2.line(frame, pts[i-1], pts[i],
                 (0, int(140*(1-ratio)), 255), max(1,int(8*ratio)), cv2.LINE_AA)

# ── Zones & Wall ──────────────────────────────────────────────
def draw_zones(frame):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (WALL_X, h), (60, 15, 8), -1)
    cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
    seg = 30
    for y in range(0, h, seg*2):
        cv2.rectangle(frame, (WALL_X-4, y),   (WALL_X+4, y+seg),   WHITE, -1)
    for y in range(seg, h, seg*2):
        cv2.rectangle(frame, (WALL_X-4, y),   (WALL_X+4, y+seg),   RED,   -1)
    cv2.putText(frame, "GARAGE", (10, h-20),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, LGRAY, 2, cv2.LINE_AA)
    cv2.putText(frame, "TRACK", (WALL_X+15, h-20),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, GREEN, 2, cv2.LINE_AA)

# ── Checkpoints ───────────────────────────────────────────────
def draw_checkpoints(frame):
    next_cp = next((i for i,c in enumerate(cp_cleared) if not c), None)
    for i, (cx, cy) in enumerate(CHECKPOINTS):
        if cx >= FW or cy >= FH: continue
        lbl = f"CP{i+1}"
        if cp_cleared[i]:
            cv2.circle(frame, (cx,cy), CP_RADIUS, GREEN, -1)
            cv2.circle(frame, (cx,cy), CP_RADIUS, WHITE, 2)
            cv2.putText(frame, lbl, (cx-15,cy+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)
        else:
            col = YELLOW if i == next_cp else GRAY
            cv2.circle(frame, (cx,cy), CP_RADIUS, col, 3)
            cv2.putText(frame, lbl, (cx-15,cy+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

# ── HUD ───────────────────────────────────────────────────────
def draw_hud(frame):
    px1,py1,px2,py2 = 10,10,390,240
    ov = frame.copy()
    cv2.rectangle(ov, (px1,py1),(px2,py2), DARK_BG, -1)
    cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame,(px1,py1),(px2,py2), YELLOW, 2)
    cv2.rectangle(frame,(px1+3,py1+3),(px2-3,py2-3), NAVY, 1)
    cv2.rectangle(frame,(px1,py1),(px2,py1+5), RED, -1)

    cv2.putText(frame, "ORACLE STRATEGY WALL",
                (20,42), cv2.FONT_HERSHEY_DUPLEX, 0.68, YELLOW, 2, cv2.LINE_AA)
    cv2.line(frame,(20,52),(px2-10,52), YELLOW, 1)

    drs_txt = f"{drs_speed} KM/H" if race_state=="RACING" else "--- KM/H"
    cv2.putText(frame, f"DRS : {drs_txt}",
                (20,78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LGRAY, 2, cv2.LINE_AA)

    t_txt = f"{current_time:6.2f}s" if race_state=="RACING" else " --.-  "
    t_col = YELLOW if race_state=="RACING" else LGRAY
    cv2.putText(frame, f"CURRENT : {t_txt}",
                (20,108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_col, 2, cv2.LINE_AA)

    cv2.line(frame,(20,118),(px2-10,118), GRAY, 1)
    lb = sorted(leaderboard)
    for rank, sfx in enumerate(["1ST","2ND","3RD"]):
        if rank < len(lb):
            txt, col = f"{sfx}: {lb[rank]:.2f}s", GREEN
        else:
            txt, col = f"{sfx}: --.-", GRAY
        cv2.putText(frame, txt, (20, 140+rank*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv2.LINE_AA)

    cv2.line(frame,(20,218),(px2-10,218), GRAY, 1)
    cv2.putText(frame, session_msg,
                (20,236), cv2.FONT_HERSHEY_SIMPLEX, 0.52, session_color, 2, cv2.LINE_AA)

# ── Flash overlay ─────────────────────────────────────────────
def draw_flash(frame):
    now = time.time()
    if now < flash_until:
        alpha = (flash_until - now) / 0.6
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]), RED, -1)
        cv2.addWeighted(ov, alpha*0.45, frame, 1-alpha*0.45, 0, frame)

# ── Main Loop ─────────────────────────────────────────────────
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("[WARN] Camera read failed — exiting.")
        break

    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)
    fh, fw = 720, 1280

    draw_zones(frame)

    # Hand tracking
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts_ms += 33
    result = hand_lm.detect_for_video(mp_img, frame_ts_ms)

    pinching         = False
    pinch_x, pinch_y = car_x, car_y

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        lm8 = lms[8]; ix, iy = int(lm8.x*fw), int(lm8.y*fh)
        lm4 = lms[4]; tx, ty = int(lm4.x*fw), int(lm4.y*fh)
        dist = math.hypot(ix-tx, iy-ty)
        mx, my = (ix+tx)//2, (iy+ty)//2
        if dist < PINCH_DIST:
            pinching       = True
            pinch_x, pinch_y = mx, my
            cv2.circle(frame,(mx,my),10,GREEN,-1)
            cv2.circle(frame,(mx,my),13,GREEN, 2)
        else:
            cv2.circle(frame,(ix,iy),10,RED,-1)

    # Dragging
    prev_x = car_x
    if pinching:
        if math.hypot(pinch_x-car_x, pinch_y-car_y) <= GRAB_DIST:
            is_dragging = True
    else:
        if is_dragging and car_x >= WALL_X:
            # PENALTY 2: dropped on track
            _stop(snd_engine)
            reset_car()
            race_state    = "INVALID"
            flash_until   = time.time() + 0.6
            invalid_until = time.time() + 2.0
            set_msg("PENALTY: CAR DROPPED ON TRACK!", RED)
        is_dragging = False

    if is_dragging:
        car_x, car_y = pinch_x, pinch_y
        trail.append((car_x, car_y))

        # GARAGE -> TRACK: start race
        if prev_x < WALL_X and car_x >= WALL_X and race_state in ("IDLE", "INVALID"):
            race_state   = "RACING"
            start_time   = time.time()
            current_time = 0.0
            cp_cleared   = [False, False, False]
            _play(snd_engine, loops=-1)
            set_msg("GO GO GO!", GREEN)

        # TRACK -> GARAGE: end lap
        if prev_x >= WALL_X and car_x < WALL_X and race_state == "RACING":
            if all(cp_cleared):
                lap = round(current_time, 2)
                leaderboard.append(lap)
                _stop(snd_engine); _play(snd_box)
                race_state = "IDLE"
                reset_lap()
                set_msg(f"LAP COMPLETE: {lap:.2f}s", GREEN)
            else:
                # PENALTY 1: missed checkpoints
                _stop(snd_engine)
                flash_until   = time.time() + 0.6
                invalid_until = time.time() + 2.0
                race_state    = "INVALID"
                reset_lap()
                set_msg("LAP INVALIDATED - MISSED CHECKPOINTS", RED)

        # Checkpoint detection
        if race_state == "RACING":
            nxt = next((i for i,c in enumerate(cp_cleared) if not c), None)
            if nxt is not None:
                cpx, cpy = CHECKPOINTS[nxt]
                if math.hypot(car_x-cpx, car_y-cpy) < CP_RADIUS:
                    cp_cleared[nxt] = True
                    n = sum(cp_cleared)
                    set_msg(f"CHECKPOINT {n}/3 CLEARED!", YELLOW)

    # INVALID timeout
    if race_state == "INVALID" and time.time() > invalid_until:
        race_state = "IDLE"
        set_msg("AWAITING DEPLOYMENT", YELLOW)

    # Live timer + DRS
    if race_state == "RACING":
        current_time = time.time() - start_time
        drs_speed    = random.randint(280, 320)

    # Render
    draw_checkpoints(frame)
    draw_trail(frame)
    draw_f1_car(frame, car_x, car_y)
    draw_flash(frame)
    draw_hud(frame)

    cv2.imshow("CQ Hacks: Oracle Red Bull F1 Simulator", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] Quit.")
        break

# ── Cleanup ───────────────────────────────────────────────────
_stop(snd_engine)
cap.release()
cv2.destroyAllWindows()
hand_lm.close()
if _PYGAME_OK:
    pygame.mixer.quit()
print("[INFO] Session ended.")
