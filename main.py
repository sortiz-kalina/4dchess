import sys
import math
import time
import threading
import numpy as np
import cv2
import mediapipe as mp
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE, K_q, K_r, K_SPACE
from OpenGL.GL import *
from OpenGL.GLU import *

from chess_logic import ChessGame

#configs
WINDOW_W, WINDOW_H  = 1000, 720
CAM_INDEX           = 0
TARGET_FPS          = 60
ROT_SENSITIVITY     = 140.0   # degrees of rotation per unit nose deviation
ROT_LERP            = 0.10   # smoothing factor (lower = smoother / more lag)
PULSE_SPEED         = 2.5    # glow pulse in Hz
BOARD_TEX_SIZE      = 256
TUNNEL_DEPTH        = 8       # how many chess rings recede into the tunnel
TUNNEL_SCALE        = 1.8     # size of the nearest ring
TUNNEL_SPACING      = 3.0     # distance between rings
FOV                 = 90

COL_GLOW            = (1.0,  0.92, 0.1)   # yellow highlight
COL_BORDER          = (0.85, 0.85, 0.85)  # edge lines between faces

#abst tracking
class Tracking:
    lock          = threading.Lock()
    nose_x        = 0.5
    nose_y        = 0.5
    detected      = False
    pip_frame     = None
    finger_x      = 0.5
    finger_y      = 0.5
    hand_detected = False
    pinching      = False
    open_palm     = False   # open palm triggers wall switch

tracking = Tracking()

game = ChessGame()

last_wall_switch = 0.0

#camera thread continuously updates the tracking state with the latest nose position and frame.
def camera_thread():
    from mediapipe.tasks.python import vision
    from mediapipe.tasks import python as mp_python

    base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(0)
    print("Camera opened:", cap.isOpened())
    if not cap.isOpened():
        print("WARNING: webcam not found — tunnel runs without tracking.")
        return
    
    palm_frame_count = 0   
    PALM_FRAMES_REQUIRED = 8
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # face tracking
        results = detector.detect(mp_image)
        if results.face_landmarks:
            lm = results.face_landmarks[0][1]  # nose tip
            h, w = frame.shape[:2]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 9, (0, 200, 255), -1)
            with tracking.lock:
                tracking.nose_x   = lm.x
                tracking.nose_y   = lm.y
                tracking.detected = True
        else:
            with tracking.lock:
                tracking.detected = False

        # hand tracking — one single block
        hand_results = hand_detector.detect(mp_image)
        if hand_results.hand_landmarks:
            tip   = hand_results.hand_landmarks[0][12]  # middle fingertip
            thumb = hand_results.hand_landmarks[0][4]   # thumb tip
            h, w  = frame.shape[:2]
            cv2.circle(frame, (int(tip.x * w),   int(tip.y * h)),   9, (255, 100, 0), -1)  # finger dot
            cv2.circle(frame, (int(thumb.x * w), int(thumb.y * h)), 9, (255, 0,   0), -1)  # thumb dot
            dist = math.hypot(tip.x - thumb.x, tip.y - thumb.y)

            # open palm — skip thumb, only check 4 fingers for reliability
            # open palm — fingers must be clearly above knuckles by a margin
            lms        = hand_results.hand_landmarks[0]
            fingertips = [8, 12, 16, 20]
            knuckles   = [5,  9, 13, 17]
           #fingertip .04 above knuckle to remove false positives from natural slight finger bends  hand sizes and camera distances
            open_palm  = all((lms[kn].y - lms[ft].y) > 0.04 for ft, kn in zip(fingertips, knuckles))

            with tracking.lock:
                tracking.finger_x      = tip.x
                tracking.finger_y      = tip.y
                tracking.hand_detected = True
                tracking.pinching      = dist < 0.05
                tracking.open_palm     = open_palm
        else:
            with tracking.lock:
                tracking.hand_detected = False
                tracking.pinching      = False
                tracking.open_palm     = False

        # pip_frame updated LAST so all dots are already drawn
        with tracking.lock:
            tracking.pip_frame = frame.copy()

#chess board rendering
def make_chess_texture():
    img = np.zeros((BOARD_TEX_SIZE, BOARD_TEX_SIZE, 3), dtype=np.uint8)
    sq  = BOARD_TEX_SIZE // 8
    for r in range(8):
        for c in range(8):
            col = np.array([240, 217, 181], dtype=np.uint8) if (r+c) % 2 == 0 \
                  else np.array([81, 130, 74], dtype=np.uint8)
            img[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = col
    return np.flipud(img)

def upload_texture(arr):
    tid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    h, w = arr.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, arr.tobytes())
    return tid

# finger position to chess board mapping
def pos_to_square(x, y):
    """
    
    #MIGHT CHANGE LATER
    """
    RANGE = 0.4   # how much finger movement covers the whole board
    dx =  (x - 0.5) / RANGE
    dy = -((1.0 - y) - 0.5) / RANGE   
    col = int(np.clip((dx + 1) / 2 * 8, 0, 7))
    row = int(np.clip((dy + 1) / 2 * 8, 0, 7))
    return row, col

# graphics fns
TUNNEL_FACE_UVS = [(0,0),(1,0),(1,1),(0,1)]

def draw_pieces(walls, active_wall, game):
    def lerp(a, b, t):
        return a + (b - a) * t

    quadric = gluNewQuadric()
    for i, wall in enumerate(walls):
        board = game.boards[i]
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece:
                    # Compute center position of the square
                    sq = 1.0 / 8.0
                    u = (col + 0.5) * sq
                    v = (row + 0.5) * sq
                    p0 = np.array(wall[0], dtype=float)
                    p1 = np.array(wall[1], dtype=float)
                    p2 = np.array(wall[2], dtype=float)
                    p3 = np.array(wall[3], dtype=float)
                    # Bilinear interpolation for center
                    bottom = np.array([lerp(p0[j], p1[j], u) for j in range(3)])
                    top = np.array([lerp(p3[j], p2[j], u) for j in range(3)])
                    center = np.array([lerp(bottom[j], top[j], v) for j in range(3)])
                    glPushMatrix()
                    glTranslatef(*center)
                    # Orient based on wall
                    if i in [0, 1]:  # side walls
                        glRotatef(90, 0, 1, 0)  # sideways
                    # Color
                    if piece.color == 'white':
                        glColor3f(1, 1, 1)
                    else:
                        glColor3f(0, 0, 0)
                    # Draw simple shape based on type
                    if piece.type == 'pawn':
                        # Pawn: cone shape
                        glPushMatrix()
                        glTranslatef(0, 0.04, 0)
                        gluCylinder(quadric, 0.06, 0.02, 0.08, 12, 1)
                        glPopMatrix()
                        gluSphere(quadric, 0.02, 8, 8)  # small ball on top
                    elif piece.type == 'rook':
                        # Rook: main cylinder with 4 turrets
                        gluCylinder(quadric, 0.05, 0.05, 0.10, 12, 1)
                        for i in range(4):
                            glPushMatrix()
                            glRotatef(i * 90, 0, 1, 0)
                            glTranslatef(0.03, 0.10, 0)
                            gluCylinder(quadric, 0.015, 0.015, 0.02, 8, 1)
                            glPopMatrix()
                    elif piece.type == 'knight':
                        # Knight: L-shape approximation with cylinders
                        gluCylinder(quadric, 0.04, 0.04, 0.08, 12, 1)
                        glPushMatrix()
                        glTranslatef(0.02, 0.08, 0)
                        glRotatef(45, 0, 0, 1)
                        gluCylinder(quadric, 0.02, 0.02, 0.04, 8, 1)
                        glPopMatrix()
                    elif piece.type == 'bishop':
                        # Bishop: cylinder with a mitre-like top
                        gluCylinder(quadric, 0.04, 0.04, 0.12, 12, 1)
                        glPushMatrix()
                        glTranslatef(0, 0.12, 0)
                        gluCylinder(quadric, 0.06, 0.02, 0.02, 8, 1)
                        glPopMatrix()
                    elif piece.type == 'queen':
                        # Queen: taller with crown-like top
                        gluCylinder(quadric, 0.06, 0.06, 0.14, 12, 1)
                        for i in range(5):
                            glPushMatrix()
                            glRotatef(i * 72, 0, 1, 0)
                            glTranslatef(0.04, 0.14, 0)
                            gluCylinder(quadric, 0.01, 0.01, 0.02, 6, 1)
                            glPopMatrix()
                    elif piece.type == 'king':
                        # King: tallest with cross
                        gluCylinder(quadric, 0.07, 0.07, 0.16, 12, 1)
                        # Cross on top
                        glPushMatrix()
                        glTranslatef(0, 0.16, 0)
                        gluCylinder(quadric, 0.02, 0.02, 0.04, 8, 1)  # vertical
                        glPushMatrix()
                        glTranslatef(0, 0.02, 0)
                        glRotatef(90, 0, 0, 1)
                        gluCylinder(quadric, 0.02, 0.02, 0.04, 8, 1)  # horizontal
                        glPopMatrix()
                        glPopMatrix()
                    glPopMatrix()

def draw_tunnel(tex_id, highlight_row, highlight_col, pulse, hx, hy, active_wall):
    """
    Draw TUNNEL_DEPTH receding chess rings.
    hx, hy: nose deviation (-1..1) shifts the vanishing point for parallax.
    highlight_row, highlight_col: which square on the nearest ring to glow
    active_wall: which of the 4 walls (0=right,1=left,2=top,3=bottom) is active
    """
    for depth in range(TUNNEL_DEPTH):
        z_near = -(depth       * TUNNEL_SPACING)
        z_far  = -((depth + 1) * TUNNEL_SPACING)

        # Scale rings down with depth for perspective feel
        s_near = TUNNEL_SCALE * (1.0 - depth       * 0.08)
        s_far  = TUNNEL_SCALE * (1.0 - (depth + 1) * 0.08)

        # Parallax offset — farther rings shift more with nose
        parallax = depth * 0.18
        ox = hx * parallax
        oy = hy * parallax

        # Depth fade — far rings are darker
        fade = max(0.15, 1.0 - depth * 0.12)

        # 4 wall faces of this ring: right, left, top, bottom
        # Each face is a quad connecting near and far ring edges
        walls = [
            # right wall (index 0)
            [( s_near, -s_near, z_near), ( s_near,  s_near, z_near),
             ( s_far + ox,  s_far + oy, z_far),  ( s_far + ox, -s_far + oy, z_far)],
            # left wall (index 1)
            [(-s_near, -s_near, z_near), (-s_near,  s_near, z_near),
             (-s_far + ox,  s_far + oy, z_far),  (-s_far + ox, -s_far + oy, z_far)],
            # top wall (index 2)
            [(-s_near,  s_near, z_near), ( s_near,  s_near, z_near),
             ( s_far + ox,  s_far + oy, z_far),  (-s_far + ox,  s_far + oy, z_far)],
            # bottom wall (index 3)
            [(-s_near, -s_near, z_near), ( s_near, -s_near, z_near),
             ( s_far + ox, -s_far + oy, z_far),  (-s_far + ox, -s_far + oy, z_far)],
        ]

        # draw walls with chess texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        for i, wall in enumerate(walls):
            # brighten the active wall so it's clearly the playable board
            if i == active_wall:
                glColor3f(1.0, 1.0, 1.0)
            else:
                glColor3f(fade, fade, fade)
            glBegin(GL_QUADS)
            for (x, y, z), (u, v) in zip(wall, TUNNEL_FACE_UVS):
                glTexCoord2f(u, v)
                glVertex3f(x, y, z)
            glEnd()

        # back wall — slightly darker than side walls so it reads as tunnel end
        back_s = s_far
        back_wall = [
            (-back_s + ox, -back_s + oy, z_far),
            ( back_s + ox, -back_s + oy, z_far),
            ( back_s + ox,  back_s + oy, z_far),
            (-back_s + ox,  back_s + oy, z_far),
        ]
        glColor3f(fade * 0.7, fade * 0.7, fade * 0.7)
        glBegin(GL_QUADS)
        for (x, y, z), (u, v) in zip(back_wall, TUNNEL_FACE_UVS):
            glTexCoord2f(u, v)
            glVertex3f(x, y, z)
        glEnd()

        # Draw grid lines on this ring for visual clarity
        glDisable(GL_TEXTURE_2D)
        glColor3f(fade * 0.4, fade * 0.4, fade * 0.4)
        glLineWidth(1.0)
        for wall in walls:
            glBegin(GL_LINE_LOOP)
            for (x, y, z) in wall:
                glVertex3f(x, y, z)
            glEnd()
        glBegin(GL_LINE_LOOP)
        for (x, y, z) in back_wall:
            glVertex3f(x, y, z)
        glEnd()

        # Highlight square on the nearest ring only
        if depth == 0:
            draw_tunnel_highlight(walls, highlight_row, highlight_col, pulse, active_wall)
            draw_pieces(walls, active_wall, game)

    glDisable(GL_TEXTURE_2D)


def draw_tunnel_highlight(walls, row, col, pulse, active_wall):
    """
    glow effect on the active square
    w bilinear interpolation
    highlights a single square on the active wall
    """
    wall = walls[active_wall]

    sq = 1.0 / 8.0
    u0 = col * sq
    u1 = (col + 1) * sq
    v0 = row * sq
    v1 = (row + 1) * sq

    # get the 4 corners of the wall — order depends on which wall
    p0 = np.array(wall[0], dtype=float)
    p1 = np.array(wall[1], dtype=float)
    p2 = np.array(wall[2], dtype=float)
    p3 = np.array(wall[3], dtype=float)

    def lerp(a, b, t):
        return a + (b - a) * t

    # bilinear interpolation across the quad
    # u goes along p0->p1 edge, v goes along p0->p3 edge
    bottom_left  = lerp(lerp(p0, p1, u0), lerp(p3, p2, u0), v0)
    bottom_right = lerp(lerp(p0, p1, u1), lerp(p3, p2, u1), v0)
    top_right    = lerp(lerp(p0, p1, u1), lerp(p3, p2, u1), v1)
    top_left     = lerp(lerp(p0, p1, u0), lerp(p3, p2, u0), v1)

    # push slightly off surface toward camera
    if active_wall == 0:  # right
        normal = np.array([-1, 0, 0])
    elif active_wall == 1:  # left
        normal = np.array([1, 0, 0])
    elif active_wall == 2:  # top
        normal = np.array([0, -1, 0])
    elif active_wall == 3:  # bottom
        normal = np.array([0, 1, 0])
    normal = normal * 0.02

    corners = [
        bottom_left  + normal,
        bottom_right + normal,
        top_right    + normal,
        top_left     + normal,
    ]

    glDisable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)

    # Filled glow (additive)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    r, g, b = COL_GLOW
    glColor4f(r, g, b, 0.22 + 0.22 * pulse)
    glBegin(GL_QUADS)
    for c in corners:
        glVertex3f(*c)
    glEnd()

    # Border (normal alpha blend for clean lines)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(1.0, 0.95, 0.1, 0.65 + 0.35 * pulse)
    glLineWidth(3.5)
    glBegin(GL_LINE_LOOP)
    for c in corners:
        glVertex3f(*c)
    glEnd()
    glLineWidth(1.0)

    glDisable(GL_BLEND)


def build_hud_surface(font, row, col, detected, active_wall):
    surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    wall_names = ["Right", "Left", "Top", "Bottom"]
    lines = [
        f"Active wall  : {wall_names[active_wall]}",
        f"Target square: {chr(ord('A') + col)}{row + 1}",
        f"Nose tracked : {'YES ●' if detected else 'searching...'}",
        "Pinch=select  Palm=next wall  R=reset  ESC=quit",
    ]
    for i, line in enumerate(lines):
        shadow = font.render(line, True, (0, 0, 0))
        surf.blit(shadow, (11, 11 + i * 22))
        color = (80, 255, 130) if detected else (255, 180, 60)
        text  = font.render(line, True, color)
        surf.blit(text, (10, 10 + i * 22))
    return surf

def add_pip(hud_surf, pip_frame, detected):
    if pip_frame is None:
        return
    pw, ph = 192, 144
    small  = cv2.resize(pip_frame, (pw, ph))
    small  = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    cam_s  = pygame.surfarray.make_surface(np.transpose(small, (1, 0, 2)))
    x, y   = WINDOW_W - pw - 12, 12
    border = (60, 220, 80) if detected else (220, 60, 60)
    pygame.draw.rect(hud_surf, border, (x - 2, y - 2, pw + 4, ph + 4), 2)
    hud_surf.blit(cam_s, (x, y))

def draw_hud_overlay(hud_surf):
    """Upload hud pygame Surface as a full-screen OpenGL quad."""
    data = pygame.image.tostring(hud_surf, "RGBA", False)
    tid  = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_W, WINDOW_H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, WINDOW_W, WINDOW_H, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D); glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(WINDOW_W, 0)
    glTexCoord2f(1, 1); glVertex2f(WINDOW_W, WINDOW_H)
    glTexCoord2f(0, 1); glVertex2f(0, WINDOW_H)
    glEnd()
    glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glDeleteTextures([tid])
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW);  glPopMatrix()

#main
def main():
    threading.Thread(target=camera_thread, daemon=True).start()

    pygame.init()
    pygame.display.set_mode((WINDOW_W, WINDOW_H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Chess Tunnel — Nose Tracking")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 15)

    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(FOV, WINDOW_W / WINDOW_H, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.04, 0.02, 0.06, 1.0)

    tex = upload_texture(make_chess_texture())

    rot_x, rot_y       = 0.0, 0.0
    target_x, target_y = 0.0, 0.0
    t0             = time.time()
    was_pinching   = False   # tracks previous frame pinch state
    was_palm_open  = False   # tracks previous frame palm state
    current_wall   = 3       # start on bottom wall (0=right,1=left,2=top,3=bottom)
    last_wall_switch = 0.0   # cooldown timer for wall switching

    print("Chess tunnel running — move your head to shift the view.")
    print("Raise your middle finger to select a square, pinch to confirm.")
    print("Open palm to switch to next wall.\n")

    while True:
        clock.tick(TARGET_FPS)
        pulse = (math.sin((time.time() - t0) * PULSE_SPEED * 2 * math.pi) + 1) / 2

        # Events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key in (K_ESCAPE, K_q):
                    pygame.quit(); sys.exit()
                if event.key == K_r:
                    target_x, target_y = 0.0, 0.0

        # Read tracking
        with tracking.lock:
            nx        = tracking.nose_x
            ny        = tracking.nose_y
            detected  = tracking.detected
            pip_frame = tracking.pip_frame
            finger_x  = tracking.finger_x
            finger_y  = tracking.finger_y
            hand_det  = tracking.hand_detected
            pinching  = tracking.pinching
            palm_open = tracking.open_palm

        # open palm switches to next wall with 1 second cooldown
        now = time.time()
        if palm_open and not was_palm_open and (now - last_wall_switch) > 1.0:
            current_wall     = (current_wall + 1) % 4
            last_wall_switch = now
            wall_names = ["Right", "Left", "Top", "Bottom"]
            print(f"Switched to {wall_names[current_wall]} wall")
        was_palm_open = palm_open

        # Update target rotation from nose deviation
        if detected:
            target_x = -(ny - 0.5) * ROT_SENSITIVITY
            target_y =  (nx - 0.5) * ROT_SENSITIVITY

        rot_x += (target_x - rot_x) * ROT_LERP
        rot_y += (target_y - rot_y) * ROT_LERP

        # use finger for square selection only — nose never touches this
        # don't update square while palm is open to avoid accidental switches
        if hand_det and not palm_open:
            row, col = pos_to_square(finger_x, finger_y)
        else:
            row, col = 7, 0   # park at A1 when no hand or palm is open

        # pinch triggers selection on the frame it first becomes True
        if pinching and not was_pinching:
            game.select_square(current_wall, row, col)
        was_pinching = pinching  # update for next frame

        # 3D render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0, -1)

        # Nose rotates the viewing angle — full sensitivity so you can reach all walls
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)

        # pass nose deviation for parallax
        nose_dx = (nx - 0.5) * 2   # -1..1
        nose_dy = (ny - 0.5) * 2
        draw_tunnel(tex, row, col, pulse, nose_dx, nose_dy, current_wall)

        # 2D overlay
        hud = build_hud_surface(font, row, col, detected, current_wall)
        add_pip(hud, pip_frame, detected)
        draw_hud_overlay(hud)

        pygame.display.flip()

if __name__ == "__main__":
    main()