import cv2
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

writer = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (640, 480),
)
writer_yellow = cv2.VideoWriter(
    "output_markup_yel.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (640, 480),
)

# Константы скоростей
SPEED_FORWARD = np.array([0.44, 0.0])
SPEED_BACKWARD = np.array([-0.44, 0])
SPEED_LEFT = np.array([0, 1])
SPEED_RIGHT = np.array([0, -1])
SPEED_BOOST_MULTIPLIER = 1.5

RENDER_PARAMS = ["human", "top_down"]
current_render_params = RENDER_PARAMS[0]

# python3 main.py --map-name=udem1
parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-udem1-v0")
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=42, type=int, help="seed")
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


def move_right(current_angle):
    global is_move_right
    action = [0, 0]

    angle_deg = np.rad2deg(current_angle)
    delta = 5

    if -delta <= angle_deg <= delta:
        is_move_right = False
    else:
        if 0 < angle_deg <= 180:
            action = SPEED_RIGHT
        elif -180 <= angle_deg <= 0:
            action = SPEED_LEFT

    return action


def move_left(current_angle):
    global is_move_left
    action = [0, 0]

    angle_deg = np.rad2deg(current_angle)
    delta = 5

    if (angle_deg > 0 and angle_deg >= 180 - delta) or (angle_deg < 0 and angle_deg <= -180 + delta):
        is_move_left = False
    else:
        if 0 <= angle_deg < 180:
            action = SPEED_LEFT
        elif -180 <= angle_deg < 0:
            action = SPEED_RIGHT

    return action


def move_forward(current_angle):
    global is_move_forward
    action = [0, 0]

    angle_deg = np.rad2deg(current_angle)
    delta = 5

    if 90 - delta <= angle_deg <= 90 + delta:
        is_move_forward = False
    else:
        if -90 <= angle_deg <= 90:
            action = SPEED_LEFT
        else:
            action = SPEED_RIGHT

    return action


def move_back(current_angle):
    global is_move_back
    action = [0, 0]

    angle_deg = np.rad2deg(current_angle)
    delta = 5

    if -90 - delta <= angle_deg <= -90 + delta:
        is_move_back = False
    else:
        if angle_deg > -90:
            if angle_deg > 90:
                action = SPEED_LEFT
            else:
                action = SPEED_RIGHT
        else:
            action = SPEED_LEFT

    return action


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global current_render_params

    global is_move_right
    global is_move_left
    global is_move_forward
    global is_move_back

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        writer.release()
        writer_yellow.release()
        env.close()
        sys.exit(0)

    # Смена вида камеры на TAB
    elif key_handler[key.TAB]:
        if current_render_params == RENDER_PARAMS[0]:
            current_render_params = RENDER_PARAMS[1]
        elif current_render_params == RENDER_PARAMS[1]:
            current_render_params = RENDER_PARAMS[0]

    # Автоматический поворот на JILK
    elif key_handler[key.J]:
        is_move_left = True
    elif key_handler[key.I]:
        is_move_forward = True
    elif key_handler[key.L]:
        is_move_right = True
    elif key_handler[key.K]:
        is_move_back = True

    elif key_handler[key.TAB]:
        view_mode = RENDER_PARAMS[1] if view_mode == RENDER_PARAMS[0] else RENDER_PARAMS[0]

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def get_bot_image(obs):

    to_show = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    hsv2rgb_yel_image = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2RGB)

    lower_gray = np.array([160, 160, 160])
    upper_gray = np.array([200, 200, 200])
    mask_gray = cv2.inRange(obs, lower_gray, upper_gray)

    cv2.imshow("gray2rgb", hsv2rgb_yel_image)
    cv2.imshow("camera image view", to_show)
    cv2.imshow("hsv format", hsv_image)
    cv2.imshow("yellow mask", mask_yellow)
    cv2.imshow("gray mask", mask_gray)

    cv2.waitKey(0)

is_move_right = False
is_move_left = False
is_move_forward = False
is_move_back = False

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global current_render_params

    global is_move_right
    global is_move_left
    global is_move_forward
    global is_move_back

    global is_view_image

    action = np.array([0.0, 0.0])

    if key_handler[key.W]:
        action += SPEED_FORWARD
    if key_handler[key.S]:
        action += SPEED_BACKWARD
    if key_handler[key.A]:
        action += SPEED_LEFT
    if key_handler[key.D]:
        action += SPEED_RIGHT
    if key_handler[key.SPACE]:
        action = np.array([0.0, 0.0])
    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= SPEED_BOOST_MULTIPLIER

    if is_move_right:
        action = move_right(env.cur_angle)
    if is_move_left:
        action = move_left(env.cur_angle)
    if is_move_forward:
        action = move_forward(env.cur_angle)
    if is_move_back:
        action = move_back(env.cur_angle)

    obs, reward, _, _ = env.step(action)

    if key_handler[key.F]:
        if not is_view_image:
            get_bot_image(obs)
            is_view_image = True
        else:
            is_view_image = False

    print(obs.shape)
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    print("bot position = ", env.cur_pos)
    print("bot angle_rad=", env.cur_angle)
    print(f"bot angle_deg=", np.rad2deg(env.cur_angle))

    hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    hsv2rgb_yel_image = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2RGB)

    writer_yellow.write(hsv2rgb_yel_image)

    bgr_image = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    writer.write(bgr_image)

    env.render(current_render_params)


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
