import cv2
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv


# python3 main.py --map-name=udem1
parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-udem1-v0")
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument(
    "--draw-curve", action="store_true", help="draw the lane following curve"
)
parser.add_argument(
    "--draw-bbox", action="store_true", help="draw collision detection bounding boxes"
)
parser.add_argument(
    "--domain-rand", action="store_true", help="enable domain randomization"
)
parser.add_argument(
    "--dynamics_rand", action="store_true", help="enable dynamics randomization"
)
parser.add_argument(
    "--frame-skip", default=1, type=int, help="number of frames to skip"
)
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
    action = [0, 0]
    angle_deg = np.rad2deg(env.cur_angle)
    delta = -3 

    if delta <= angle_deg <= abs(delta):
        tap_move_right = False
    else: # Не довернулись
        action = [0, -0.5]

    return action

def move_left(current_angle):
    action = [0, 0]
    angle_deg = np.rad2deg(env.cur_angle)
    delta = -3 

    if 180 - abs(delta) <= angle_deg <= 180 or -180 <= angle_deg <=-180 + abs(delta):
        tap_move_left = False
    else: # Не довернулись
        action = [0, 0.5]

    return action
def move_up(current_angle):
    action = [0, 0]
    angle_deg = np.rad2deg(env.cur_angle)
    delta = -3

    if 90 + delta <= angle_deg <= 90 + (abs(delta)):
        tap_move_up = False
    else: # Не довернулись
        action = [0, -0.5]

    return action

def move_down(current_angle):
    action = [0, 0]
    angle_deg = np.rad2deg(env.cur_angle)
    delta = -3 

    if -90 + delta <= angle_deg <= -90 + abs(delta):
        tap_move_down = False
    else: # Не довернулись
        action = [0, 0.5]

    return action

RENDER_PARAMS = ["human", "top_down"]
view_mode = RENDER_PARAMS[0]
tap_move_right = False
tap_move_left = False
tap_move_up = False
tap_move_down = False


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global tap_move_right
    global tap_move_left
    global tap_move_up
    global tap_move_down
    global view_mode

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    elif key_handler[key.TAB] and view_mode == RENDER_PARAMS[0]:
        view_mode = RENDER_PARAMS[1]
    elif key_handler[key.TAB] and view_mode == RENDER_PARAMS[1]:
        view_mode = RENDER_PARAMS[0]

    elif symbol == key.D:
        tap_move_right = True
    elif symbol == key.A:
        tap_move_left = True
    elif symbol == key.W:
        tap_move_up = True
    elif symbol == key.S:
        tap_move_down = True

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        # [-1, 1] - |+-1|: максимальная скорость (~0.30м/c)
        # 1 -> 0 : 0.5 (~ в 2 раза меньше скорость!)
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]: 
        action += np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action += np.array([0, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    if tap_move_right:
        action = move_right(env.cur_angle)
    if tap_move_left:
        action = move_left(env.cur_angle)
    if tap_move_up:
        action = move_up(env.cur_angle)
    if tap_move_down:
        action = move_down(env.cur_angle)

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    # obs - картинка (в виде трехмерной матрицы)
    # done = True|False
    
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    print("bot position = ", env.cur_pos)
    print("bot angle_rad=", env.cur_angle)
    print(f"bot angle_deg=", np.rad2deg(env.cur_angle))

    env.render(view_mode)


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()