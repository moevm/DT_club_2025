import cv2
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

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

    global camera_image

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        writer_image.release()
        writer_mask_yellow.release()
        env.close() 
        sys.exit(0)
        
    # Смена вида камеры на TAB
    elif key_handler[key.TAB]:
        if current_render_params == RENDER_PARAMS[0]: 
            current_render_params = RENDER_PARAMS[1]
        elif current_render_params == RENDER_PARAMS[1]: 
            current_render_params = RENDER_PARAMS[0]

    #Автоматический поворот на JILK
    elif key_handler[key.J]:
        is_move_left = True
    elif key_handler[key.I]:
        is_move_forward = True
    elif key_handler[key.L]:
        is_move_right = True
    elif key_handler[key.K]:
        is_move_back = True

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def filter_small_contours(contours, min_contour_area):
    filtered = []

    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:
            filtered.append(contour)

    return filtered


def get_yellow_mask(hsv_image):
    lower = np.array([20, 100, 100])
    upper = np.array([30, 255, 255])
    return cv2.inRange(hsv_image, lower, upper)


def get_grey_mask(rgb_image):
    grey_lower = np.array([160, 160, 160])
    grey_upper = np.array([200, 200, 200])
    return cv2.inRange(rgb_image, grey_lower, grey_upper)


def get_red_mask(hsv_image):
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    return cv2.bitwise_or(mask1, mask2)


def get_filtered_contours(mask, min_contour_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return filter_small_contours(contours, min_contour_area)



def draw_contours_on_image(rgb_image, contours):
    result = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 100), 2)
    return result


def process_bot_image(obs):
    bgr_image = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    mask_yellow = get_yellow_mask(hsv_image)
    mask_grey = get_grey_mask(obs)
    mask_red = get_red_mask(hsv_image)

    red_contours = get_filtered_contours(mask_red, 50)
    result_contours = draw_contours_on_image(obs, red_contours)

    return bgr_image, hsv_image, mask_yellow, mask_grey, mask_red, result_contours


def get_bot_image(obs):
    bgr_image, hsv_image, mask_yellow, mask_grey, mask_red, result_contours = process_bot_image(obs)

    cv2.imshow("camera image view", bgr_image)
    cv2.imshow("hsv format", hsv_image)
    cv2.imshow("red mask", mask_red)
    cv2.imshow("yellow mask", mask_yellow)
    cv2.imshow("grey mask", mask_grey)
    cv2.imshow("red contours", result_contours)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


is_move_right = False
is_move_left = False
is_move_forward = False
is_move_back = False

is_view_image = False 

writer_image = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (640, 480), # (witch, height)
)  

writer_mask_yellow = cv2.VideoWriter(
    "mask.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (640, 480), # (witch, height)
)  

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
        # [-1, 1] - |+-1|: максимальная скорость (~0.30м/c)
        # 1 -> 0 : 0.5 (~ в 2 раза меньше скорость!)
        action += SPEED_FORWARD
    if key_handler[key.S]: 
        action += SPEED_BACKWARD
    if key_handler[key.A]:
        action += SPEED_LEFT
    if key_handler[key.D]:
        action += SPEED_RIGHT
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= SPEED_BOOST_MULTIPLIER

    if is_move_right:
        action  = move_right(env.cur_angle)
    if is_move_left:
        action  = move_left(env.cur_angle)
    if is_move_forward:
        action  = move_forward(env.cur_angle)
    if is_move_back:
        action  = move_back(env.cur_angle)



    obs, reward, done, info = env.step(action) # -> return as RGB format
    bgr_image, hsv_image, mask_yellow, mask_grey, mask_red, result_contours = process_bot_image(obs)

    if key_handler[key.F]:
        if not is_view_image:
            get_bot_image(obs)
            is_view_image = True
    else:
        is_view_image = False 

    # obs - картинка (в виде трехмерной матрицы)
    # done = True|False

    
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    print("bot position = ", env.cur_pos)
    print(obs.shape)


    yellow_bgr = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)
    writer_mask_yellow.write(yellow_bgr)
    writer_image.write(bgr_image)

    env.render(current_render_params)

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()

