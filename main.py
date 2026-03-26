import cv2
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv


# MOVEMENT
speed = 0.44

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

def move_to_target(current_angle, target, delta = 3):
    action = [0, 0]
    angle_deg = np.rad2deg(current_angle)
    # угол в [-180, 180]
    angle_deg = ((angle_deg + 180) % 360) - 180
    
    if abs(angle_deg - target) <= delta:
        return action
    
    # Определение направления кратчайшего пути
    difference = (target - angle_deg + 180) % 360 - 180
    action = [0, speed if difference > 0 else -speed]
    
    return action
 
def move_down(current_angle):
    return move_to_target(current_angle, -90)

def move_up(current_angle):
    return move_to_target(current_angle, 90)

def move_right(current_angle):
    return move_to_target(current_angle, 0)

def move_left(current_angle):
    return move_to_target(current_angle, 180)


RENDER_PARAMS = ["human", "top_down"]

writer = cv2.VideoWriter( 
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (640, 480), #(width, height) (столбцы, строки)
)

writer_yellow = cv2.VideoWriter( 
    "output_markup_yel.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
    (640, 480), #(width, height) (столбцы, строки)
)

view_mode = RENDER_PARAMS[0]
tap_move_right = False
tap_move_left = False
tap_move_up = False
tap_move_down = False
is_view_image = False
red_stop = False
ignore_red_stop = False
last_steering = 0


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global tap_move_right, tap_move_left, tap_move_up, tap_move_down, view_mode, ignore_red_stop

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

    elif key_handler[key.TAB]:
        view_mode = RENDER_PARAMS[1] if view_mode == RENDER_PARAMS[0] else RENDER_PARAMS[0]

    elif symbol == key.RIGHT:
        tap_move_right = True
        tap_move_left = False
        tap_move_up = False
        tap_move_down = False
    elif symbol == key.LEFT:
        tap_move_right = False
        tap_move_left = True
        tap_move_up = False
        tap_move_down = False
    elif symbol == key.UP:
        tap_move_right = False
        tap_move_left = False
        tap_move_up = True
        tap_move_down = False
    elif symbol == key.DOWN:
        tap_move_right = False
        tap_move_left = False
        tap_move_up = False
        tap_move_down = True

    elif symbol == key.F:
        is_view_image = True
    
    #остановка на красный цвет
    elif symbol == key.I:
        ignore_red_stop = True
    elif symbol == key.J:
        ignore_red_stop = False

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def get_bot_image(obs):

    bgr_image = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    hsv2rgb_yel_image = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2RGB)

    lower_gray = np.array([160, 160, 160])
    upper_gray = np.array([200, 200, 200])
    mask_gray = cv2.inRange(obs, lower_gray, upper_gray)

    '''
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 120, 70])
    upper_red_2 = np.array([180, 255, 255])
    mask_red_1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
    hsv2rgb_red_image = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2RGB)
    '''

    cv2.imshow("camera image view", bgr_image)
    cv2.imshow("gray2rgb", hsv2rgb_yel_image)
    cv2.imshow("hsv format", hsv_image)
    cv2.imshow("gray mask", mask_gray)
    '''
    cv2.imshow("red mask", hsv2rgb_red_image)

    h,  w = obs.shape[0], obs.shape[1]
    mask_red[0:h//2, :] = 0

    contours, _ = cv2.findContours(image=mask_red, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 25]
    image_with_contours = cv2.drawContours(image=bgr_image.copy(), contours=contours, contourIdx=-1, color=(0, 255, 0), thickness = 3)

    max_contours = max(contours, key=cv2.contourArea)
    dist = cv2.pointPolygonTest(max_contours, (w // 2, h - 1), True)

    print(np.abs(dist))

    cv2.imshow("red mask", mask_red)
    cv2.imshow('red contours', image_with_contours)
    '''

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lane_follow(obs):

    global last_steering

    h, w = obs.shape[:2]
    hsv_image = cv2.cvtColor(obs,cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(
        hsv_image[h // 2 : h - 1, :], np.array([20, 100, 100]), np.array([30, 255, 255])
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #фильтрация пикселей
    contours = [contour for contour in contours if cv2.contourArea(contour) > 10]

    lx = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            lx = int(M["m10"] / M["m00"])

            center_x = w / 2
            deviation = center_x - lx
            if deviation > 0:
                last_steering = 1
            else:
                last_steering = -1


    steering_angle = 0.0
    if lx is not None:
        center_x = w / 2
        deviation = center_x - lx
        steering_angle = deviation / center_x
    else:
        if last_steering == 1:
            steering_angle = 1.0
        elif last_steering == -1:
            steering_angle = -1.0
        else:
            steering_angle = 0.0


    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return steering_angle

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    global red_stop, is_view_image

    action = np.array([0.0, 0.0])

    if key_handler[key.W]:
        # [-1, 1] - |+-1|: максимальная скорость (~0.30м/c)
        # 1 -> 0 : 0.5 (~ в 2 раза меньше скорость!)
        action += np.array([speed, 0])
    if key_handler[key.S]: 
        action -= np.array([speed, 0])
    if key_handler[key.A]:
        action += np.array([0, 1])
    if key_handler[key.D]:
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

    if red_stop and not ignore_red_stop:
        action = np.array([0, 0])

    if key_handler[key.X]:
        obs = env.render_obs()
        steering_angle = lane_follow(obs)
        action += np.array([speed / 2, steering_angle])

    obs, reward, done, info = env.step(action) # -> return as RGB format

    h,  w = obs.shape[0], obs.shape[1]

    hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 120, 70])
    upper_red_2 = np.array([180, 255, 255])
    mask_red_1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    mask_red[0:h//2, :] = 0

    contours, _ = cv2.findContours(image=mask_red, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 25]

    if not ignore_red_stop:
        if contours:
            max_contours = max(contours, key=cv2.contourArea)
            dist = cv2.pointPolygonTest(max_contours, (w // 2, h - 1), True)

            dist_abs = np.abs(dist)
            print(dist_abs)

            if dist_abs < 150:
                red_stop = True
            else:
                red_stop = False
        else:
            red_stop = False
    else:
        red_stop = False



    if key_handler[key.F]:
        if not is_view_image:
            get_bot_image(obs)
            is_view_image = True
        else:
            is_view_image = False
    # obs - картинка (в виде трехмерной матрицы)
    # done = True|False
    
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

    bgr_image = cv2.cvtColor(obs,cv2.COLOR_BGR2RGB)
    writer.write(bgr_image)

    env.render(view_mode)


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()