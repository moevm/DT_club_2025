import cv2
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

# SPEED_SETTINGS (ABS_MEANING!!!)
CONST_UP_SPEED = np.array([0.44, 0.0])
CONST_DOWN_SPEED = np.array([0.44, 0])
CONST_LEFT_SPEED = np.array([0, 1])
CONST_RIGHT_SPEED = np.array([0, 1])

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

def m_right(current_angle):
    global s_m_right
    action = [0, 0]
    
    angle_deg = np.rad2deg(current_angle)
    dlt = s_m_right[1]
    
    p_p=s_m_right[1]-180
    if p_p <= -180:
        abs(p_p)
        p_p=p_p-180
        
    p_l=s_m_right[1]+180
    if p_l >= 180:
        p_l=p_l-180
        
    if dlt-6<=angle_deg<=dlt+6:
        s_m_right=False
    elif p_p<=angle_deg<=p_l:
        action =[0,1]
    else:
        action =[0, -1]
        
    return action

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global s_m_right
    
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        writer.release()
        writer_y.release()
        env.close()
        sys.exit(0)
    elif symbol == key.L:
        s_m_right=[True,0]
    elif symbol == key.K:
        s_m_right=[True,-90]
    elif symbol == key.J:
        s_m_right=[True,180]
    elif symbol == key.I:
        s_m_right=[True,90]

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def g_b_image(obs):
    to_show=cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    cv2.imshow("camera image view", to_show)
    
    
    hsv_image=cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    lower_y=np.array([20,100,100])
    upper_y=np.array([30,255,255])
    mask_yellow=cv2.inRange(hsv_image,lower_y,upper_y)
    
    lower_g=np.array([160,160,160])
    upper_g=np.array([200,200,200])
    mask_gray=cv2.inRange(obs,lower_g,upper_g)
    
    lower_r1 = np.array([1, 100, 100])
    upper_r1 = np.array([10, 254, 254])
    lower_r2 = np.array([160, 100, 100])
    upper_r2 = np.array([179, 254, 254])
    mask_r1 = cv2.inRange(hsv_image, lower_r1, upper_r1)
    mask_r2 = cv2.inRange(hsv_image, lower_r2, upper_r2)
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)
    
    h,  w = obs.shape[0], obs.shape[1]
    mask_r[0:h//2, :] = 0
    
    r_image = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)

    
    cv2.imshow("yellow mask", mask_yellow)
    cv2.imshow("gray mask", mask_gray)
    cv2.imshow("red mask", r_image)


    contours, _ = cv2.findContours(image=mask_r, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 25]
    image_with_contours = cv2.drawContours(image=to_show.copy(), 
    contours=contours, contourIdx=-1, color=(0, 255, 0), thickness = 3)

    if contours:
        max_contours = max(contours, key=cv2.contourArea)
        dist = cv2.pointPolygonTest(max_contours, (w // 2, h - 1), True)

        print(np.abs(dist))

    cv2.imshow("red mask", mask_r)
    cv2.imshow('red contours', image_with_contours)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

RENDER_PARAMS = ["human", "top_down"]
writer=cv2.VideoWriter (
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
(640, 480),
)

writer_y=cv2.VideoWriter (
    "outpat.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    20,
(640, 480),
)

def lane_follow(obs):
    h,w=obs.shape[:2]
    
    hsv_image=cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(
        hsv_image[h//2:h-1, : ], np.array([20,100,100]), np.array([30,255,255])
    )
       
    global last_steering

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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


    cv2.imshow('mask', mask)
    
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return steering_angle
    
global bgr_image1
last_steering=0
boolcam=True
s_m_right=False
s_v_image=False

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global boolcam
    global s_m_right
    global s_v_image 
    
    action = np.array([0.0, 0.0])

    if key_handler[key.W]:
        # [-1, 1] - |+-1|: максимальная скорость (~0.30м/c)
        # 1 -> 0 : 0.5 (~ в 2 раза меньше скорость!)
        action += CONST_UP_SPEED
    if key_handler[key.S]: 
        action -= CONST_DOWN_SPEED
    if key_handler[key.A]:
        action += CONST_LEFT_SPEED
    if key_handler[key.D]:
        action -= CONST_RIGHT_SPEED
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action += np.array([-0.44, 0.])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action += np.array([0, -1])
    
    if s_m_right:
        action=m_right(env.cur_angle)
    
    if key_handler[key.X]:
        obs = env.render_obs()
        steering_angle = lane_follow(obs)
        action += np.array([0.4 / 2 , steering_angle])
    
    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    # obs - картинка (в виде трехмерной матрицы)
    # done = True|False
    
    h,w = obs.shape[0],obs.shape[1]

    hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    lower_r1 = np.array([0, 150, 90])
    upper_r1 = np.array([10, 255, 255])
    lower_r2 = np.array([170, 200, 120])
    upper_r2 = np.array([175, 255, 255])
    mask_r1 = cv2.inRange(hsv_image, lower_r1, upper_r1)
    mask_r2 = cv2.inRange(hsv_image, lower_r2, upper_r2)
    mask_r = cv2.bitwise_or(mask_r1, mask_r2)

    mask_r[0:h//2, :] = 0

    contours, _ = cv2.findContours(image=mask_r, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 25]

    if contours:
        max_contours = max(contours, key=cv2.contourArea)
        dist = cv2.pointPolygonTest(max_contours, (w // 2, h - 1), True)

        dist_abs = np.abs(dist)
        print(dist_abs)

        if dist_abs < 100:
            red_s = True
        else:
            red_s = False
    else:
        red_s = False
    
    if key_handler[key.F]:
        if not s_v_image:
            g_b_image (obs) 
            s_v_image = True
    else:
        s_v_image = False

    print(obs.shape)
    
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    print("bot position = ", env.cur_pos)
    print("bot angel=", env.cur_angle)
    print(f"bot angel=", np.rad2deg(env.cur_angle))

    bgr_image=cv2.cvtColor(obs,cv2.COLOR_RGB2BGR)
    writer.write(bgr_image)
    
    hsv_image=cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    lower_y=np.array([20,100,100])
    upper_y=np.array([30,255,255])
    mask_yellow=cv2.inRange(hsv_image,lower_y,upper_y)
    bgr_image1=cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)
    writer_y.write(bgr_image1)

    target_pos=[1.0,0,1.0]
    dist=np.sqrt((env.cur_pos[0] - target_pos[0])**2 + (env.cur_pos[2] - target_pos[2])**2)

       #изменение положения камеры
    if key_handler[key. TAB] and (boolcam == True or boolcam ==False):
        boolcam = not boolcam
                  
    if boolcam == True:
        env.render("human")
    else:
        env.render("top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
