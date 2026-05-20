import cv2
import argparse

import gym
import numpy as np
import pyglet
from pyglet.window import key
from pyapriltags import Detector

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

apriltag_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)


SPEED_FORWARD = np.array([0.44, 0.0])
SPEED_BACKWARD = np.array([-0.44, 0.0])
SPEED_LEFT = np.array([0.0, 1.0])
SPEED_RIGHT = np.array([0.0, -1.0])
SPEED_BOOST_MULTIPLIER = 1.5

RED_STOP_DISTANCE = 150
RED_MIN_CONTOUR_AREA = 25
RED_STOP_SECONDS = 3.0
RED_IGNORE_SECONDS = 2.0

LANE_MIN_CONTOUR_AREA = 10
LANE_FORWARD_SPEED = 0.22
LANE_KP = 0.02
LANE_MAX_STEERING = 1.0
LANE_SINGLE_LINE_STEERING = 0.8
LANE_SMOOTHING = 0.55

YELLOW_MAX_X_RATIO = 0.75
GREY_MIN_X_RATIO = 0.45
MIN_LANE_WIDTH_PIXELS = 80

RENDER_PARAMS = ["human", "top_down"]
current_render_params = RENDER_PARAMS[0]


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

    action = np.array([0.0, 0.0])

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

    action = np.array([0.0, 0.0])

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

    action = np.array([0.0, 0.0])

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

    action = np.array([0.0, 0.0])

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
    global current_render_params

    global is_move_right
    global is_move_left
    global is_move_forward
    global is_move_back

    global is_show_masks
    global red_stop
    global red_stop_timer
    global red_ignore_timer

    global last_steering
    global last_steering_angle

    global lane_follow_enabled
    global x_was_pressed
    global is_april_tag_detect

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        red_stop = False
        red_stop_timer = 0.0
        red_ignore_timer = 0.0
        last_steering = 0
        last_steering_angle = 0.0
        lane_follow_enabled = False
        env.reset()
        env.render()

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        writer.release()
        writer_yellow.release()
        cv2.destroyAllWindows()
        env.close()
        pyglet.app.exit()

    elif symbol == key.TAB:
        if current_render_params == RENDER_PARAMS[0]:
            current_render_params = RENDER_PARAMS[1]
        else:
            current_render_params = RENDER_PARAMS[0]

    elif symbol == key.F:
        is_show_masks = not is_show_masks

        if not is_show_masks:
            cv2.destroyAllWindows()

    elif symbol == key.X:
        if not x_was_pressed:
            lane_follow_enabled = not lane_follow_enabled
            x_was_pressed = True
            print("lane_follow_enabled =", lane_follow_enabled)

    elif symbol == key.J:
        is_move_left = True

    elif symbol == key.I:
        is_move_forward = True

    elif symbol == key.L:
        is_move_right = True

    elif symbol == key.K:
        is_move_back = True

    elif symbol == key.Q:
        is_april_tag_detect = True


@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    global x_was_pressed

    if symbol == key.X:
        x_was_pressed = False


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


def get_grey_mask(hsv_image):
    lower = np.array([0, 0, 120])
    upper = np.array([180, 80, 255])
    return cv2.inRange(hsv_image, lower, upper)


def get_red_mask(hsv_image):
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    return cv2.bitwise_or(mask1, mask2)


def get_filtered_contours(mask, min_contour_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return filter_small_contours(contours, min_contour_area)


def is_red_line_close(mask_red):
    h, w = mask_red.shape[:2]

    mask = mask_red.copy()
    mask[0 : h // 2, :] = 0

    contours = get_filtered_contours(mask, RED_MIN_CONTOUR_AREA)

    if not contours:
        return False

    max_contour = max(contours, key=cv2.contourArea)

    distance = cv2.pointPolygonTest(max_contour, (w // 2, h - 1), True)
    distance_abs = abs(distance)

    print("red line distance =", distance_abs)

    return distance_abs < RED_STOP_DISTANCE


def detect_apriltags(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = apriltag_detector.detect(gray, estimate_tag_pose=False)

    for tag in tags:
        corners = tag.corners.astype(int)
        center = tuple(tag.center.astype(int))

        for i in range(4):
            cv2.line(
                frame,
                tuple(corners[i]),
                tuple(corners[(i + 1) % 4]),
                (0, 255, 0),
                2,
            )

        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"id={tag.tag_id}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        print(f"AprilTag найден: id={tag.tag_id}, center={tag.center}")
    
    to_show = frame[..., ::-1]
    cv2.imshow("AprilTag", to_show)
    cv2.waitKey(1)

    return tags


def get_median_x_from_contours(contours):
    all_x = []

    for contour in contours:
        for point in contour:
            all_x.append(point[0][0])

    if not all_x:
        return None

    return float(np.median(all_x))


def get_contour_median_x(contour):
    all_x = []

    for point in contour:
        all_x.append(point[0][0])

    if not all_x:
        return None

    return float(np.median(all_x))


def get_yellow_x(contours_yellow, image_width):
    filtered_contours = [contour for contour in contours_yellow if cv2.contourArea(contour) > LANE_MIN_CONTOUR_AREA]

    if not filtered_contours:
        return None

    candidate_contours = []

    for contour in filtered_contours:
        x = get_contour_median_x(contour)

        if x is not None and x < image_width * YELLOW_MAX_X_RATIO:
            candidate_contours.append(contour)

    if not candidate_contours:
        return None

    return get_median_x_from_contours(candidate_contours)


def get_grey_x(contours_grey, yellow_x, image_width):
    filtered_contours = [contour for contour in contours_grey if cv2.contourArea(contour) > LANE_MIN_CONTOUR_AREA]

    if not filtered_contours:
        return None

    candidates = []

    for contour in filtered_contours:
        x = get_contour_median_x(contour)

        if x is None:
            continue

        if yellow_x is not None:
            if x > yellow_x + MIN_LANE_WIDTH_PIXELS:
                candidates.append((cv2.contourArea(contour), x))
        else:
            if x > image_width * GREY_MIN_X_RATIO:
                candidates.append((cv2.contourArea(contour), x))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def smooth_steering(steering_angle):
    global last_steering_angle

    steering_angle = np.clip(
        steering_angle,
        -LANE_MAX_STEERING,
        LANE_MAX_STEERING,
    )

    steering_angle = LANE_SMOOTHING * steering_angle + (1.0 - LANE_SMOOTHING) * last_steering_angle

    last_steering_angle = steering_angle

    return steering_angle


def lane_follow(obs):
    global last_steering
    global last_steering_angle

    h, w = obs.shape[:2]
    hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    lower_half = hsv_image[h // 2 : h - 1, :]

    mask_yellow = cv2.inRange(
        lower_half,
        np.array([20, 100, 100]),
        np.array([30, 255, 255]),
    )

    mask_grey = cv2.inRange(
        lower_half,
        np.array([0, 0, 120]),
        np.array([180, 80, 255]),
    )

    contours_yellow, _ = cv2.findContours(
        mask_yellow,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    contours_grey, _ = cv2.findContours(
        mask_grey,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    yellow_x = get_yellow_x(contours_yellow, w)
    grey_x = get_grey_x(contours_grey, yellow_x, w)

    has_yellow = yellow_x is not None
    has_grey = grey_x is not None

    steering_angle = 0.0

    if has_yellow and has_grey:
        line_center_x = (yellow_x + grey_x) / 2.0
        image_center_x = w / 2.0

        delta_center = line_center_x - image_center_x

        steering_angle = -LANE_KP * delta_center

        if steering_angle > 0:
            last_steering = 1
        elif steering_angle < 0:
            last_steering = -1
        else:
            last_steering = 0

        print("yellow_x =", yellow_x)
        print("grey_x =", grey_x)
        print("line_center_x =", line_center_x)
        print("delta_center =", delta_center)

    elif has_grey and not has_yellow:
        steering_angle = LANE_SINGLE_LINE_STEERING
        last_steering = 1

    elif has_yellow and not has_grey:
        steering_angle = -LANE_SINGLE_LINE_STEERING
        last_steering = -1

    else:
        if last_steering == 1:
            steering_angle = LANE_SINGLE_LINE_STEERING
        elif last_steering == -1:
            steering_angle = -LANE_SINGLE_LINE_STEERING
        else:
            steering_angle = last_steering_angle

    steering_angle = smooth_steering(steering_angle)

    print("lane steering =", steering_angle)

    return steering_angle


def draw_contours_on_image(rgb_image, contours):
    result = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 100), 2)
    return result


def process_bot_image(obs):
    bgr_image = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    mask_yellow = get_yellow_mask(hsv_image)
    mask_grey = get_grey_mask(hsv_image)
    mask_red = get_red_mask(hsv_image)

    red_contours = get_filtered_contours(mask_red, 50)
    result_contours = draw_contours_on_image(obs, red_contours)

    return bgr_image, hsv_image, mask_yellow, mask_grey, mask_red, result_contours


def show_bot_images(bgr_image, hsv_image, mask_yellow, mask_grey, mask_red, result_contours):
    cv2.imshow("camera image view", bgr_image)
    cv2.imshow("hsv format", hsv_image)
    cv2.imshow("red mask", mask_red)
    cv2.imshow("yellow mask", mask_yellow)
    cv2.imshow("grey mask", mask_grey)
    cv2.imshow("red contours", result_contours)

    cv2.waitKey(1)


is_move_right = False
is_move_left = False
is_move_forward = False
is_move_back = False

is_show_masks = False

red_stop = False
red_stop_timer = 0.0
red_ignore_timer = 0.0

last_steering = 0
last_steering_angle = 0.0

lane_follow_enabled = False
x_was_pressed = False
is_april_tag_detect = False


def update(dt):
    global current_render_params

    global is_move_right
    global is_move_left
    global is_move_forward
    global is_move_back

    global red_stop
    global red_stop_timer
    global red_ignore_timer

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

    if lane_follow_enabled:
        obs_copy = env.render_obs()
        steering_angle = lane_follow(obs_copy)
        action = np.array([LANE_FORWARD_SPEED, steering_angle])

    if is_april_tag_detect:
        obs_copy = env.render_obs()
        detect_apriltags(obs_copy)

    if red_stop:
        action = np.array([0.0, 0.0])

    obs, reward, _, _ = env.step(action)

    bgr_image, hsv_image, mask_yellow, mask_grey, mask_red, result_contours = process_bot_image(obs)

    red_line_close = is_red_line_close(mask_red)

    if red_stop:
        red_stop_timer += dt

        if red_stop_timer >= RED_STOP_SECONDS:
            red_stop = False
            red_stop_timer = 0.0
            red_ignore_timer = RED_IGNORE_SECONDS

    elif red_ignore_timer > 0:
        red_ignore_timer -= dt

        if red_ignore_timer < 0:
            red_ignore_timer = 0.0

    elif red_line_close:
        red_stop = True
        red_stop_timer = 0.0

    if is_show_masks:
        show_bot_images(
            bgr_image,
            hsv_image,
            mask_yellow,
            mask_grey,
            mask_red,
            result_contours,
        )

    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    print("bot position = ", env.cur_pos)
    print("obs shape =", obs.shape)
    print("lane_follow_enabled =", lane_follow_enabled)
    print("red_stop =", red_stop)
    print("red_stop_timer =", red_stop_timer)
    print("red_ignore_timer =", red_ignore_timer)
    print("last_steering =", last_steering)
    print("last_steering_angle =", last_steering_angle)

    yellow_bgr = cv2.cvtColor(mask_yellow, cv2.COLOR_GRAY2BGR)

    writer_yellow.write(yellow_bgr)
    writer.write(bgr_image)

    env.render(current_render_params)


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

pyglet.app.run()

env.close()
