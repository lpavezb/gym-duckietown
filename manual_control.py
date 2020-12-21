#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2
from pupil_apriltags import Detector


parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='loop_sign_go')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Parametros para el detector de lineas
lower_yellow = np.array([40,0.2,0.2]) 
upper_yellow = np.array([80,0.7,0.6]) 
lower_white = np.array([0,0.0,0.4])
upper_white = np.array([255,0.3,0.8])
lower_red = np.array([340,0.4,0.4])
upper_red = np.array([360,0.9,0.80])
min_area = 300

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


velocity = 0.2

k_p = 19
k_d = 8

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Handler para reiniciar el ambiente
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

def update(dt):
    """
    Funcion que se llama en step.
    """
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action[0]+=0.44
    if key_handler[key.DOWN]:
        action[0]-=0.44
    if key_handler[key.LEFT]:
        action[1]+=1
    if key_handler[key.RIGHT]:
        action[1]-=1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5
   
    # aquí se obtienen las observaciones y se setea la acción
    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    #print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if done:
        print('done!')
        env.reset()
        env.render()

    # Detección de lineas
    # El objetivo de hoy es detectar las lineas ajustando los valores del detector
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

    obs = obs/255.0
    obs = obs.astype(np.float32)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()
    
    # Filtro por color https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
    
    #Cambiar tipo de color de BGR a HSV
    tags = at_detector.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)
    print(tags)
    if len(tags) > 0:
    	corners = tags[0].corners
    	castint = lambda x: int(x)
    	x1, y1= list(map(castint, corners[0]))
    	x2, y2 = list(map(castint, corners[2]))
    	cv2.rectangle(frame, (int(x1), int(y1)), (int(x2),int(y2)), (0,0,255), 1)
    cv2.imshow("patos", frame)
    cv2.waitKey(1)



    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()