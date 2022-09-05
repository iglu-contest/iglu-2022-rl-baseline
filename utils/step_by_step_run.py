import sys
sys.path.append("./utils/")
sys.path.append("../")
from step_by_step_enjoy import download_weights, make_agent
from wrappers.target_generator import RandomFigure, CustomFigure
from wrappers.common_wrappers import  VisualObservationWrapper,  \
    Discretization,  flat_action_space,ColorWrapper, JumpAfterPlace
from loggers import VideoLogger, Logger, R1_score, SuccessRateFullFigure, Statistics,StatisticsLogger

from gridworld.env import GridWorld
from gridworld.tasks.task import Task
from wrappers.reward_wrappers import RangetRewardFilledField
from wrappers.multitask import TargetGenerator, SubtaskGenerator
import numpy as np
from enjoy import make_iglu
import os
import cv2
import gym
import torch
os.makedirs("./video_step_by_step_run", exist_ok=True)

def castom_tasks():
    tasks = dict()   
    
#     t1 = np.zeros((9,11,11))
#     t1[0, 1:4, 1:4] = 1
#     tasks['[0, 1:4, 1:4]'] = t1
    
#     t2 = np.zeros((9,11,11))
#     t2[0:2, 1:4, 1:4] = 1
#     tasks['[0:2, 1:4, 1:4]'] = t2
    
    t3 = np.zeros((9,11,11))
    t3[0:5, 4, 4] = 1
    t3[1, 4, 4] = 0
    t3[3, 4, 4] = 0
    t3[0, 8, 7] = 1
    tasks['[0:7, 4, 4]'] = t3
    
#     t4 = np.zeros((9,11,11))
#     t4[0, 4:8, 4:8] = 1
#     tasks['[0, 4:8, 4:8]'] = t4
    
#     t5 = np.zeros((9,11,11))
#     t5[0:3, 8:10, 8:10] = 1
#     tasks['[0:3, 8:10, 8:10]'] = t5
    
    return tasks

if __name__ == "__main__":
    agent = make_agent()

    custom_grid = np.ones((9, 11, 11))
    env = GridWorld(render=True, select_and_place=True, discretize=True, max_steps=1000, render_size=(64, 64))
    render = True

    env.set_task(Task("", custom_grid))

    figure_generator = CustomFigure
    figure_generator.row_figure[0, 1:4, 1:4] = 1
    figure_generator.generator_name = '[0, 1:4, 1:4]'
    
    tasks = castom_tasks()
    env = TargetGenerator(env, fig_generator=figure_generator,  tasks = tasks)
    env = SubtaskGenerator(env)
    env = VisualObservationWrapper(env)

    env = JumpAfterPlace(env)
    env = ColorWrapper(env)
    env = RangetRewardFilledField(env)
    
     # Loggers
    env = Statistics(env)
    env = R1_score(env)
   # env = SuccessRateFullFigure(env)
    env = StatisticsLogger(env, st_name = "custom_zoya_tasks_mhb2.csv")

    obs = env.reset()
    obs.keys()


    i = 0
    filenames = []
    for k in range(10):
        done = False
        obs = env.reset()
       
        step = 0 
        tryies = 0
        while not done:
       #   print(i)
          img = obs['obs']
          act = agent.act([obs])
          obs, reward, done, info = env.step(act[0])
          if ((act[0] > 5) and (act[0] < 12)):
                  tryies += 1
                 # obs, reward, done, info = env.step(5)
                  #obs, reward, done, info = env.step(5)
              #    done = env.one_round_reset()
          img = img.astype(np.uint8)
          fname = "video_step_by_step_run/frame_%d.jpg"%i
          cv2.imwrite(fname,img) 
          filenames.append(fname)
          step += 1
          i+=1
        print("builded")
        print(obs['grid'].sum(axis = 0))
        print("figure")
        print(env.figure.figure.sum(axis = 0))
        print("tryies / blocks")
        print(tryies, "/",len(np.where(env.figure.figure!=0)[0]))


    import imageio
    with imageio.get_writer('video.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    import shutil
    shutil.rmtree('./video_step_by_step_run')