#!/bin/bash

cd baku

export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=6
# python  train.py \
#     agent=baku suite=aloha dataloader=aloha \
#     suite/task=aloha \
#     suite.task_make_fn.height=480 \
#     suite.task_make_fn.width=640 \
#     suite.pixel_keys=["top","right_wrist","left_wrist"] \
#     data_fold="aloha_h480w640-joints_baku_aloha_2arm_ep100"  \
#     info="bs8_j2j_2task_M2arm_D2arm_top-right-left_step10w_ep100" 


python  train.py \
    agent=baku suite=aloha dataloader=aloha \
    suite/task=aloha \
    suite.task_make_fn.height=240 \
    suite.task_make_fn.width=320 \
    suite.pixel_keys=["top","angle"] \
    data_fold="aloha_h240w320_joints_top-angle_baku_aloha_2arm_ep200"  \
    info="bs32_j2j_2task_M2arm_D2arm_top-angle_ep200" 