#!/bin/bash

file=./tools/infer_seg_voc.py
inferset=val
crf=true

cpt=./00_sota/voc/checkpoints/model_sota.pth
python $file --model_path $cpt --infer_set $inferset  --crf_post $crf