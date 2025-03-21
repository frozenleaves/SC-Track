#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: config.py.py
# @Author: Li Chengxin 
# @Time: 2023/6/30 22:12


# Global configuration variables. If you are not sure what these parameters are, please do not modify them,
# except for those marked with *

from __future__ import annotations

from typing import List, Tuple

AVAILABLE_RANGE_COEFFICIENT: float = 1.5 # The available range coefficient of candidate screening for matching algorithms

RAW_INPUT_IMAGE_SIZE: Tuple = (2048, 2048)   # * image size (width, height)

GAP_WINDOW_LEN = 10 # Maximum number of lost track

ENTER_DIVISION_THRESHOLD = 50 # Starting from the completion of cell division, the second division is not allowed until the value is exceeded

USING_IMAGE_FOR_TRACKING = False   # * Using additional image information for tracking

RECORD_SPEED: bool = False    # *  Record tracking speed

SMOOTH_WINDOW_LEN = 10  # Window length of smooth classification algorithm

# TRACKING_CHECK_WINDOW_LEN: int = 100  # Tracking algorithm backtracking window length

PROB_THRESHOLD: float = 0.6  # Probability threshold of smooth classification algorithm

N_CLASS = 4    # *  Number of annotation categories

CLASS_NAME: List | None = ['G1', 'S', 'G2', 'M']   # *  Class name list
# CLASS_NAME = None

EXPORT_TRACKING_VISUALIZATION = True

INCLUDE_CELL_CYCLE = True

FILTER_THRESHOLD = 10
