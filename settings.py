#!/usr/bin/python
# -*- encoding:utf-8 -*-

import os
################
# Project Configuration
################
# Path to project
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
# Path to source data or called input data
SOURCE_DATA = BASE_PATH + '/src_data/'
# Path to static files such as stopwords
STATIC_DIR = BASE_PATH + '/static_data/'
# Path to save Hierarchical Bayes model
BAYES_DIR = BASE_PATH + '/static_data/HIERARCHICAL_BAYES_MODEL/'
