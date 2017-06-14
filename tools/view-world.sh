#!/bin/bash
# Runs gazebo with the correct plugin path
# can also specify an extra model path (generated after an evolutionary run for example)
WORLD=$1
MODELS=$2
CMD=gazebo
GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:`pwd`/../build GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:`pwd`/../../revolve/tools/models:`pwd`/models:$2 $CMD -u $WORLD
