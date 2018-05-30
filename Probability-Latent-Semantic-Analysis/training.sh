#!/bin/bash

if [ ! -d "Topic" ]; then
  mkdir Topic
fi

if [ ! -d "exp" ]; then
  mkdir exp
fi

python main.py
python visualization.py
