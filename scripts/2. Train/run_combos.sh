#!/bin/bash

combos='global'

for combo in $combos
do 
  echo Running combo $combo
  python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserGlobalModel' --fast_dev_run
done
