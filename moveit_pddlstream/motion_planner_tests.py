#!/usr/bin/env python

import yaml
import os
import matplotlib
import moveit_pddlstream_planner
import json
import numpy as np


with open(os.path.expanduser("~") + '/ws_moveit/src/moveit_pddlstream/src/knowledge_base.json') as f:
    knowledge_base = yaml.safe_load(f)


# x: 0.3 -- 0.7
# y: -0.3 -- 0.3
# z: assume 0

valid_poses = []
for i in range(100):
    poses_x = np.random.random(3) * 0.1 - 0.05
    poses_y = np.random.random(3) * 0.1 - 0.05
    j = 0
    for obj in knowledge_base['objects']:
        knowledge_base['objects'][obj]['pose'][0] += poses_x[j]
        knowledge_base['objects'][obj]['pose'][1] += poses_y[j]
        j += 1
    with open('knowledge_base.json', 'w') as outfile:
        json.dump(knowledge_base, outfile, indent=4)

    try:
        moveit_pddlstream_planner.main()
        valid_poses.extend(np.vstack(poses_x, poses_y))
    except:
        print('failed')

valid_poses = np.array(valid_poses)
print(valid_poses)
np.savetxt('valid_poses.txt', valid_poses)
