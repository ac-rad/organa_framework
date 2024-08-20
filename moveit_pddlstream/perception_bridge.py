#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import json


class Perception:
    def __init__(self):
        self.data = None
        self.objects = {}
        self.initial_angles = []
        self.subscriber = rospy.Subscriber("perception", String, self.callback)

    def callback(self, data):
        self.data = json.loads(data.data)

    def wait_for_data(self):
        rate = rospy.Rate(10)
        while self.data is None:
            rate.sleep()
        for d in self.data:
            if d["type"] == "object":
                name = d["data"]["name"]
                self.objects[name] = d["data"]["position"]
            elif d["type"] == "joints":
                self.initial_angles = d["data"]
        # self.objects['beaker'][0] = self.objects['beaker'][0] + 0.07
        # self.objects['beaker'][1] = self.objects['beaker'][1] + 0.02
        # self.objects['beaker'][2] = self.objects['beaker'][2] + 0.04
        # self.objects['beaker2'][0] += 0.1
        # self.objects['beaker2'][1] -= 0.18
        # self.objects['beaker2'][2] += 0.19
        # print("objects:", self.objects)
