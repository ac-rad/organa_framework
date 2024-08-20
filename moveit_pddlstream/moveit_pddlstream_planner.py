#!/usr/bin/env python
# Python 2/3 compatibility imports
from argparse import ArgumentError
from re import T
from imp import load_dynamic
from colorsys import hls_to_rgb
from fileinput import close
from string import printable
from tokenize import PlainToken
from six.moves import input

import sys
import os
import copy
import time
import rospy
import numpy as np
import pickle
import math
import yaml
import json

from std_msgs.msg import Bool, Header, String
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
from moveit_msgs.msg import RobotState
from tf.transformations import quaternion_from_euler
from elion_examples.srv import *
from trac_ik_python.trac_ik import IK
from scipy.spatial.transform import Rotation as R
import time
import websocket
from datetime import datetime
import pytz
from autolab_core import RigidTransform as rt

# PDDLStream path should be changed in each environment
sys.path.append('/root/git/tests/frankapy_control_test_scripts/ws_moveit/pddlstream')


from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.utils import read, Profiler
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.algorithms.meta import create_parser, solve
from pddlstream.language.constants import (
    And,
    PDDLProblem,
    print_solution,
    Minimize,
    Equal,
)
from moveit_pddlstream.move_group import MoveGroupPythonInterfaceTutorial
from moveit_pddlstream.xdl_parse import XDLParser


ARM_DOF = 7  # the robot ee is only rotated, does not have the 8th dof.
ELECTRODE_LENGTH = 0.09
GRIPPER_LENGTH = 0.15
PREINSERTIOND_DIST = 0.2

HARDWARE_SOCKET_ADDRESS = {
    "PUMP":"ws://127.0.0.1:13260/",
    "WASH":"ws://127.0.0.1:13260/",
    "MEASURE":"ws://127.0.0.1:13254/",
    "POLISH":"ws://127.0.0.1:13256/"
}


try:
    from math import pi, tau, dist, fabs, cos, sin
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sin, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


def filter_outlier(raw_array, outlier_filt: float = 2):
    # filter outliers of 2nd column of a 2D array
    mean = np.mean(raw_array[:,1])
    std = np.std(raw_array[:, 1])
    for i in range(0, raw_array.shape[0]):
        if (raw_array[i, 1] < mean - std * outlier_filt) or (raw_array[i, 1] > mean + std * outlier_filt):
            raw_array[i, :] = [np.nan, np.nan]
    return raw_array

def avg_vi(cv, block_size: int = 16, outlier_filt: float = 2):
    # calculate the block average to denoize the CV plot
    new_cv_len = cv.shape[0] // block_size
    new_cv = np.zeros((new_cv_len, 2))
    for i in range(0, new_cv_len):
        filtered_block = filter_outlier(cv[i * block_size : (i+1) * block_size, :], outlier_filt)
        avg_voltage = np.nanmean(filtered_block[:, 0])
        avg_current = np.nanmean(filtered_block[:, 1])
        new_cv[i] = [avg_voltage, avg_current]
    return new_cv

def analyze_CV(file_name, ph_value):
    cv_raw = np.genfromtxt(file_name, delimiter=',')

    # Extract cycle 4
    cycle_4 = cv_raw[:, 3] == 2
    cv_raw = cv_raw[cycle_4]
    cv_raw = cv_raw[:, 1:3]
    
    # Make plot smooth
    cv_avg = avg_vi(cv_raw, 16, 1)
    cv_avg_cropped = cv_avg[(cv_avg[:, 0] < 0.4) & (cv_avg[:, 0] > -1.4)]
    maximum = cv_avg_cropped[:, 1].argmax()
    minimum = cv_avg_cropped[:, 1].argmin()
    potential = (cv_avg_cropped[maximum, 0] + cv_avg_cropped[minimum, 0]) / 2.0
    peaks={'pH': ph_value, 'eV': potential}
    return peaks

def call_ik(env, pose, seed, solve_type, bounds=None, attempts=None):
    if not bounds:
        bounds = [0.01, 0.01, 0.01]
    if not attempts:
        attempts = 10

    ik_solver = IK(
        "panda_link0", "virtual_eef_link", timeout=0.005, solve_type=solve_type
    )

    if type(seed) == RobotState:
        seed_state = seed.joint_state.position[:ARM_DOF]
    else:
        seed_state = seed

    quaternion = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]
    target_jt_state = RobotState()
    try:
        for i in range(attempts):
            target_jt_state.joint_state.name = [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ]
            target_jt_state.joint_state.position = ik_solver.get_ik(
                seed_state,
                # X, Y, Z end-effector target pos
                pose.position.x,
                pose.position.y,
                pose.position.z,
                quaternion[0],
                quaternion[1],
                quaternion[2],
                quaternion[3],
                0.001,
                0.001,
                0.001,  # X, Y, Z bounds
                bounds[0],
                bounds[1],
                bounds[2],  # Rotation X, Y, Z bounds
            )
            if target_jt_state.joint_state.position:
                valid_state = env.check_state_validity(target_jt_state)
            else:
                valid_state = False
            if valid_state:
                break

        if target_jt_state and valid_state:
            return target_jt_state
        else:
            rospy.logwarn("Error: No IK Solution")
            return None
    except rospy.ServiceException:
        rospy.logwarn("Error: No IK Solution")
        return None

def create_problem(env, polishing_pose, washing_pose, measurement_pose, goals):
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(
        os.path.join(directory, "../../conf/pddl/electrochemistry/domain_copy.pddl")
    )
    stream_pddl = read(
        os.path.join(directory, "../../conf/pddl/electrochemistry/stream.pddl")
    )

    def check_collision(grasp_pose, otherblock, otherblock_pos):
        # Create the target object
        env.add_object(name="tmp", position=otherblock_pos)
        grasp_state = call_ik(
            pose=grasp_pose, seed=env.initial_state, solve_type="Speed"
        )
        # Remove the target object
        env.remove_object(name="tmp")
        if grasp_state is None:  # IK fails (collision)
            return False
        else:
            return True

    # def find_move(electrode, curr_state, initial_pose, target_pose):
    def find_move(curr_state, tmp_target_pose):
        print("------------------------------------------")
        print("----------- find_move --------------------")
        print("------------------------------------------")

        # get pre-approach pose
        pre_pose = copy.deepcopy(tmp_target_pose)
        post_target_pose = copy.deepcopy(tmp_target_pose)
        target_pose = copy.deepcopy(tmp_target_pose)
        target_pose.position.z -= PREINSERTIOND_DIST

        pre_target_state = call_ik(env, pose=pre_pose, seed=curr_state, solve_type="Distance")
        if pre_target_state is None:
            print("pre target state is none")
        target_state = call_ik(env, pose=target_pose, seed=pre_target_state, solve_type="Distance")
        if target_state is None:
            print("target state is none")
        post_target_state = call_ik(env, pose=post_target_pose, seed=target_state, solve_type="Distance")
        if pre_target_state is None:
            print("post target state is none")
        
        move_pre_target_trajectory = env.find_constrained_plan(goal_state=pre_target_state, start_state=curr_state, time=5)
        move_target_trajectory = env.find_constrained_plan(goal_state=target_state, start_state=pre_target_state, time=5)
        move_post_target_trajectory = env.find_constrained_plan(goal_state=post_target_state, start_state=target_state, time=5)

        if move_pre_target_trajectory is None or move_target_trajectory is None or move_post_target_trajectory is None:
            return
        else:
            yield post_target_state, [move_pre_target_trajectory, move_target_trajectory, move_post_target_trajectory]

    def update_time(delta_time):
        print("----------------------------")
        print("------- update_time --------")
        print("----------------------------")
        print("old_time", old_time)
        print("delta_time", delta_time)
        yield delta_time,


    initial_pose = env.move_group.get_current_pose().pose
    initial_state = env.move_group.get_current_state()
    initial_state.joint_state.name = initial_state.joint_state.name[:ARM_DOF]      # Ignore DoF for fingers
    initial_state.joint_state.position = initial_state.joint_state.position[:ARM_DOF]
    env.initial_state = initial_state

    # init = [('empty', 'arm'), ('handpose', initial_pose), ('athandpose', initial_pose),
    #         ('state', initial_state), ('atstate', initial_state)]

    constant_map = {
        "measurement_pose": measurement_pose,
        "polishing_pose": polishing_pose,
        "washing_pose": washing_pose,
        "measurement_station": "measurement_station",
        "ph_station" : "ph_station",
        "robot_agent": "robot_agent",
        "pump_agent": "pump_agent",
        "polishing_action": "polishing_action",
        "washing_action": "washing_action",
        "moving_action": "moving_action",
        "transferring_action": "transferring_action",
        "water": "water",
        "buffer": "buffer",
        "nacl": "nacl",
        "quinone": "quinone",
        "target_solution": "target_solution",
    }

    init = [
        ("at_pose", "electrode", initial_pose),
        ("electrode", "electrode"),
        ("solution", "measurement_solution"),
        ("solution", "quinone"),
        ("solution", "buffer"),
        ("solution", "water"),
        ("solution", "nacl"), 
        ("solution", "target_solution"), 
        ("beaker", "water_container"),
        ("beaker", "buffer_container"),
        ("beaker", "quinone_container"),
        ("beaker", "nacl_container"),
        ("beaker", "wasting_station"),
        ("beaker", "measurement_station"),
        ("beaker", "ph_station"),
        ("object_pose", initial_pose),
        ("object_pose", polishing_pose),
        ("object_pose", washing_pose),
        ("object_pose", measurement_pose),
        ("agent", "robot_agent"),
        ("agent", "pump_agent"),
        ("action", "polishing_action"),
        ("action", "washing_action"),
        ("action", "moving_action"),
        ("action", "transferring_action"),
        Equal(("total-cost",), 0.0),
        Equal(("delta_time",), 0.01),
        Equal(("polishing_time",), 60.0),
        Equal(("washing_time",), 40.0),
        Equal(("transferring_time",), 120.0),
        Equal(("moving_time",), 10.0),
        ("robot_state", initial_state),
        ("at_state", initial_state),
        ("beaker_contains", "ph_station", "water"),
    ]


    # full experiment
    if goals[0] == "characterized":
        goal1 = ["characterized"]
        goal2 = ["ph_measured"]
        goal3 = ["beaker_cleaned", "measurement_station"]
        goal4 = ["beaker_cleaned", "ph_station"]
        goal5 = ["beaker_contains", "ph_station", "water"]
        goal = And(goal1, goal2, goal3, goal4, goal5)
    else:
        return

    samples = []
    roadmap = []
    stream_map = {
        "find_move": from_gen_fn(find_move),
    }
    """print("-----------------------------")
    print("init", init)
    print("goal", goal)
    print("-----------------------------")"""
    problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)
    return problem, samples, roadmap


class RobotPlanner:
    def __init__(self):
        parser = create_parser()
        self.args = parser.parse_args()
        print("Arguments:", self.args)
    
        self.plans = []
        self.hardware_plans = []
        self.goals = []
        self.output_xdl_path = None
        self.kb_path = None
        self.knowledge_base = None
        self.initial_angles = None
        self.isRobExcuteDone = False
        self.isHardwareExcuteDone = False
        self.cv_filename = None
        self.buffer_ph = 4
        self.current_ph =10000

        rospy.init_node("moveit_pddlstream", anonymous=True)
        rate = rospy.Rate(1)  # 1 Hz

        self.env = MoveGroupPythonInterfaceTutorial()
        self.rviz_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        rospy.loginfo("Robot/hardware subscriber is ready to receive commands.")
        self.robot_execute_sub = rospy.Subscriber("/robot_execute", String, self.robot_execute_callback)
        self.robot_execute_status_pub = rospy.Publisher("/robot_execute_status", Bool, queue_size = 10)
        self.robot_execute_start_pub = rospy.Publisher("/planning/execute", String, queue_size = 10)
        self.viewpose_sub = rospy.Subscriber("/viewpose_execute", Bool, self.viewpose_callback_execution)
        self.viewpose_execute_status_pub = rospy.Publisher("/viewpose_execute_status", Bool, queue_size=10)
        rospy.spin()

    def viewpose_callback_execution(self, data):
        if data.data:
            rospy.loginfo("Request to goto viewpose from task manager !")
            curr_state = self.env.move_group.get_current_state()
            curr_state.joint_state.name = curr_state.joint_state.name[:ARM_DOF]      # Ignore DoF for fingers
            curr_state.joint_state.position = curr_state.joint_state.position[:ARM_DOF]
            target_state = copy.deepcopy(curr_state)
            target_state.joint_state.position = [ 6.71515064e-01, -1.19781497e+00,  4.89822817e-04, -2.69177437e+00, 5.18479210e-01,  1.05301683e+00,  1.81695329e+00]
            
            move_trajectory = self.env.find_constrained_plan(goal_state=target_state, start_state=curr_state, time=10)
            move_trajectory = self.env.move_group.retime_trajectory(
                        self.env.move_group.get_current_state(),
                        move_trajectory,
                        velocity_scaling_factor=0.07,
                        acceleration_scaling_factor=0.07,
                        algorithm="iterative_spline_parameterization",
                    )
            print("...........executing robot plan -> goto viewpose.........")
            self.env.execute_plan(move_trajectory)
            isAtViewPose = True
            self.viewpose_execute_status_pub.publish(True)

    def robot_execute_callback(self, data):    
        rospy.loginfo("Request to execute exp plan from task manager !")
        self.robot_execute_start_pub.publish("start_execution")
        self.kb_path = data.data.split("!")[0]
        self.output_xdl_path = data.data.split("!")[1]
        self.goals = []       # reset goal

        # handle request 
        self.parse_plan()
        if len(self.goals) != 0:
            self.plan()
            self.visualize()
            self.isRobExcuteDone = self.execute()
        if self.isRobExcuteDone: 
            rospy.loginfo("Robot finished execution. Sending status back to task manager...")
            self.robot_execute_status_pub.publish(self.isRobExcuteDone)
            self.plans = []
            self.hardware_plans = []
            self.goals = []

    def measure_ph(self):
        previous_ph = 0
        while abs(self.current_ph-previous_ph)/self.current_ph > 0.1:
            previous_ph = self.current_ph
            time.sleep(10)
            self.send_command_to_hardware("PUMP", "get_pH")
        print(f'Final Ph Value: {self.current_ph}')

    def on_message(self, ws, message):
        if message == 'Done':
            ws.close()
        else:
            self.current_ph = float(message.split(',')[-1])

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws, close_status_code, close_msg):
        return

    def send_command_to_hardware(self,hardware, command):
        ws = websocket.WebSocketApp(HARDWARE_SOCKET_ADDRESS[hardware],
                                    on_message=self.on_message,
                                    on_open=lambda ws: ws.send(command),
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        ws.run_forever()
        time.sleep(5)

    def measure_cv(self):
        now = datetime.now()
        est_now = now.astimezone(pytz.timezone("America/Toronto"))
        log_path = "/home/sf3202msi/franka_arm_infra/tests/frankapy_control_test_scripts/cv_data"
        filename = est_now.strftime(f"ph_{self.buffer_ph}_%Y%m%d_%H%M%S.csv")
        full_filepath = f'{log_path}/{filename}' 
        self.send_command_to_hardware("MEASURE", full_filepath)
        docker_filepath = f'/root/git/tests/frankapy_control_test_scripts/cv_data/{filename}'
        return docker_filepath

    def update_knowledge_base(self):
        with open(self.kb_path, 'w') as outfile:
            json.dump(self.knowledge_base, outfile, indent=4)

    def write_knowledge_base(self, updated_kb):
        with open(self.kb_path, "w") as f:
            json.dump(updated_kb, f, indent=4)
        return
    
    def get_grasp_pose(self, obj_pose, z_offset = 0, x_offset=0, y_offset=0):
        obj_pose = copy.deepcopy(obj_pose)
        obj_pose.position.z += ELECTRODE_LENGTH + PREINSERTIOND_DIST # to get pre approach pose 
        obj_pose.position.x -= GRIPPER_LENGTH   # calibration for gripper length

        obj_pose.position.z += z_offset
        obj_pose.position.x += x_offset
        obj_pose.position.y += y_offset
        return obj_pose

    def get_target_poses(self, initial_pose):
        washing_beaker=copy.deepcopy(np.array(self.knowledge_base['objects']['wash_beaker']['pose']))
        exp_beaker=copy.deepcopy(np.array(self.knowledge_base['objects']['experiment_beaker']['pose']))
        polishing_plate=copy.deepcopy(np.array(self.knowledge_base['objects']['polish_plate']['pose']))

        polishing_pose_org = copy.deepcopy(initial_pose)
        polishing_pose_org.position.x = polishing_plate[0]      
        polishing_pose_org.position.y = polishing_plate[1]
        polishing_pose_org.position.z = polishing_plate[2]

        washing_pose_org = copy.deepcopy(initial_pose)
        washing_pose_org.position.x = washing_beaker[0]
        washing_pose_org.position.y = washing_beaker[1]
        washing_pose_org.position.z = washing_beaker[2]

        experiment_pose_org = copy.deepcopy(initial_pose)
        experiment_pose_org.position.x = exp_beaker[0]
        experiment_pose_org.position.y = exp_beaker[1]
        experiment_pose_org.position.z = exp_beaker[2]

        return washing_pose_org, experiment_pose_org, polishing_pose_org

    def parse_goals(self, goal_list):
        reagent_amounts = []
        goal_experiments = []
        for goal in goal_list:
            if type(goal) == list:
                if goal[0] == 'pump':
                    # construct pump command REAGENT_DISPENSEBEAKER_VOLUME
                    if goal[2] == "experiment_beaker": goal[2]="beaker"
                    if "buffer" in goal[1]: self.buffer_ph = goal[1].replace("buffer", "pH").split("pH")[1]
                    pump_msg=f'{goal[1]}_{goal[2]}_{goal[-1]}'
                    reagent_amounts.append(goal[-1])
                    self.hardware_plans.append(['pump', pump_msg])
                elif goal[0] == 'stir_reagent':
                    pump_msg=f'beaker_beaker_{5}'
                    self.hardware_plans.append(['stir_reagent', pump_msg])
                elif goal[0] == 'monitor' :
                    self.hardware_plans.append(goal)
                elif goal[0] == 'RunCV':
                    self.goals = ["characterized"]

    def parse_plan(self):
        xdl_parser = XDLParser(self.output_xdl_path, self.kb_path)
        xdl_parser.parse()
        # load knowledge base and generate plan
        print(".....PARSE XDL PLAN FROM LANGUAGE.....")
        with open(self.kb_path, 'r') as f:
            self.knowledge_base = json.load(f)
        self.parse_goals(self.knowledge_base['goals'])

    def plan(self, max_time=200):
        print("..... PDDLSTREAM PLANING .....")

        self.initial_angles = [0, -tau / 8, 0, -3 * tau / 8, 0, tau / 4, tau / 8]
        self.env.go_to_initial_joint_state(self.initial_angles)

        initial_pose = self.env.move_group.get_current_pose().pose
        washing_pose_org, experiment_pose_org, polishing_pose_org = self.get_target_poses(initial_pose)
        polishing_pose = self.get_grasp_pose(polishing_pose_org, z_offset=0., x_offset=0.03, y_offset=0.015)
        washing_pose = self.get_grasp_pose(washing_pose_org, z_offset=0.0)
        experiment_pose = self.get_grasp_pose(experiment_pose_org, z_offset=0.03, y_offset=0.005)

        start = time.time()
        problem, samples, roadmap = create_problem(self.env, polishing_pose, washing_pose, experiment_pose, self.goals)
        constraints = PlanConstraints(max_cost=1.25)

        with Profiler(field="tottime", num=10):
            self.solution = solve(
                problem,
                algorithm="binding",
                unit_costs=self.args.unit,
                search_sample_ratio=1,
                reorder=False,
                planner="ff-astar2",
                # max_iterations= 2,
                debug=False,
                verbose =True,
                # max_effort=2,
                initial_complexity=4,
                visualize= True
            )
        print("======================================")
        print("=============== PLAN =================")
        print("======================================")
        print("execution time for planning:", time.time() - start)
        print("cost", self.solution.cost)
        for i, action in enumerate(self.solution.plan):
            if action.name =='move':
                print(f"step {i}: {action.name}")
            else:
                print(f"step {i}: {action.name} , {action.args}")
        print("======================================")
        print("======================================")

    def load_plan(self, filename):
        with open(filename, "rb") as f:
            self.plans = pickle.load(f)

    def visualize(self):
        def rviz_display(plans):
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = env.move_group.get_current_state()
            for plan in plans:
                if plan["type"] == "robot":
                    display_trajectory.trajectory.append(plan["plan"])
            # Publish
            self.rviz_publisher.publish(display_trajectory)

        env = self.env
        env.go_to_initial_joint_state(self.initial_angles)

        input("Press Enter to Construct plan list and Visualize plan:")
        plan = None
        post_target_plan = None
        target_plan = None
        pre_target_plan = None
        for i, action in enumerate(self.solution.plan):
            if action.name == "move":
                pre_target_plan = action.args[-1][0]
                target_plan = action.args[-1][1]
                post_target_plan = action.args[-1][2]
                self.plans.append({"type": "robot", "plan": pre_target_plan})
                self.plans.append({"type": "robot", "plan": target_plan})
            elif action.name == "polish":
                self.plans.append({"type": "polish", "plan": 30})               # for real experiment: 30 s
                self.plans.append({"type": "robot", "plan": post_target_plan})
            elif action.name == "wash_electrode":
                self.plans.append({"type": "wash", "plan": 30})                 # for real experiment: 30 s
                self.plans.append({"type": "robot", "plan": post_target_plan})
            elif action.name == "measure_redux_potential":
                self.plans.append({"type": "CV", "plan": None})
                self.plans.append({"type": "robot", "plan": post_target_plan})
            elif action.name == "empty_clean_ph_station":
                pump_msg = f'empty_pH'
                self.plans.append({"type": "pump", "plan": pump_msg})
            elif action.name == "empty_clean_measurement_station":
                pump_msg = f'flush'
                self.plans.append({"type": "pump", "plan": pump_msg})
                pump_msg = f'water_beaker_15'
                self.plans.append({"type": "pump", "plan": pump_msg})
                pump_msg = f'beaker_waste_16'
                self.plans.append({"type": "pump", "plan": pump_msg})
            elif action.name == "transfer_target_solution":
                pump_msg = f'beaker_ph_15'
                self.plans.append({"type": "pump", "plan": pump_msg})
            elif action.name == "add_water_ph_station":
                pump_msg = f'water_ph_15'
                self.plans.append({"type": "pump", "plan": pump_msg})
            elif action.name == "measure_ph":
                self.plans.append({"type": "ph", "plan": None})
            elif action.name == "transfer_quinone":
                reagent = "quinone"
                volume = None
                dispense_beaker = None
                for goal in self.knowledge_base['goals']:
                    if goal[0] == "pump":
                        if goal[1] == reagent:
                            volume = goal[-1]
                            dispense_beaker = "beaker"
                            break
                pump_msg=f'{reagent}_{dispense_beaker}_{volume}'              #pump msg format: REAGENT_DISPENSEBEAKER_VOLUME
                self.plans.append({"type": "pump", "plan": pump_msg})
            elif action.name == "transfer_liquid":
                reagent = None
                volume = None
                dispense_beaker = None
                for goal in self.knowledge_base['goals']:
                    if goal[0] == "pump":
                        if action.args[0] in goal[1]:
                            reagent = goal[1]
                            volume = goal[-1]
                            dispense_beaker = "beaker"
                            break
                pump_msg = f'{reagent}_{dispense_beaker}_{volume}'             #pump msg format: REAGENT_DISPENSEBEAKER_VOLUME
                self.plans.append({"type": "pump", "plan": pump_msg})
            elif action.name == "mix_solution":
                pump_msg = f'beaker_beaker_{5}'
                self.plans.append({"type": "pump", "plan": pump_msg})

        for i, plan in enumerate(self.plans):
            if plan["type"] =='robot':
                print(f"step {i}: {'robot'}")
            else:
                print(f"step {i}: {plan}")

        if len(self.plans) > 0:
            rviz_display(self.plans)
            print("plan length", len(self.plans))
            time.sleep(2.0)
            print("going to initial configuration.")
        env.go_to_initial_joint_state(self.initial_angles)

        now = time.strftime("%Y%m%d%H%M")
        with open("plans_{}.pkl".format(now), "wb") as f:
            pickle.dump(self.plans, f)

    def execute(self):
        env = self.env
        env.go_to_initial_joint_state()
        answer = input("Execute the plan? [y/n]: ")
        env.execution_publisher.publish("start_execution")
        time.sleep(5)
        if answer == "y":
            for plan_dict in self.plans:
                if plan_dict["type"] == "robot":
                    plan = plan_dict["plan"]
                    plan = self.env.move_group.retime_trajectory(
                        env.move_group.get_current_state(),
                        plan,
                        velocity_scaling_factor=0.07,
                        acceleration_scaling_factor=0.07,
                        algorithm="iterative_spline_parameterization",
                    )
                    print("...........executing robot plan.........")
                    start = time.time()
                    env.execute_plan(plan)
                    print("!!!!!!!! execution time for robot action:", time.time() - start)

                elif plan_dict["type"] == "polish":
                    print("...........executing polish plan.........")
                    self.send_command_to_hardware("POLISH", str(plan_dict["plan"]))
                elif plan_dict["type"] == "wash":
                    print("...........executing wash plan.........")
                    self.send_command_to_hardware("WASH", "start_stir")    # send stirring command
                    time.sleep(int(plan_dict["plan"]))
                    self.send_command_to_hardware("WASH", "stop_stir")
                elif plan_dict["type"] == "CV":
                    print("...........measuring CV.........")
                    start = time.time()
                    self.cv_filename = self.measure_cv()
                    print("!!!!!!!! execution time for measuring CV:", time.time() - start)
                    print(".............measure_cv done ............")
                elif plan_dict["type"] == "pump":
                    print("...........executing pump plan.........")
                    start = time.time()
                    self.send_command_to_hardware("PUMP", str(plan_dict["plan"]))  
                    output = str(plan_dict["plan"])
                    time_diff=time.time() - start
                    print(f"!!!!!!!! execution time for {output}: {time_diff}")
                elif plan_dict["type"] == "ph":
                    print("...........measuring ph .........")
                    start = time.time()
                    self.measure_ph()
                    observations = analyze_CV(self.cv_filename, self.current_ph)
                    self.knowledge_base["observation"].append(observations)    # update observation in kb
                    self.update_knowledge_base()
                    print("!!!!!!!! execution time for measuring ph:", time.time() - start)
                    print(".............measure_ph done ............")

            print("Finish execution")
        else:
            print("Finish without executing the plan")
        
        env.go_to_initial_joint_state()
        return True

    def execute_hardware(self):
        print(".....EXECUTE HARDWARE PLAN.....")
        for step, goal in enumerate(self.hardware_plans):
            print(f'Step {step+1}: {goal[-1]}')
            if goal[0] == "pump" or goal[0] == "stir_reagent":
                self.send_command_to_hardware("PUMP", goal[-1])
            if goal[0] == "flush":
                self.send_command_to_hardware("PUMP", "flush")
            elif goal[0] == "monitor":
                print("measuring Ph.....")
                ph_value = measure_ph()
        return True


def main():
    planner = RobotPlanner()
    