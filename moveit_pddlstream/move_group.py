#!/usr/bin/env python
from __future__ import print_function

from hashlib import algorithms_available

import copy
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import shape_msgs.msg
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetStateValidity, GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_euler
from elion_examples.srv import *

try:
    from math import pi, tau, dist, fabs, cos, sin
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sin, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from std_msgs.msg import String, Header
from moveit_commander.conversions import pose_to_list

GRASPING_GROUP_NAME = "hand"
ARM_GROUP_NAME = "panda_arm"

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class MoveGroupPythonInterfaceTutorial(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        group_name = ARM_GROUP_NAME
        move_group = moveit_commander.MoveGroupCommander(group_name)
        move_group.set_max_velocity_scaling_factor(0.1)
        move_group.set_max_acceleration_scaling_factor(0.1)
        # trajectory_publisher = rospy.Publisher(
        #     "joint_state",
        #     sensor_msgs.msg.JointState,
        #     queue_size=20,
        # )

        self.gripper_publisher = rospy.Publisher(
            "franka_gripper", String, queue_size=10)
        self.execution_publisher = rospy.Publisher(
            "/planning/execute", String, queue_size=10)

        planning_frame = move_group.get_planning_frame()

        eef_link = move_group.get_end_effector_link()

        group_names = robot.get_group_names()

        self.box_name = ""
        self.group_name = group_name
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        # self.trajectory_publisher = trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        try:
            # Wait for 10 seconds and assumes we don't want IK
            rospy.wait_for_service('compute_ik', timeout=10.0)
            self.compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
        except rospy.ROSException:
            rospy.logwarn(
                "MoveIt was not loaded and arm teleop will not be available")
            self.compute_ik = None

        try:
            rospy.wait_for_service('plan_kinematic_path', timeout=10.0)
            self.motion_planner = rospy.ServiceProxy(
                'plan_kinematic_path', GetMotionPlan)
        except rospy.ROSException:
            rospy.logwarn(
                "MoveIt was not loaded and planner will not be available")
            self.motion_planner = None

        rospy.wait_for_service('/constraint_planning', timeout=10.0)
        try:
            self.elion_planner = rospy.ServiceProxy(
                '/constraint_planning', elionpy)
        except rospy.ServiceException as e:
            print("Elion service call failed: {}".format(e))
            sys.exit()

        rospy.wait_for_service('compute_cartesian_path', timeout=10.0)
        try:
            self.cartesian_planner = rospy.ServiceProxy(
                '/compute_cartesian_path', GetCartesianPath)
        except rospy.ServiceException as e:
            print("Cartesian path planning service call failed: {}".format(e))
            sys.exit()

        rospy.wait_for_service('/check_state_validity', timeout=10.0)
        try:
            self.moveit_state_validity_srv = rospy.ServiceProxy(
                '/check_state_validity', GetStateValidity)
        except rospy.ServiceException as e:
            print("State validity service call failed: {}".format(e))
            sys.exit()

    def check_state_validity(self, robot_state_):
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = self.move_group.get_name()
        req.robot_state = robot_state_
        response = self.moveit_state_validity_srv.call(req)
        return response.valid

    def go_to_initial_joint_state(self, joint_angles=None):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        if joint_angles is None:
            joint_goal[0] = 0
            joint_goal[1] = -tau / 8
            joint_goal[2] = 0
            joint_goal[3] = -3 * tau / 8
            joint_goal[4] = 0
            joint_goal[5] = tau / 4
            joint_goal[6] = tau / 8
        else:
            assert (len(joint_angles) == 7)
            joint_goal[0:7] = joint_angles

        move_group.go(joint_goal, wait=True)
        move_group.stop()

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def find_constrained_plan(self, start_state, goal_state, time=None, tolerances=None, orientation=None,
                              collisions=None):
        if not time:
            time = 5
        if not tolerances:
            tolerances = [-1, -1, -1]

        request = moveit_msgs.msg.MotionPlanRequest()
        request.planner_id = 'PRMstar'
        request.allowed_planning_time = time
        request.start_state = start_state
        attached_obj = self.scene.get_attached_objects()
        if attached_obj:
            # obj = moveit_msgs.msg.AttachedCollisionObject()
            # obj.link_name = "virtual_eef_link"
            # print('asdf', attached_obj)
            # obj.object = attached_obj[list(attached_obj.keys())[0]]
            # obj.touch_links = ["dynamixel_robotiq_joint", "robotiq_85_left_inner_knuckle_joint",
            #                    "robotiq_85_left_finger_tip_joint","robotiq_85_left_knuckle_joint",
            #                    "robotiq_85_right_inner_knuckle_joint","robotiq_85_right_finger_tip_joint",
            #                    "robotiq_85_right_knuckle_joint"]
            request.start_state.attached_collision_objects = [attached_obj[list(attached_obj.keys())[0]]]

        orientation_constraint = moveit_msgs.msg.OrientationConstraint()
        orientation_constraint.header.frame_id = "panda_link0"
        orientation_constraint.link_name = "virtual_eef_link"
        if not orientation:
            qx, qy, qz, qw = quaternion_from_euler(tau / 2, -tau / 4, 0)
            orientation_constraint.orientation.x = qx
            orientation_constraint.orientation.y = qy
            orientation_constraint.orientation.z = qz
            orientation_constraint.orientation.w = qw
        else:
            orientation_constraint.orientation = orientation
        orientation_constraint.absolute_x_axis_tolerance = tolerances[0]
        orientation_constraint.absolute_y_axis_tolerance = tolerances[1]
        orientation_constraint.absolute_z_axis_tolerance = tolerances[2]
        request.path_constraints.orientation_constraints = [
            orientation_constraint]
        request.path_constraints.name = "AngleAxis"

        if collisions is None:
            collisions = []

        try:
            response = self.elion_planner(request, goal_state, collisions)
            plan = response.trajectory.trajectory
            #print("goal state comparison", goal_state, plan.joint_trajectory.points[-1])
            return plan
        except rospy.service.ServiceException as e:
            print("Error:", e)
            print("No constrained motion plan found!")


    def find_grasp_plan_linear(self, start_state, goal_state, sample_plan):
        print("*************************************************")
        print("*************************************************")
        print("*************************************************")
        print("*************************************************")
        print("[find_grasp_plan_linear] sample_plan: \n", sample_plan)
        plan = copy.deepcopy(sample_plan)
        print("*************************************************")
        print("[find_grasp_plan_linear] plan: \n", plan)
        print("*************************************************")
        print("[find_grasp_plan_linear] plan: \n", plan.joint_trajectory.points)
        print("*************************************************")
        print("[find_grasp_plan_linear] type(plan): \n", type(plan.joint_trajectory.points))

        plan.joint_trajectory.points=[]
        print("*************************************************")

        print("[find_grasp_plan_linear] plan: \n", plan)


        start_point= moveit_msgs.msg.trajectory_msgs.msg.JointTrajectoryPoint()
        start_point.positions= start_state.joint_state.position

        goal_point= moveit_msgs.msg.trajectory_msgs.msg.JointTrajectoryPoint()
        goal_point.positions= goal_state.joint_state.position
        
        print("*************************************************")
        print("[find_grasp_plan_linear]  goal_state: \n", goal_state)
        print("[find_grasp_plan_linear]  start_state: \n", start_state)
        
        print("[find_grasp_plan_linear]  goal_point: \n", goal_point)
        print("[find_grasp_plan_linear]  start_point: \n", start_point)
        
        print("*************************************************")
        plan.joint_trajectory.points.append(start_point)
        plan.joint_trajectory.points.append(goal_point)

        print("[find_grasp_plan_linear]  plan: \n", plan)

        print("*************************************************")
        print("*************************************************")
        
        return plan

    def find_grasp_plan(self, start_state, goal_state, bounding_seed):
        request = moveit_msgs.msg.MotionPlanRequest()
        request.planner_id = 'PRMstar'
        request.allowed_planning_time = 20
        request.start_state = start_state
        position_constraint = moveit_msgs.msg.PositionConstraint()
        position_constraint.header.frame_id = "panda_link0"
        position_constraint.link_name = "virtual_eef_link"

        position_constraint.constraint_region = moveit_msgs.msg.BoundingVolume()
        position_constraint.constraint_region.primitives = []
        box = shape_msgs.msg.SolidPrimitive()
        #box.type = 1
        box.type = shape_msgs.msg.SolidPrimitive.BOX
        box.dimensions = [0.05, 0.05, 0.2]
        position_constraint.constraint_region.primitives.append(box)
        position_constraint.constraint_region.primitive_poses = [bounding_seed]
        request.path_constraints.position_constraints = [position_constraint]
        request.path_constraints.name = "box"

        collisions = []

        try:
            print("----------------------------------")
            print("----------------------------------")
            print('request: \n',request,'goal_state: \n' ,goal_state, \
            'collisions: \n', collisions)
            print("----------------------------------")
            print('BOX PLAN')
            response = self.elion_planner(request, goal_state, collisions)
            plan = response.trajectory.trajectory
            print("goal state comparison", goal_state,
                  plan.joint_trajectory.points[-1])
            print("----------------------------------")
            print("joint_trajectory", plan.joint_trajectory.points)
            print("----------------------------------")
            return plan
        except rospy.service.ServiceException as e:
            print("Error:", e)
            print("No constrained motion plan found!")

    def find_motion_plan(self, start_state, goal_state):
        request = moveit_msgs.msg.MotionPlanRequest()
        request.planner_id = 'PRMstar'
        request.group_name = self.group_name
        request.allowed_planning_time = 5
        request.num_planning_attempts = 3
        request.start_state = start_state
        constraints = moveit_msgs.msg.Constraints()
        constraints.name = "goal_state"
        joint_constraints = []
        for i in range(len(goal_state.joint_state.name)):
            goal_constraint = moveit_msgs.msg.JointConstraint()
            goal_constraint.joint_name = goal_state.joint_state.name[i]
            goal_constraint.position = goal_state.joint_state.position[i]
            goal_constraint.tolerance_above = 0.01
            goal_constraint.tolerance_below = 0.01
            goal_constraint.weight = 100
            joint_constraints.append(goal_constraint)
        constraints.joint_constraints = joint_constraints
        request.goal_constraints = [constraints]
        try:
            response = self.motion_planner(request)
            print('=========response', response)
            plan = response.motion_plan_response.trajectory

            return plan
        except rospy.service.ServiceException as e:
            print("Error:", e)
            print("No motion plan found!")

    def execute_plan(self, plan):
        move_group = self.move_group
        move_group.execute(plan, wait=True)

    def wait_for_state_update(
            self, box_is_known=False, box_is_attached=False, timeout=4
    ):

        box_name = self.box_name
        scene = self.scene

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            is_known = box_name in scene.get_known_object_names()
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            rospy.sleep(0.1)
            seconds = rospy.get_time()

        return False

    def add_box(self, name="box", position=(0, 0, 0), size=(0.05, 0.05, 0.05), timeout=4):
        x, y, z = position
        size_x, size_y, size_z = size
        self.scene_interface.addBox(name, size_x, size_y, size_z, x, y, z)

        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def add_object(self, name="box", position=(0, 0, 0), size=(0.05, 0.05, 0.05), timeout=4):
        object_pose = geometry_msgs.msg.PoseStamped()
        object_pose.header.frame_id = "panda_link0"
        object_pose.pose.orientation.w = 1.0
        x, y, z = position
        object_pose.pose.position.x = x
        object_pose.pose.position.y = y
        object_pose.pose.position.z = z
        self.scene.add_box(name, object_pose, size=size)
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def attach_box(self, box_name="box", timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link
        group_names = self.group_names

        grasping_group = GRASPING_GROUP_NAME
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)

        return self.wait_for_state_update(
            box_is_attached=True, box_is_known=False, timeout=timeout
        )

    def detach_box(self, box_name="box", timeout=4):
        scene = self.scene
        eef_link = self.eef_link

        scene.remove_attached_object(eef_link, name=box_name)

        return self.wait_for_state_update(
            box_is_known=True, box_is_attached=False, timeout=timeout
        )

    def remove_object(self, name="box", timeout=4):
        scene = self.scene

        scene.remove_world_object(name)

        return self.wait_for_state_update(
            box_is_attached=False, box_is_known=False, timeout=timeout
        )
