#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool, String
from organa.srv import GetExpPlan, GetExpPlanResponse

import os
import numpy as np
import openai
from tqdm import tqdm
import sys
sys.path.append('/root/git/frankapy/catkin_ws/src/organa/')
sys.path.append('/root/git/frankapy/catkin_ws/src/organa/clairify/')

import utils
from clairify.xdlgenerator.nlp2xdl import generate_xdl

DEBUG_MODE = True

#openai.api_key = os.environ["OPENAI_API_KEY"]


def prompt_llm(
    instructions,
    system_prompt="You are a robot that performs chemistry experiments.\
        A chemist will ask you to plan an experiment.",
    model="gpt-4",
):
    """prompt_llm.

    Parameters
    ----------
    instructions :
        instructions
    system_prompt :
        system_prompt
    model :
        model
    """
    response = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instructions},
        ],
    )
    return response["choices"][0]["message"]["content"]

def format_prompt(prompts, memories, init_conditions_dict):
    """format_prompt.

    Parameters
    ----------
    prompts :
        prompts
    memories :
        memories
    """
    next_prompt = (
        prompts["initial_prompt"].format(
            init_conditions_dict["goal"],
            init_conditions_dict["setup"],
            init_conditions_dict["thought"],
            init_conditions_dict["action"],
            init_conditions_dict["expected_obs"],
        )
        + "\n"
    )
    feedback_prompt = ""
    if len(memories) > 0:
        memory_prompt = prompts["memory_prompt"] + "\n"
        for mem in memories:
            memory_prompt += "<Thought>" + mem["thought"] + "</Thought>\n"
            memory_prompt += "<Action>" + mem["action"] + "</Action>\n"
            memory_prompt += "<Observation>" + \
                mem["observation"] + "</Observation>\n"
        next_prompt += memory_prompt + "\n"
        last_exp = memories[-1]
        feedback_prompt = prompts["feedback_prompt"].format(
            "<Human Feedback>" +
            last_exp["human_feedback"] + "</Human Feedback>",
            "<System Feedback>" +
            last_exp["system_feedback"] + "</System Feedback>",
        )
    next_prompt += (
        prompts["goal_prompt"].format(init_conditions_dict["goal"])
        + "\n"
        + feedback_prompt
        + "\n"
        + prompts["rules_prompt"]
    )
    return next_prompt

def process_output(llm_output):
    """process_output.

    Parameters
    ----------
    llm_output :
        llm_output
    """

    thought = llm_output.split("<Thought>")[1].split("</Thought>")[0]
    action = llm_output.split("<Action>")[1].split("</Action>")[0]
    expected_output = llm_output.split("<Expected Observation>")[1].split(
        "</Expected Observation>"
    )[0]
    return {"thought": thought, "action": action, "expected_output": expected_output}


def format_observations(observed_pH, observed_eV):
    """format_observations.

    Parameters
    ----------
    observed_pH :
        observed_pH
    observed_eV :
        observed_eV
    """
    assert len(observed_pH) == len(observed_eV)
    observation = ""
    for ii in range(len(observed_eV)):
        observation += "After running experiment {}, the pH was {} and the potential was {}. ".format(
            ii, observed_pH[ii], observed_eV[ii]
        )
    return observation[:-1]


def dict2prompt(plan_dict):
    """dict2prompt.

    Parameters
    ----------
    plan_dict :
        plan_dict
    """
    return """<Thought>{0}</Thought>
<Action>{1}</Action>
<Expected Observation>{2}</Expected Observation>""".format(
        plan_dict["thought"], plan_dict["action"], plan_dict["expected_output"]
    )


def rationalize_observation(plan_dict, observations, prompts):
    """rationalize_observation.

    Parameters
    ----------
    plan_dict :
        plan_dict
    observations :
        observations
    prompts :
        prompts
    """
    rationalize_prompt = prompts["rationalize_experiment_prompt"].format(
        dict2prompt(plan_dict), observations
    )
    can_rationalize = False
    if DEBUG_MODE:
        rationalization = "<YES>"
    else:
        rationalization = prompt_llm(rationalize_prompt)
    if "<YES>" in rationalization:
        can_rationalize = True
    return can_rationalize


def ping_human():
    """ping_human."""
    human_feedback = ""
    while human_feedback == "":
        human_feedback = input("")
    return human_feedback


def run_clairify(plan, system_constraints):
    """run_clarify."""
    _, xdl, errors = generate_xdl(
        plan, system_constraints["hardware"], system_constraints["reagents"]
    )
    return xdl

def start_up(prompts):
    """start_up."""
    human_input = input(
        "Hello, my name is ORGANA. I am a robot chemist. Is there an experiment I can help you perform today? "
    )
    start_up_prompt = prompts["start_up_prompt"].split("\n")
    print("\n".join(start_up_prompt))
    timeout = 0
    question = ""
    init_conditions_satisfied = 0
    while timeout < 10:
        output = prompt_llm(
            "Question: " + question + "\n" + "Answer: " + human_input,
            system_prompt="\n".join(start_up_prompt),
        )
        print("output", output)
        output = output.split("\n")
        question = ""
        for line in output:
            if "[Item" in line:
                item_num = line.split("[")[1].split("]")[0]
                for ii in range(len(start_up_prompt)):
                    if (
                        "[Item" in start_up_prompt[ii]
                        and item_num == start_up_prompt[ii].split("[")[1].split("]")[0]
                    ):
                        if (
                            len(start_up_prompt[ii].split(":::")[1]) == 0
                            and len(human_input) > 0
                        ):
                            init_conditions_satisfied += 1
                            print(
                                "another init_conditions_satisfied",
                                init_conditions_satisfied,
                            )
                        start_up_prompt[ii] += human_input
            if "Question" in line:
                question = line.split("]")[1]
        print("*********")
        print("\n".join(start_up_prompt))
        if init_conditions_satisfied == 6:
            print("Ok, thank you! I will start the experiment and let you know if I have any questions.")
            print("returning start up prompt.......",
                  "\n".join(start_up_prompt))
            return start_up_prompt
        if question == "":
            question = prompt_llm(
                "What question do you want to ask next?",
                system_prompt="\n".join(start_up_prompt),
            )
        human_input = input(question + " ")
        timeout += 1
    return -1

def process_init_conditions(start_up_prompt):
    """process_init_conditions.

    Parameters
    ----------
    start_up_prompt :
        start_up_prompt
    """
    init_conditions_dict = {"num_repeats": 1}
    for line in start_up_prompt:
        if "[Item 1]" in line:
            init_conditions_dict["goal"] = line.split(":::")[1]
        elif "[Item 2]" in line:
            init_conditions_dict["setup"] = line.split(":::")[1]
        elif "[Item 3a]" in line:
            init_conditions_dict["thought"] = line.split(":::")[1]
        elif "[Item 3b]" in line:
            init_conditions_dict["action"] = line.split(":::")[1]
        elif "[Item 3c]" in line:
            init_conditions_dict["expected_obs"] = line.split(":::")[1]
        elif "[Item 4]" in line:
            init_conditions_dict["num_repeats"] = int(line.split(":::")[1])
    return init_conditions_dict


class NLPServer:
    def __init__(self):
        rospy.init_node('my_service_server')
        service = rospy.Service('nlp_service', GetExpPlan, self.handle_request)
        rospy.Subscriber('/robot_exp_status', String, self.task_manage_callback)
        rospy.loginfo("Service server is ready to receive requests.")

        # initialization
        self.prompts = utils.load_prompts()
        if DEBUG_MODE:
            start_up_prompt = self.prompts["sample_start_up"].split("\n")
        else:
            start_up_prompt = start_up(self.prompts)
        self.init_conditions_dict = process_init_conditions(start_up_prompt)
        self.step = 0
        self.total_step = self.init_conditions_dict["num_repeats"]
        self.constraints_dict = utils.load_constraints()
        self.isExpCompleted = False
        self.isPlanGenerated = False

    def generate_xdl(self, plan_dict, output_xdl_path):
        if DEBUG_MODE and f"sample_xdl_step_{self.step}" in self.prompts:
            xdl_plan = self.prompts[f"sample_xdl_step_{self.step}"]
        else:
            xdl_plan = run_clairify(plan_dict['action'], self.constraints_dict) ## CALL CLAIRIFY 

        print("XDL PLAN..................", xdl_plan)
        with open(output_xdl_path, 'w') as f:
            knowledge_base = f.write(xdl_plan)

    def generate_plans(self, output_xdl_path):
        memories = []
        if self.step < self.total_step:
            next_prompt = format_prompt(self.prompts, memories, self.init_conditions_dict)
            print(f"step {self.step} PROMPT TO LLM:::", next_prompt)
            if self.step == 0:
                processed_output = process_output(next_prompt)
            else:
                if DEBUG_MODE and f"sample_plan_step_{self.step}" in self.prompts:
                    llm_output = self.prompts[f"sample_plan_step_{self.step}"]
                else:
                    llm_output = prompt_llm(next_prompt)
                print("llm_output for next plan.....", llm_output)
                processed_output = process_output(llm_output)
            print("EXPERIMENT PLAN........", processed_output)
            
            # XDL generation
            self.generate_xdl(processed_output, output_xdl_path)
            self.isPlanGenerated = True
            self.isExpCompleted = False  
            self.step += 1 

    def handle_request(self, req):    
        rospy.loginfo("Request to generate experiment plan received !")
        output_xdl_path = req.output_xdl_path
        self.generate_plans(output_xdl_path)
        return GetExpPlanResponse(self.isPlanGenerated, self.isExpCompleted)
    
    def task_manage_callback(self, data):
        rospy.loginfo("Received iteration completion from task manager: %s", data.data)
    

if __name__ == "__main__":
    nlp_server = NLPServer()
    rospy.spin()
