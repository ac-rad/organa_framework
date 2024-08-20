#!/usr/bin/env python

import json
import os
import sys
import tempfile
import types
import time
from copy import deepcopy
from pprint import pprint

import numpy as np
import openai
import rospy
import matplotlib.pyplot as plt
from electrochemistry.srv import (GoToViewPose, GoToViewPoseResponse,
                                  RobotExecute, RobotExecuteResponse)
from electrochemistry_perception.srv import getViews2Pose
from std_msgs.msg import Bool, String
from tqdm import tqdm
from playsound import playsound

from postprocessing_functions import electrochemistry

sys.path.append('/root/git/frankapy/catkin_ws/src/organa/')
sys.path.append('/root/git/frankapy/catkin_ws/src/organa/clairify/')
import io
import tkinter as tk
from tkinter import BOTH, YES
from datetime import datetime

import cv2
from dotenv import load_dotenv
from fpdf import FPDF
from PIL import Image, ImageTk

import utils
from clairify.xdlgenerator.nlp2xdl import generate_xdl

import speech_recognition as sr
r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    r.adjust_for_ambient_noise(source)  # we only need to calibrate once, before we start listening


DEBUG_MODE = False
AUDIO_MODE = True

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

def load_postprocessing_functions():
    """load_postprocessing_functions."""
    modules = [electrochemistry]
    funcname2func = {}
    for module in modules:
        for elem in dir(module):
            if type(getattr(module, elem)) == types.FunctionType:
                funcname2func[elem] = getattr(module, elem)
    return funcname2func

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

def format_prompt(prompts, memories, memory_summaries, init_conditions_dict):
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
    memory_prompt = ""
    if len(memories) > 0:
        memory_prompt = prompts["memory_prompt"] + "\n"

        if len(memory_summaries) > 0:  # if there are any summaries, add them
            for ii, mem_summ in enumerate(memory_summaries):
                if ii == 0:
                    mem_summ = (
                        "This is a summary of what happened during the first three experiments:\n"
                        + mem_summ
                    )
                else:
                    mem_summ = (
                        "This is a summary of what happened during the next three experiments:\n"
                        + mem_summ
                    )
                memory_prompt += mem_summ + "\n\n"

        start = len(memory_summaries) * 3  # memories not in summaries
        memory_prompt += "This is what happened during the most recent experiments:\n"
        for mem in memories[start:]:
            memory_prompt += "<Thought>" + mem["thought"] + "</Thought>\n"
            memory_prompt += "<Action>" + mem["action"] + "</Action>\n"
            memory_prompt += f"<Observation>{mem['observation']}</Observation>\n"
        next_prompt += memory_prompt + "\n\n"
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
    return next_prompt, memory_prompt

def process_output(llm_output):
    """process_output.

    Parameters
    ----------
    llm_output :
        llm_output
    """

    thought = llm_output.split("<Thought>")[1].split("</Thought>")[0]
    action = llm_output.split("<Action>")[1].split("</Action>")[0]
    expected_output = "Unsure what expeected output should be."
    if "<Expected Observation>" in llm_output:
        expected_output = llm_output.split("<Expected Observation>")[1].split(
            "</Expected Observation>"
        )[0]
    elif "<Expected Output>" in llm_output:
        expected_output = llm_output.split("<Expected Output>")[1].split(
            "</Expected Output>"
        )[0]
    return {"thought": thought, "action": action, "expected_output": expected_output}

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
        dict2prompt(plan_dict), observations,''
    )
    can_rationalize = False
    if DEBUG_MODE:
        rationalization = "<YES>"
    else:
        rationalization = prompt_llm(rationalize_prompt)
    if "<YES>" in rationalization:
        can_rationalize = True
    else:
        print("rationalization prompt", rationalize_prompt)
        print("rationalizing.........", rationalization)
    return can_rationalize, rationalization

def ping_human():
    """ping_human."""
    human_feedback = ""
    while human_feedback == "":
        human_feedback = input("")
    return human_feedback

def run_clairify(plan, system_constraints, experiment_type):
    """run_clarify."""
    _, xdl, errors = generate_xdl(
        plan,
        system_constraints[f"hardware_{experiment_type.lower()}"],
        system_constraints[f"reagents_{experiment_type.lower()}"],
    )
    return xdl

def start_up(prompts):
    text = ""
    prev_text = ""
    if AUDIO_MODE:
        prompt = "Hello, my name is ORGANA. I am a robot chemist. Can I help you with anything today? "
        wait = play_audio_file("audio/voice_1.mp3", block=True)
        # human_input = listen_to_human()
        human_input = "I would like to run a chemistry experimnet"
    else:
        human_input = input(
            "Hello, my name is ORGANA. I am a robot chemist. Can I help you with anything today? "
        )

    start_up_prompt = prompts["start_up_prompt_FULL"].split("\n")
    print("\n".join(start_up_prompt))
    timeout = 0
    question = ""
    init_conditions_satisfied = 0
    counter = 1
    i=0
    while timeout < 10:
        counter += 1
        output = prompt_llm(
            "Question: " + question + "\n" + "Answer: " + human_input,
            system_prompt="\n".join(start_up_prompt),
        )
     
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
            play_audio_file(f"audio/voice_{counter}.mp3")
            if AUDIO_MODE:
                pass
            print(
                "Ok, thank you! I will start the experiment and let you know if I have any questions."
            )
            print("returning start up prompt.......",
                  "\n".join(start_up_prompt))
            return start_up_prompt
        if question == "":
            question = "What is the weather today?"
            print("QUESTION:::::", question)
        if AUDIO_MODE:
            play_audio_file(f"audio/voice_{counter}.mp3")
            human_input = listen_to_human()
            i+=1
        else:
            human_input = input(question + " ")
        print("HUMAN INPUT", human_input)
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
            init_conditions_dict["num_repeats"] = int(line.split(":::")[1].replace('.', ''))
    return init_conditions_dict

def organa_speak(prompt):
   audio = generate(
       text=prompt,
       voice="Bella",
       model="eleven_multilingual_v2"
   )
   play(audio)

def play_audio_file(path, block=True):
    """play_audio_file.

    Parameters
    ----------
    path :
        path
    """
    playsound(path, block=block)
    # duration = librosa.get_duration(filename=path)
    return True

def listen_to_human():
    audio_file = f'test_{int(time.time())}.wav'
    with sr.Microphone() as m:
        print("listning.............")
        audio = r.listen(m, timeout=10)
    # with open(audio_file, 'wb') as file:
    #     wav_data = audio.get_wav_data()
    #     file.write(wav_data)
    print("done listening")
    # print('Saved audio to, ', audio_file)

    text = r.recognize_whisper_api(audio, api_key=openai.api_key)
    print(text)
    return text

def follow(thefile):
    """follow.

    Parameters
    ----------
    thefile :
        thefile
    """
    thefile.seek(0, os.SEEK_END)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line

def clean_text(text):
    text = text.replace("\x00", "").replace("\x1b[2K", "")
    question_end = text.rfind("?")
    if question_end != -1:
        text = text[question_end + 1:]
    pattern = r"\[.*?\]"
    text = re.sub(pattern, "", text)
    pattern = r"\(.*?\)"
    text = re.sub(pattern, "", text)
    text = " ".join(text.split())
    return text

def resize_image(event):
    new_width = event.width
    new_height = event.height
    image = copy_of_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    label.config(image = photo)
    label.image = photo #avoid garbage collection

def ground_perception(img_path, knowledgebase_path, prompts):   
    # AUDIO_MODE =  False
    knowledgebase = utils.load_knowledgebase(knowledgebase_path)
    knowledgebase_updated = deepcopy(knowledgebase)
    object2desc = {}
    img = cv2.imread(img_path)
    colours = [(250, 218, 221), (158,161,212)]
    questions = ["What will this object be used for?"]#, "What is the max volume of this container (in mL)?"]
    questions_audio = ["audio/name_vessel.mp3", "audio/name_volume.mp3"]
    idx2key = {0: "user_descr"}#, 1: "max_vol"}
    target_names = ["wash_beaker", "experiment_beaker", "ph_beaker", "scale", "polish_plate"]

    obj_dictionary=knowledgebase['objects']
    for object_ in obj_dictionary:
        if obj_dictionary[object_]['object_type'] != "beaker" and obj_dictionary[object_]['object_type'] != "polishing_pad": continue
        bbox = obj_dictionary[object_]['bbox_xyxy']
        object2desc[object_] = {}
        for ii, question in enumerate(questions):
            image_w_box = utils.draw_rectangle(img, bbox, is_opaque=True, bbox_color=colours[ii])
            image_w_box = utils.add_T_label(image_w_box, label=question, bbox=bbox, thickness=1, text_bg_color=colours[ii])
            window = tk.Tk()
            window.title("ENTER hardware information")
            window.configure(background='grey')
            
            img_tk = ImageTk.PhotoImage(Image.fromarray(image_w_box))
            panel = tk.Label(window, image = img_tk)
            
            def retrieve_input(event=None):
                if AUDIO_MODE:
                    wait = play_audio_file(questions_audio[ii], block=True)
                    inputValue = listen_to_human()
                    print(inputValue)
                else:
                    inputValue = textBox.get("1.0", "end-1c")
                object2desc[object_][idx2key[ii]] = inputValue
                window.destroy()

            def center_window(width=300, height=200):
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()
                x = (screen_width/2) - (width/2)
                y = (screen_height/2) - (height/2)
                window.geometry('%dx%d+%d+%d' % (width, height, x, y))

            textBox=tk.Text(window, height=3, width=30)
            textBox.pack()

            buttonCommit=tk.Button(window, height=1, width=10, text="Enter",
                                command=lambda: retrieve_input())
            buttonCommit.pack()
            panel.pack()
            
            center_window(1300, 800)
            if AUDIO_MODE:
                window.after(1000, buttonCommit.invoke)

            window.mainloop()
    play_audio_file(f"audio/voice_8.mp3")
    # LLM Matchings
    user_names = [object2desc[object_]['user_descr'] for object_ in object2desc if 'user_descr' in object2desc[object_]]
    matching = prompt_llm(prompts['hardware_grounding'].format(target_names, user_names))
    matching = matching[matching.index("[start matching]") + 17:matching.index("[end matching]")].split('\n')
    user2targetname = {}
    for item in matching:
        if item == "": continue
        k, v = item.split("|")
        user2targetname[k] = v
    
    for object_ in obj_dictionary:
        if object_ in object2desc and "user_descr" in object2desc[object_]:
            user_descr = object2desc[object_]["user_descr"]
            name = user2targetname[object2desc[object_]["user_descr"]]
            # max_vol = prompt_llm(prompts['volume_prompt'].format(object2desc[object_]['max_vol']))
            # assert "<UNIT ERROR>" not in max_vol
            # max_vol = max_vol[max_vol.index("[start volume]")+15:max_vol.index("\n[end volume]")]
        else:
            del knowledgebase_updated["objects"][object_]
            continue
        
        knowledgebase_updated["objects"][name] = knowledgebase_updated["objects"][object_]
        knowledgebase_updated["objects"][name]['user_descr'] = user_descr
        if name in ["wash_beaker", "experiment_beaker", "ph_beaker"]:
            # knowledgebase_updated["objects"][name]['max_vol'] = int(max_vol.split(" ")[0])
            knowledgebase_updated["objects"][name]['content'] = []
        del knowledgebase_updated["objects"][object_]

    # update knowledge base
    with open(knowledgebase_path,'w+') as f:
        json.dump(knowledgebase_updated, f)
    return True


class TaskManager:
    def __init__(self, kb_path, log_path):
        rospy.init_node('task_manager')
        self.robot_execute_pub = rospy.Publisher('/robot_execute', String, queue_size=20)
        self.robot_execute_status_sub = rospy.Subscriber("/robot_execute_status", Bool, self.robot_execute_callback)
        self.viewpose_execute_status_sub = rospy.Subscriber("/viewpose_execute_status", Bool, self.viewpose_execute_callback)
        self.viewpose_execute_pub = rospy.Publisher("/viewpose_execute", Bool, queue_size = 10)

        # initialization
        self.isAtViewPose=False
        self.isRobotCompleted = False
        self.memories = []
        self.memory_summaries = []
        self.processed_output=None
        self.kb_path = kb_path
        self.log_path = log_path
        self.xdl_path = None
        self.prompts = utils.load_prompts()

        if DEBUG_MODE:
            start_up_prompt = self.prompts["start_up_pourbaix"].split("\n")
        else:
            start_up_prompt = start_up(self.prompts)
        
        self.init_conditions_dict = process_init_conditions(start_up_prompt)
        funcname2func = load_postprocessing_functions()
        self.experiment_type = prompt_llm(
            self.prompts["experiment_type"].format(self.init_conditions_dict["goal"])
        )
        self.postprocessing_function = funcname2func[f"postprocessing_{self.experiment_type.lower()}"]
        self.step = 0
        self.total_step = self.init_conditions_dict["num_repeats"]
        self.constraints_dict = utils.load_constraints(self.experiment_type)
        if DEBUG_MODE:
            self.observations = [{'pH': 3.81, 'eV': -0.431}, {'pH': 4.80, 'eV': -0.469}]
        else:
            self.observations = []
        self.summary_pdf = self.load_log()
        self.pdf = self.load_log()

    def viewpose_execute_callback(self, data):
        if data.data:
            self.isAtViewPose = True
    
    def robot_execute_callback(self, data):
        if data.data:
            self.isRobotCompleted = True
            print("Robot complete execution!")

    def call_robot_execute(self):  # robot execution code with pddlstream
        rospy.loginfo("Call robot_execution successful")
        robot_msg = f'{self.kb_path}!{self.xdl_path}'
        self.robot_execute_pub.publish(robot_msg)
    
    def call_goto_viewpose_service(self):
        rospy.loginfo("Call view_pose_service successful")
        self.viewpose_execute_pub.publish(True)
        
    def call_perception_service(self):
        rospy.wait_for_service("/perception_server")
        service_proxy = rospy.ServiceProxy("/perception_server", getViews2Pose)
        perception_msg = f'{self.kb_path}!{self.log_path}'
        try:
            response = service_proxy(perception_msg)
            rospy.loginfo("Call perception_server successful")
            return True
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed:")

    def format_experiment(self, processed_output, step):
        """format_experiment.
        Parameters
        ----------
        processed_output :
            processed_output
        step :
            step
        """
        text = ""
        key2preamble = {
            "thought": "This was the rationalization behind the experiment:",
            "action": "This was the experiment protocol that was done:",
            "expected_output": "This was the expected output from the experiment:",
            "observation": "This was the actual output from the experiment:",
            "system_feedback": "This was the system feedback:",
            "human_feedback": "A human was asked to intervene in the experiment, and this was their feedback:",
        }
        for key in [
            "thought",
            "action",
            "expected_output",
            "observation",
            "system_feedback",
            "human_feedback",
        ]:
            if processed_output[key] not in [None, "", "None", "none"]:
                text += f'{key2preamble[key]}\n'
                text += f'{processed_output[key]}\n\n'
        return text

    def format_observations(self, observation):
        """format_observations.

        Parameters
        ----------
        observed_pH :
            observed_pH
        observed_eV :
            observed_eV
        """
        observed_pH = [observation["pH"]]
        observed_eV = [observation["eV"]]
        assert len(observed_pH) == len(observed_eV)
        observation = ""
        for ii in range(len(observed_eV)):
            observation += "After running the experiment, the pH was {} and the potential was {}. ".format(
                observed_pH[ii], observed_eV[ii]
            )
        return observation[:-1]

    def generate_xdl(self, plan_dict, output_xdl_path):
        if DEBUG_MODE and f"sample_xdl_step_{self.step}" in self.prompts:
            xdl_plan = self.prompts[f"sample_xdl_step_{self.step}"]
        else:
            xdl_plan = run_clairify(plan_dict['action'], self.constraints_dict, self.experiment_type) ## CALL CLAIRIFY 

        print("XDL PLAN..................", xdl_plan)
        with open(output_xdl_path, 'w') as f: xdl_file = f.write(xdl_plan)
        return True 

    def generate_plans(self):
        next_prompt, memory_prompt = format_prompt(self.prompts, self.memories, self.memory_summaries, self.init_conditions_dict)
        print(f"step {self.step} PROMPT TO LLM:::", next_prompt)
        if self.step == 0:
            self.processed_output = process_output(next_prompt)
        else:
            if DEBUG_MODE and f"sample_plan_step_{self.step}" in self.prompts:   # CHANGE THIS CONDITION
                llm_output = self.prompts[f"sample_plan_step_{self.step}"]
            else:
                print("...call llm at the end of experiment...")
                llm_output = prompt_llm(next_prompt)
            print("llm_output for next plan.....", llm_output)
            self.processed_output = process_output(llm_output)
        
        print(".......EXPERIMENT PLAN........")
        # XDL generation
        self.xdl_path = f'{self.log_path}/plan_{self.step+1}.xdl'
        isXDLGenerated = self.generate_xdl(self.processed_output, self.xdl_path) 
        return isXDLGenerated
 
    def load_log(self):
        """load_log."""
        # Initialize PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        return pdf

    def gen_summary(self):
        _, memory_prompt = format_prompt(
            self.prompts, self.memories, self.memory_summaries, self.init_conditions_dict
        )
        final_summary = prompt_llm(self.prompts["final_summary"].format(memory_prompt))

        self.summary_pdf.set_x(10)
        self.summary_pdf.set_font("Helvetica", size=11)
        self.summary_pdf.multi_cell(150, 5, final_summary)

        global TEMP_IMG_PATH
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_img:
            data = self.postprocessing_function(self.observations, save_path=temp_img.name)
            # data = self.postprocessing_function(observations, save_path=TEMP_IMG_PATH)
            print("FINAL RESULTS................")
            pprint(data)

            self.summary_pdf.ln(10)  # New line
            self.summary_pdf.multi_cell(
                150, 5, "After performing the experiments, these are the results:\n"
            )
            self.summary_pdf.multi_cell(150, 5, data["text"])
            self.summary_pdf.ln(10)  # New line
            # self.summary_pdf.image(TEMP_IMG_PATH, w=150)
            self.summary_pdf.set_x(10)
            self.summary_pdf.image(temp_img.name, w=150)
            self.summary_pdf.ln(10)  # New line

    def save_report(self, save_name='experiment_logfile'):
        buffer1 = io.BytesIO()
        self.pdf.output(buffer1)

        buffer2 = io.BytesIO()
        self.summary_pdf.output(buffer2)

        now = datetime.now()
        formatted_datetime = now.strftime("%Y%m%d%H%M%S")
        save_path = f"{save_name}_{formatted_datetime}.pdf"
        print("saving to........", save_path)
        utils.combine_fpdfs(buffer2, buffer1, save_path)

    def nlp_postprocess(self):
        def write_to_pdf():
            exp_text = (
                f"....................Log for experiment {self.step+1}....................\n\n"
            )
            self.pdf.set_x(10)
            self.pdf.set_font("Helvetica", size=12, style="B")
            self.pdf.ln(2)  # New line
            self.pdf.multi_cell(150, 5, exp_text)
            self.pdf.ln(2)  # New line

            exp_text = self.format_experiment(self.processed_output, self.step)
            self.pdf.set_x(10)
            self.pdf.set_font("Helvetica", size=11)
            self.pdf.multi_cell(150, 5, exp_text)

            # save image to log_file
            # global TEMP_IMG_PATH
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_img:
                data = self.postprocessing_function(
                    self.observations, save_path=temp_img.name)
                self.pdf.multi_cell(
                    150, 5, "After performing the experiments, these are the results:\n"
                )
                self.pdf.multi_cell(150, 5, data["text"])
                self.pdf.set_x(10)
                self.pdf.image(temp_img.name, w=150)
                print("RESULTS................")
                pprint(data)
            self.pdf.ln(10)  # New line
            print("******")

        if DEBUG_MODE:
            observed_pH = 3.31     # for testing only
            observed_eV = -0.38
            observation = {'pH': observed_pH, 'eV': observed_eV}
            system_feedback = ""
            human_feedback = ""
        else:
            knowledgebase = utils.load_knowledgebase(self.kb_path)    # load observation from kb
            observation = knowledgebase["observation"]
            system_feedback = ""
            human_feedback = "" 

        can_rationalize, rationalization = rationalize_observation(self.processed_output, observation, self.prompts)
        human_feedback = ""
        if not can_rationalize:
            print("pinging human......")
            human_feedback = input(
                "Observations don't make sense::: {}. Any feedback on the experiment? ".format(
                    rationalization
                )
            )
            
        self.processed_output["observation"] = observation
        self.processed_output["system_feedback"] = system_feedback
        self.processed_output["human_feedback"] = human_feedback
        self.observations = observation
        print("SUMMARY OF CURRENT EXPERIMENT......", self.processed_output)
        self.memories.append(self.processed_output)
        write_to_pdf()
        return True


if __name__ == "__main__":
    # log path
    log_path = "/root/git/frankapy/catkin_ws/src/organa/exp_log"
    kb_path = f"{log_path}/knowledge_base_pddlstream.json"
    perception_output_img_path=f"{log_path}/result.jpg"

    # state variables
    isPlanGenerated = False
    isXDLReady = False
    isNLPPostProcessDone = True
    isPerceptionDone = False
    isGroundPerceptionDone = False

    # initialize the server
    start = time.time()
    task_manager = TaskManager(kb_path, log_path)
    print("!!!!!!execution time for organa:", time.time() - start)
    start = time.time()
    task_manager.call_goto_viewpose_service()
    task_manager.isAtViewPose = True
    while True:
        if task_manager.isAtViewPose:
            task_manager.isAtViewPose = False
            isPerceptionDone = task_manager.call_perception_service()
            print("!!!!!!execution time for perception:", time.time() - start)
            start = time.time()
            #isPerceptionDone = True
        elif isPerceptionDone == True and isGroundPerceptionDone == False:
            isGroundPerceptionDone = ground_perception(img_path=perception_output_img_path, knowledgebase_path=task_manager.kb_path, prompts=task_manager.prompts)
            print("!!!!!!execution time for grounding perception:", time.time() - start)
            start = time.time()
            #isGroundPerceptionDone = True
        elif task_manager.isRobotCompleted:
            print("!!!!!!execution time for robot and hardware execution:", time.time() - start)
            start = time.time()
            task_manager.isRobotCompleted = False
            isNLPPostProcessDone = task_manager.nlp_postprocess()
            task_manager.step += 1
            print("!!!!!!execution time for NLP postprocessing:", time.time() - start)
            start = time.time()
            print("!!!!!!execution time for the iteration:", time.time())
        elif isXDLReady:
            isXDLReady = False
            task_manager.call_robot_execute()
            
        elif isPerceptionDone == True and isGroundPerceptionDone == True and isNLPPostProcessDone == True and task_manager.step < task_manager.total_step:
            isNLPPostProcessDone = False
            isXDLReady = task_manager.generate_plans()
            print("!!!!!!execution time for XDL generation:", time.time() - start)
            start = time.time()
        elif task_manager.step == task_manager.total_step:
            break

    
    print("......experiment finished!.......")
    task_manager.gen_summary()
    task_manager.save_report()
    print("!!!!!!execution time for report generation:", time.time() - start)
    print("!!!!!!execution time for whole experiment:", time.time())
    rospy.spin()
