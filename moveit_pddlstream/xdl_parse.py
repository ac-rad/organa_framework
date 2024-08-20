from cmath import exp
import xml.etree.ElementTree as ET
import json
import argparse


class XDLParser:
    def __init__(self, xdl_filename, kb_filename):
        self.xdl_filename = xdl_filename
        self.kb_filename = kb_filename
        self.goals = []
        self.knowledge_base = {}

    def parse_procedure(self, root):
        for step in root:
            if step.tag == 'Add':
                vessel = step.attrib['vessel']
                reagent = step.attrib['reagent'].split(' solution')[0].replace(" ", "")
                if reagent == "NaCl": reagent="nacl"
                volume = step.attrib['volume'].split(" ")[0]
                self.goals.append(('pump', reagent, vessel, volume))
            elif step.tag == 'Stir':
                self.goals.append(['stir_reagent'])
            elif step.tag == 'RunCV':
                self.goals.append(['RunCV'])
            elif step.tag == 'Monitor':
                vessel = step.attrib['vessel']
                quantity = step.attrib['quantity']
                self.goals.append(('monitor', vessel, quantity))
            elif step.tag == "Transfer":
                from_vessel = step.attrib['from_vessel']
                to_vessel = step.attrib['to_vessel']
                self.goals.append(('flush', from_vessel, to_vessel))
        self.knowledge_base['goals'] = self.goals
        if "observation" not in self.knowledge_base:
            self.knowledge_base['observation'] = []

    def parse_hardware(self, root):
        for component in root.iter('Component'):
            name = component.attrib['id'].replace(' ', '')
            if name not in self.knowledge_base['objects'].keys():
                if 'object_type' in component:
                    vessel_type = component.attrib['type']
                else:
                    vessel_type = 'glass'
                self.knowledge_base['objects'][name] = {
                    'object_type': vessel_type,
                    'content': [],
                    "pose": None,
                }

    def parse_reagents(self, root):
        reagent_dict={}
        for reagent in root.iter('Reagent'):
            reagent = reagent.attrib['name']
            name = reagent.split(' solution')[0]
            name = name.replace(" ", "_")
            if name == "NaCl": name="nacl"
            reagent_dict[name] = {
                'type': "reagent",
                'content': [name],
                "volume": None,
            }
        self.knowledge_base['reagents']=reagent_dict

    def parse_synthesis(self, root):
        for hardware in root.iter('Hardware'):
            self.parse_hardware(hardware)
        for reagents in root.iter('Reagents'):
            self.parse_reagents(reagents)
        for procedure in root.iter('Procedure'):
            self.parse_procedure(procedure)

    def load_knowledge_base(self):
        with open(self.kb_filename, 'r') as kb:
            self.knowledge_base = json.load(kb) 

    def update_knowledge_base(self):
        with open(self.kb_filename, 'w') as outfile:
            json.dump(self.knowledge_base, outfile, indent=4)

    def parse(self):
        self.load_knowledge_base()
        tree = ET.parse(self.xdl_filename)
        root = tree.getroot()
        
        if root.tag == 'Synthesis':
            self.parse_synthesis(root)
        else:
            for child in root:
                if child.tag == 'Synthesis':
                    self.parse_synthesis(child)

        self.update_knowledge_base()


if __name__ == '__main__':
    # testing 
    xdl_parser = XDLParser('/root/git/frankapy/catkin_ws/src/organa/output.xdl', 'kb_perception.json')
    xdl_parser.parse()
    