import os
import warnings
import json
import random
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

from models.utils import preprocess 
from models.ScienceBot import generate_answer

with open('intents.json', 'r') as file:
        data = json.load(file)
        
def match_intent(tokens, intent):
    words = set(tokens)
    patterns = intent['patterns']

    for pattern in patterns:
        pattern_words = preprocess(pattern)
        if words.intersection(pattern_words):
            return True
    return False

def generate_response(text):
    tokens = preprocess(text)

    for intent in data['intents']:
        if match_intent(tokens, intent):
                return intent['tag'], random.choice(intent['responses'])

    return None, None

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        request = text.lower()
        tag, response = generate_response(request)
        if not tag:
            response = generate_answer(request)
        output.append(response)

    return SimpleText(dict(text=output))
