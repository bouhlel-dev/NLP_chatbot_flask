# import required packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import json
import random
import tensorflow as tf
import numpy as np
from nltk_utils import tokenize, bag_of_words, generate_chatbot_response
from data_preprocess import vocabulary, tags

# open the intent file
with open('intent.json', 'r') as json_data:
    intents = json.load(json_data)

# load the saved chatbot brain
chatbot_brain = tf.keras.models.load_model('Barista.keras')
bot_name = "Ahmed Mohsen"

def get_response(sentence):
        # Preprocess the user request to make it understandable by the AI model
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, vocabulary= vocabulary)
    X = X.reshape(1,-1)

    # Predict the user intent
    pred_proba, intent_tag = generate_chatbot_response(chatbot_brain,X,tags)

    # if the chatbot is sure that it understood the user request
    if pred_proba > 0.75:
        for intent in intents['intents']:
            if intent_tag == intent['tag']:
                return random.choice(intent['responses'])
    # else
    else:
        return "Sorry I do not understand.. Could you make it simpler please !"






# Create the console chatbot app

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

