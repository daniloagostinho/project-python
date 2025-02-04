from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS  # Import the CORS extension

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# If you want to restrict CORS to a specific origin, you can specify it:
# CORS(app, resources={r"/chat": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/*": {"origins": ["*"]}})

# Load pre-trained model and tokenizer
# model_name = "microsoft/DialoGPT-medium"
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Dictionary to hold conversation history per user (if needed)
user_histories = {}

@app.route('/')
def home():
    return 'API is running!'
    
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    user_id = request.json.get('user_id', 'default')

    # Get the conversation history for the user
    chat_history_ids = user_histories.get(user_id)

    # Encode the new user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input to the chat history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_p=0.95,
        top_k=60,
        temperature=0.7,
    )

    # Update the user's conversation history
    user_histories[user_id] = chat_history_ids

    # Decode the response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return jsonify({'reply': response})

if __name__ == '__main__':
    app.run()
