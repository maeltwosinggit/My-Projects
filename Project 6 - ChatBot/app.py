from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client with the API key from the environment variable
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message")
    conversation_history = data.get("history", [])

    if user_message:
        # Add the user message to the conversation history
        conversation_history.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            max_tokens=4096  # Adjust this value as needed
        )

        bot_message = response.choices[0].message.content
        # Add the bot message to the conversation history
        conversation_history.append({"role": "assistant", "content": bot_message})

        return jsonify({"message": bot_message, "history": conversation_history})
    return jsonify({"message": "No message provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)