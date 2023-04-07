from flask import Flask, request, render_template
from chatbot import get_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.form['user_input']
    bot_response = get_response(user_input)
    return str(bot_response)

if __name__ == '__main__':
    app.run(debug=True)