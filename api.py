from flask import Flask, request
import logging

app = Flask(__name__)
logging.basicConfig(filename='api_log.txt', level=logging.INFO)


@app.route('/')
def hello():
    return 'Hello, face!'

@app.route('/api/face_data', methods=['POST'])
def handle_post():
    data = request.get_json()
    logging.info(f"Received data: {data}")
    return 'Data received and logged successfully!'

if __name__ == '__main__':
    app.run()