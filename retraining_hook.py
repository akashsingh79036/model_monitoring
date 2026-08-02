import subprocess
from flask import Flask, request
from dotenv import load_dotenv

# load the env variables
load_dotenv()

app = Flask(__name__)

# add home route
@app.route('/')
def home():
    return "Retraining Hook is running! \n"

@app.route('/alert', methods=['POST'])
def alert_hook():
    alertname = "ModelPredictionBias"
    data = request.json
    if data is not None:
        # Check if the data contains 'alerts' key
        if "alerts" in data:
            # check if the alert name is matched
            if data["groupLabels"].get("alertname") == alertname:
                subprocess.run(["C:/Program Files/Git/bin/bash.exe", "./push_to_github.sh"])
                return "Triggered", 200
            
    return "No action", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)