import flask
from GetPrediction import getPrediction
from flask import request
from flask_cors import CORS
import json

# initializing the Flask API
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# applying CORS policy to the API
CORS(app)


# POST request should be made with an image URL
# (in Postman -> select POST method -> Body -> raw -> JSON -> give the URL in JSON format )
# {
#     "url": "https://ag.umass.edu/sites/ag.umass.edu/files/fact-sheets/images/lateblightTom_lesion3.jpg"
# }

# endpoint to get the prediction
@app.route('/predict/', methods=['POST'])
def home():
    # getting the URL from the request body of the POST request
    req_data = request.get_json()
    url = req_data['url']

    # calling the method to get the prediction by passing URL as a parameter
    result = getPrediction(url)

    # returning the results as a JSON object
    return {
        "predicted_result": result['predicted_result'],
        "value": str(result['value']),
        "msg": "200OK"
    }


# running the Flask app
app.run()
