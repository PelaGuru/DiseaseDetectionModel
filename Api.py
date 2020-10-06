import flask
from GetPrediction import getPrediction
from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/predict/', methods=['POST'])
# @app.route('/predict/<url>')
def home():
    req_data = request.get_json()
    url = req_data['url']

    # url = request.args.get('url')
    print(url)
    # url= txt.replace("123", "%2F")

    result = getPrediction(url)
    # return ("<h1> %s </>" % result)
    # print(result)
    return {
        "predicted_disease": result,
        "msg": "200OK"
    }


app.run()
