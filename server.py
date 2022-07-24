from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import os
from min_example import interface, load_model
app = Flask(__name__, static_folder="../client/dist", template_folder="../client")
port = int(os.getenv('PORT', 8080))
CORS(app)
config, model, smlstok, tdatstok, comstok = load_model()

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/test")
def test():
	return "Test Successful!"

@app.route("/handshake", methods=['GET'])
def handshake():
	return "success"

@app.route('/submit_payload', methods=['POST'])
def submit_payload():
    # get the parameters from the POST request(these would be required as parameters for your main()/Interface function)
    payload_json = request.get_json(force=True)
	# print(payload_json)
    # parse the payload_json to get your parameters:
    input_code = payload_json['input_code']
	# call your main()/Interface function
    output_explanation, output_dict, dict1 = interface(input_code, config, model, smlstok, tdatstok, comstok)
    # make sure that result is in dictionary format
    result = { "summary": output_explanation, "intermediate_output": output_dict, "input_dict":dict1 }
    return jsonify(result)

if __name__ == "__main__":

	app.run(host='0.0.0.0', port=int(port))
