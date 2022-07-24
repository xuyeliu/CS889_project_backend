from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import os
import numpy as np
import json
from min_example import interface, load_model
app = Flask(__name__, static_url_path='', static_folder="static")
port = int(os.getenv('PORT', 8080))
CORS(app)

@app.route("/")
def index():
	return app.send_static_file("index.html")

@app.route("/test")
def test():
	return "Test Successful!"

@app.route("/handshake", methods=['GET'])
def handshake():
	return "success"

@app.route('/submit_payload', methods=['POST'])
def submit_payload():
    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """
        def default(self, obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32,
                                np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    config, model, smlstok, tdatstok, comstok = load_model()

    # get the parameters from the POST request(these would be required as parameters for your main()/Interface function)
    payload_json = request.get_json(force=True)
	# print(payload_json)
    # parse the payload_json to get your parameters:
    input_code = payload_json['input_code']
	# call your main()/Interface function
    output_explanation, output_dict, dict1 = interface(input_code, config, model, smlstok, tdatstok, comstok)
    # make sure that result is in dictionary format
    result = { "summary": output_explanation, "intermediate_output": output_dict, "input_dict":dict1 }
    return json.dumps(result, cls=NumpyEncoder)

if __name__ == "__main__":

	app.run(host='0.0.0.0', port=int(port))
