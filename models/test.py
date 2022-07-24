import pickle
# import unicode
import json
import numpy as np
class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# data = pickle.load("/Users/xuyeliu/Downloads/layer_output (2).json")
with open('/Users/xuyeliu/Downloads/input.json', 'rb') as fp:
    data = pickle.load(fp)
    # print(data[2])
    print(type(data))
with open('input.json', 'w') as fp:
    fp.write(json.dumps(data, cls=NDArrayEncoder))
