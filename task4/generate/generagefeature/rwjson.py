import json
import sys
obj_num_file = sys.argv[1]
with open(obj_num_file) as f:
        data = json.load(f)
for key in data.keys():
    data[key] = int(data[key])

with open(sys.argv[2], 'w') as w:
        json.dump(data, w)

