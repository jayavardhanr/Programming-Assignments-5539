import json
import gzip
import pprint
with open('/Users/jayavardhanreddy/Github_Data_Files/test_json.json', 'r') as f:
    for line in f:
        while True:
            try:
                jfile=json.loads(line)
                break
            except ValueError:
                line+=next(f)
    print(json.dumps(jfile, indent=4))