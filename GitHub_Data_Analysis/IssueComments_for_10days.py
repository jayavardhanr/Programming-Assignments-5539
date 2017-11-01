import json
from pandas.io.json import json_normalize
import json

#IssueComments for 10 days
list_json=[]
line_number=0
with open('/Users/jayavardhanreddy/Github_Data_Files/IssueCommentEvent_Payload_2017-09-10_to_2017-09-19.json') as f:
    for line in f:
        while True:
            try:
                jfile = json.loads(line)
                list_json.append(jfile)
                #print(jfile)
                break
            except ValueError:
                # Not yet a complete JSON value
                line += next(f)
        line_number+=1

print('line_number',line_number)
print('list_json',len(list_json))
df_issues=json_normalize(list_json)
print('df_issues',len(df_issues))
df_issues.to_csv('IssueComments_from_2017-09-10_to_2017-09-19.csv',sep='\t', encoding='utf-8')
