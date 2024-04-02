import json

#df=open('datasplits.json','r')
df=open('datasplits_60000.json','r')
str=df.read()
js_data = json.loads(str)

print(len(js_data['test']))
print(len(js_data['train']))
print(len(js_data['validation']))
