import json
from urllib.request import urlopen

url='http://ipinfo.io/json'
response=urlopen(url)
data=json.load(response)

print(data)
data.popitem()
data.pop('org')
new_data=str(data)
print(new_data)
