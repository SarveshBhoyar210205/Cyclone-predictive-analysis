from twilio.rest import Client
from location_creator import new_data

account_sid = 'AC9099abdae31954c928738815d55d0008'
auth_token = 'a8138a6a27b115707a15fec26813fd69'
client = Client(account_sid, auth_token)

sms="SOS"
message=sms+new_data

message = client.messages.create(
  from_='+19382018409',
  body=new_data,
  to='+919930400475'
)

print(message.sid)