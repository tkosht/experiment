#! /usr/bin/sh

url=http://192.168.1.11:8000/api/dify/receive
api_key="123456"

curl -X POST -sSL $url \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $api_key" \
     -d '
{
    "point": "ping"
}
' 

