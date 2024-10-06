#! /usr/bin/sh

url=http://192.168.1.11:8000/api/dify/receive
api_key="123456"

curl -X POST -sSL $url \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $api_key" \
     -d '
{
    "point": "app.external_data_tool.query",
    "params": {
        "app_id": "61248ab4-1125-45be-ae32-0ce91334d021",
        "tool_variable": "weather_retrieve",
        "inputs": {
            "location": "London"
        },
        "query": "How'"'"'s the weather today?"
    }
}
' 

