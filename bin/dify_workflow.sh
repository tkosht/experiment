#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

. ./.env

api_key=$DIFY_API_KEY_MARKDOWN_SPIDER
user="tkosht"


curl -X POST 'http://192.168.1.11/v1/workflows/run' \
--header 'Authorization: Bearer '$api_key \
--header 'Content-Type: application/json' \
--data-raw '{
  "inputs": {
    "file_or_url": {
      "transfer_method": "remote_url",
      "url": "https://www.fastretailing.com/jp/ir/financial/",
      "type": "document"
    }
  },
  "response_mode": "blocking",
  "user": "'$user'"
}'

