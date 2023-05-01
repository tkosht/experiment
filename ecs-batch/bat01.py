# -*- coding: utf-8 -*-
import boto3
import sys
import datetime
import pytz

now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
endpoint_url = 'http://localstack:4566/'
s3 = boto3.resource("s3", region_name='ap-northeast-1',endpoint_url=endpoint_url, aws_access_key_id='dummy',aws_secret_access_key='dummy')
bucket = s3.Bucket('my-bucket')
src_object_body = bucket.put_object(Key='key01',Body= now.isoformat())

print (src_object_body)

