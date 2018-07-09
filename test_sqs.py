import urllib

import json

def lambda_handler(event, context):

    for record in event['Records']:
        print(record['body'])
        task = json.loads(record['body'])[0]
        doTask(task)


def doTask(task):
    taskId = task['UID']
    sourceImageName = task['fileName']
    operation = task['operation']
    print("taskId=" + taskId)
    print("sourceImageName=" + sourceImageName)
    print("operation=" + operation)


