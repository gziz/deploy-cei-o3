
import json
import numpy as np
import torch # Import torch to test with inputs
import ComponentModel as CM # Test the API of ComponentModel


def lambda_handler(event, context):

    try:

        body = json.loads(event['body'])
        context = body['context']
        componentes = len(context[0])
        horas = len(context)
        input = np.array(context).reshape(horas, 1, componentes)
        input = torch.tensor(input).float()

        modelReal = CM.Net(from_file=True,file_path="./models/modelApodaca4.2FixedD.pt")
        (h_t,c_t) = modelReal.ini_hidden(1)

        out = modelReal(input,h_t,c_t, future = 24,with_iteration=True)

        output = [list(x) for x in out.squeeze().detach().numpy().astype('float')]

        return {
            "statusCode": 200,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True

            },
            "body": json.dumps({'Preds': output})
        }

    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
