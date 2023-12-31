import os
from random import randint, random

import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    ri = randint(0, 100)
    r = random()
    log_param('param1', ri)

    print(ri, r)
    log_metric('foo', r)
    log_metric('foo', r + 1)
    log_metric('foo', r + 2)

    path = "output"

    if not os.path.exists(path):
        os.makedirs(path)

    with open("%s/test.txt" % (path), 'w') as f:
        f.write("hello world")

    log_artifacts("%s" % path)
