# fastscore.schema.0: double
# fastscore.schema.1: double

import random

# modelop.score
def action(datum):
    yield datum + random.random()
