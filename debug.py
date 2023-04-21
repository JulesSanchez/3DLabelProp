from tqdm import tqdm
import time
for k in tqdm(range(100000)):
    for j in tqdm(range(500),leave=False):
        time.sleep(0.01)