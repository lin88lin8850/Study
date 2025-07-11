import requests
import time
import threading
import random
TASK_SEND_INTERVAL = 0.5

server_url = "http://0.0.0.0:8000/"


def send_request(prompt):
    try:
        request = requests.get(server_url, json={"prompt": prompt})
        print(f"Request for prompt '{prompt}': {request.json()}")
    except Exception as e:
        print(f"Error sending request for prompt '{prompt}': {e}")

prompts = [
    "a b c d e",
    "f g h i j k l m",
    "n o p q r s t",
    "u v w x y z",
]


random.seed(42)
threads = []
for i in range(10):
    t = threading.Thread(target=send_request, args=(random.choice(prompts),))
    threads.append(t)

for t in threads:
    time.sleep(TASK_SEND_INTERVAL)
    t.start()

for t in threads:
    t.join()

print("All requests sent.")
