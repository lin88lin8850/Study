import sys
import time
import asyncio
import uvicorn

from fastapi import FastAPI, Request



BATCH_INFERENCE_TIME = 0.5
CONTEXT_SWITCH_AWAIT_TIME = 0.001


class Response:
    def __init__(self):
        self.gen_tokens = []
        self.steps =  []
        self.finished = False


class AsyncLLMEngine:
    def __init__(self):
        self.llm = None
        self.requests = {}
        self.responses = {}
        self.running = False
        self.step_cnt = 0
        self.events = {}
        self.gen_prompt_id = self.gen_id()
        
    def gen_id(self):
        cur_id = 0
        while True:
            yield cur_id
            cur_id += 1

    def add_new_request(self, prompt):
        prompt_id = next(self.gen_prompt_id)
        prompt_segments = prompt.split(" ")
        self.requests[prompt_id] = [0,prompt_segments]
        self.responses[prompt_id] = Response()
        event = asyncio.Event()
        self.events[prompt_id] = event
        return prompt_id, event

    def print_step(self): 
        lines = []
        for prompt_id, response in self.responses.items():
            if len(response.gen_tokens) == 0:
                lines.append('')
            else:
                line = [' ' for _ in range(max(index for index in response.steps)+1) ]
                for token, step in zip(response.gen_tokens, response.steps):
                    line[step] = token
                line.insert(0, f"resp_{prompt_id} : ")
                if response.finished:
                    line.append("---------------Done")
                lines.append(' '.join(line))
        sys.stdout.flush()
        print("\n".join(lines))
        
    
    def step_batch(self):
        time.sleep(BATCH_INFERENCE_TIME) # simulate inference time
        finished_ids = []
        for prompt_id, prompt_info in self.requests.items():
            prompt_index, prompt_segments = prompt_info
            if prompt_index >= len(prompt_segments):
                finished_ids.append(prompt_id)
                continue
            gen_token = prompt_segments[prompt_index].upper()
            response = self.responses[prompt_id]
            response.gen_tokens.append(gen_token)
            response.steps.append(self.step_cnt)
            prompt_info[0] += 1
        return finished_ids
    
    async def step(self):
        self.running = True
        await asyncio.sleep(CONTEXT_SWITCH_AWAIT_TIME)
        finished_ids = self.step_batch()
        self.step_cnt += 1
        for prompt_id in self.requests.keys():
            # asyncio.Event 用于不同协程之间的同步
            # 某些协程可能在等待 event.wait()，只有当事件被 set 之后，这些协程才会被唤醒继续执行。
            # 当一次 batch 推理（step）完成后，通知所有等待该 prompt_id 结果的协程“你可以继续了”。
            self.events[prompt_id].set() # 将事件对象状态设置为“已触发”（set），从而唤醒所有等待这个事件的协程。
        for finished_id in finished_ids:
            self.requests.pop(finished_id)
            self.responses[finished_id].finished = True
        self.running = False
        self.print_step()


    async def generate_async(self, prompt):
        prompt_id, event = self.add_new_request(prompt)
        while not self.responses[prompt_id].finished:
            if not self.running:
                await self.step()
            else:
                try:
                    await asyncio.wait_for(event.wait(), timeout=1)
                except:
                    continue
                event.clear()
        response = " ".join(token for token in self.responses[prompt_id].gen_tokens)
        # self.responses.pop(prompt_id)
        self.events.pop(prompt_id)
        # print("before return : ", response)
        return response

app = FastAPI()
engine = AsyncLLMEngine()

@app.get("/")
async def generate(request: Request):
    request_dict = await request.json()
    prompt = request_dict.get("prompt", "")

    response = await engine.generate_async(prompt)

    return {"message": response}

if __name__ == "__main__":
    

    uvicorn.run("async_llm_server:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
