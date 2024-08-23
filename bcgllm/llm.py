import time
from abc import abstractmethod, ABC
from datetime import datetime

import requests
from dashscope import get_tokenizer
from openai import OpenAI

tokenizer = get_tokenizer('qwen-turbo')


def get_token_num(s):
    tokens = tokenizer.encode(s)
    return len(tokens)


class LLMClient(ABC):

    @abstractmethod
    def request(self, request_text) -> str:
        pass


class MyClient(LLMClient):
    def __init__(self, url: str, authorization: str):
        self.url = url
        self.authorization = authorization
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": self.authorization
        }
        self.request_num = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.time_consumed = 0.0

    def request(self, request_text) -> str:
        print('request start at:', datetime.now())
        print('request text:')
        print(request_text)
        data = {
            "messages": request_text
        }
        response_text = None
        t0 = time.time()
        try:
            self.request_num += 1
            input_tokens = get_token_num(request_text)
            print('input_tokens:', input_tokens)
            self.input_tokens += input_tokens
            response = requests.post(self.url, json=data, headers=self.headers, timeout=(10, 60))
            # Check if the request was successful
            if response.status_code == 200:
                response.encoding = 'utf-8'
                print("Response from server:")
                print(response.text)  # Assuming the response is text
                response_text = response.text
                output_tokens = get_token_num(response_text)
                print('output_tokens:', output_tokens)
                self.output_tokens += output_tokens
            else:
                print("Error:", response.status_code, response.text)
        except requests.exceptions.RequestException as e:
            # Handle any errors that occur during the request
            print("HTTP Request failed:", e)
        t = time.time() - t0
        print('time_consumed:', t)
        self.time_consumed += t
        print('request end at:', datetime.now())
        return response_text


class DeepSeekClient(LLMClient):
    def __init__(self, api_key=None, base_url="https://api.deepseek.com/"):
        self.request_num = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.time_consumed = 0.0
        self.temperature = 0.7
        self.api_key = api_key
        self.base_url = base_url
        self.model = "deepseek-chat"

    def request(self, request_text) -> str:
        print('request start at:', datetime.now())
        print('request text:')
        print(request_text)
        response_text = None
        t0 = time.time()
        try:
            self.request_num += 1
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=180)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": request_text},
                ],
                temperature=self.temperature
            )
            input_tokens = response.usage.prompt_tokens
            print('input_tokens:', input_tokens)
            self.input_tokens += input_tokens
            response_text = response.choices[0].message.content
            print("Response from server:")
            print(response_text)
            output_tokens = response.usage.completion_tokens
            print('output_tokens:', output_tokens)
            self.output_tokens += output_tokens
        except Exception as e:
            print("Client Request failed:", e)
        t = time.time() - t0
        print('time_consumed:', t)
        self.time_consumed += t
        print('request end at:', datetime.now())
        return response_text


class GPTClient(LLMClient):
    def __init__(self, api_key=None, base_url="https://api.zhizengzeng.com/v1"):
        self.request_num = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.time_consumed = 0.0
        self.temperature = 0.7
        self.api_key = api_key
        self.base_url = base_url
        # self.model = "gpt-3.5-turbo"
        self.model = "gpt-4o"

    def request(self, request_text) -> str:
        print('request start at:', datetime.now())
        print('request text:')
        print(request_text)
        response_text = None
        t0 = time.time()
        try:
            self.request_num += 1
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=180)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": request_text},
                ],
                temperature=self.temperature
            )
            input_tokens = response.usage.prompt_tokens
            print('input_tokens:', input_tokens)
            self.input_tokens += input_tokens
            response_text = response.choices[0].message.content
            print("Response from server:")
            print(response_text)
            output_tokens = response.usage.completion_tokens
            print('output_tokens:', output_tokens)
            self.output_tokens += output_tokens
        except Exception as e:
            print("Client Request failed:", e)
        t = time.time() - t0
        print('time_consumed:', t)
        self.time_consumed += t
        print('request end at:', datetime.now())
        return response_text
