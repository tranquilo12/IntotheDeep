import json
import aiohttp
from typing import Tuple


class Interpreter:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def run(self, code: str) -> Tuple[str, str]:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json={"code": code}) as response:
                result = await response.text()
                result = json.loads(result)
                return result["stdout"], result["stderr"]


async def execute_code_locally(code: str) -> Tuple[str, str]:
    repl = Interpreter(endpoint="http://localhost:8888/execute")
    output, error = await repl.run(code)
    return output, error
