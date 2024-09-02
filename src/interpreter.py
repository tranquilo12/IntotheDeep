import aiohttp
import json
from typing import Tuple
from pydantic import BaseModel, Field


##########################################################
########## Local Python Interpreter Related ##############
##########################################################
class Interpreter(BaseModel):
    endpoint: str = Field(default="http://localhost:8888/execute")

    class Config:
        arbitrary_types_allowed = True  # Allow the aiohttp ClientSession

    async def run(self, code: str) -> Tuple[str, str]:
        """
        Execute the provided code using an HTTP POST request to the specified endpoint.

        Parameters
        ----------
        code : str
            The code to execute.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the standard output and standard error from the code execution.
        """
        session_timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            async with session.post(self.endpoint, json={"code": code}) as response:
                result = await response.text()
                result = json.loads(result)
                return result["stdout"], result["stderr"]
