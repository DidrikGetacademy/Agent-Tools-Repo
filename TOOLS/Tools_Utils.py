###Importing a tool collection from any MCP server###import os
from smolagents import ToolCollection, CodeAgent
import os
from mcp import StdioServerParameters
from smolagents import HfApiModel

model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
    agent.run("Please find a remedy for hangover.")

###############################################################################################################