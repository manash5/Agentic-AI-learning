from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

import asyncio


async def main(): 
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python", 
                "args": ["mathserver.py"], 
                "transport": "stdio"
            }, 
            "weather": {
                "url": "http://localhost:8000/mcp", # Ensure the server is running here 
                "transport": "streamable_http"
            }
        }
    )

    import os 
    os.environ["GROQ_API_KEY"]= os.getenv("GROQ_API_KEY")

    tools = await client.get_tools() # Establish a connection with all the mcp tools in the server 
    model = ChatGroq(model="qwen/qwen3.6-27b")
    agent = create_react_agent(
        model, tools 
    )



    math_repsonse = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3+5) x 12?"}]}
    )

    print("Math Response: ", math_repsonse['messages'][-1].content)


    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
    )

    print("Weather Response: ", weather_response['messages'][-1].content)


asyncio.run(main())