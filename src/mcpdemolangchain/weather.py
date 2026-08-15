from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weeather")

@mcp.tool()
async def get_weather(locaiton: str)-> str: 
    """Get the weather location"""
    return "it's always raining in california"

if __name__ == "__main__":
    mcp.run(transport="streamable-http") 