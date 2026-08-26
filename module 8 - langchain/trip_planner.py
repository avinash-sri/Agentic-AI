from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    fake_data = {"Mumbai": "32°C, humid", "Delhi": "38°C, sunny", "Goa": "30°C, humid"}
    return fake_data.get(city, "No data available")

@tool
def calculate_budget(total_budget: float, days: int) -> str:
    """Calculate a daily budget breakdown for a trip given total budget and number of days."""
    daily = total_budget / days
    return f"Daily budget: {daily:.0f} (Stay: {daily*0.4:.0f}, Food: {daily*0.3:.0f}, Activities: {daily*0.3:.0f})"


agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather, calculate_budget],
    system_prompt="You are a helpful travel planning assistant."
)

result = agent.invoke({"messages": [{"role": "user", "content": "Plan a 4-day Goa trip with a 20000 budget."}]})
print(result["messages"][-1].content)

result1 = agent.invoke({"messages": [{"role": "user", "content": "Plan a 4-day Goa trip with 20000 budget."}]})
history = result1["messages"]  # grab the full message list back out

history.append({"role": "user", "content": "Actually make it 5 days."})
result2 = agent.invoke({"messages": history})  # feed the whole thing back in
print(result2["messages"][-1].content)