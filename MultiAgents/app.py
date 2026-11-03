
from smolagents import CodeAgent, GoogleSearchTool, InferenceClientModel, DuckDuckGoSearchTool
from calTool import calculate_cargo_travel_time
from subagent import sub_agent
from reasoning import check_reasoning_and_plot

model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=8096)

task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
Also give me some supercar factories with the same cargo plane transfer time."""



manager = CodeAgent(
    model=model,
    tools=[calculate_cargo_travel_time],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    max_steps=20,
    final_answer_checks=[check_reasoning_and_plot],
    managed_agents=[sub_agent]
)

detailed_report = manager.run(f"""
You're an expert analyst. You make comprehensive reports after visiting many websites.
Don't hesitate to search for many queries at once in a for loop.
For each data point that you find, visit the source url to confirm numbers.

{task}
""")



manager.visualize()