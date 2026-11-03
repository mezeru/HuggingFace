import os
from PIL import Image
from smolagents import CodeAgent, GoogleSearchTool, InferenceClientModel, DuckDuckGoSearchTool
from calTool import calculate_cargo_travel_time
from visitWebPageTool import VisitWebpageTool

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

visit_web_page = VisitWebpageTool()

sub_agent = CodeAgent(
    tools=[calculate_cargo_travel_time, DuckDuckGoSearchTool(), visit_web_page],
    model=model,
    name="web_search_agent",
    additional_authorized_imports=[],
    description="Searches Web Site for queries",
    max_steps=5
)