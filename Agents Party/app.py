from huggingface_hub import login
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool
import time
import datetime
from langfuse import get_client
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from dotenv import load_dotenv, dotenv_values 

load_dotenv()


 
langfuse = get_client()

if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

login()



model = InferenceClientModel(
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    max_tokens=2048,
    temperature=0.5
)

@tool
def get_time_taken_by_butler_for_tasks(task: str) -> str:
    """
    Returns the time taken by the butler to prepare the party.
    Args:
        task (str): The task for which the time taken is to be calculated. Allowed values are:
                    - "Prepare the drinks"
                    - "Decorate the mansion"
                    - "Set up the menu"
                    - "Prepare the music and playlist"
    """
    if task == "Prepare the drinks":
        return "The butler takes 30 minutes to prepare the drinks."
    elif task == "Decorate the mansion":
        return "The butler takes 2 hours to decorate the mansion."
    elif task == "Set up the menu":
        return "The butler takes 1 hour to set up the menu."
    elif task == "Prepare the music and playlist":
        return "The butler takes 45 minutes to prepare the music and playlist."
    return "The bulter can take 20-30 minutes to complete the task."

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

SmolagentsInstrumentor().instrument()

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), suggest_menu, get_time_taken_by_butler_for_tasks],
    model=model,
    additional_authorized_imports=["datetime", "time"],
)


agent.run("   Alfred needs to prepare for the party. Here are the tasks: Prepare the drinks, Decorate the mansion, Set up the menu, Prepare the music and playlist. " \
"If we start right now, at what time will the party be ready?")

langfuse.flush()