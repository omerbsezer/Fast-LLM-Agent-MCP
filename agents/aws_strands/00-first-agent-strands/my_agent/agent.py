from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator, current_time, python_repl

MODEL= "us.amazon.nova-pro-v1:0"

bedrock_model = BedrockModel(
    model_id=MODEL,
    temperature=0.3,
    top_p=0.8,
)

# Define a custom tool as a Python function using the @tool decorator
@tool
def letter_counter(word: str, letter: str) -> int:
    """
    Count occurrences of a specific letter in a word.

    Args:
        word (str): The input word to search in
        letter (str): The specific letter to count

    Returns:
        int: The number of occurrences of the letter in the word
    """
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0

    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")

    return word.lower().count(letter.lower())

# Create an agent with tools from the strands-tools example tools package
agent = Agent(model=bedrock_model, tools=[calculator, current_time, python_repl, letter_counter])

# Ask the agent a question that uses the available tools
message = """
I have 3 requests:
1. What is the time right now?
2. Calculate 3111696 / 74088
3. Tell me how many letter R's are in the word "strawberry" 🍓

Use your python tools to confirm that the script works before outputting it
"""
agent(message, stream=False)
