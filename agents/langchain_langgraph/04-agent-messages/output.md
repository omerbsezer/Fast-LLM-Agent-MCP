python3 agent.py 
[SystemMessage] You are a helpful weather assistant. Use tools when asked about weather.

[HumanMessage] What's the weather in Istanbul?

[AIMessage] content='[{'type': 'text', 'text': '<thinking> To determine the current weather in Istanbul, I need to use the `get_weather` tool with the city set to "Istanbul". </thinking>\n'}, {'type': 'tool_use', 'name': 'get_weather', 'input': {'city': 'Istanbul'}, 'id': 'tooluse_1hquSSDeEKRMLRa1r75UqL'}]' | tool_calls=['get_weather']

[ToolMessage] Sunny, 28°C (tool_call_id=tooluse_1hquSSDeEKRMLRa1r75UqL)

[AIMessage] Final answer: The current weather in Istanbul is sunny with a temperature of 28°C.