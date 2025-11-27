from crewai import Agent, Task, Crew
from crewai import LLM

llm = LLM(model="ollama/llama3", base_url="http://127.0.0.1:11434/v1")

agent = Agent(
    role="Tester",
    goal="Say hello from local Ollama",
    backstory="Running fully offline",
    llm=llm,
    verbose=True
)

task = Task(
    description="Say: 'Hello! I am running on local Ollama llama3:8b via CrewAI.'",
    expected_output="One sentence",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)
result = crew.kickoff()
print("Result:", result)