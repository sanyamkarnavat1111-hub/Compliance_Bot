# ProcurementAgenticSystem Crew

Welcome to the ProcurementAgenticSystem Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.13 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/procurement_agentic_system/config/agents.yaml` to define your agents
- Modify `src/procurement_agentic_system/config/tasks.yaml` to define your tasks
- Modify `src/procurement_agentic_system/crew.py` to add your own logic, tools and specific args
- Modify `src/procurement_agentic_system/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the procurement-agentic-system Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew

The procurement-agentic-system Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the ProcurementAgenticSystem Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.


## Openrouter API configuation

OPENAI_API_BASE= "https://openrouter.ai/api/v1"
OPENAI_MODEL_NAME= "openrouter/qwen/qwen3-32b"
OPENAI_API_KEY = ""

OPENROUTER_API_KEY = ""



## Some sample file running locations

python procurement_agentic_system/src/procurement_agentic_system/tools/custom_tool.py
python embedding_management.py
python embedding_generation.py
python chunk_processing.py
python pipeline.py

<!-- For running crewai -->
python procurement_agentic_system/src/procurement_agentic_system/main.py (from /home/aman/Desktop/RFP_100_accurate/code)

## Working directory tree strcture

.
├── archieved
│   ├── main.py
│   ├── rfp_proposal_eval.py
│   └── worker.py
├── chunk_processing.py
├── ea_standard_eval.py
├── embedding_db
│   ├── embeddings.npy
│   └── metadata.json
├── embedding_generation.py
├── embedding_management.py
├── embeddings.index
├── output
│   ├── 01
│   │   ├── tmp_dummy.txt
│   │   ├── tmp_output_1_translate.txt
│   │   ├── tmp_output_1.txt
│   │   ├── tmp_output_2_translate.txt
│   │   ├── tmp_output_2.txt
│   │   └── tmp_output_all_procecssed.txt
│   └── 02
│       └── tmp_output_1.txt
├── pdf_parsing.py
├── pipeline.py
├── procurement_agentic_system
│   ├── config
│   ├── final_evaluation_table.html
│   ├── pyproject.toml
│   ├── README.md
│   ├── reviewed_evaluation_table.html
│   ├── src
│   │   └── procurement_agentic_system
│   │       ├── config
│   │       │   ├── agents.yaml
│   │       │   └── tasks.yaml
│   │       ├── crew.py
│   │       ├── __init__.py
│   │       ├── main.py
│   │       └── tools
│   │           ├── custom_tool.py
│   │           ├── __init__.py
│   ├── tools
│   └── uv.lock
├── sample_data
│   └── 01
│       ├── 01 rfp.pdf
│       └── 02 ea standard.pdf
├── tmp.txt
└── translation_gpt_openai.py

182 directories, 1315 files
