from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from logger_config import get_logger
import re

logger = get_logger(__file__)

# Create CrewAI agent
def create_agent(model_choice="openrouter/qwen/qwen3-32b"):
    llm = LLM(
        model=model_choice,
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=40_000,
        timeout=300
    )
    
    requirements_processor = Agent(
        role="Requirements Processing Specialist",
        goal="Remove semantically duplicate requirements and group similar ones under specialized category names while maintaining exact original descriptions and all metadata",
        backstory="You are an expert in semantic analysis, data cleaning, and requirement organization with years of experience in processing procurement and tender requirements. You specialize in both identifying true duplicates versus similar but distinct requirements and creating meaningful categorizations. You ensure no important information is lost during processing while organizing requirements into logical groups with descriptive category names.",
        llm=llm,
        verbose=True
    )
    
    return requirements_processor

# Create CrewAI task
def create_task(requirements_processor, requirements_data):
    """
    Create task for processing requirements data
    
    Args:
        requirements_processor: Agent for processing requirements
        requirements_data: Dictionary containing scored and unscored requirements
    """
    
    processing_task = Task(
        description=f"""
        Process the following requirements data by performing BOTH deduplication and aggregation:
        
        INPUT DATA:
        {json.dumps(requirements_data, indent=2)}
        
        STEP 1 - DEDUPLICATION:
        1. Examine both scored and unscored requirements for semantic duplicates
        2. Consider requirements as duplicates only if they have substantially the same meaning and intent
        3. Preserve the most complete version when duplicates are found
        4. Maintain all original metadata (weight_percentage, context) exactly as provided
        5. Do not modify any requirement text, context, or weight information
        6. Focus on true duplicates, not just similar requirements
        
        STEP 2 - AGGREGATION:
        1. Analyze semantic similarity between the deduplicated requirements
        2. Create meaningful category names that reflect the grouped requirements' common theme
        3. ABSOLUTELY PRESERVE all original requirement text, context, and weight_percentage exactly as provided
        4. Group requirements that share similar themes, purposes, or domains
        5. Each category should contain requirements that logically belong together
        6. Category names should be descriptive and professional
        
        CRITICAL RULES:
        - NEVER modify requirement text content
        - NEVER modify context information  
        - NEVER modify weight_percentage values
        - ONLY create category names as keys
        - Original content must remain 100% intact
        """,
        agent=requirements_processor,
        expected_output="""
        A JSON dictionary with specialized category names as keys and arrays of requirements as values:
        {
          "Language_and_Documentation_Requirements": [
            {
              "requirement": "Original requirement text (completely unchanged)",
              "weight_percentage": "Original weight (if applicable)",
              "context": "Original context (completely unchanged)"
            }
          ],
          "Bidder_Selection_Criteria": [
            {
              "requirement": "Original requirement text (completely unchanged)", 
              "weight_percentage": "Original weight (if applicable)",
              "context": "Original context (completely unchanged)"
            }
          ],
          "Technical_Specifications": [
            {
              "requirement": "Original requirement text (completely unchanged)",
              "context": "Original context (completely unchanged)"
            }
          ]
        }
        
        Note: Categories are examples - create appropriate names based on actual content.
        Maintain the exact structure with requirement, weight_percentage (for scored), and context fields.
        Requirements should be deduplicated first, then grouped into appropriate categories.
        """,
    )
    
    return processing_task

# Function to execute the crew workflow
def process_requirements(requirements_data, model_choice="openrouter/qwen/qwen3-32b"):
    """
    Process requirements data through deduplication and aggregation
    
    Args:
        requirements_data: Dictionary with scored and unscored requirements
        model_choice: LLM model to use
        
    Returns:
        Processed and aggregated requirements
    """
    try:
        logger.info("Starting requirements processing workflow")
        
        # Create agent
        requirements_processor = create_agent(model_choice)
        
        # Create task
        processing_task = create_task(requirements_processor, requirements_data)
        
        # Create crew
        crew = Crew(
            agents=[requirements_processor],
            tasks=[processing_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute workflow
        logger.info("Executing crew workflow")
        result = crew.kickoff()
        
        logger.info("Requirements processing completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in requirements processing: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Sample input data
    sample_data = {
  "scored_requirements": [
    {
      "requirement": "Compliance with work scope",
      "weight_percentage": "40%",
      "context": "The proposal must demonstrate compliance with the work scope as part of the technical evaluation criteria."
    },
    {
      "requirement": "Quality assurance and operation",
      "weight_percentage": "25%",
      "context": "The proposal must address quality assurance and operational requirements as part of the technical evaluation criteria."
    },
    {
      "requirement": "Experience in similar projects",
      "weight_percentage": "20%",
      "context": "The proposal must include evidence of experience in similar projects as part of the technical evaluation criteria."
    },
    {
      "requirement": "Post-sale support",
      "weight_percentage": "15%",
      "context": "The proposal must outline post-sale support plans as part of the technical evaluation criteria."
    }
  ],
  "unscored_requirements": [
    {
      "requirement": "Bids must be submitted in Arabic with English copy",
      "context": "All bids must be submitted in Arabic and accompanied by an additional copy in English (Section 1.10)."
    },
    {
      "requirement": "Bid validity period of 90 days",
      "context": "Offers must remain valid for ninety days from the date set for opening bids (Section 1.10)."
    },
    {
      "requirement": "Submission deadline of 2023-04-11 and 4:00 PM on 2023-04-12",
      "context": "Final Bid Submission Date is 2023-04-12 by 4:00 PM, with last date for receiving bids on 2023-04-11 (Chunk 1 Table)."
    },
    {
      "requirement": "100% global service availability SLA",
      "context": "The solution must provide a global service availability level agreement (SLA) of 100% (Scope of Work 1.0)."
    }
  ]
}
    
    # Process requirements
    result = process_requirements(sample_data)
    print(result)