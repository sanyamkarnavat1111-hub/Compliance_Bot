from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
import logging
from dotenv import load_dotenv
import os
import json
from logger_config import get_logger

# Removed module-level logger: logger = get_logger(__file__)

class AnalysisCrew:
    def __init__(self, logger_instance=None):
        # Configure logging
        self.logger = logger_instance if logger_instance is not None else get_logger(__name__) # Use passed logger or default
        # Removed basicConfig as it's handled by logger_config
        
        # Load environment
        load_dotenv()
        
        self.model_name = "openrouter/qwen/qwen3-32b"
        
        # Initialize LLM
        self.llm = LLM(
            model=self.model_name,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3
        )

    def create_scored_requirements_analyzer(self):
        """Create agent for analyzing scored requirements"""
        scored_agent = Agent(
            role="Scored Requirements Analysis Expert",
            goal="Analyze scored requirements from proposal evaluation data and provide comprehensive insights with individual requirement details",
            backstory="An expert evaluator specializing in analyzing scored requirements with deep understanding of proposal evaluation metrics and scoring systems.",
            llm=self.llm,
            verbose=True
        )

        scored_task = Task(
            description=(
                f"# TASK: Analyze Scored Requirements from Proposal Evaluation\n\n"
                f"## INPUT DATA\n"
                f"The following scored requirements data needs to be analyzed:\n"
                f"```json\n{json.dumps(self.proposal_evaluation_data.get('scored', []), indent=2)}\n```\n\n"
                "## ANALYSIS REQUIREMENTS\n"
                "1. **Preserve all individual scored requirements** with their specific details, scores, and analysis\n"
                "2. **Extract and categorize all strengths** from the scored requirements\n"
                "3. **Identify and categorize all concerns** from the scored requirements\n"
                "4. **Analyze risk patterns** across all scored requirements\n"
                "5. **Provide a comprehensive summary** of the overall scored requirements performance\n\n"
                "## ANALYSIS APPROACH\n"
                "- FIRST: List each individual scored requirement with its specific score, weight, and key findings\n"
                "- THEN: Group similar strengths together and provide consolidated insights\n"
                "- Identify common patterns in concerns and their potential impact\n"
                "- Assess risk levels and their implications for the overall proposal\n"
                "- Consider scoring patterns and their significance\n"
                "- CRITICAL: Ensure ALL scored requirements from the input are included in the individual_requirements section\n\n"
                "## EXPECTED OUTPUT FORMAT\n"
                "Return a JSON object with the following structure:\n"
                "```json\n"
                "{\n"
                '  "individual_requirements": [\n'
                '    {\n'
                '      "requirement": "Exact requirement text",\n'
                '      "assigned_score": "Actual score received",\n'
                '      "requirement_score": "Weight/percentage for this requirement",\n'
                '      "key_strengths": ["Main strengths for this specific requirement"],\n'
                '      "key_concerns": ["Main concerns for this specific requirement"],\n'
                '      "compliance_summary": "Brief summary of how well this requirement was addressed"\n'
                '    }\n'
                '  ],\n'
                '  "technical_strengths": ["List of consolidated strengths from all scored requirements"],\n'
                '  "technical_concerns": ["List of consolidated concerns from all scored requirements"],\n'
                '  "technical_risks": ["List of identified risks and their implications"],\n'
                '  "scoring_summary": {\n'
                '    "total_scored_requirements": "Number of scored requirements analyzed",\n'
                '    "total_possible_score": "Sum of all requirement weights",\n'
                '    "total_achieved_score": "Sum of all assigned scores",\n'
                '    "overall_percentage": "Overall scoring percentage"\n'
                '  },\n'
                '  "technical_summary": "Comprehensive summary of scored requirements analysis"\n'
                "}\n"
                "```\n\n"
                "## CRITICAL INSTRUCTIONS\n"
                "- MANDATORY: Include ALL scored requirements from the input data in the individual_requirements section\n"
                "- Each individual requirement must preserve its original requirement text, scores, and analysis\n"
                "- Focus on extracting meaningful patterns and insights in the consolidated sections\n"
                "- Avoid redundancy in the consolidated output\n"
                "- Provide actionable insights in the summary\n"
                "- Consider the scoring context when analyzing strengths and concerns\n"
                "- Make sure to not include pricing or financial information\n"
                "- Double-check that the count in total_scored_requirements matches the actual number of requirements processed"
            ),
            expected_output="A comprehensive JSON analysis of scored requirements with individual requirement details, consolidated strengths, concerns, risks, scoring summary, and overall summary",
            agent=scored_agent
        )

        scored_crew = Crew(
            agents=[scored_agent],
            tasks=[scored_task],
            process=Process.sequential,
            manager_llm=self.llm,
            model=self.model_name,
            verbose=True
        )

        return scored_crew

    # def create_unscored_requirements_analyzer(self):
    #     """Create agent for analyzing unscored requirements"""
    #     unscored_agent = Agent(
    #         role="Unscored Requirements Analysis Expert",
    #         goal="Analyze unscored requirements from proposal evaluation data and provide comprehensive insights",
    #         backstory="An expert evaluator specializing in analyzing unscored requirements with deep understanding of proposal evaluation criteria and risk assessment.",
    #         llm=self.llm,
    #         verbose=True
    #     )

    #     # unscored_task = Task(
    #     #     description=(
    #     #         f"# TASK: Analyze Unscored Requirements from Proposal Evaluation\n\n"
    #     #         f"## INPUT DATA\n"
    #     #         f"The following unscored requirements data needs to be analyzed:\n"
    #     #         f"```json\n{json.dumps(self.proposal_evaluation_data.get('unscored', []), indent=2)}\n```\n\n"
    #     #         "## ANALYSIS REQUIREMENTS\n"
    #     #         "1. **Extract and categorize all strengths** from the unscored requirements\n"
    #     #         "2. **Identify and categorize all concerns** from the unscored requirements\n"
    #     #         "3. **Analyze risk patterns** across all unscored requirements\n"
    #     #         "4. **Provide a comprehensive summary** of the overall unscored requirements performance\n\n"
    #     #         "## ANALYSIS APPROACH\n"
    #     #         "- Group similar strengths together and provide consolidated insights\n"
    #     #         "- Identify common patterns in concerns and their potential impact\n"
    #     #         "- Assess risk levels and their implications for the overall proposal\n"
    #     #         "- Consider the qualitative nature of unscored requirements\n\n"
    #     #         "## EXPECTED OUTPUT FORMAT\n"
    #     #         "Return a JSON object with the following structure:\n"
    #     #         "```json\n"
    #     #         "{\n"
    #     #         '  "technical_strengths": ["List of consolidated strengths from all unscored requirements"],\n'
    #     #         '  "technical_concerns": ["List of consolidated concerns from all unscored requirements"],\n'
    #     #         '  "technical_risks": ["List of identified risks and their implications"],\n'
    #     #         '  "technical_summary": "Comprehensive summary of unscored requirements analysis"\n'
    #     #         "}\n"
    #     #         "```\n\n"
    #     #         "## CRITICAL INSTRUCTIONS\n"
    #     #         "- Focus on extracting meaningful patterns and insights\n"
    #     #         "- Avoid redundancy in the output\n"
    #     #         "- Provide actionable insights in the summary\n"
    #     #         "- Consider the qualitative nature of unscored requirements\n"
    #     #         "- Make sure to no include a pricing or financial information"
    #     #     ),
    #     #     expected_output="A comprehensive JSON analysis of unscored requirements with strengths, concerns, risks, and summary",
    #     #     agent=unscored_agent
    #     # )

    #     unscored_task = Task(
    #         description=(
    #             f"# TASK: Analyze Unscored Requirements from Proposal Evaluation\n\n"
    #             f"## INPUT DATA\n"
    #             f"The following unscored requirements data needs to be analyzed:\n"
    #             f"```json\n{json.dumps(self.proposal_evaluation_data.get('unscored', []), indent=2)}\n```\n\n"
    #             "## ANALYSIS REQUIREMENTS\n"
    #             "1. **Extract and categorize all strengths** from non-mandatory unscored requirements and any mandatory requirements with contradictory statements in the proposal\n"
    #             "2. **Identify and categorize all concerns** from non-mandatory unscored requirements and any mandatory requirements with contradictory statements in the proposal\n"
    #             "3. **Analyze risk patterns** across non-mandatory unscored requirements and any mandatory requirements with contradictory statements\n"
    #             "4. **Provide a comprehensive summary** of the overall performance of the analyzed unscored requirements\n\n"
    #             "## ANALYSIS APPROACH\n"
    #             "- Filter out mandatory terms, conditions, or instructions unless they contain contradictory statements in the proposal\n"
    #             "- Group similar strengths together from the filtered requirements and provide consolidated insights\n"
    #             "- Identify common patterns in concerns from the filtered requirements and their potential impact\n"
    #             "- Assess risk levels and their implications for the overall proposal based on the filtered requirements\n"
    #             "- Consider the qualitative nature of the analyzed unscored requirements\n"
    #             "- Highlight contradictions in mandatory requirements explicitly in the output\n\n"
    #             "## EXPECTED OUTPUT FORMAT\n"
    #             "Return a JSON object with the following structure:\n"
    #             "```json\n"
    #             "{\n"
    #             '  "technical_strengths": ["List of consolidated strengths from non-mandatory unscored requirements and mandatory requirements with contradictions"],\n'
    #             '  "technical_concerns": ["List of consolidated concerns from non-mandatory unscored requirements and mandatory requirements with contradictions"],\n'
    #             '  "technical_risks": ["List of identified risks and their implications from the analyzed requirements"],\n'
    #             '  "technical_summary": "Comprehensive summary of the analyzed unscored requirements, focusing on non-mandatory requirements and any contradictory mandatory requirements"\n'
    #             "}\n"
    #             "```\n\n"
    #             "## CRITICAL INSTRUCTIONS\n"
    #             "- Exclude mandatory terms, conditions, or instructions from the analysis unless contradictory statements are present in the proposal\n"
    #             "- Highlight any contradictions in mandatory requirements explicitly in the strengths, concerns, risks, and summary\n"
    #             "- Focus on extracting meaningful patterns and insights from non-mandatory unscored requirements and any contradictory mandatory requirements\n"
    #             "- Avoid redundancy in the output\n"
    #             "- Provide actionable insights in the summary\n"
    #             "- Consider the qualitative nature of the analyzed requirements\n"
    #             "- Do not include pricing or financial information in the analysis"
    #         ),
    #         expected_output="A comprehensive JSON analysis of unscored requirements with strengths, concerns, risks, and summary, excluding mandatory requirements without contradictions",
    #         agent=unscored_agent
    #     )

    #     unscored_crew = Crew(
    #         agents=[unscored_agent],
    #         tasks=[unscored_task],
    #         process=Process.sequential,
    #         manager_llm=self.llm,
    #         model=self.model_name,
    #         verbose=True
    #     )

    #     return unscored_crew

    def create_final_analysis_agent(self, scored_analysis, unscored_analysis):
        """Create agent for final comprehensive analysis"""
        final_agent = Agent(
            role="Final Proposal Analysis Expert",
            goal="Provide comprehensive final analysis combining insights from both scored and unscored requirements with detailed individual requirement tracking",
            backstory="An expert evaluator specializing in comprehensive proposal analysis with deep understanding of both quantitative and qualitative evaluation factors.",
            llm=self.llm,
            verbose=True
        )

        final_task = Task(
            description=(
                f"# TASK: Final Comprehensive Proposal Analysis\n\n"
                f"## SCORED REQUIREMENTS ANALYSIS\n"
                f"```json\n{json.dumps(scored_analysis, indent=2)}\n```\n\n"
                f"## UNSCORED REQUIREMENTS ANALYSIS\n"
                f"```json\n{json.dumps(unscored_analysis, indent=2)}\n```\n\n"
                "## ANALYSIS REQUIREMENTS\n"
                "1. **Provide overall assessment** combining insights from both scored and unscored analyses\n"
                "2. **Generate strategic recommendations** based on the combined analysis\n"
                "3. **Outline next steps** for proposal improvement or decision-making\n\n"
                "## ANALYSIS APPROACH\n"
                "- Synthesize insights from both quantitative (scored) and qualitative (unscored) analyses\n"
                "- Identify cross-cutting themes and patterns\n"
                "- Provide strategic-level recommendations\n"
                "- Consider both immediate and long-term implications\n\n"
                "## EXPECTED OUTPUT FORMAT\n"
                "Return a JSON object with the following structure:\n"
                "```json\n"
                "{\n"
                '  "overall_assessment": ["List of key assessment points combining both analyses"],\n'
                '  "recommendation": ["List of strategic recommendations"],\n'
                '  "next_steps": ["List of actionable next steps"]\n'
                "}\n"
                "```\n\n"
                "## CRITICAL INSTRUCTIONS\n"
                "- Provide high-level strategic insights\n"
                "- Focus on actionable recommendations\n"
                "- Consider both strengths and areas for improvement\n"
                "- Provide clear next steps for decision-making\n"
                "- Make sure to no include a pricing or financial information"
            ),
            expected_output="A comprehensive final analysis with overall assessment, recommendations, and next steps",
            agent=final_agent
        )

        final_crew = Crew(
            agents=[final_agent],
            tasks=[final_task],
            process=Process.sequential,
            manager_llm=self.llm,
            model=self.model_name,
            verbose=True
        )

        return final_crew

    def process_analysis_results(self, result):
        """Extract and clean JSON from analysis results"""
        try:
            if isinstance(result, str):
                # Try to extract JSON from the string if needed
                if '```json' in result:
                    result = result.split('```json')[1].split('```')[0].strip()
                return json.loads(result)
            else:
                return result
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON from result: {str(e)}") # Use self.logger
            return {"error": f"Failed to parse result: {str(e)}"}

    def save_analysis_outputs(self, scored_analysis, unscored_analysis, final_analysis, output_dir):
        """Save all analysis outputs to files"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save scored requirements analysis
            with open(os.path.join(output_dir, "scored_requirements_analysis.json"), "w", encoding='utf-8') as f:
                json.dump(scored_analysis, f, indent=2, ensure_ascii=False)
            
            # Save unscored requirements analysis
            with open(os.path.join(output_dir, "unscored_requirements_analysis.json"), "w", encoding='utf-8') as f:
                json.dump(unscored_analysis, f, indent=2, ensure_ascii=False)
            
            # Save final analysis
            with open(os.path.join(output_dir, "final_analysis.json"), "w", encoding='utf-8') as f:
                json.dump(final_analysis, f, indent=2, ensure_ascii=False)

            # Return all analyses
            final_json =  {
                "scored_requirements_analysis": scored_analysis,
                "unscored_requirements_analysis": unscored_analysis,
                "final_analysis": final_analysis
            }

            with open(os.path.join(output_dir, "aggregated_analysis.json"), "w", encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Successfully saved all analysis outputs to files") # Use self.logger

            return final_json, os.path.join(output_dir, "aggregated_analysis.json")
            
        except Exception as e:
            self.logger.error(f"Error saving analysis outputs to files: {str(e)}") # Use self.logger

    def start_crew_process(self, formatted_eval_path):
        """Run the complete three-stage analysis process"""
        try:
            # Validate input parameter
            if formatted_eval_path is None:
                self.logger.error("formatted_eval_path is None") # Use self.logger
                return {"error": "No evaluation data path provided"}, None
            
            if not os.path.exists(formatted_eval_path):
                self.logger.error(f"File does not exist: {formatted_eval_path}") # Use self.logger
                return {"error": f"File does not exist: {formatted_eval_path}"}, None
            
            with open(formatted_eval_path, "r", encoding="utf-8") as f:
                self.proposal_evaluation_data = json.load(f)

            # Stage 1: Analyze scored requirements
            self.logger.info("Starting scored requirements analysis...") # Use self.logger
            
            scored_crew = self.create_scored_requirements_analyzer()
            scored_result = scored_crew.kickoff()
            scored_analysis = self.process_analysis_results(scored_result.raw)
            
            self.logger.info("Reading unscored requirements analysis...") # Use self.logger
            unscored_analysis = self.proposal_evaluation_data.get('unscored', [])
            
            # # Stage 2: Analyze unscored requirements
            # self.logger.info("Starting unscored requirements analysis...")
            # unscored_crew = self.create_unscored_requirements_analyzer()
            # unscored_result = unscored_crew.kickoff()
            # unscored_analysis = self.process_analysis_results(unscored_result.raw)
            
            # Stage 3: Final comprehensive analysis
            self.logger.info("Starting final comprehensive analysis...") # Use self.logger
            final_crew = self.create_final_analysis_agent(scored_analysis, unscored_analysis)
            final_result = final_crew.kickoff()
            final_analysis = self.process_analysis_results(final_result.raw)
            
            # Save all outputs
            final_analysis_json, final_analysis_json_path = self.save_analysis_outputs(scored_analysis, unscored_analysis, final_analysis, os.path.dirname(formatted_eval_path))
        
            return final_analysis_json, final_analysis_json_path
            
        except Exception as e:
            self.logger.error(f"Error in crew process: {str(e)}") # Use self.logger
            return {"error": f"Failed to complete analysis: {str(e)}"}, None

if __name__ == "__main__":
    # Example usage
    evaluator = AnalysisCrew(logger_instance=get_logger(__name__)) # Pass module logger for example
    results = evaluator.start_crew_process(formatted_eval_path="outputs_proposal_eval/formatted_eval.json")
    evaluator.logger.info(json.dumps(results, indent=2)) # Use the instance logger to print results
