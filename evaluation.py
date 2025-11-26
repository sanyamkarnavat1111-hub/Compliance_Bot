from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import time
import random
from langchain.schema import HumanMessage

class Evaluation:
    def __init__(self, result_html_path, rfp_text, ea_standard_text, entity_text, output_file="final_outputs/evaluations/evaluation_report.txt"):
        load_dotenv()
        self.api_keys = os.getenv('GOOGLE_API_KEY')
        self.result_html_path = result_html_path
        self.rfp_text = rfp_text
        self.ea_standard_text = ea_standard_text
        self.entity_text = entity_text
        self.llm = None  # Initialize LLM client later
        self.output_file = output_file

    def extract_text(self, file_path):
        """Extracts text from a file based on its extension."""
        with open(file_path, "r", encoding="utf-8") as file:
            if file_path.endswith(".html"):
                soup = BeautifulSoup(file, "html.parser")
                return soup.get_text(separator="\n")
            else:
                return file.read()

    def generate_prompt(self, result_html_text, rfp_text, ea_standard_text, entity_text):
        """Generates the evaluation prompt for Gemini."""
        return  f"""
You are an expert evaluator tasked with assessing the **RFP HTML report**, which evaluates the alignment of RFP requirements with Enterprise Architecture (EA) standards and Entity requirements for Government procurement processes.

**DEFINITIONS:**
- **RFP Requirements**: The original Request for Proposal document outlining the project’s needs, tasks, and specifications.
- **RFP HTML Report**: The generated evaluation report (Result HTML) that assesses how well the RFP requirements align with EA standards and Entity requirements, including improvement recommendations.

**PRIMARY EVALUATION FOCUS:**
Evaluate how accurately the **RFP HTML report** identifies and documents the alignment (or misalignment) between the RFP requirements and both EA standards and Entity requirements, focusing on True Positive (correct alignments) and True Negative (correctly identified misalignments). Provide constructive feedback to enhance the report’s quality.

**EVALUATION CRITERIA:**

1. **Gap Analysis Coverage** - How effectively does the **RFP HTML report** identify and address gaps?
   - Assess coverage of discrepancies between RFP requirements and both EA standards and Entity requirements.
   - Evaluate identification of missing Entity requirements.
   - Review completeness of technical specification alignments.

2. **RFP HTML Report Coverage Assessment** - How well does the **RFP HTML report** assess the completeness of RFP requirements?
   - Check if the report addresses missing sections in RFP requirements against EA standards and Entity requirements.
   - Evaluate technical criteria and evaluation weightings documented in the report.
   - Assess compliance framework coverage reported.

3. **EA Requirement Alignment** - How accurately does the **RFP HTML report** document alignment of RFP requirements with EA standards and Entity requirements?
   - Verify technology stack recommendations match EA standards and Entity-approved lists.
   - Check security and architecture standard compliance as reported.
   - Assess infrastructure and deployment requirement coverage documented.

**ASSESSMENT APPROACH:**
- Focus on constructive evaluation rather than critical assessment.
- Identify how the **RFP HTML report** accurately documents alignment of RFP requirements with EA standards and Entity requirements, emphasizing True Positive (correctly identified alignments) and True Negative (correctly identified misalignments).
- Suggest enhancements where appropriate without demanding strict compliance.
- Balance compliance needs with practical implementation considerations.
- Recognize that section names and terminology may vary between the **RFP HTML report**, EA standards, and Entity requirements while maintaining similar meaning or scope.
- Consider that RFP requirement sections may use broader (superset) or narrower (subset) terminology than EA standards and Entity requirements while still addressing the same needs.

**INPUT DATA:**
Result HTML (**RFP HTML Report**):
{result_html_text}

RFP Requirements:
{rfp_text}

EA Standards:
{ea_standard_text}

Entity Requirements:
{entity_text}

**OUTPUT FORMAT:**

## 1. Gap Analysis
**Identified Gaps:**
- Document key gaps between RFP requirements and both EA standards and Entity requirements as identified in the **RFP HTML report**.
- Note missing Entity requirements flagged in the report.
- Highlight technical specification misalignments addressed.

**Gap Coverage Assessment:**
- Evaluate comprehensiveness of gap identification in the **RFP HTML report**.
- Note any significant gaps that might benefit from additional attention.
- Assess completeness of gap analysis across all requirement areas.

## 2. RFP HTML Report Coverage
**Current Coverage Strengths:**
- Identify RFP requirement sections that the **RFP HTML report** notes as aligning well with EA standards and Entity requirements.
- Note areas where the report indicates the RFP requirements already meet standards effectively.

**Enhancement Areas:**
- Document recommended additions in the **RFP HTML report** to improve RFP requirement completeness.
- Evaluate suggested structural improvements reported.
- Assess proposed evaluation criteria enhancements documented.

**Implementation Considerations:**
- Note practical aspects of implementing recommended changes as outlined in the **RFP HTML report**.
- Consider resource and timeline implications where relevant.

## 3. EA Requirement Alignment
- Assess how well the **RFP HTML report** aligns RFP requirements with EA standards and Entity requirements.

**EVALUATION SUMMARY:**
- **Strengths:** Key positive aspects of the **RFP HTML report**’s alignment analysis.
- **Enhancement Opportunities:** Areas where the **RFP HTML report** could be refined or expanded to improve accuracy.

- **Overall Quality Score:** Rate the accuracy of the **RFP HTML report** in documenting True Positive (correctly identified alignments) and True Negative (correctly identified misalignments) RFP requirements with criteria MUst be include : [EA standards with Entity requirements] (0-10 scale). [Accuracy also me in FLoating point not complusory to give in roundfigure way.]

- Be careful to note: is anything is Missing in **RFP HTML report** that EA_stadard or Entity have but not in RFP requirements? and my **RFP HTML report** missed it to mention or describe. [IT"S MAIN AGENDA OF THIS EVALUTION, IF THIS CONDITION MATCH THEN GIVE 1% HIPE IN FINAL SCORE, IF NOT THEN CUT % SCORE BASED ON HOW MUCH IT'S MISSED] 

- ALSO TELL EVRRY TIME : If accuracy broke even 0.10% then describe for what reason ?? 

- DO not Hallusinate everytime THink Carefully and then give the answer.
- **Overall Assessment:** Summary of the **RFP HTML report**’s accuracy and value in evaluating RFP requirement alignment.

**BALANCED ASSESSMENT PRINCIPLES:**
- Recognize that EA standards and Entity requirements often provide recommended rather than mandatory lists.
- Allow for justified alternatives where technical requirements support them.
- Focus on improving **RFP HTML report** accuracy while maintaining practical feasibility.
- Provide constructive feedback that supports iterative improvement.
- Understand that **RFP HTML report** sections and both EA standards and Entity requirements may use different terminology while covering similar content.
- Consider semantic similarity rather than exact naming matches when evaluating section coverage.
- Do not ask for or check for the detailed assessment of the potential impact of each gap and more specific recommendations for addressing them.

**NOTE:**
- Ensure the response is based solely on the **RFP HTML report**’s accuracy in evaluating how RFP requirements align with EA standards and Entity requirements.
- Do not include personal thoughts in the final result; provide answers based only on the **RFP HTML report**’s alignment analysis.
- After finalizing the answer, re-evaluate the result to ensure it meets all conditions, checking for missing elements or mistakes. Refine until confident it is the final answer.
- If no personal thoughts are included, specify at the end of the Summary: "I have not included any personal thoughts in this final result; it is based solely on the **RFP HTML report**’s accuracy in evaluating how RFP requirements align with EA standards and Entity requirements."
- Make sure to not look for exact or same names or items, there can be some deviation in the name, It can be synonyms or related terms or superset name. -> And also make sure to not cut accuracy for it.

Note: When evaluating technology choices and section coverage, consider that EA standards and Entity requirements typically provide guidance rather than strict mandates. Section names in RFP requirements may differ from EA standards but can still address the same needs through related, broader, or more specific terminology. Recommendations in the **RFP HTML report** should support compliance while allowing for justified technical decisions based on project needs.
"""

    def make_api_call(self, prompt: str) -> str:
        """Make API call to Gemini with provided prompt"""
        print('called make api call function')
        
        content = [{"type": "text", "text": prompt}]


        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.15,
            google_api_key=self.api_keys
        )
    
        response = self.llm.invoke([HumanMessage(content=content)])
        print(response)
        return response.content

    def run(self):
        print("Extracting input files...")
        result_html_text = self.extract_text(self.result_html_path)

        print("Generating evaluation prompt...")
        prompt = self.generate_prompt(result_html_text, self.rfp_text, self.ea_standard_text, self.entity_text)

        print("Sending to Gemini...")
        reports = []
        for i in range(5):
            report = self.make_api_call(prompt)
            reports.append(report)


            output_dir = os.path.dirname(self.output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            base_name, ext = os.path.splitext(os.path.basename(self.output_file))
            output_path = os.path.join(output_dir, f"{base_name}_{i+1}{ext}")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        print(reports)
        print(f"\nEvaluation complete. Report saved to: {self.output_file}")

if __name__ == "__main__":
    # Update these paths to point to your actual files
    result_html_path = "final_outputs/report_for_evaluation.html"
    
    rfp_text = """

    """

    EA_standard_text = """

    """

    entity_text = """
    Scope of Work, Project Selivery, Preferred Application Frameworks,Accepted Database,Accepted Operating System,Requirement of SSL Certificate,Non-Functional,Project Deliverables
    """
    evaluation = Evaluation(result_html_path, rfp_text, EA_standard_text, entity_text)
    evaluation.run()