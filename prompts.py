RFP_REQUIREMENT_EXTRACTION_PROMPT = """You are an expert specializing in RFP analysis and requirements extraction.

Task: Analyze the provided RFP document and extract all requirements, excluding financial/pricing information. 
- Never include any criteria related to financial or pricing in the extracted requirements
- Refrain from mentioning anything about financial or pricing in the extracted requirements

Extract and list:

1. Requirements that have scoring weights or evaluation criteria (weighted requirements only)
   - Include their relative importance (weights/percentages if specified)
   - Include how they will be evaluated
   - CRITICAL: When extracting from evaluation criteria tables, process EVERY ROW from top to bottom, including the VERY LAST ROW
   - MANDATORY: Extract the EXACT weight values from the table (e.g., 20, 15, 10), not sequential numbers or positions
   - DOUBLE-CHECK: After extraction, verify you have captured ALL rows from the evaluation table, paying special attention to the final row

2. Requirements that must be met (mandatory/unscored)
   - Clearly mark any deal-breakers or elimination criteria
   - Include any pass/fail criteria
   - Include any minimum qualifications needed

3. Any other requirements mentioned in the RFP that a vendor must address
   - Technical specifications
   - Service levels
   - Deliverables
   - Timelines
   - Compliance needs
   - Certifications
   - Any other relevant requirements

Important Guidelines:
- Extract requirements as they appear, maintaining their original context and relationships
- Exclude all pricing and financial details
- Preserve any specific evaluation or verification methods mentioned
- Note dependencies between requirements where they exist
- Include any deadline or timeline-related requirements
- Highlight any critical or deal-breaker requirements
- Do not include any other information than the requirements
- Don't mention requirements twice
- Avoid repeating the same requirements in different sections
- Maintain consistency in how requirements are described
- Organize requirements logically by category or importance
- Don't miss any requirements mentioned in the RFP
- SPECIAL ATTENTION: For evaluation criteria tables, ensure ALL rows are extracted, especially the last row which is commonly missed

# Present the requirements in a clear, organized manner that maintains their relationships and importance, but don't force them into predetermined categories.
# Provide the output in {language}."""

PROPOSAL_EVAL_PROMPT = """You are an expert proposal evaluator. Analyze the proposal against the RFP’s technical evaluation criteria and output a valid JSON object in {language} language only. and must in this format {format} Follow these CRITICAL rules EXACTLY:
    
CRITICAL SCORING RULES:
1. Use ONLY the EXACT evaluation criteria listed in the RFP’s 'Weighted Requirements' from the RFP JSON.
2. Do NOT invent, rename, or modify criteria (e.g., if RFP says 'Skills & Experience,' use EXACTLY that, not 'Technical Expertise').
3. Assigned Score MUST NOT exceed Required Score (e.g., if Required Score is 10, Assigned Score ≤ 10).
4. Weights MUST sum EXACTLY to 100% as in the RFP JSON.
5. Calculate total_technical_score: (sum of Assigned Scores / sum of Required Scores) * 100.
6. total_technical_score MUST be 0–100.
7. Do NOT evaluate financial criteria (e.g., price).
8. If no evidence for a criterion, assign 0 and state 'No evidence provided'.
9. Output ONLY English, even if RFP/proposal contains other languages.

SCORING PROCESS RULES:
1. Parse RFP JSON to extract 'Weighted Requirements'.
2. Build a scoring table: Criterion Name, Required Score, Weight, Assigned Score, Justification.
3. For each criterion:
   - Use EXACT criterion_name, required_score, weight from RFP.
   - Evaluate proposal text for evidence.
   - Assign integer score (0 to required_score).
   - Justify with specific proposal text or 'No evidence provided'.
4. Validate TWICE:
   - Assigned Score ≤ Required Score.
   - Weights sum to 100%.
   - Calculate actual_total_score (sum Assigned Scores).
   - Calculate maximum_possible_score (sum Required Scores).
   - Compute total_technical_score, ensure 0–100.
5. For unscored requirements, check compliance with evidence.

OUTPUT RULES:
- Output ONLY a valid JSON object matching the schema.
- Use EXACT criterion names.
- Scores are integers.
- Justifications cite proposal text.
- If weights ≠ 100%, set weight_verification.is_valid to false, explain in summary.
- No prose, comments, or non-JSON content.

RFP JSON: {rfp_requirements}
Proposal Text: {proposal_text}

* MUST FOLLOW THIS RULES:
1) Use ONLY 'Weighted Requirements' criteria for Scoring.
2) EXACT criterion names.
3) Assigned Score ≤ Required Score. if Assigned Score > Required Score then rethink and set proper evaluted     score but never higher the require score.
4) MUST Weights sum to 100%, NEVER More than 100% with combine of all criterin assign score in scored requirements.
5) Score 0 if no evidence, justify clearly.
6) No financial criteria.
"""



format_1 = """
{
    "title": "Proposal Evaluation Report",
    "overview": {
        "rfp_summary": "",
        "evaluation_scope": "",
        "methodology": ""
    },
    "scored_requirements": {
        "evaluation_criteria": [
            {
                "criterion": "",
                "score":,
                "assigned_score":,
                "justification": ""
            }
        ],
        "scoring_summary": {
            "total_technical_score":,
            "pass_fail_status": "",
            "minimum_score_required":,
            "conclusion": "",
            "explanation": ""
        }
    },
    "unscored_requirements": {
        "requirements": [
            {
                "category": "",
                "requirements": [
                    {
                        "name": "",
                        "description": "",
                        "evaluation": {
                            "compliance_status": "",
                        }
                    }
                ],
                "category_assessment": ""
            }
        ]
    },
    "analysis": {
        "strengths": [],
        "concerns": [],
        "risks": []
    },
    "conclusion": {
        "overall_assessment": "",
        "recommendation": "",
        "next_steps": []
    }
}"""

format_2 = {
    "scored_requirements": {
        "evaluation_criteria": [
            {
                "criterion_name": "Name of criterion from RFP",
                "requirements": "Detailed requirements for this criterion from RFP",
                "proposal_compliance": "Detailed assessment of how proposal meets requirements",
                "required_score": "Minimum required score for this criterion",
                "assigned_score": "Actual score assigned to proposal for this criterion",
                "justification": "Detailed explanation with specific examples from proposal"
            }
        ],
        "overall_assessment": {
            "total_technical_score": "Sum of all weighted scores",
            "technical_strengths": [
                "List of key technical strengths identified",
                "With specific references to proposal sections"
            ],
            "technical_weaknesses": [
                "List of key technical weaknesses identified",
                "With specific references to proposal sections"  
            ],
            "summary": "Comprehensive overview of technical evaluation"
        }
    },
    "unscored_requirements": {
        "requirements": [
            {
                "category": "Category or section of unscored requirements",
                "requirements": [
                    {
                        "name": "Name of the unscored requirement",
                        "description": "Detailed description of the requirement from RFP",
                        "evaluation": {
                            "compliance_status": "Status indicating if requirement is met, partially met, or not met",
                        }
                    }
                ],
                "category_assessment": "Overall assessment of compliance for this category of requirements"
            }
        ]
    },
    "analysis": {
        "strengths": ["List of major proposal strengths identified during evaluation"],
        "concerns": ["List of concerns or issues that may impact implementation or success"],
        "risks": ["List of potential risks identified in the proposal that could affect project outcomes"]
    },
    "conclusion": {
        "overall_assessment": "Comprehensive summary of the entire proposal evaluation",
        "recommendation": "Clear recommendation on whether to accept, reject, or request modifications",
        "next_steps": ["List of recommended actions to take following this evaluation"]
    }
}

format_2_arabic = {
    "المتطلبات_المُسجَّلة_بالدرجات": {
        "معايير_التقييم": [
            {
                "اسم_المعيار": "اسم المعيار من RFP",
                "المتطلبات": "المتطلبات التفصيلية لهذا المعيار من RFP",
                "امتثال_الاقتراح": "تقييم تفصيلي لكيفية تلبية الاقتراح للمتطلبات",
                "الدرجة_المطلوبة": "الحد الأدنى للدرجة المطلوبة لهذا المعيار",
                "الدرجة_المخصصة": "الدرجة الفعلية المخصصة للاقتراح لهذا المعيار",
                "التبرير": "تفسير تفصيلي مع أمثلة محددة من الاقتراح"
            }
        ],
        "التقييم_العام": {
            "إجمالي_الدرجة_الفنية": "مجموع جميع الدرجات الموزونة",
            "نقاط_القوة_الفنية": [
                "قائمة بنقاط القوة الفنية الرئيسية المحددة",
                "مع الإشارات المحددة إلى أقسام الاقتراح"
            ],
            "نقاط_الضعف_الفنية": [
                "قائمة بنقاط الضعف الفنية الرئيسية المحددة",
                "مع الإشارات المحددة إلى أقسام الاقتراح"
            ],
            "الملخص": "نظرة شاملة على التقييم الفني"
        }
    },
    "المتطلبات_غير_المُسجَّلة_بالدرجات": {
        "المتطلبات": [
            {
                "الفئة": "فئة أو قسم المتطلبات غير المسجلة بالدرجات",
                "المتطلبات": [
                    {
                        "الاسم": "اسم المتطلب غير المسجل بالدرجات",
                        "الوصف": "وصف تفصيلي للمتطلب من RFP",
                        "تقييم": {
                            "حالة_الامتثال": "حالة توضح ما إذا كان المتطلب ملبى، ملبى جزئيًا، أو غير ملبى"
                        }
                    }
                ],
                "تقييم_الفئة": "تقييم شامل للامتثال لهذه الفئة من المتطلبات"
            }
        ]
    },
    "التحليل": {
        "نقاط_القوة": ["قائمة بنقاط القوة الرئيسية للاقتراح المحددة أثناء التقييم"],
        "المخاوف": ["قائمة بالمخاوف أو المشكلات التي قد تؤثر على التنفيذ أو النجاح"],
        "المخاطر": ["قائمة بالمخاطر المحتملة المحددة في الاقتراح والتي قد تؤثر على نتائج المشروع"]
    },
    "الاستنتاج": {
        "التقييم_العام": "ملخص شامل لتقييم الاقتراح بأكمله",
        "التوصية": "توصية واضحة بشأن قبول أو رفض أو طلب تعديلات",
        "الخطوات_التالية": ["قائمة بالإجراءات الموصى بها بعد هذا التقييم"]
    }
}


JSON_TO_HTML_CONVERTER_PROMPT = """
Your task is to convert the provided JSON data into well-structured HTML format.

Here is the JSON data you'll receive: {json_response}

Generate the HTML content in {language}

Important Instructions:
- Convert all JSON fields into appropriate HTML elements with semantic markup
- When providing Arabic output ensure to use this html tag (<table class="table table-bordered" dir=rtl>, <tr>, <th>, <td>)
- When providing en output ensure to use this html tag (<table class="table table-bordered">, <tr>, <th>, <td>)
- If the language is English, do not add dir="rtl"; only Arabic output should be in RTL format
- Ensure proper nesting and indentation of HTML elements
- Include data attributes where relevant
- Format numbers and scores appropriately
- The HTML output will be injected into an existing div container
- Maintain consistent formatting and style throughout
- Preserve all data from the original JSON
- Output should be in the same language as input
- Do not skip any fields or sections
"""

# Add to prompts.py

LARGE_PROPOSAL_CHUNK_EVAL_PROMPT = """You are an expert proposal evaluator analyzing a large proposal document that has been split into chunks.

This is chunk {chunk_number} of {total_chunks}.

Your task is to evaluate this chunk of the proposal against the RFP requirements provided. Focus only on the content in this chunk.

Proposal chunk text:
{proposal_chunk}

Analyze this chunk against the RFP requirements I'll provide. For each requirement mentioned in this chunk, evaluate compliance and assign appropriate scores.

Remember:
1. Only evaluate content present in this chunk
2. Note which requirements are addressed in this chunk
3. If a requirement is partially addressed, make note of what's missing
4. This is just one part of the full evaluation

Provide the output in {language}
"""
CHUNK_WITH_CONTEXT_EVAL_PROMPT = """You are an expert proposal evaluator analyzing a large proposal in chunks. This is chunk {chunk_number} of {total_chunks}.

CRITICAL SCORING INSTRUCTIONS:
1. Use ONLY the exact evaluation criteria mentioned in the RFP requirements
2. NEVER invent or modify criteria names - use them exactly as specified in the RFP
3. Assigned scores MUST NEVER exceed the required scores for each criterion
4. If a criterion has a required score of X, the maximum assigned score must be X
5. Always document the exact scoring methodology from the RFP for each criterion
6. Ensure all scores are properly weighted according to the RFP specifications
7. Be consistent with previous chunk evaluations

Your task is to evaluate this chunk of the proposal against the RFP requirements, taking into account the evaluations from previous chunks.

Previous chunks evaluation summary:
{previous_evaluations}

For this chunk, focus on:
1. Finding new evidence related to the evaluation criteria
2. Identifying any requirements that weren't addressed in previous chunks
3. Maintaining consistency with previous evaluations
4. Ensuring assigned scores never exceed required scores

Here is the current proposal chunk:
{proposal_chunk}

Ensure that:
- You use ONLY the exact criteria names from the RFP
- Assigned scores never exceed required scores
- You maintain consistency with previous chunk evaluations
- You provide detailed justification with specific references to the proposal

Provide the output in {language}."""

CHUNK_WITH_FULL_CONTEXT_EVAL_PROMPT = """You are an expert proposal evaluator analyzing a large proposal document that has been split into chunks.

This is chunk {chunk_number} of {total_chunks}.

Your task is to analyze this chunk of the proposal against the RFP requirements provided. Focus on collecting evidence rather than making final judgments.

For each requirement mentioned in this chunk:
1. Note which requirements are addressed
2. Collect relevant evidence and quotes
3. Identify strengths and weaknesses
4. DO NOT assign final scores at this stage

Previous chunk evaluation:
{previous_evaluations_json}

Proposal chunk text:
{proposal_chunk}

Provide the output in {language}
"""

FORMAT_OUTPUT_TO_JSON_PROMPT = """You are a data formatting specialist. Your task is to take the evaluation content provided and structure it into a proper JSON format according to the specified schema.

The input is an unstructured evaluation report. Your job is to:
1. Extract all relevant information from the evaluation
2. Structure it according to the JSON schema provided
3. Ensure the output is valid JSON that can be parsed by `json.loads()`
4. Maintain all the original content and scores from the evaluation

Here is the evaluation content to format:
{evaluation_content}

Format your response using exactly this JSON structure:
{json_schema}

Important formatting requirements:
- All keys and string values must be enclosed in double quotes
- There should be no trailing commas after the last item in any object or array
- The JSON structure must be valid and parsable
- Do not include any special characters or formatting indicators like ```json or 'n'
- Ensure all numerical values are properly formatted as numbers without quotes
- The total_technical_score must be scaled to 100 as specified in the evaluation
- All output should be in {language}

Only provide the formatted JSON as your response, without any additional explanation or commentary.
"""

RFP_REQUIREMENT_WITH_CONTEXT_EXTRACTION_PROMPT = """You are an expert specializing in RFP analysis and requirements extraction working with large RFP documents split into chunks.

This is chunk {chunk_number} of {total_chunks}.

Task: Analyze this chunk of the RFP document and extract all requirements, taking into account the requirements already extracted from previous chunks.

Previous chunks extraction summary:
{previous_extractions}

For this chunk, focus on:
1. Identifying new requirements not mentioned in previous chunks
2. Finding additional details for requirements partially covered in previous chunks
3. Noting any modifications or clarifications to previously extracted requirements
4. Maintaining consistency with previous extractions

Extract and list:

1. Requirements that have scoring weights or evaluation criteria (weighted requirements only)
   - Include their relative importance (weights/percentages if specified)
   - Include how they will be evaluated
   - CRITICAL: When extracting from evaluation criteria tables, process EVERY ROW from top to bottom, including the VERY LAST ROW
   - MANDATORY: Extract the EXACT weight values from the table (e.g., 20, 15, 10), not sequential numbers or positions
   - DOUBLE-CHECK: After extraction, verify you have captured ALL rows from the evaluation table, paying special attention to the final row

2. Requirements that must be met (mandatory/unscored)
   - Clearly mark any deal-breakers or elimination criteria
   - Include any pass/fail criteria
   - Include any minimum qualifications needed

3. Any other requirements mentioned in this chunk that a vendor must address
   - Technical specifications
   - Service levels
   - Deliverables
   - Timelines
   - Compliance needs
   - Certifications
   - Any other relevant requirements

Important Guidelines:
- Extract requirements as they appear, maintaining their original context and relationships
- Exclude all pricing and financial details
- Preserve any specific evaluation or verification methods mentioned
- Note dependencies between requirements where they exist
- Include any deadline or timeline-related requirements
- Highlight any critical or deal-breaker requirements
- Focus only on requirements present in this chunk
- Clearly indicate if a requirement appears to continue from a previous chunk or into the next chunk
- Maintain consistency with previous chunk extractions
- SPECIAL ATTENTION: For evaluation criteria tables, ensure ALL rows are extracted, especially the last row which is commonly missed

RFP chunk text:
{rfp_chunk}

Present the requirements from this chunk in a clear, organized manner that maintains their relationships and importance.
Provide the output in {language}."""


COMBINE_RFP_CHUNK_RESULTS_PROMPT = """You are an expert in RFP analysis and requirements extraction, now combining results from multiple chunk extractions into a final comprehensive requirements document.

Your task is to:
1. Combine all requirements extracted from different chunks
2. Eliminate any duplications while preserving all unique details
3. Resolve any inconsistencies between chunk extractions
4. Organize requirements logically by category and importance
5. Ensure the final document reflects the complete RFP

When combining results:
- Merge similar requirements while preserving all unique details
- Resolve any contradictions between chunk extractions
- Ensure the final extraction reflects the complete RFP
- Maintain the original context and relationships between requirements
- Verify that all requirements from individual chunks are included

The final output should:
1. Present all weighted requirements with their scoring criteria and relative importance
2. List all mandatory/unscored requirements with clear identification of deal-breakers
3. Include all other technical, service, compliance, and timeline requirements
4. Be organized in a clear, logical structure

Important Guidelines:
- Exclude all pricing and financial details
- Preserve any specific evaluation or verification methods mentioned
- Note dependencies between requirements where they exist
- Include any deadline or timeline-related requirements
- Highlight any critical or deal-breaker requirements
- Do not include any other information than the requirements
- Don't mention requirements twice
- Avoid repeating the same requirements in different sections
- Maintain consistency in how requirements are described
- Organize requirements logically by category or importance

Present the combined requirements in a clear, organized manner that maintains their relationships and importance.
Provide the output in {language}."""

COMBINE_CHUNK_RESULTS_PROMPT = """You are an expert proposal evaluator combining results from multiple chunk evaluations into a final comprehensive evaluation in {language}.

CRITICAL SCORING INSTRUCTIONS:

RFP Requirements: {rfp_text}

TAKE RFP REQUIREMENTS TO EVALUATE THE PROPER SCORE:
1. Use ONLY the exact evaluation criteria listed in the RFP's 'Weighted Requirements'.
2. NEVER invent, rename, or modify criteria names—use them EXACTLY as specified.
3. Assigned scores MUST NEVER exceed the required scores for each criterion (e.g., if required_score is X, assigned_score ≤ X).
4. If a criterion lacks evidence, assign 0 and justify with 'No evidence provided'.
5. Ensure weights sum EXACTLY to 100%.
6. Re-evaluate scores using the proposal text and RFP requirements to resolve chunk inconsistencies.

Your task:
- Combine evidence from all chunks for each criterion.
- Resolve inconsistencies in scores across chunks by re-evaluating against RFP requirements.
- Ensure assigned_score ≤ required_score for every criterion.
- Provide detailed justifications referencing the proposal text.
- Calculate the total_technical_score accurately.

Scoring Process:
1. Extract criteria, required_score, and weight from RFP 'Weighted Requirements'.
2. For each criterion in chunk results:
   - Verify criterion exists in RFP.
   - Cap assigned_score at required_score.
   - If assigned_score exceeds required_score or is inconsistent, reassign based on proposal evidence and RFP.
   - Provide justification citing proposal text.
3. Validate TWICE:
   - assigned_score ≤ required_score.
   - Weights sum to 100%.
4. Calculate total_technical_score:
   - total_raw_score = sum(assigned_score for all criteria).
   - max_possible_score = sum(required_score for all criteria).
   - total_technical_score = MIN(100, MAX(0, (total_raw_score / max_possible_score) * 100)).

Output:
- JSON matching the schema: {format}.
- Include all RFP criteria, even if not in chunks.
- Detailed justifications for each score.
- total_technical_score between 0 and 100.

Strict Rules:
- EXACT criterion names and Required score from RFP.
- assigned_score ≤ required_score.
- Weights = 100%.
- Validate scores TWICE before outputting."""

