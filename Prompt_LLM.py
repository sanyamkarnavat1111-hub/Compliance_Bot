import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm(model_provider: str):
    """Initialize and return the appropriate LLM based on model_provider."""
    if model_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://api.openai.com/v1"
        )
    elif model_provider == "opensource":
        return ChatOpenAI(
            model="qwen/qwen3-32b",
            # model="qwen/qwen2.5-vl-32b-instruct",
            # model= "deepseek/deepseek-r1-distill-llama-70b",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            # openai_api_key="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",
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0
        )
    else:
        raise ValueError(f"Invalid model_provider: {model_provider}")
    
# standard_English_refiner_prompt = PromptTemplate(
#     input_variables=["standard_text"],
#     template="""
#     You are an expert in Language Translation Specific in Arabic to english. Your task is to JUST convert the give Standard Document into English Language.
    
#     # Instruction :
#     - JUst convert the Standard Document into English Language.
#     - Standard Docoment have sensitive information in form of : Section, Lines, Tables, Bullet point, without Bullet point, Percentage wise, markdown, without markdown, etc. [Completely analyse the Standard Document]
#     - Now convert the Standard Document into English Language. [IN AS IT IS FORMATE]
#     - Do not add or remove any information from the Standard Document.
    
#     1. Carefully examine the entire content (including text, tables, sections, and bullet points).
#     2. Identify **every unique concept, technology, platform, protocol, system, architecture, tool, or specification** mentioned in the document — even if it appears only once or inside another context.
      
#     NOTE* : 
#     - Follow same structure with proper understanding of Standard Document, tables,section,bullet point,sections,pages,etc.
#     -carefully translate with context meaning what used by Original standard.
#     - DO not miss any single info to MENTION IN DESCRIPTION AND TOPIC [ include every page within section,title,topic,higlights,Tables ].
#     - Never add any outside word from the standard document or your own word.just translate the Standard Document into English Language.
    
#     Standard Document:
#     {standard_text}
#   """
# )

standard_English_refiner_prompt = PromptTemplate(
  input_variables=["standard_text"],
  template="""
  You are an expert in Language Translation Specific in Arabic to english. Your task is to JUST convert the give Standard Document into English Language.
  
  Instruction :
  - JUst convert the Standard Document into English Language.
  - Standard Docoment have sensitive information in form of : Section, Lines, Tables, Bullet point, without Bullet point, Percentage wise, markdown, without markdown, etc. [Completely analyse the Standard Document]
  - Now convert the Standard Document into English Language. [IN AS IT IS FORMATE]
  
  
  Standard Document:
  {standard_text}
  
  NOTE* 
  - DO not miss any single info to MENTION IN DESCRIPTION AND TOPIC [ include every page within section,title,topic,higlights,Tables ].
  - Do not add your own word or remove any information from the Standard Document.just translate all presented word in the Standard Document to English Language.
  """
)

# REact:

# Answer the following Instructions as best you can. You have access to the Standard Document:

        # Use the following format:

        # Question: Consider whole Standard Document and the Instructions,what is the answer to the MY INSTRUCTION?
        # Thought: you should always think about what to do
        # Action: the action to take, USE STANDARD DOCUMENT to answer the Instructions,
        
        
        # Action Input: Use the Action Input to answer the Instructions.
        # Observation: the result of the action, IS ALL Standard Document INFO ARE CONVERED TO THE ACTION INPUT? IF NOT ASK AGAIN. and then repeat the process until you have all the information OF Standard Document and the Instructions are covered in the Action Input.
        # ... (this Thought/Action/Action Input/Observation can repeat N times)
        # Thought: I now know the final answer
        # Final Answer: the final answer to the original input Instructions

        # Begin!
        
# Task 1 Prompt: Extract Topics
task1_prompt = PromptTemplate(
    input_variables=["language", "max_iterations", "standard_text"],
    template="""
    You are an expert in RFP analysis with deep knowledge of standard documents. Your task is to analyze the standard document and extract a comprehensive list of section topics with detailed descriptions for RFP evaluation.   

    Instructions:
      - Analyze the standard document thoroughly from top to bottom, capturing ALL relevant details, including technical, operational, and emerging requirements ETC..
      - Standard Document can Contain Theory, Paragraphs, section, or Tables form of INFO, you have to analyse all this things in perfect manner.
      
      - After analysing the standard document, extract each and every section that would be Section TOPIC.
      - TOPICS can be every Main Focus things, Strictly mentioned or Mentioned with Requirements.
      - Do not miss any section or topic, even if it seems minor or implicit.
      - Ensure to capture all technical terms, requirements, and specifications mentioned in the standard.
    
      # Identify ALL relevant section topics:
        - Core technical areas (e.g., system, storage, framework, operating system, network, security, database, API, cloud, infrastructure, hardware, software, specifications, performance metrics, percentages,Licenses and every things related to it or mentioned in the standard).
        - Emerging or cross-cutting themes (e.g., sustainability, data governance, accessibility, interoperability, or other non-technical but critical requirements implied or stated in the standard).
        
      # For each topic, provide a concise description that:
        - Includes ALL relevant terms from the standard(ensuring no term is missed).
        - Potential discrepancies or ambiguities in the standard that could affect RFP alignment.
        - Exclude all financial, pricing, and cost-related content from topics and descriptions.
      
    # CRITICAL: Output MUST be a JSON list of objects in English ONLY, with each object containing "topic" and "description" fields. 
    - Ensure descriptions are specific, avoiding vague or generic statements
    - Return the JSON list without markdown code blocks (e.g., ```json).
    - Retry up to {max_iterations} times if the output is incomplete, inaccurate, or misses critical topics.
    - Standard Document can contain Theory, Paragraphs, section, pages and Tables form of INFO also it can be in mixed form, you have to analyse all this things in perfect manner. Need to Understand first whole Info of Standard Document. Then start making TOPIC and DESCRIPTION.
    
    Standard Document:
    {standard_text}
    
    NOTE* 
    - DO not miss any single info to MENTION IN DESCRIPTION AND TOPIC. EVERY SINGLE WORD IS IMPORTANT.
    - Mentioned Properlly In description LIke for what reason it mentioned in Standard with There Actully usecase whateve mentioned in Standard Document.
    - Cover EVERY detail in the Standard Document to be in TOPIC AND DESCRIPTION.
    - Do not add your own things in TOPIC AND DESCRIPTION. just used from what Standard Document Mentioned Info ONLY.
    - Don't change even meaning of Standard document, You have to translate it with same concept,keyword,meaning,section,list,bullet point,table.you dont have any right to miss any information or give the wrong information in translated english text.
    - Recheck and Confirm before you FInalise your answer. 
    - For each topic, provide a concise description in English that includes all relevant terms,  whole Response must be in English Language only for both Topic and Description.
    """
)

# from langchain.prompts import PromptTemplate

# task1_prompt = PromptTemplate(
#     input_variables=["language", "standard_text_chunk", "uncovered_lines", "iteration"],
#     template="""
# You are an expert in RFP analysis with deep knowledge of standard documents. Your task is to analyze the provided standard document chunk and extract a comprehensive list of section topics with detailed descriptions for RFP evaluation. The output must be EXCLUSIVELY in {language}, cover EVERY detail in the chunk, and contain NO information not explicitly stated in the chunk.

# # Instructions:
# - Analyze the provided document chunk thoroughly, capturing ALL details, including technical, operational, security, architectural, and other requirements in text, tables, bullet points, or other formats.
# - Convert the chunk into a JSON list of objects, each with "topic" and "description" fields in {language} ONLY.
# - **Topic Definition**: Each topic must correspond to a specific requirement, section, or concept (e.g., a table row, bullet point, or sentence). Topics must be granular (e.g., separate encryption from network configurations).
# - **Description**: Quote or paraphrase the chunk exactly, including ALL relevant details. Do NOT summarize, omit details, or add speculative information (e.g., no references to platforms like Azure unless stated).
# - **Coverage**: Ensure every line in the chunk is represented in at least one topic or description. Use the provided uncovered lines to prioritize missing content.
# - **No Hallucinations**: Output must contain only information explicitly stated in the chunk. Exclude any inferred or external details.
# - **Table Handling**: Create a topic for each table row or logical group, capturing all table content verbatim.
# - **JSON Formatting**: Produce valid JSON with proper Arabic encoding (UTF-8). Every object MUST have both "topic" and "description" fields with non-empty strings. Avoid trailing commas, invalid characters, or syntax errors.
# - **Iteration Focus**: This is iteration {iteration}. Prioritize uncovered lines to fill gaps from previous iterations.

# # ReAct Process:
# Question: What is the complete list of topics and descriptions for the document chunk?
# Thought: Analyze the chunk line by line, identifying each requirement as a topic with a detailed description.
# Action: Extract topics and descriptions, focusing on uncovered lines.
# Action Input: Use the document chunk and uncovered lines below.
# Observation: Verify all chunk content is covered. Identify missed lines.
# ... (Repeat until all content is covered)
# Thought: All chunk content is covered with accurate topic-description pairs.
# Final Answer: A JSON list of topics and descriptions in {language}.

# Document Chunk:
# {standard_text_chunk}

# Uncovered Lines (prioritize these):
# {uncovered_lines}
# """
# )

# Task 2 Prompt: Intent Check

task2_intent_prompt =  PromptTemplate(
    input_variables=["dynamic_topic", "existing_topics"],
    template="""
    You are an expert in RFP analysis. Your task is to determine if the dynamic topic is a duplicate of any existing topic based on name, meaning, and semantic intent, or Redudant. Output a JSON object with a single key "is_duplicate" set to true if the dynamic topic matches an existing topic, false otherwise.

    Instructions:
    - Language might be Arabic or English so carefully understand the Symentic meanings then take action, to sure determine if the dynamic topic is a duplicate of any existing topic based on name, meaning, and semantic intent, or Redudant.
    - IF "Duplicate" means *same concept*, even if the wording is different (e.g. "Database Systems" and "Supported Databases", "Project Delivery'" and "Project Deliverables" are duplicates).
    - "Redundant" means the topic is already covered under a broader topic.
    - You must compare by **topic name**, and **semantic intent**, not just exact words.
    # Steps:
    1. Compare the topic name and meaning of dynamic_topic against the list below.
    2. The existing_topics list may contain Arabic or English names. Analyze semantically and cross-lingually.
    3. If the topic name or meaning already exists in the list, even with different wording or language, it is a duplicate.
    - Compare the dynamic topic '{dynamic_topic}' with the Existing Topics (name): {existing_topics}.
    # Think, Observe and take action. Before you finalize the answer rethink about your desissions and take actions.
    - Consider the topic name, description, and semantic intent, if it match any of this then Return: {{"is_duplicate": true}} otherwise return  Return: {{"is_duplicate": false}}
    """
)

# task2_intent_prompt = PromptTemplate(
#     input_variables=["dynamic_topic", "existing_topics", "dynamic_topic_description"],
#     template="""
#     You are an expert in RFP analysis. Your task is to determine if the dynamic topic is a duplicate or redundant compared to any existing topic based on name, meaning, semantic intent, and description. Output a JSON object with a single key "is_duplicate" set to true if the dynamic topic matches an existing topic, false otherwise.

#     Instructions:
#     - Topics may be in Arabic or English. Perform robust cross-lingual semantic analysis to compare meanings, accounting for synonyms and translations (e.g., "قواعد البيانات" == "Databases").
#     - A "duplicate" means the dynamic topic represents the *same concept* as an existing topic, even with different wording (e.g., "Database Systems" and "Supported Databases" are duplicates).
#     - A "redundant" topic is fully covered by a broader existing topic (e.g., "Accepted Databases" under "Databases and Technical Configurations").
#     - Specific technical requirements (e.g., "Accepted Operating System") should be treated as distinct unless explicitly covered by an existing topic’s name or intent.
#     - Compare both the topic name and the provided description (if any) to ensure accurate semantic matching.
#     - If no description is provided for the dynamic topic, rely on the topic name and inferred intent, but prioritize precision to avoid false positives.

#     Steps:
#     1. Translate and normalize the dynamic topic '{dynamic_topic}' and existing topics {existing_topics} to a common semantic space (e.g., map Arabic terms to English equivalents).
#     2. Compare the dynamic topic name, inferred intent, against each existing topic’s name and implied intent.
#     3. If the dynamic topic’s concept or scope is fully covered by an existing topic (e.g., as a subset or synonym), mark it as duplicate.
#     4. If the dynamic topic represents a specific requirement not explicitly addressed in existing topics, mark it as unique.
#     5. Re-evaluate the comparison to ensure no false positives or negatives occur, especially for specific technical terms.

#     Example:
#     - Dynamic Topic: "Accepted Database", Existing Topic: "قواعد البيانات والإعدادات الفنية" → Duplicate (same concept: databases).
#     - Dynamic Topic: "Preferred Application Frameworks", Existing Topic: None → Unique (no overlap).
#     - Dynamic Topic: "SSL Certificate", Existing Topic: "الأمن السيبراني" → Duplicate (SSL is a subset of cybersecurity).

#     Input:
#     - Dynamic Topic: '{dynamic_topic}'
#     - Existing Topics: {existing_topics}

#     Output:
#     Return: {{"is_duplicate": true}} if the topic is a duplicate or redundant, otherwise {{"is_duplicate": false}}.
#     """
# )


# 2nd way to add The fixed sections: 


task2__intent_prompt =  PromptTemplate(
    input_variables=["dynamic_topic", "existing_topics"],
    template="""
    You are an expert in RFP analysis. Your task is to determine if the dynamic topic is a duplicate of any existing topic based on name, meaning, and semantic intent, or Redudant. Output a JSON object with a single key "is_duplicate" set to true if the dynamic topic matches an existing topic, false otherwise.

    Instructions:
    - Language might be Arabic or English so carefully understand the Symentic meanings then take action, to sure determine if the dynamic topic is a duplicate of any existing topic based on name, meaning, and semantic intent, or Redudant.
    - IF "Duplicate" means *same concept*, even if the wording is different (e.g. "Database Systems" and "Supported Databases", "Project Delivery'" and "Project Deliverables" are duplicates).
    - "Redundant" means the topic is already covered under a broader topic.
    - You must compare by **topic name**, and **semantic intent**, not just exact words.
    # Steps:
    1. Compare the topic name and meaning of dynamic_topic against the list below.
    2. The existing_topics list may contain Arabic or English names. Analyze semantically and cross-lingually.
    3. If the topic name or meaning already exists in the list, even with different wording or language, it is a duplicate.
    - Compare the dynamic topic '{dynamic_topic}' with the Existing Topics (name): {existing_topics}.
    # Think, Observe and take action. Before you finalize the answer rethink about your desissions and take actions.
    - Consider the topic name, description, and semantic intent, if it match any of this then Return: List which are unique, do not add anything else 
    """
)

# Task 2 Prompt: Description Generation
task2_description_prompt = PromptTemplate(
    input_variables=["topic", "language"],
    template="""
    You are an expert in RFP analysis. Generate a concise description (50-100 words) for the topic '{topic}' in english. 
    
    - The description should include relevant technical terms (e.g., system, storage, framework, operating system, network, security, database, API, cloud, infrastructure) and focus on its role in RFP evaluation. 
    - See proper given **standard Document** to generate the description.
    - Ensure the description is specific to the topic cover ALL INFORMATION WHICH MENTIONED IN STANDARD DOCUMENT.
    - No single thinks missed, and no technical term should be left out [INSHORT COVER ALL THE INFO RELATE THE TOPIC].
    - The description should highlight specific requirements, potential gaps, or alignment needs with enterprise or entity-specific priorities.
    - Exclude financial, pricing, or cost-related content. Output a JSON object: {{"topic": "{topic}", "description": "<description>"}}.

    Used standard Document to generate the Description:
    {standard_text}

    Instructions:
    - Ensure the description is specific ONLY to the topic's intent, Not include any other aspects.
    - Strictly output in English language, No matter what main keywords are they, Just Generate everythings from every little word to big main keyword in English language.
    """
)

# Task 2 Prompt: Translation (for Arabic)
task2_translation_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    Translate the topic '{topic}' to Arabic only. Return only the translated text.
    """
)

# Task 3 Prompt: Generate HTML - Original
# task3_prompt = PromptTemplate(
#     input_variables=["topic", "description", "rfp_context", "language"],
#     template="""
#     You are an expert in RFP analysis. Your task is to evaluate the RFP against the topic's description for the given section topic. The output must be in {language} and formatted as an HTML table row with two columns: "Elements" (left) and "Proposed Improvements" (right).

#     Instructions:
#     - Analyze the RFP context thoroughly, ensuring EVERY word relevant to the Description, and MUST BE ACCURATELLY MENTIONED.
#     - Do not miss any technical term from RFP which are relevant to the topic Description.[Strickliy analyse all thing which are relevant to the topic description]
#     - The topic '{topic}' is and Description is {description}.
#     - Use the topic description as the standard requirement.
#     - Generate a detailed evaluation in the following format:
#       <tr>
#         <td>{topic}</td>
#         <td>
#           <b>Detailed Example:</b><br>
#           <b>Standard Requirement</b>: Quote the topic description<br>
          
#           <b>Current RFP Status</b>:
          
#             Generated base on Given[TOPIC AND THERE DESCRIPTION]

#             - If Topic is more then one Word-meaning then try to understand both of them and then analyse the RFP context.
            
#             - Describe ALL relevant content from the RFP context which are aligns with standard TOPIC and there description. Makesure to answer Every sentance of the description. Some times description may contain more then one sentance or different meaning also there, so you have to analyse the RFP context based on that.
            
#             - For every topics : Describe what things ACCEPT BY THE RFP CONTEXT based on the topic.

#             - Describe WHat is mentioned in RFP context are aligns with the topic and description, with proper naming conventions and technical terms(do not missed to mention Properlly). Doesn't matter is same match with the Standard but show what is mentioned in RFP context.
            
#             - If any gaps, then describe with proper word what exactlly mentioned In RFP and what Standard Tell, and if any misalignments then describe 
            
#             - Before you tell "Yes this is Exist" or "No this is not exist" In RFP context. You have to RETHINK and REANALYSE the RFP context Proper context.
          
#           <b>Impact</b>: Explain implications of gaps or misalignments<br>
          
#           <b>Recommended Improvement</b>: Provide detailed solutions to address gaps<br>
          
#           <b>Implementation Notes</b>: Include specific steps or considerations
#         </td>
#       </tr>
#     - Ensure no RFP content is missed; include all technical terms (e.g., system, storage, framework, operating system, network, security, database, API, cloud, infrastructure).
#     - Just return HTML Content only, do not add any other text like : ```html and first and last ```.
#     - Strictly output in {language} ONly.
#     - Return the HTML row content only.

#     Topic Description (Standard Requirement):
#     {description}

#     RFP Context (Complete Content):
#     {rfp_context}
#     """
# )

#OLd and production 1 Json Prompt
from langchain_core.prompts import PromptTemplate

# task3_prompt = PromptTemplate(
#     input_variables=["topic", "description", "rfp_context", "language"],
#     template="""
#     You are an expert in RFP analysis and compliance evaluation. Your task is to evaluate the RFP against the topic's description for the given section topic. The output must be EXCLUSIVELY in {language} language and formatted as a JSON object with the following fields: topic, standard_requirement, rfp_status, impact, recommended_improvement, implementation_notes. Do NOT generate Chinese characters or non-{language} text.

#     **Instructions**:
#     - Analyze the RFP context thoroughly, ensuring EVERY relevant word from the Description is addressed.
#     - Do not miss any technical term from the RFP context relevant to the topic description (e.g., system, storage, framework, operating system, network, security, database, API, cloud, infrastructure).
#     - The topic is '{topic}' and the description is '{description}'.
#     - Use the topic description as the standard requirement.
#     - Generate a detailed evaluation in the following JSON format:
#       {{
#         "topic": "{topic}",
#         "standard_requirement": "Quote the topic description",
#         "rfp_status": "Detailed analysis of RFP alignment",
#         "impact": "Implications of gaps or misalignments",
#         "recommended_improvement": "Detailed solutions to address gaps",
#         "implementation_notes": "Specific steps or considerations"
#       }}
#     - For `rfp_status`:
#       - If the topic has multiple meanings, analyze all relevant aspects.
#       - Describe ALL relevant content from the RFP context that aligns with the topic and description, using proper naming conventions and technical terms.
#       - For each sentence in the description, evaluate whether the RFP context meets it.
#       - List what the RFP context mentions, even if it partially aligns.
#       - Identify gaps or misalignments, comparing the RFP content to the standard.
#       - Before stating alignment (e.g., "exists" or "does not exist"), re-analyze the RFP context.
#     - Ensure all fields are non-empty and relevant.
#     - Return only the JSON content, without markdown code blocks (e.g., ```json).

#     **Topic Description (Standard Requirement)**:
#     {description}

#     **RFP Context (Complete Content)**:
#     {rfp_context}
    
#     NOTE: The generated output contains {language} language text only, No any single word in other language.every thinks in {language} only. 
    
#     - Must Generated every single word in value of Json key in {language} language only.
#     - No matter what main keywords are they, Just Generate everythings from every little word to big main keyword in {language} language.
#     """
# ) 

task3_prompt = PromptTemplate(
    input_variables=["topic", "description", "rfp_context", "language"],
    template="""
    You are an expert in RFP analysis and compliance evaluation. Your task is to evaluate the RFP against the topic's description for the given section topic. The output must be EXCLUSIVELY in {language} language for all fields including the topic and description, which MUST use the exact input topic '{topic}' verbatim. The output must be a JSON object with the following fields: topic, standard_requirement, rfp_status, impact, recommended_improvement, implementation_notes. 

    **Instructions**:
    - Analyze the RFP context thoroughly, ensuring EVERY relevant word from the Description is addressed.
    - Do not miss any technical term from the RFP context relevant to the topic description (e.g., system, storage, framework, operating system, network, security, database, API, cloud, infrastructure).
    - The topic is '{topic}' and the description is '{description}'.
    - Use the topic description as the standard requirement.
    - Generate a detailed evaluation in the following JSON format:
      {{
        "topic": "topic",
        "standard_requirement": "Quote the topic description",
        "rfp_status": "Detailed analysis of RFP alignment",
        "impact": "Implications of gaps or misalignments",
        "recommended_improvement": "Detailed solutions to address gaps",
        "implementation_notes": "Specific steps or considerations"
      }}
    
    - For `rfp_status`:
      - If the topic has multiple meanings, analyze all relevant aspects.
      - Describe ALL relevant content from the RFP context that aligns with the topic and description, using proper naming conventions and technical terms.
      - For each sentence in the description, evaluate whether the RFP context meets it.
      - List what the RFP context mentions, even if it partially aligns.
      - Identify gaps or misalignments, comparing the RFP content to the standard.
      - Before stating alignment (e.g., "exists" or "does not exist"), re-analyze the RFP context.
    - Ensure all fields are non-empty and relevant.
    - Return only the JSON content, without markdown code blocks (e.g., ```json).
    

    **Topic Description (Standard Requirement)**:
    {description}

    **RFP Context (Complete Content)**:
    {rfp_context}
    
    NOTE: -always generate topic and description field in {language} language Only. The input might be in english but if it is in {language} language then it must be in {language} language only.. Every word in value of JSON keys (except topic) must be in {language} language.
    """
)


task3_prompt_3 = PromptTemplate(
    input_variables=["topic", "description", "rfp_context", "language"],
    template="""
    You are an expert in RFP analysis and compliance evaluation. Your task is to evaluate the RFP against the topic's description for the given section topic. The output must be EXCLUSIVELY in English language for all fields including the topic and description, which MUST use the exact input topic '{topic}' verbatim. The output must be a JSON object with the following fields: topic, standard_requirement, rfp_status, impact, recommended_improvement, implementation_notes. 

    **Instructions**:
    - Analyze the RFP context thoroughly, ensuring EVERY relevant word from the Description is addressed.
    - Do not miss any technical term from the RFP context relevant to the topic description (e.g., system, storage, framework, operating system, network, security, database, API, cloud, infrastructure).
    - The topic is '{topic}' and the description is '{description}'.
    - Use the topic description as the standard requirement.
    - Generate a detailed evaluation in the following JSON format:
      {{
        "topic": "topic",
        "standard_requirement": "Quote the topic description",
        "rfp_status": "Detailed analysis of RFP alignment",
        "impact": "Implications of gaps or misalignments",
        "recommended_improvement": "Detailed solutions to address gaps",
        "implementation_notes": "Specific steps or considerations"
      }}
    
    - For `rfp_status`:
      - If the topic has multiple meanings, analyze all relevant aspects.
      - Describe ALL relevant content from the RFP context that aligns with the topic and description, using proper naming conventions and technical terms.
      - For each sentence in the description, evaluate whether the RFP context meets it.
      - List what the RFP context mentions, even if it partially aligns.
      - Identify gaps or misalignments, comparing the RFP content to the standard.
      - Before stating alignment (e.g., "exists" or "does not exist"), re-analyze the RFP context.
    - Ensure all fields are non-empty and relevant.
    - Return only the JSON content, without markdown code blocks (e.g., ```json).
    

    **Topic Description (Standard Requirement)**:
    {description}

    **RFP Context (Complete Content)**:
    {rfp_context}
    
    NOTE: -always generate topic and description field in english language Only. Every word in value of JSON keys (except topic) must be in english language.
    """
)

task3_translation_prompt = PromptTemplate(
    input_variables=["json_text"],
    template="""
    You are an expert in translation from English or arabic to Arabic. Your task is to translate the provided JSON object to Arabic, ensuring all fields of Json contain only Arabic text. Preserve the exact meaning, technical terms, context, and JSON structure.

    Instructions:
    - Translate all fields, including 'topic', to Arabic using standard technical terms. Use the following mappings for common terms:
      - 'system' → 'نظام'
      - 'database' → 'قاعدة بيانات'
      - 'operating system' → 'نظام التشغيل'
      - 'network' → 'شبكة'
      - 'security' → 'أمان'
      - 'API' → 'واجهة برمجة التطبيقات'
      - 'cloud' → 'سحابة'
      - 'infrastructure' → 'بنية تحتية'
      - 'storage' → 'تخزين'
      - 'framework' → 'إطار عمل'
      
      - Manage original key in Object you can't chnage:
        {{
        "topic": 
        "standard_requirement": 
        "rfp_status": 
        "impact": 
        "recommended_improvement":
        "implementation_notes": 
      }}
      this key must be same in Arabic and English in Object.
    - For terms not listed, use contextually appropriate Arabic translations.
    - Preserve the JSON structure, including all keys and their order, with no trailing commas or invalid syntax.
    - Ensure the output contains only Arabic text (characters in the range U+0600–U+06FF, plus spaces, digits, and punctuation).
    - Handle special characters, formatting, and punctuation correctly to produce valid JSON.
    - Return only the translated JSON object as a string, without markdown code blocks (e.g., ```json).
    - If any field cannot be translated accurately, use the most contextually appropriate Arabic term and log a warning.


    NOTE: Your main focus must on only translate the input Json into Completely Arabic Language.with proper meaning and formate. No any single word in other language. 
    
    Input JSON:
    {json_text}
    
    - all things must be in "ARABIC LANGUAGE" only.
"""
)


task3_strict_translation_prompt = PromptTemplate(
    input_variables=["json_text"],
    template="""
    You are an expert in translation, specializing in converting English to Arabic. Your task is to translate any English words in the provided JSON object to Arabic, ensuring ALL fields are in Arabic. Preserve technical terms (e.g., Microsoft SQL Server, MySQL) as proper nouns or transliterate them (e.g., MySQL → ماي إس كيو إل). The topic field must remain unchanged.

    Instructions:
    - Identify and translate ALL English words in the JSON to Arabic, ensuring natural and contextually appropriate language.
    - Preserve globally recognized proper nouns (e.g., Microsoft, Oracle) as-is or transliterate technical terms (e.g., MySQL → ماي إس كيو إل, CosmosDB → كوزموس دي بي).
    - Do not modify the JSON structure or the topic field content.
    - Retain any Arabic text as-is.
    - Output only the JSON object, without markdown code blocks (e.g., ```json).

    JSON Input:
    {json_text}
    
    NOTE*s
    - you Expert in Stricklly converter given Json object language to Arabic language.
    - input Json may have mix language of Arabic and english, your task is to convert whole Json object language to Arabic language, with 100% garantee work.
    - Ensure all fields (topic, standard_requirement, rfp_status, impact, recommended_improvement, implementation_notes) are generate in Arabic language only, No matter what input is, except for preserved proper nouns or transliterated terms.
    """
) 