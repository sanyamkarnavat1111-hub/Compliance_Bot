from main import RFPCompleteness

rfp_completeness = RFPCompleteness()
id, result, error_user, error_tech = rfp_completeness.is_complete(
    id="1",
    model="openai",
    rfp_url="https://compliancebotai.blob.core.windows.net/compliancebotdev/rfp/document/67b5be9c749273GORz1739964060.pdf",
    ea_standard_eval_url="https://compliancebotai.blob.core.windows.net/compliancebotdev/ministry/document/6747f7dd5c51cgK33W1732769757.docx",
    output_language="english"
)