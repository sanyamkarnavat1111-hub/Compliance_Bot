import base64
import json
import hmac
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
# Assuming llog is imported globally or available

# PKCS7 padding functions (from ED_testing.py)
def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    """Apply PKCS7 padding to data."""
    padding_len = block_size - (len(data) % block_size)
    return data + bytes([padding_len]) * padding_len

def pkcs7_unpad(data: bytes) -> bytes:
    """Remove PKCS7 padding from data and validate padding consistency."""
    if not data:
        raise ValueError("Invalid PKCS7 padding: empty input.")
    padding_len = data[-1]
    if padding_len < 1 or padding_len > 16:  # Block size is 16 for AES
        raise ValueError("Invalid PKCS7 padding length.")
    if len(data) < padding_len:
        raise ValueError("Invalid PKCS7 padding: block too short.")
    # Validate that all padding bytes match padding_len
    if data[-padding_len:] != bytes([padding_len]) * padding_len:
        raise ValueError("Invalid PKCS7 padding bytes.")
    return data[:-padding_len]


def _compute_laravel_mac(key: bytes, iv_b64: str, value_b64: str) -> str:
    """Compute Laravel-compatible HMAC (hex) over iv_b64 + value_b64 using the raw key bytes."""
    mac = hmac.new(key, (iv_b64 + value_b64).encode("utf-8"), hashlib.sha256).hexdigest()
    return mac

# ... existing code ...
def laravel_decrypt_raw(encrypted_value: str, app_key_base64: str) -> str:
    """Decrypt a Laravel-encrypted string (Crypt::encryptString)."""
    key = base64.b64decode(app_key_base64.strip())
    if len(key) not in (16, 24, 32):
        raise ValueError("Invalid Laravel APP_KEY length after base64 decode (expected 16/24/32 bytes).")

    print(f"DEBUG: Encrypted value (raw): {encrypted_value}")
    try:
        try:
            decoded_payload_bytes = base64.b64decode(encrypted_value)
        except Exception:
            # Fallback: maybe it's raw JSON already
            decoded_payload_bytes = encrypted_value.encode("utf-8")
        print(f"DEBUG: Decoded payload (bytes): {decoded_payload_bytes}")
        payload = json.loads(decoded_payload_bytes)
        print(f"DEBUG: Parsed payload (dict): {payload}")
    except json.JSONDecodeError as jde:
        # Explicitly print the problematic bytes if JSON decoding fails
        print(f"ERROR: JSON Decode Error: {jde}")
        print(f"ERROR: Attempted to decode these bytes as JSON: {decoded_payload_bytes}")
        raise ValueError(f"Failed to decode JSON from encrypted value: {jde}")
    except Exception as e:
        print(f"ERROR: Unexpected error during payload decoding: {e}")
        raise ValueError(f"Unexpected error during payload decoding: {e}")

    iv_b64 = payload.get('iv')
    value_b64 = payload.get('value')
    mac_hex = payload.get('mac')

    if not isinstance(iv_b64, str) or not isinstance(value_b64, str):
        raise ValueError("Invalid payload: 'iv' and 'value' must be strings.")

    # If a MAC is present, verify it just like Laravel does
    if mac_hex is not None and mac_hex != "":
        expected_mac = _compute_laravel_mac(key, iv_b64, value_b64)
        if not hmac.compare_digest(mac_hex, expected_mac):
            raise ValueError("Invalid MAC: payload may be tampered or key is incorrect.")

    iv = base64.b64decode(iv_b64)
    ciphertext = base64.b64decode(value_b64)

    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)
    print(f"DEBUG: Raw decrypted bytes before unpadding: {decrypted}")

    return pkcs7_unpad(decrypted).decode("utf-8")


def laravel_encrypt_raw(plain_text: str, app_key_base64: str) -> str:
    """Encrypt a string so Laravel can decrypt with Crypt::decryptString."""
    key = base64.b64decode(app_key_base64.strip())
    if len(key) not in (16, 24, 32):
        raise ValueError("Invalid Laravel APP_KEY length after base64 decode (expected 16/24/32 bytes).")

    # Step 1: Generate random IV (16 bytes for AES-256-CBC)
    iv = get_random_bytes(16)

    # Step 2: Pad plaintext
    padded = pkcs7_pad(plain_text.encode("utf-8"))

    # Step 3: Encrypt
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(padded)

    # Step 4: Build payload (same as Laravel). Laravel includes a MAC (HMAC-SHA256 hex)
    iv_b64 = base64.b64encode(iv).decode("utf-8")
    value_b64 = base64.b64encode(ciphertext).decode("utf-8")
    mac_hex = _compute_laravel_mac(key, iv_b64, value_b64)

    payload = {
        "iv": iv_b64,
        "value": value_b64,
        "mac": mac_hex,
    }

    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


class EncryptionUtil:
    """
    Utility class for encrypting and decrypting data using Laravel's AES-256-CBC encryption scheme.
    """
    def __init__(self, app_key: str):
        """
        Initializes the EncryptionUtil with the Laravel APP_KEY.

        Args:
            app_key (str): The Laravel application key, typically prefixed with "base64:".
        """
        self.laravel_app_key_base64 = None
        app_key = (app_key or "").strip()
        if app_key.startswith("base64:"):
            self.laravel_app_key_base64 = app_key[len("base64:"):].strip()
        else:
            # If not prefixed, assume it's directly the base64 encoded key
            self.laravel_app_key_base64 = app_key.strip()
            # raise ValueError("APP_KEY must be prefixed with \"base64:\" for Laravel encryption.")

        if not self.laravel_app_key_base64:
            raise ValueError("Laravel APP_KEY is empty or invalid after parsing.")

    def encrypt_text(self, plaintext: str) -> str:
        """
        Encrypts a plaintext string using Laravel's encryption scheme.

        Args:
            plaintext (str): The string to encrypt.

        Returns:
            str: The base64 encoded JSON payload (Laravel encrypted string).
        """
        if not self.laravel_app_key_base64:
            raise RuntimeError("Laravel APP_KEY not set for encryption.")
        try:
            return laravel_encrypt_raw(plaintext, self.laravel_app_key_base64)
        except Exception as e:
            # llog if available
            raise ValueError(f"Laravel encryption failed: {e}")

    def decrypt_text(self, encrypted_text: str) -> str:
        """
        Decrypts a Laravel encrypted string using the stored Laravel APP_KEY.

        Args:
            encrypted_text (str): The base64 encoded JSON payload from Laravel.

        Returns:
            str: The decrypted plaintext string.

        Raises:
            ValueError: If decryption fails (e.g., invalid token, incorrect key, or Laravel key not set).\n
        """
        if not self.laravel_app_key_base64:
            raise RuntimeError("Laravel APP_KEY not set in EncryptionUtil for decryption.")

        try:
            # llog("EncryptionUtil", f"Attempting Laravel decryption: {encrypted_text}", "encryption_log")
            decrypted_string = laravel_decrypt_raw(encrypted_text, self.laravel_app_key_base64)
            # llog("EncryptionUtil", "Laravel decryption successful.", "encryption_log")
            return decrypted_string
        except Exception as e:
            # llog("EncryptionUtil", f"Laravel decryption failed: {e}", "encryption_log")
            raise ValueError(f"Laravel decryption failed for \'{encrypted_text}\': {e}")


if __name__ == "__main__":

    # Example usage with the client-provided Laravel key
    # APP_KEY_LARAVEL = "jzMfjr8K14jZQDkobv7+zZnfoVSfSxkzWAr1acSQoXw="
    APP_KEY="base64:jzMfjr8K14jZQDkobv7+zZnfoVSfSxkzWAr1acSQoXw="
    # TEST_STRING = "Hello from Python!"
    TEST_STRING = """ <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <title>Compliance Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .category-header {
            background-color: #3498db;
            color: white;
            padding: 15px;
            margin: 20px 0 10px 0;
            border-radius: 5px;
            font-size: 18px;
            font-weight: bold;
            text-transform: uppercase;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            table-layout: fixed;
        }
        th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            word-wrap: break-word;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e8f4f8;
        }
        .status-met {
            background-color: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .status-not-met {
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .status-partially-met {
            background-color: #fff3cd;
            color: #856404;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .priority-high, .priority-عالي, .priority-عالية {
            background-color: #dc3545;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }
        .priority-medium, .priority-متوسط, .priority-متوسطة {
            background-color: #fd7e14;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }
        .priority-low, .priority-منخفض, .priority-منخفضة {
            background-color: #28a745;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }
        .recommendation-cell {
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding-left: 16px;
        }
        /* RTL specific styles */
        [dir="rtl"] th {
            text-align: right;
        }
        [dir="rtl"] .recommendation-cell {
            border-left: none;
            border-right: 4px solid #007bff;
            padding-left: 12px;
            padding-right: 16px;
        }
    </style>
</head>

    <body>
        <div class="container">
            <h1>EA Standard Compliance Analysis Report</h1>
            <table>
                <thead>
                    <tr>
                        <th style="width: 20%;">EA Requirement</th>
                        <th style="width: 10%;">Status</th>
                        <th style="width: 25%;">RFP Coverage</th>
                        <th style="width: 20%;">Gap Analysis</th>
                    </tr>
                </thead>
                <tbody>
    
            <tr>
                <td>Bidder eligibility criteria: licenses, certifications, financial capacity</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP includes references to licenses (e.g., Saudi Contractors Authority certificate) and financial capacity (e.g., company capital, payment terms, bank account details, and shareholder financial information). However, there is no explicit mention of certifications as a requirement for bidder eligibility.</td>
                <td>The EA standard explicitly requires certifications as part of bidder eligibility criteria. While licenses and financial capacity are addressed in the RFP, certifications are not mentioned or required. The RFP does not clarify if certifications (e.g., ISO, industry-specific) are mandatory for bidders.</td>
            </tr>
        
            <tr>
                <td>Post-implementation support</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP includes technical support services (e.g., &#x27;Support and technical assistance services for information technology units&#x27;) and project management (e.g., &#x27;PMI methodology for project management&#x27;). It also references ongoing system maintenance and coordination with stakeholders for project continuity.</td>
                <td>The EA standard requires structured post-implementation support with defined timelines and accountability. The RFP implicitly addresses support but does not specify duration, response times, or measurable outcomes (e.g., SLAs). For example, while &#x27;technical support services&#x27; are mentioned, there is no explicit timeline or performance criteria for post-implementation activities.</td>
            </tr>
        
            <tr>
                <td>Independent quality assurance audits</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP references financial audits (e.g., &#x27;auditors must be appointed by the general assembly of shareholders&#x27; in Article 10-7) but does not mention independent audits for project deliverables, system performance, or IT processes.</td>
                <td>The EA standard mandates independent audits for quality assurance, distinct from financial compliance. The RFP&#x27;s focus on financial auditing (e.g., &#x27;reviewing financial statements&#x27;) does not satisfy this requirement. There is no mention of third-party audits for IT systems, deliverables, or operational processes.</td>
            </tr>
        
            <tr>
                <td>Supported operating systems: Windows Server, Linux (RHEL/Ubuntu), and virtualization platforms (VMware, Hyper-V)</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content does not explicitly mention any operating systems, virtualization platforms, or related infrastructure requirements. The focus is on company governance, financial terms, project management, and service scope without technical infrastructure details.</td>
                <td>The EA standard explicitly requires the inclusion of approved operating systems (Windows Server, Linux [RHEL/Ubuntu]) and virtualization platforms (VMware, Hyper-V). The RFP does not address these requirements, leaving a critical gap in technical infrastructure specification.</td>
            </tr>
        
            <tr>
                <td>Governance framework development</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP includes references to board of directors with defined roles (e.g., chairman, members from government/private sector), shareholders&#x27; meetings, and project management structures (e.g., project manager as communication point). It also outlines decision-making processes for capital changes, audits, and operational approvals.</td>
                <td>While governance roles and decision-making processes are described, the RFP lacks an explicit governance framework definition. Key elements like escalation paths, formal governance policies, and accountability structures for IT operations are not clearly articulated. The board&#x27;s responsibilities are detailed but not framed as a cohesive governance framework.</td>
            </tr>
        
            <tr>
                <td>IT operating model</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP describes operational support for shared services systems (financial, HR, procurement, etc.), project management for system enhancements, and continuity of business operations. It also mentions service delivery through a shared services ERP system and operational milestones.</td>
                <td>The IT operating model is implied through shared services system operations and project management structures but is not explicitly defined. Organizational alignment, service delivery models, and governance policies for IT operations are not formally outlined in the RFP.</td>
            </tr>
        
            <tr>
                <td>Evaluation criteria: technical first, then financial</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP includes separate technical and financial offers as required (e.g., &#x27;Separate Technical Offer&#x27; and &#x27;Separate Financial Offer&#x27; are listed as required documents for participating companies). The project also includes financial details (e.g., total value, payment milestones, VAT calculations).</td>
                <td>The RFP does not explicitly state that technical criteria are evaluated before financial criteria. While both are present, the sequence of evaluation is not clearly defined in the RFP content. The EA standard mandates a specific order (technical first, then financial), which is missing in the RFP.</td>
            </tr>
        
            <tr>
                <td>Database support: Oracle, SQL Server, PostgreSQL, MySQL</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content does not explicitly mention the use of any database technologies (Oracle, SQL Server, PostgreSQL, or MySQL) in the context of the project&#x27;s technical requirements or infrastructure. The focus of the RFP is on ERP system operations, financial management, and administrative services, with no direct reference to database systems.</td>
                <td>The EA standard explicitly requires the use of one of the four specified databases (Oracle, SQL Server, PostgreSQL, MySQL). The RFP does not address this requirement by naming any of the approved options, leaving a critical gap in compliance.</td>
            </tr>
        
            <tr>
                <td>Requirement for bid bonds (initial and final guarantees)</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes legal and operational details about company management, share transfers, board of directors, and financial policies, but there is no explicit mention of bid bonds, initial guarantees, or final guarantees.</td>
                <td>The EA standard explicitly requires bid bonds (initial and final guarantees) to be addressed in the RFP. However, the provided RFP chunks do not reference bid bonds, guarantees, or any related financial security mechanisms for bids or contracts.</td>
            </tr>
        
            <tr>
                <td>Virtual Machine (VM) scalability and high availability setup</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes general references to &#x27;basic information technology infrastructure services&#x27; and discusses system operations for financial accounting, procurement, and human resources. However, it does not explicitly address VM scalability mechanisms (e.g., auto-scaling, load balancing) or high availability architectures (e.g., clustering, failover).</td>
                <td>The EA standard explicitly mandates VM scalability and high availability as mandatory requirements. The RFP lacks specific technical requirements for virtual machine infrastructure, scalability configurations, or redundancy strategies. While the RFP mentions IT infrastructure, it does not define VM-related capabilities required by the EA standard.</td>
            </tr>
        
            <tr>
                <td>Confidentiality of submitted proposals</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP document asserts confidentiality for itself, stating it is &#x27;exclusively for the use of the Ministry of Culture&#x27; and prohibits unauthorized disclosure. It also includes a copyright notice and restrictions on reproduction. However, there is no explicit mention of confidentiality obligations for submitted proposals or their handling post-submission.</td>
                <td>The RFP addresses confidentiality for its own document but does not extend these protections to submitted proposals. There is no clause specifying secure handling, storage, or destruction requirements for proposals after submission, nor does it clarify confidentiality obligations for evaluators or third parties involved in the process.</td>
            </tr>
        
            <tr>
                <td>Ownership of submitted proposals</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP asserts copyright ownership by Tamkeen Technologies for the document itself, prohibiting unauthorized use or distribution. It also specifies that the document is for internal use by the Ministry of Culture under an agreement. However, there is no mention of ownership rights for intellectual property (IP) created in submitted proposals.</td>
                <td>The RFP establishes ownership of the RFP document but does not address ownership of IP in submitted proposals. There is no clause clarifying whether the procuring entity (e.g., Ministry of Culture) retains rights to proposal content, such as ideas, designs, or technical solutions.</td>
            </tr>
        
            <tr>
                <td>Knowledge transfer through TOGAF training and workshops</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes general project management methodology (PMI) and system operation support but does not explicitly mention TOGAF training, workshops, or structured knowledge transfer activities aligned with TOGAF standards.</td>
                <td>The EA standard explicitly requires knowledge transfer via TOGAF training and workshops. The RFP does not reference TOGAF, training programs, or workshops for knowledge transfer. While the RFP mentions project management and system operations, these are unrelated to the EA-mandated TOGAF-based knowledge transfer requirements.</td>
            </tr>
        
            <tr>
                <td>Activation and operation of Enterprise Architecture (EA) frameworks</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes operational details about ERP systems, shared services, and IT project management (e.g., financial accounting systems, procurement systems, human resources management systems). However, there is no explicit mention of Enterprise Architecture (EA) frameworks, their activation, or operational processes.</td>
                <td>The EA standard explicitly requires activation and operation of EA frameworks, but the RFP does not address this requirement. While the RFP references IT systems and project management, these are operational implementations rather than EA frameworks (e.g., TOGAF, Zachman, etc.). No EA framework activation or operational methodology is described.</td>
            </tr>
        
            <tr>
                <td>Availability for remote sessions in Jeddah and Dammam</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP does not mention any provision for remote sessions or availability in Jeddah or Dammam. All references to the company&#x27;s operations focus on its Riyadh headquarters.</td>
                <td>The EA standard explicitly requires availability for remote sessions in Jeddah and Dammam, but the RFP contains no information about remote capabilities or geographic availability beyond Riyadh.</td>
            </tr>
        
            <tr>
                <td>Arabic and English language proficiency for communication and documentation</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes company descriptions, service offerings, administrative details, and qualification criteria but does not explicitly state requirements for Arabic and English language proficiency for communication and documentation.</td>
                <td>The EA standard explicitly mandates Arabic and English language proficiency for communication and documentation. The RFP does not mention this requirement in any section, including company profiles, service descriptions, qualification criteria, or contractual obligations. While Arabic is implied as the primary language in Saudi Arabia, English proficiency is not addressed at all.</td>
            </tr>
        
            <tr>
                <td>Capability to provide 24/7 support during implementation and transition</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes general references to system maintenance, service continuity, and operational support for systems like financial accounting, procurement, and human resources. It mentions &#x27;ensuring business continuity by maintaining the current system&#x27; and &#x27;enhancing and developing the existing services.&#x27; However, there is no explicit or implicit mention of 24/7 support during implementation or transition phases.</td>
                <td>The EA standard explicitly requires 24/7 support during implementation and transition. The RFP does not address this requirement directly. While it references service continuity and system operations, it lacks language confirming round-the-clock availability, staffing, or response times during these critical phases. No keywords like &#x27;24/7,&#x27; &#x27;round-the-clock,&#x27; or &#x27;continuous support&#x27; are present in the context of implementation or transition.</td>
            </tr>
        
            <tr>
                <td>Digital transformation alignment with EA principles</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP mentions the Ministry of Culture&#x27;s alignment with Vision 2030 and the National Transformation Program, and references the project&#x27;s objective to implement a system meeting functional requirements for financial and administrative affairs. It also notes the company&#x27;s role in supporting cultural and economic initiatives through technical solutions.</td>
                <td>While the RFP discusses digital transformation initiatives (e.g., Vision 2030, ERP system extension), it does not explicitly state how these efforts align with EA principles (e.g., architecture frameworks, governance, strategic integration). The EA standard requires direct linkage to EA principles, which is absent in the RFP content.</td>
            </tr>
        
            <tr>
                <td>Escalation and dispute resolution mechanism defined in contract</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP includes general contractual terms (e.g., penalties for delays, bank guarantees, communication details) but does not explicitly define an escalation or dispute resolution mechanism in the contract. There is no dedicated section or description outlining procedures for resolving disputes or escalating issues.</td>
                <td>The EA standard explicitly requires a defined escalation and dispute resolution mechanism in the contract. The RFP lacks any structured process, steps, or roles for addressing disputes (e.g., mediation, arbitration, escalation paths, timelines). While contact information for Tamkeen Technologies personnel is provided, this does not constitute a formal mechanism as required.</td>
            </tr>
        
            <tr>
                <td>Minimum 5-10 years of experience in Enterprise Architecture projects</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP mentions the company&#x27;s establishment in 2013 and &#x27;proven experience in similar projects,&#x27; but does not explicitly specify that resources must have 5-10 years of experience in Enterprise Architecture projects. The focus is on company history, ERP system operations, and general IT services rather than individual experience duration in EA.</td>
                <td>The EA standard explicitly requires a minimum of 5-10 years of experience in Enterprise Architecture projects for resources. The RFP does not address this specific experience duration requirement for EA roles or team members. While the company has a 10-year operational history, this does not equate to individual resource experience requirements as mandated by the EA standard.</td>
            </tr>
        
            <tr>
                <td>Proven track record of at least 3 similar government-level projects</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP explicitly mentions one government-level project: the Ministry of Culture&#x27;s Shared Services ERP System - Phase Two (two-month extension). It also references Tamkeen Technologies&#x27; general involvement with public sector entities like the Ministry of Labor, Human Resources Development Fund, and Saudi Contractors Authority. However, no additional specific government projects beyond the ERP system are detailed in the RFP content.</td>
                <td>The EA standard mandates at least three distinct government-level projects. While the RFP provides one explicit project (Ministry of Culture ERP system) and implies broader public sector engagement, it lacks documentation of two additional specific government projects with similar scope and scale to satisfy the &#x27;at least 3&#x27; requirement.</td>
            </tr>
        
            <tr>
                <td>Compliance with quality and safety standards</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP includes compliance with statutory obligations such as VAT registration, social insurance, Zakat compliance, and certifications from Saudi authorities (e.g., Ministry of Labor, General Authority of Zakat &amp; Tax). It also references project goals to &#x27;improve service quality&#x27; and &#x27;raise the level of services and data.&#x27;</td>
                <td>The EA standard requires general compliance with quality and safety standards, but the RFP does not explicitly state adherence to formal quality/safety frameworks (e.g., ISO standards) or technical safety protocols. While statutory compliance is documented, there is no direct alignment with the EA standard&#x27;s phrasing of &#x27;quality and safety standards,&#x27; leaving a gap in explicit acknowledgment of the requirement.</td>
            </tr>
        
            <tr>
                <td>Integration with emerging technologies (AI, IoT, RPA, VR)</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content focuses on ERP system operations, financial management, procurement, human resources systems, and infrastructure services. It mentions mobile application development and system enhancements but does not explicitly reference AI, IoT, RPA, or VR.</td>
                <td>The RFP does not address any of the explicitly listed emerging technologies (AI, IoT, RPA, VR) in the EA standard. While the RFP discusses system enhancements and mobile applications, these are not tied to the specified emerging technologies.</td>
            </tr>
        
            <tr>
                <td>Configuration of EA toolsets</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP mentions system adjustments and service enhancements (e.g., improving financial accounting, human resources, and procurement systems) but lacks specifics on configuration management processes like version control, rollback strategies, or parameter settings.</td>
                <td>Configuration management practices (e.g., version control, rollback strategies) are not explicitly addressed, creating a potential compliance gap.</td>
            </tr>
        
            <tr>
                <td>Cloud compatibility: Microsoft Azure, AWS, or local government-approved cloud services</td>
                <td><span class="status-partially-met">Partially Met</span></td>
                <td>The RFP references cloud environment usage in the context of the Ministry of Culture&#x27;s ERP system extension project, stating that the proposal must comply with functional requirements &#x27;either at the Ministry&#x27;s headquarters or on the cloud environment.&#x27; This indicates general acknowledgment of cloud compatibility but does not explicitly name Microsoft Azure, AWS, or a local government-approved cloud service.</td>
                <td>The RFP does not explicitly specify which cloud service (Microsoft Azure, AWS, or local government-approved) is being used. While cloud environment usage is referenced, the absence of named approved options creates ambiguity about compliance with the EA standard&#x27;s explicit list of permitted services.</td>
            </tr>
        
            <tr>
                <td>Use of approved programming languages: Java, .NET, Python, JavaScript (for integration)</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content does not explicitly mention the use of any programming languages (Java, .NET, Python, or JavaScript) in the context of system development, integration, or operations. The technical proposal and scope of work focus on system operations, enhancements, and administrative tasks without specifying programming language requirements.</td>
                <td>The EA standard explicitly requires the use of approved programming languages (Java, .NET, Python, JavaScript). The RFP does not address this requirement, leaving no evidence that the project will utilize any of the listed languages. Non-approved languages are not mentioned, but the absence of approved options constitutes a gap.</td>
            </tr>
        
            <tr>
                <td>Integration with enterprise directory services (Active Directory, LDAP)</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content does not explicitly mention integration with Active Directory or LDAP. It discusses ERP system operations, financial management, and project management methodologies (e.g., PMI), but no directory integration requirements are specified.</td>
                <td>The EA standard explicitly requires integration with enterprise directory services (Active Directory or LDAP). The RFP does not address this requirement or specify the use of any approved directory service.</td>
            </tr>
        
            <tr>
                <td>Sustainability initiatives (Green IT, energy efficiency)</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes company policies, financial details, organizational structure, and an ERP system extension project. However, there is no explicit mention of sustainability initiatives, Green IT practices, or energy efficiency measures in the provided chunks.</td>
                <td>The EA standard explicitly requires addressing sustainability initiatives (Green IT and energy efficiency). The RFP does not include any references to these requirements, such as energy-efficient IT infrastructure, sustainable resource management, or Green IT strategies.</td>
            </tr>
        
            <tr>
                <td>Cybersecurity compliance with ISO 27001</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP mentions general information security services (e.g., &#x27;information security solutions&#x27;) but does not explicitly reference ISO 27001 compliance.</td>
                <td>The RFP lacks explicit ISO 27001 requirements. No direct mention of &#x27;ISO 27001&#x27; or its controls (e.g., risk management, access control policies) is found in the RFP text.</td>
            </tr>
        
            <tr>
                <td>Cybersecurity compliance with NIST</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP references IT services and project management methodologies (e.g., PMI) but does not mention NIST compliance or its Cybersecurity Framework (CSF).</td>
                <td>The RFP lacks explicit NIST references. No alignment with NIST CSF core functions (Identify, Protect, Detect, Respond, Recover) or specific NIST SP 800-series controls is evident.</td>
            </tr>
        
            <tr>
                <td>Mandatory use of SSL certificates</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP mentions &#x27;information security solutions&#x27; and &#x27;technical solutions and information security services&#x27; but does not explicitly reference SSL certificates. No mention of TLS/SSL in data transmission or API security sections.</td>
                <td>SSL certificates are explicitly required by the EA standard but are not mentioned in the RFP. General security references do not satisfy the specific requirement.</td>
            </tr>
        
            <tr>
                <td>Mandatory use of AES-256 encryption</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP references &#x27;information security solutions&#x27; and &#x27;software development&#x27; but does not specify AES-256 encryption. No mention of encryption standards for data at rest or in transit.</td>
                <td>AES-256 encryption is explicitly required by the EA standard but is not mentioned in the RFP. General security references do not satisfy the specific requirement.</td>
            </tr>
        
            <tr>
                <td>Mandatory use of secure protocols</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP includes vague terms like &#x27;secure protocols&#x27; in the context of &#x27;technical solutions&#x27; but does not define or specify them. No mention of protocols like HTTPS, SFTP, or SSH.</td>
                <td>The EA standard requires &#x27;secure protocols&#x27; as a mandatory item, but the RFP lacks explicit mention of this requirement. Vague references to &#x27;security&#x27; are insufficient.</td>
            </tr>
        
            <tr>
                <td>Service Level Agreements (SLAs) including 99.9% uptime</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes administrative, financial, and project management details (e.g., company certifications, payment terms, project duration, PMI methodology). However, there is no explicit mention of Service Level Agreements (SLAs) or a 99.9% uptime requirement in any section of the provided RFP chunks.</td>
                <td>The EA standard explicitly mandates the inclusion of 99.9% uptime in SLAs. The RFP does not address this requirement, as no reference to SLAs, uptime guarantees, or performance metrics related to system availability is present in the content. This omission creates a direct gap against the EA standard.</td>
            </tr>
        
            <tr>
                <td>Business Continuity &amp; Disaster Recovery - RTO 4h, RPO 30m</td>
                <td><span class="status-not-met">Not Met</span></td>
                <td>The RFP content includes general corporate governance, financial management, and IT service descriptions (e.g., ERP system operations, shared services), but no explicit mention of RTO/RPO metrics, recovery time objectives, or data retention/recovery requirements.</td>
                <td>The EA standard explicitly requires RTO 4h and RPO 30m, but the RFP does not address these specific recovery time and data loss thresholds. While IT system operations are mentioned (e.g., ERP, financial systems), there is no quantified requirement for system availability, data backup frequency, or disaster recovery timelines.</td>
            </tr>
        
                </tbody>
            </table>
        </div>
    </body>
    </html>"""

    try:
        print("Main_EncryptionUtil", "Testing Laravel Encryption/Decryption")
        encryptor_laravel = EncryptionUtil(APP_KEY)
        print("Main_EncryptionUtil", "EncryptionUtil (Laravel) initialized successfully.")

        # encrypted_laravel_test = encryptor_laravel.encrypt_text(TEST_STRING)
        # print(f"🔒 Encrypted for Laravel: {encrypted_laravel_test}")

        encrypted_laravel_test = "eyJpdiI6ICJOMms2T1pKeWhqVFd2M0wvZ0tZTS9BPT0iLCAidmFsdWUiOiAiOGRoRE0wTVRlRFU3eWpNV0Z5VG50dlUxcEt2WEplRDRhUWM1RlM4d0tSYjk3WGJBdjJMdmF6MXpQRlBjOWZsU1cvUDFORm5RbkE0aEpaaGIyaTBmRzVzdGM3bHgvbFRCUG4vM052UFFJcWNnd2RGY3piYXlEL2xwaTlXemlWU3BHVkN5OEZQVGhpdVI4S3RUTkk3YkF0YzVHZkhnZjlHSTBQUDlIRCtVR0JTNCtGZGZKaE4vTitNVFViQjY5UCs4b0JqV2w3ZVRzdjArSG9IYXkxamYvQ3hwaGkrcHpzMDM5b2xkaFlzejFZZDljbXNzU2RMT3MxUUt2dGZhZHU1ejhBbXlGc2pkYWFaR1JkSk96YTlhVkFyeEY4Ymd4TFBnYVhoNys5QkdVb1BQYWFrMituTFFJUXI0WmhzZXRMb2xSZDFMUDdGOXdObFdHT1JOTERQVFpqSjVJQ1hCcGZLeXFUOEUraitWbytMUGt3WDhHa1FnM0drVm43aHc4SytpMDlVNnI3V1RHclR1eU5EZEhyOG1yRTBtMFhoczhXNVMrNEZuelNsNmxnWkVsbjNvdDFYV1Mzc20xdVNGZFFLOG15L3o5QVFlT29GcW5yRUVWZnJEMlVyalpKcWVOSnIraFpvL2R0WUVyazU2QmhUY1NvWjBVY1VJWXU3ejJnRXFWdnVGZ29UaGEzWUIxdTdmRktkaENza0ZiR1NxL0hEQlMzTm5TV3V1a0hua2txSFFRaGxZVjNCemQ1aXZvMmNvSDVIRjBsMEVYdFd1aG9UdDdhWEdIOUV0UTFlcHlROHNGZGZ4aW9NdVpZdGxnRGExTW1xa01kYmM2ZExuUEFZTjhCOUxNYXlJNnJlZ1cxc3VYd1ZRT3BWSXhVRTVYTmdlRHpIS0oxcGNpVjgzaGl2VXlCT0UyTitCcnRHcFNoRjNWSk5LdGp0aG5PK3pTQzlBS2xJWGRoS3VtMExvbXJ4Rm1xcGZoczk0VkJVT1ZPVEJRWEk1b2tuYTRjMlJWakZkREtlVTZad205azJaK0luYUhXbVA3WTJzc2I5NWs2WnJyVW5XTkUzMWZ5OCtiS09EY3NtOGhlQWpQM1dYWDlFczhHYjIydVgwVUJ0cUtVNitQMU1HaXBQQ3BvMGRWeGttRm80ZC9STXlERW5uN3VUMWl0Y3FCbGkxMjF2bkV2MXdsNTR0MW54ang0UlRFckZjVUcxdjlZZGpmYnZSTWRiZ3lqL0lud2FxcVB4Ry9JMHdTeUVmbUFpRCs3SDJib2thd2JoM28zVE55VkREZWR4SExUY0JJaXRVNWFlZmw3MXlEZFMrcCtJdzdXNExORGlKOGE5dkFRZHo5MG5DRk9ROUFVd0d5SXFmOXFwOXd1N3hEblN3TzJQdUNUMm54am5rT0ZobW9BUkszOXgrRGRQN2xBbWxad3JFSjMwS2lXOWg5RlJlK1VWRDk4aFBpM1U3TVhPU0svbDdmbkN2M3M0M3dsdTdJWWxnMmVKZUpiaXUvWERCb1lCYWNpRU95VjF5Tk0wQ0p1cXR3QU9TU0xVZWNxOU1qVnJkSzdZaXFlYVY3ZE43bVlaTldIMDEzaldKbEdTN3oySUZTTE93bDc1VXUvaU5wU1V4WGdBL05MNzVRUHJWbVZsSTB2Y1NsT0VnOEJGbDFpZXNKS2FPcXg4b3F3M3gzSnduUlNuZk1Nam1SREtobldjMUlIK2pTNC9YeUxYVjRkZlBhbWFwMFExdytreC8vUTNFTklVeUtvcCtMc2RuaTFJVWV5S0g3TjQ2b29sZ0t1eTBhdlhNYlp1cFdUYzg1Q2orUDdVdTdVMGMxMUpBNDY2T0RETjdHN21kbGY0VDVkN3NGOHNaaWlxSGxqbUY2MlVQbVpQV2NkbmwwQ3huNHBuRFRiSE9mOHdWWHBwa3N1c0VIVkhlUUFQbENOb3grRHpZSXpPZjYxV0tTa3N5U2Znay9vdHJaY3Z6UEkzOTJyTGlLVktURFVEamVQTjJwVlVQRnQxdm1kZWt6ZVF0UjRQM3d4U3A3T0VEY1piaEhBaXprbGxQeTRsZlZueC9qTmJPMXR0RFlaeWVHZWZwVCtVSDl6S2JMVXVnQ0N1VVNlTi91WVlXWlZNTGZ6OHB0ZThSUzhWMmVYcmtxVGQrMGhUUFBnT1p2Q3dKOGVMUnBoeklCdm43dUJIL042QXA0VGVQWXVvcDJXNjAxTWRNTHMrSmZDQjI0Vnhzd2hoMVFiSFV0S0d5U1czMzZ0K3pDYnVLSUJBWFdXOVQ5TGJTUHB0R3UxVEZwZitvN0JkMXloQmlNT0s4aU9mbHN5YmxLTExOZ2RFcURQUXQ1Q2RIVTZodU4wRHIxaTlmWitoRlJ0b1QrRWN5c2o5NXgxWmw2N21TMC9oSWNFbmVoTzdsRkQwQnNvNnI4Y05za3VhTXI0ZVlzbllZei9ub2V1YTd6NzdkcEdqM0JDeXVadVpXeDB5SkxPNWNybnhyOFppbU1ic0xtVFliUTJteUhNckwrSFZCVXVyMC9rVTM2NmUzNzRIZnJTRXFkWkM2ZUtrWU5iWDB3RnEvM0pGZnJ2QXhwQU56ZnZaZ2tkWlhWNUloZkxBUzJmc2RnMHNzQzAwRGJWM1BSbkFZQ1FSV05NbmFjM0lYZGJRMGZudm04eGpidzNUc1p5KzY3SHJTckJlcG4xcGhKS2hDNnZCdlZPbjlac1VmK2s4eGtDR1pLWGpta1VYWXFWTm1xVzk4Q2w4cWl6eUdrMmEvWXV4dGJtV3NhbFdvNnhZMlpiNXBxWjArQWlSbVpQSCtoemdqZENzRURFR3RGQ0xSZ2l0SjQ3UlBrMjRMekhEYnlwZEc1REsxcmFEdzZWa0pqMzF3akc0Y2E0ajJYMkVkUHBpSUl6YkIwQktPc2djL1orQ05VcGMwL2dqTlNFVTY5ZlJETllNM21lcGNJNnhsbDkrR2E1d2U3Q1NjL0Zaa0JNZWxNSXpyQ3Z5azM1dzJvWHQybkh6cVlxSXkwckRiTVhMdlNSUXovUERNQlVZUFN3bGdzZUhaMWFmV2Y2c0xNZitZTlhHWDVOMGNmUmsxU0pEcTczMDg3MmdWYkZhNmdQUWhuZWU5WDROZjFCRVFocllyUGtsa2V0SnVNZG95QkZQYUh5OG9HdjBDbk9SNkRvTUh5T3R4UUhaUXpoeGloc08zREE1akxabC9Idmp3ZEhhZ0Z3cXNCa2RJNHdtZTVjZUZJWU1nbThMV1UxS2FiZ2pFYm91UFlSczM3bWg4ME5EOFMwenk0ZnZXUEk0NHhrTG9YZnNvT3JUWGFGNjVZRnBCeHNOWmkxRGpWa2owOHQwbWlERGRXdkl0NEVKbVEyZTBqQWlEV2pkM2pHSndtU0RWcXIyczhmQkZJL3FnZk0vSW9RMWJjbVZwWVJYVTF0eTNYaXpHODFlVzFvTXNrcExWSXFSTldEUENqS010eEpwMzFXSHl5a2hndmp0bnVjVm1ZWUJJZ2d5TWVuV2FaM0IvbmxOK3VBcmpJWmdIT2RyUkpNZkRwWXFmMk53MkJpMmRtbzJxS0tWS21wb0FFU2JxSTRrRG1PNzdaUGxHZGNuNXl2UTVNVXhsYS9TRTUxOStzMGZHNUVBYnY5ZDVLVi9EWHk3WGRBTXhIT25NdnNFa1VuOEtIdmNuc2pBZDVaMytLUzJkaVpRbXFNbG1DSXNFS0ovOHdDaGw1OXc2dWFHcWlualJHblV1c1BSQnlwMFlsbk5hSkJ0L1NubGdkWWRqckMrcXY3N0NTRUhGTDR1N1Q0SlJ5Z09CTko5UGNGeWtvS3hpa0o3RWZtVUxYZXZSczBCU213dlFuZVplSCtaeUF5bDhwLzBaeU9aRDNCTjYwdThsQmlzRzF1VkVac2R1aXZMdldSSUcrSTdiVlNaWWVDcGhXOVpkUHZaWFdJYkFlNWtiSGdoVG9odFdoU0hXVlhuemVMeUpuSGNZZGNwZUExTllGWmVCcENqZ0RxNEY2NG1LbG5BMVJ1Q25UVm5FVStkOU9NYXJuUjIxMDY4cUhhd21KTzlDMkl4b2RsemNGVGg5Y1R4cklTcUFkUml0dFFwMkwxMWJDY1dBZVF2Z2YrM0RxMjErVkhOUlJHbXRaZ2VZd3NpMW9mWm5oc1JxVzFIU0xiTHk1T2JodzFUQzM0cnhrRlU1eWNhNitmdi9NbmxMcjVIaUp4OHcvR2NRaE54ZVhMRTJUU2pZUUpWWUtQWlZjVUtEakxIWnVnc2JzanZwcVhrM3RkSEtmQjZ2QW1GbFVycUFrLy94d0FMZnBsTVd2SHVSSmxCR1JoSllrZ01Rczh0bXFWM3d2U0NEWTdOK21yZDY2czF2TEJDMHh2SjQ4RnZ6Mk1aY1UzcFJXV1RrTW16OHV1YU50c3ppMjZBblptU2h3VTdmeGRxWVQwOGRrcFNlUXM3MTFFZ3lDaC9iZmE3ODlzZTlmd2xLMXpZVDczTlVldGtUd3NBdXVTRlQ3am5VWTZJTjYrYjZyQThIV2VGTmgyZzAweForSHFhNWp0K0hQdWRzN056NWZodExHSFZkRU5HRGhrUHNFNXlicjNCTlBXSEdwQk9qY3NnVVZyQ1J2SVJ1bld5TDVsUjd4SDFCeDhxYUVTUk5TUW1MTmtwbFZDOHI5cWZBdFhlRTQ5Wll5RmpsVHdiM1ZWOGxaelZJNHRWRkFqekpQWkhLWWdzWTlEeFNvU0t6U0NWUWRJMUk1QmI3aGNQRU0ycklJOUxFRElqeGlqZTV4TVdta0I5WEUxVW9MVW4rZHhWOWxCOTg4R1ozMlZhM0ZDbnZBalU0bjQzRnE4U3p0cm1JK0lYbENRVzRSQ0JSOC8zeTNrajB5bmZVanMvUk5OaklNMm8vd3JiNkVmc0wyVGNBa2xWSmNsYTYzS0ZlSFFBbHZLNGZOSm9LNmxDdUd2Qml6MVViRUp5eU1wZ0NRZFJ0Qm5FbDZPZXVrMjh6MmM4aUF5UkJoYVdRM1Y3bkNjLzA5T0VoUWo5ZFozRlZZY3BBSzJrNnJwMmFhWWZkUFpQNzk0NEZ3MVRoNHdsRVJoZWM3aFBOY3p2U3JyOEdpTTRrY0paMzdsNEI0bU50WVNBQU9YdFFKSGhVc1lOQVd4YmFCOXlWRjRNNXJQa3EwbUxnKzZqMXVDT3N5Vm1QUldhZXJnN29WQkhIWW9JaGJxZEI2YXovakVBU2ZBTXhNVlQySnhtbU5UZFhYY0ROVnRxS1A2amJySjZUNkxCWFAxQXl4WmtzK2ZJSnBGZDhmS2d2UXVIeFY2YndDcTJmOWhHaWozVXNxdzQ1cEpXeUpnMnhnK245UERGWTlrazBoQVFNUmk4SURPZHZBSXliTFBNU09uSHlGb1JxOUptME9mQ0kwWUdjK0UrS24rdDNCdUx2WFI2NVU4N1VrV294L0tVMlQySUdrOEE0NnJtbXNJaVVwaE83ZnY5ZnVaV29ObjZleXBGQTVDMzF5SFd3ZzV2NnIzc2FOeUNKZ0sweXZYekhXbFVhRFJibGZBMjFpRGJ3Y01KcSt1R3JwRlJNOGlnZGoweDQvcWJHcTVVVDdKaStORlgvK294V1NwejZCWTkrcUh4a0dJK1lnQXd1UGIydWptdFFvYTdscDdvdXlQUEFFMjM4ajZsWlJRMVVKQ1d1VEJlckQwdlJsa05nV2xLVHYxcEI2TERzcHhRTjhNbHVVanBlUGdVYUptaWZxNHhxYzYrdzA4UnA0YTRHWkk4Yk9CK0grbnJhN3ljdGU2NlpMaEJBbXYwN0VHUmpnSHpFR05CVHB0aU1KUklXSVdTRXNCc1l3TU9qSlZtQTB3TmRzbDY3SCtiNkpXK2VCMmFWQUpYRnhhWE4rNkJ1SWQwY2RVeXB1Yk4yWWFCL0ZhK1RMU3Q1L0w2bWsvTGwrTmtiWUMrSFBxNDF6SkNMdnRpdHV5UzNyQWJnZTZoT3dZVHI5akpUZW5SWW5iSWdnL2N3Mk51QUR2TUlEZzNSVHIxcVBEZFpIRTZXTUdMRWxoNnBOQ1ZGSXBmb3JQbi85VTVFNXlid3ZXQmxlYWtIOUJlVmNlbCtrYWFuZGJleXhidmdtazcvNHdobFU4R3FTYXJMVFRJZ1kzMUdZdDdCbGlUV3Nud0dZdld3SVk0Z0MrMzdnYmU4OEZmSXZFeEtTakJQUEZ4MlhqZXZ4cGxxK0pEeEY1UWptNy8yWVNxbUR2Q3hQNi9iS2xJY2U0TWNPWis0WkVmK2pkbWlwK09zak9pVktwN0pjZ2Vic2xUSEx4dTNiNG03WjZBdVZYb3FIOEJtbFZkNzExamVSeXd2Y0RuMzNVd2FFS3FFWUtrdU1NWm8wellqMmdMMHlsOHRyRDBISjd0WTNsd1YvdUVQOU5SM3hvdU4xKzE3azZvMXVoZUtHK0NaK092ZkhSYUExOTVQdkwrbVFicHIxY3dkWE5sM2ViRWNqS2RSUldkWElLelNKUmhHMGJuNCtycEdkYmZuZUR0Z3pCOEpteDdXN3U0MEMyOGY0Ym8zZHV1bTc1NUpKTGlNRnliMHM0N3ZncjJsRkVCM0hTeUxlR0xuZStuRVZBSVFUbDZ3bmlTZ1BMU2tPUmJBTWZXS1poaW9zMGc2MEJmOTZGT00zQjVQbDRreXBhaDIwWjdkRlJISXVPK1FTaXp2VFp5RnlrTFNxSUtaYUMraU5sVXE2ZnR2K3llK2VHN1NobGNXS0Z5b0o5TnRoV3p0K2lNaGtmcVNyNWR3V2lEOWZBWUZ1bE5rcTg2YUVLRGtWUmpSSHhEd25STjM0Q1VxTzIvVmJJWDlGYnBHN3dUY3daR2l4Rk96MFpFNlFmdmd1bUY1c2llTWZGL3pxQXBFRkVCN0N5OFh2S1RTMTVNV3JueW1sbjBJUWdlbWhreUlOdU0xQUNVelMwQnJNaDZSTE5qS3R0MTlzRytwNkN1YWVlTVJWWWtDV0EvdFFGR1FVWEpiaUdvZ0FCMU1TdGxnYWViZHBVUDIyZ2tJL0RocklLaGdoYW5tUldIaDVwb0lLWDUydzhISXRQZXhhU0pFN1hyMi9DM1NKTGQvNDl4MzZiREZxbnhMMXI5VEhaY2diellkb2lqaGd3VlRoblExN3lGT3YvYzhNanpvWGZpM2YzSXJrdElQaC9Sbmg0Rm1PUWo3NmE0NXV4OWVXQUVzNFg3YmtqaDFvK3QrTWJQRU1ZZ1RNTkd1aUFKSjNKY3dLczd0ZG5XN1NIZ2VoVzFTc3JJZnliQzQ4eFd4em9wWFhmaXdFMnZqektDY3ZZNkQxOHRJRjYxTmVqQUczVng2MWtSekw5UkI5TDNKSnFrOEN0enFYNjZ6OEhkUWVxNUt5SWRaZUJxYU1scTRieWZibnhhaERmREhjWWlucklseExSSEw5c2UxNEFTdlFVd0drVE16bkl0RXBESEE2RCtQQ3M0RGVZa2VkcnNQQVBXMFhscHZqV0h4V3lOczR6SnRkZmhUVnN4dmU0czBlcHRuNWpFUTFDRG11akl0WjIveEZUYTA0TG1kZ1NhVWNZT3hQZitHekR0WE5pWEJncVJuRHJFTTZ6R1lXOVYzbmdrYnEwYndnSWkvemdMazRvQnBqV1RsbGhtdUZWUFMwOER0ZjBiSk5aUmoydk1lNHljMzhWTVBFUDdHdGNaVmRwSjBRTDJ6QmFPTHYxMGdjRTJFeHppVXM3ZkNid0F5d3YrRkltZjlXMWJacFhPeUUwKzZiekVvSXh0Sm5pOUd2UkJhbjliYlVDNGVTdFNuNUZlSnRiTkhRS0xZbzByUVdWdjJ6SFI4akYydjVqS2g1Nm0vSVR3QXkvdGlBY1VRcXdRZTdWN2dhTXVybGxmRmZxaEVUUnNKRmxuT3Vuby9rMk93cHBLTG8weFlSK3hJNE5tT2lvSWdXSGZkNk1ETkRWTGlCRGtYbElnejB2a24vdTdNQ3dsZFlvK1R2ZENieXdkSUZ5NHIxMUZYTm9tRUNpdkJYRnRveFlwWElNSkNIaUdBTzVKZWN3Ti9iY0EwRFMvVzZJd2l5TGp4L3ExWXFLS0s2Y3RMajdyeTljZXlEOVZTQU1oQlRFMVJNV05DV0tKQUx4dEhKRk5JT2FYcXZPb3ZYc3VhNWYyU1JQU3hPOEpPMHJiNTRobmR4YkdtU3JGeW96ZVU5eGxZY2t5K3VUQWx5WEsrQ2lIVkt5MDlkaEwzWTRTYktoeTVybEVIekw1NU5jaEtRYTJlVDltRHArcGJ3ZGNXdE9vV1dvSlNCOTZUTjZkV1d0T2FHZTRDdTJ4dkZqaitQTlhGdWEySlB4ZFR2MHVBTVRneVdmdmJjQXM1NlEzbXJCNUNTcWNweklYSHI2NDlyb2RQYlg0SDhFRlpFVDU5TWxQQTNUT3BtelpwbDJyUlF1M25YMDZ2cGRyL2wxa1FCTk5aRjYwcEhzeE45OW9yZ2x2dE1ja0VOdGNjRk1tZ3RxdjErNmMwY05ITjN3ZllWR2NBWXQyS252UUZPVDRNQWI3eVZwUkRBaFo0NjRVcVdiZWhQK3Z4ck1rQmphNzNUM3VUcnhSaEUzMWttcGZFWElmaFFTSjR1OGF0am95SitaenlmVnI4NWJCazlaZEF5d0s5dWllcWVOMkZuR1RtN3hBa0FzV1Exdk4zWFJxTTl2bVIvajB6ZnBPbjJ0dCs1NC9zY2M2RVZraEVLdS9uZmhMbVFuNHg2aU5WbmVtdEVpcWdtb3dmeEVTc3NYR1BhOTRHMFhjZE9TdnBzbThacHdUQm03YytpWlN4OENjY0ZKTUdMaTBZa1F5amFsV0prYkhyQjQzNHVLRmdWaDlmWDJkQ0Q2L0RIRmhzbXh1TWxEdzY4RVFWOTV5c2d0dnBjOGl2TzBNaVU2WTdaNXpFOHlRdHN2UWY3bXFaeDlFaVVFcllNVUxVQ3RyTnltbVBwbUZJWXUxQ2JkRHNkamtHOUxGRGJEYXFKRG0rNHVuVFlRdTNja1prZ0NiWFlPaXoweUlvSFhaMlRZOWxXYnJXSFAxd29GNkt3OWR0bklwUXdxZ1laV1Jndms5QmFKUE1CZ0ZxMjY0SU16OUJULzBzMGdKcG9PdnVwSk1iRWt2SXN5VERSbkVINUwwcjRVZUNqcFZ3eXFWTkRDazdZdTJ1MUV6c3MxblhVMVFEN3ZTMGh4dUp1eWM2TER4bGZaL2RSU1V4bVhiL2dzNURnQ3ZXb2RaektYY0RNVUJOMHJ0UHdKTW0yVkg0QnBXVndreDVaWGJyM2YxZVFnRjExN1FuaTBCSklvditHYytYcWhEMUtScFlSK2ljeUx0aFpPUTJiNFIrbzJiWHRnUXhiNTdkU2FJZjhjMEtxeEtJWk1XbnZZemdVZ1FHY0RnTW12bkllbElTM0ZzU1BObE84SGRXcWpkRWdueXlzNFZUWWl2S1VhQlNkbnJzckN0Vm04WncvWVNrTVRhQzdyV0kzRUZ6aE1GYlhZRUlkYURRdGVTSmRSWmlQd2owNnBTTkNMdnR4RjQwNDVKL21pYUFKRkFiSG5MR2pTVDBFZjZ3Uk1lVzVrNFNxQm81eTRDeTBtNnBHODdIWXJMY0ZtQjJyOURtVXFmREZIWEFxam1NTTNvQVUxZTA1bUZkK21rSUZTVUdzcEE2MHdTQzErZjNyODVmUkhRV1JIbEpxVWRtdGVuc244STdRZXhFVXl0cnk1WVJhZzR2K1ZjMWZoTytMVXpoVlljczJSUmd2RGNLbWlMZFVMc1JCZm5IdHhab1hXaG5POEJmOXVpTi9xNWlYenpVdGZiekNvOGkrcVhuSFZraW4rY3I0RXZwalhHMkRaLzN6N2NoL2Q1VmJCdVJiRXVOKzlXK243NThUS2JWU2s4MGZQM1pOU1hoeVJ1eEE2bVpnVjh2YXVEcDdvczllYUd0YnBLUU9iNXVIYTdYQUZ0ZmpXcXlWSG5sVEpFNmZGK2ViUlJHdi8wZ0JJOXliTjF4aDlnMWs1Vy93S01ya1FwTWdVNDhaR0VJdnBXK1VmYUlOZU8rU09sdG9rYWtNR2swb1BjaE1oRVFEK1pkOWQ0LzNJL3VVY2RGZ1FOL2NjMGNKanl1elhJWktPZkxFZlB3bS9DVDJpNElaelc0WU5JUFBWM0JHSHA3dkgyL0FCRmxQVXN6dlZoZkpLdXBRaDJLMWlRN0E0amdRdnM5a1Eva0wvTDBmN0lZNjZGd0JhMm5LL0Z3d0R4bG83dUxaZDlLSzRzRFhGZGtxS05NaHF3ZnZIbVRyb3Z5dVl2VEo5YjdreHNPZXlHT2RiejNEb2FiN21LUFBlN0IzbjdFZTdiekswUTR4a3ZJb21oNlV3L29HUFFSOTQrb2NGRU90YXlQWlJPVHZmWUIxcHZpajJDNU13aXZmVmovRU1Lb0F0SThVc1cvSTRvUzVSak9jWmtld2tFaU0wcVhDV3g0S1c4N0l4TUpteXplYkdjTHZNOEo2VHJOTjh3VExSMklMZ3A4MnRuNWRFR2RpQWluU3NQOHd2RkkzTzZzQ2xaTTBzQU5waXJDSTgyeW10aDM5NWRHSDFIMTNaaEVVa2RQL3JxWjVreFpQTzZub3FqODdieWxIZDErYklHMER5cXpXNHh6emVObDBqSlE0MGJMNm04VDFXWUpucDZFeU4rZEtmeUxXRGhFcUxmRzA5QlJhMjJwd2hHZXE0aER2amJGZ2M5R3NxalNCcVpLWjlXVFJSY2RQaGEvYlR3cG5EVytFbCt3WEdZTDdraDJidEV2NWNGOStxRFVEM0FLTTlGdlhMYnZoMUJJbUV5WElrQ3FYTmRTbUFiY25pQm1qbmVFVjRKODB4VVVmay9HZFhjVzNEdE1hcjhyYjJyRjVBY3lNczQ1SzZOblBIcXJnQ0Fkejcyc2x3c3M2Qkl4OXZueDloZnRyeTZTZnBLb3pRRG5FYnNpd0pnQ0FMcVIycllDRG1TZjEydTBMTzNLa3NGZ0FSL2lUVndXTTZWV054aUVFZ3VzdGhDb2tDNzJnMW8vdC9mU0JvYXdIWW01OTRoUHhMaHg2WFdEcDJNUm0vN2MweDRHNXZtZStGTjVERWdGOWJJQ08zZlVGZkZHcmxiNzZTRy9iK01OQ2d2dlVtL2ExYmFKU3VQM2hQQ2JkR0hPS2xuQUdBeE5EMmU0bU1iMFZjcDBZM3Z0MFJPNHQzSE8vSnJFck4xM2ZwQWtUaXhsdDR3YmVCbmt5TEJOOWY0bEdrK05pWDNENno2OHNDVXUzeUltdmFVdDlSdldUUlU0cHk0RUdqUDhFY0J3L0VsR3poeDQ2V2p5cUtUVW4yL2VQZDdlN1RETk90bVRsOUVIZVd0Sk1ubFgxWFhkbHp1L1dBM2RIWW1Cbjl3Ry9KZXhSVWp4RW82aFhZRTlHTmd4d1lZSGdueGJHRkNGSURJRjF5ZFliczBLMXcwaHJ0aVgyV25Dd1k0M1g1M2RkUURBYmJWOGdZaytmMFBYWVI0cTV0NnA1OC93M1RFNWtqVk5HSnl0b1pSRjlicit5TVJxNnlMWXBEMmI1MytzUTdYekx0UkdjcHYyY3lzN1RRUllDZThOcXNYVlFIbEYraGZobDRXRHRDUlpYaGRveFdMbEdnc0tmQmVEVkJ2WURIOVhaWkloK3JnL2ovYU9xNlVsQmUxc05XWTFZaXp3Qk82eWF0YnhRVlI1UmVlRnJJS3Qxbk9rOWhzaDVwWGlsN2w5QkFpS0J1ZGJIS1VaV2pzRzVWVElzL2dneUJvQkRZc0lSd3lnZjRlUFdPNFlsb3VCK2N2QWtMOS9RMy9yb3YwNzhLSTN6enAvNmFNcitVVmNsS0RCNlRQd0VFN1V0NlY0K1RXOXQwdGN2WTVmbmxMdHM3TEVpS3ZuQmUrS1dERXNkeXZQejRsV0RkbjhXRG1LN1FKcWt5L0duYXBSUldySzBFaC9JME04TEhTbkhqSGFHWVo4cU5BTW1EOWd0ZkNndmlhZUVsOVFGNE0vNEdaQzhsRUJ5dDJuL3JwdlFxRTZSWWRhbUdJSkVzZFl3Q0lYa3l6QTE5bmgrQ2hqQVRJSFlQdml5amtNbXFMQmUyQ2VtTGQrYUVtOTNTdFM4Z0FWdXJhOFpVWjJNeTIxbXJOZDFYMTRJRTJBSDZHQkVRdDVkL0xqcjJId3Y1WVNWak1oV2QycWU3M1NtdzVmTXg2czR2dUFkcEU4aEVrQklkdjNqN0xOVnZTcDBhZFhUNFl0SVVDNXI0K2FEdmZMQzNtcTZOOEwwMFRXV1E0SzlRUTVqWEt1U2JibWRWYmJjYUJOdk5RYjhiOHB2WitxSm5aUDM0amRKYVlTa0ZVcldNUEdqbmdLN1c3M1RXT1FPMmxuU3NmVllpQVNGSDVrdjVNc0FjalpVamhVY0dHeWNCTUk2cmRHSUV2QW90dWVicWpnSnp6RDJ6VUxQKzI0cXRMYXk0ZHFiYTF1UXJBeXRaMExCT2lvU21EcHNsNUJtNmVlSmJKSlN4ZVdUSitpZjNFVTNteFBScmphc3Y1Ym1mVllyNWUvckZhWFpZZi9IMThUQ2JJRmdZSHFuRlJUMmlvbU0yOExUaDhlbmFVV0V6Uy82ZXNaZmxKRW5kQlhIZGRORWxKZFloeWhYZjZ6cW9pMnlKdVNOK1ZOSGFNMnRNQmwrTVowWGVrd0pvR2ExeDJTUUFwcERkSTR6RkZQMjhEU3lJRnpEMHJWdlZzRC9NSE8wZUd5bjNnYWY1S1RFenZlVGVEQm5wMGNGVk1qcFlscEZab1pqRXIvdmUwWEtnZ2loNS9qSUx2YVJ2Q2dXZWtSdEo2MzVWa3dlZ2d0eE1KdGFFZVhxTjU5RG1UNE41cmRoY3htTnlJZFdVRWFibUFhNmtqaElGeVZjVDY1emZKaTRTejRaR1lnVnMweVhCMmM2YlkwcXhiTXNEY29SNmk5cGVWajg4amhWTE5oTlVUYkp4VTBPSmkySC9GcXJDakl4MmRFVEdiRUg0b0Y4R3UrUitLeTBHTGZURE5LcWE2R28rZkI1M2xTWW5kdDZMSUprdk9SRE13a0ovQjR5NEZDRnU3T2xTZTExMXRwMVRUeE54UGkvcVRMMHZGK2wrMEoyR1k5MGJreklMVkdWOWhZTUg4NVpBNEt1aCtzMzhsV0pMMEU1NWRqYi8wRHhOQnBsenlMTk95K0ZpVjV6c3dvc21SUFhlbENVQjBhNDRpNjNHamEvdU94TzdBd0VRbFo5a0J3emtMd1p5MUtiZUNnOE5IVUdVaDRza2c3ZUlPSzRvRllmZWFFelhvT3lNNWlQcUREanVMV2tiWFdRcWtoL0xZNmtiWDhnTG9HaTJOaHBQYmxvZFVMNzV6YmVWZlBvZEJWL3drcnc1QWdRc09OMC9UbGVqb0lPcHRMWHlUZ2RuSmlwS24yYWp2MkdkVXcyY2RUSFArTDRuSDNrNmlWT0I1cFRreVFtYVBkR2Rqdy8ydVNhektpRTgzdmF1eHp4SmlHS0ZGd1IySFUyMW0ySU55SDlOWVlMWnp4OXR6dnpkMHdqSXgvMXRaZWRSa3R6QlNTTEt1WDg0cFgvUXJnMzhLclhSbWFQSEFXenhzYTFpL3AzaWJINVd5bmFTQjhVbFhPZGdQREg5dDBPcHpNSEM1bVF0VkRkRnM5UXBOM25wcVpQTWJ3TzlVVXA1VTBHdWZhemhuYmJYRHRvOStyZmJac0JTd2hHTEIyd0MzbGEwdkFXRFVtbEFtdlAxKyswSG5rNFJiRnIrQU1BSzFnY1JNalBuanNORTVMVUh6aFRzQVh3cFd4UDhrRWRiZU84ZTFrdG4zZDJYMkRMTG1veWYyZ3ZLVkxGRy84Ym5BYW9wa21WckZ5Yi96QmdmR3Z6bURETDdsSkVmSzN4cmhlZ040dHVucGFCZzI3ZTFYTEtzMjdVQVdEbVZZRWVXMXNydzJUYUUvb2dNMUF6aUV4RytEamcxVzBod1pIK0xrUUV6WnlIL0pFQ3VKczhKWEVuVnRUMjE3QXN6ZFBSN1ZIZkxWNTRUSWtIanhMdnZ2Y29xK2tINDU2emZodU4wYkoyVnlsT2VwajZHOVg0L0phOGQ4Q3d3WXNGbThXaHRRWDcvYThhL3BiNW1md2F3ajY4c3FpWTk0ZjRrS3orVGlpSHAxS3NDWEZVcTVpL3ZJdEJ2eWV4Tmc0RXloTnB6MVlzQTJjNExGQSsxNnVTSmdGRDl6WmdDTFhSeVRBdmx2bi9ydEZ4aVVnRWxIeUpnTHlwVjZkZ2Nzc1FpdmNaQXVzaCtJVGQzQmYvVzMzYlNTTFVjVXdUS2oxWktBY2pydXlBTEVaYjFhRTdRNkpxTnh6WnlRWmtVbHhsR3h1b2k2NE9namNza0JJMFFEbnJCendaTkllUFNVQnllRjYxeUQrclNraTcwdXVZdndYbVY4THFaalFvWVp3YjZhUWswMzVodS96elY5SzcxdStlTTI4Q2loTkhMM3R3SEdmSXdnSHN3ZHBzQ24wdHZGYk8rMHF5cVBkeUtyRkdwMzFJWmQrZk9EWlRyRFdvU25qcWRGamxRZUVhS3B2eEVQTmdTcXA1enVYVWVYNXRqU3RyK0RBMzA3YUE1Yk4vOHJkQlQ4anNaSjVrNHdlbGdRaWw5VGtzMVl2dWRMQStMZUNFY2YxVGtRNDBMd281VWRPNERySmhzenRDUkZpQkEvaUo3amVPaFRtU2pzZjh2eXdQRU9TcThyeHVJL2Fsby84UStPVDl3ZXc1MHcxNkxvL0sxaFZXbFdZRExYMnpEMnlWTjYvSmpISFFoS3MwaGhJUlBNWXg3b3VGRUJUV0tuNlVlaHlrUzd6QS9yWjVodjBucjdXQ0doRXJ5KzhjY1NQZnpEWVFXenFEZENvMitaWVBYdU8zRVlkTW12VXovUEVLSExGVWxNaE9CME1KVnpMdGhRbXVlZTJjQ1VLUmZaVTBESzBnWUw5bjBkMGZmNVp3NmZmMmtHTTBJTG1ZYXQxclFWdWtwMzdTcDFWTjRwYmlUTGttZTMyaEovYWNZS2xWMkZzRXkzckwrSFF6RGV1alBwa0VScGF1dVYvVk96ZHVjVzIxdDBXcEZwOVNndkVHS2ZwNUpOdVBYZS8rZ1F0R2ZFZ1hPSVV2RUw1bHhNVDF5VHh6dDRIMDlIVDhZK1YwYmJmYTM2dC8vY3A3aEUvdWtJSUVEYUpGVkgrNEx1MVFrTDQ4VDBORWFUeGVJUzZKMkdOMjczMnF5YWdHaWw2eE1hY1JTblUyUlkrWm1ZYW1oU2MwQ3dCQWgwY3VoWUFYbHBqRUs3YW1kRHpVV0JsekJRS2Y2RXdaZFdkb2RKaE1SR0s4U01JZElCb0hOK21YZm9VeEIyNnIyRVNNU2dJSk80MjVaQnVTZFdRYTl3a1VtbEY3UEhsc0FBMEo0dHloOFBYRnlDcGxRWGVtWTdleU5LSDdQSHVjVk1qandYT1g4MjNLR2lsNEtGdHVTNTR1Vm1ZaDI4R0JGdnVyZ3lkREg3eFBHTG9jMjNFNFpVK3JSVjhxcFRvT0t5Q3ZjMWxlNVFpalljUVNWUVkrMU01cy91cms4TFlOTHBLRVZqaERURFR5TlpyMG1LWXZZaGljUmFnYVY1MGwxT0R0RlBkTnpHazZZOHNkcGVFVXlDWENMM0UxdlhBMFB2NVVyQlN4aHZTZG1iR0haOEVkQ0Y3cGVPNksxYU9xWWZJaUZseEc5TE9pT0V0MzJBeW0wa2JlSGJkTkMxcWx4cm9UVEFTb1Y1ZlRTY1JYR2hSeWd0TXpSVHFpNmUwUnBpZnV0U0VaMi9TN0xZbDVYZDAzVnF3SFdCK0Rvem5sNSs0VWZWNEpVcFN5V0NldEVwNGlhU2pwWW5kR0xmejkxOThRd2FDbTBzQjR1TXNaRmF1VWllOEh2UDROeVdjS1AwOHNhQ2pEbDJYM3UrS1pwUlRQVUw5anIyQTV4QzYzZEZ0TWRUYndIQ0p3VVJpcE9TOERpSVBrMUU4YXlPRCthb0JaeUd0aVpDT0pzWUl6ODJOWXZmbmQ3WjEwUzRGV3JBL2w0UGtUMEIxQktyM3B4OTh2V21kRWM0aDMzSnBXZnJTWkRCVDJSMmRIL3V4bDg0bkJteHpDZWtncTBZdUtZWndiYndQNlovY1ordVVwWWtLTjNzSEJ3TU5nSUF6aUlmN2hZUmVQdlRHNHJFQ1RocVJEbG5sK20wNFllK0ppU2dEU3l1bEMzSlFJaVJrbTVYVWQ1bWFNTUZZOE9LSjlQUnA0VklUZTJXMmQ2anBVYTRLVzc1VGxWOFlDdzljbitlYWk4SkN0RktBK3ErQStTMHhlVzhCWEVNa3NxU2hjZ3lDNEJhN0c0MlVBOHV2bldKdzc5eTJnV3hjb0tMSm9TbTdFQkplV3JKSFlkM2ZYZnp0bGpzdmNkTWxFam5mYnVzSVlBQWR0MThsa2dsOHRUcW9pVWlramU1R0xyWkd4aWQ5Sy9sKzhla29lZERkQXF5TW1MOHJqOEdEckJnL2hBcGdseHlRMWFXWUlHUUtCamRRNmNtNW5VTkh3LytVeVRFbnhTLzA2ZHRYVlRVY2xYZlpQeTVoMzhhRXl0Q1dXMnpFTU9xOUNXWk9VT3JmcjM2bWpCbldIRTNWRGVyWW9KbnJOLzlaMEhKNGk2Sm92LzlGNnZDRU9SMkE0UW85eE5FVmtKZjIzYTVtVGt2aVowUG1wQ3MyN2N5VVgvUmxGREsxWFlMdnNQbVpCdEJUUTBQVVR4SmxxRXFDeHM3bEE0NUtqOHUwOU1QdkUzK2VVVExpY2Vzb0J0b2FWbmYzVlpoZGxEWXlrVUMzUDI1L0o0ZnhhRzJvNjBQQUpzbzRaUTBNMWVaeWV5cUNWYlFpQi93VW1Ud05EeG9PNGs3UkJJNkVPaHpyU0ZPMVE0MzdNNTkvaXBucTZlYUtZOHZNVDFJeDFjR3c2WEI4KzdaVU5HOGw2WmQ5ajIzMGFlOXRpdld6Rmx2angxbEl5YjB1WlRkOXMvZjdpcklZanAzaElIbGdLVHlxZGtTU2NKL2ZzaUt1b0RHZzdYOXdMNDY4REhjaVBGKzRoRnd3dWw0Y1pSOUZTYVlpVlFtU2wxQlNuZGdZcER6TXV0bWU2TER5cDh6UURjS1M0SjArTXNadFhNeUNpNW05aWdhdXpuNEZRa2NLbkxxT3FsSmU5NTZBRGRybWNyMHd4ekMwZjMxVnorSFJRbzluSHUzelNwcko0NzVrRkZMUmNXa01tVGxiZnFZbE1QMUpuYU5Hc3VRaDF6dlRzNTJrbWVwNHFUYXVKS1QwRytHaU56cU9ZY1pRV0d3L1QrNlFIYmoyUytieGw5VERicXVzK1B6OGhtUFN1RHR4SDhVOGlJMXZJY3JKb2xnbWtVUXhuUVhvSG1RMUNvYTV5UEkyZzRXVnF3cytQUTQvVXU3MHdPcWcyQ3EzZHMzWDJQNVJqb01tSWlhS0lUbXVTeUkxMkhNRlQrSkhJdGE1VzNxNkNucW8wRVVqSDltWmtvZmRYVXBoVUJucjBkT0hUbDVkdEsyakt6eHZRaGl2VFk4ZnVrOUdwRTJKZm0xWHJ4QnUxSGxNSFg4bmNmbS9WcnVvNnRJbUttem1yK3U0WXdaOHozQ1NNdkgrSGR6bm9NdGU3RXk4MHVFZ1EzUnIxc1hWVUdHYnhndS8zTE9tUDBQcGtXcjRMcVFMelFIYklxaWZHSTN2QjdoOHg0bXlyMlZ2c1NIWlVHUjRaV05RUGN6UzZvL01yZEdVSWd0L3hRd1Q5a1ErMFZtV1Y1YzhJVVFwMldmSVg1YWh6ZGROWTA3Y1V0dnNqblFxbXZacjJqUVBRaU1TVzdzaWppM0FVaFZNbmN5em83cEhaSjdaMzMwTG1LR250b0p3aTV6VUhzUFhkMSs4ajRZck5YcnZseTlQci8raUZYaEtkcVVjVUlnemRVRUZ1dU1panh3WVAwUFhmVTloOW55bUFGeFN6NnFaaDZmVE5nUEtOQk50VmZGWndwZEJUWWNhMm1hb3hBVGVYMVVRSjU5SlpjVFhyVW9WUEdwaWtTMHV6R25vK0tPZW4wdVl6QXMwQkVXcVA2ZG5hT0s1MnMwMU9PMGxKbHFHUk1vMnJmZGw5SDhZUnJCcGszL1pvYmlQNUJ4K01ucW9vNHdpM0Y1eVArb3J2cUE4SytGM2Q1ZG5mMXZhdEpJb3VJYzRzeXczK1B5UVNjcVJiYlhIdTRjN0lmbkl1TDI4cnEvODlTUzFlNlltUGkxRXdNRzd5S251dDQ4Q0NHQy8yMDZyNU1XMGhyTjhqbUptc3V3eW1JcGI0ZUVwanc4SHpXZDVEN0ZVTjhEbnp2RTVYR2JPM1JoTkhRUlhuYWdPVDJEQmJsUFVIQ29sa2JZb2lBV1RnSVVtZzVIRHg4YVJhbWJnNjlDNm1IWUpJZDQxbDRpYVU3dC9wczhncGJDb0x5ZjhJVk9lL01EZTg4TXNOUFBRWnVycFlPamwyUVh3Qjc4a3dQZm5GdWluVmxiV2l0MFBWV3luMG5jNXowQVZ0c2ZxeCtpRlpTK2lETGNITFRQQjhreXErZ0FKQWo2QUh6UXVLdE9TbXd2dmtUV05hcmRvYVNnQXd0RUhxMy9mN3hXc1lxdVNxbXp2cDdpY0hUc0pIWS9GQlNkeTZSa2RobnFCWm9qdmd1Q2t6ZUt2TTl0UGo0Wk5CcXlIbTI2R2ZhVVVzZCtvS25odGNLV1lsR2xLZFF0RkR5OWY2cm55VzBKcGdIYVc4MndpRjA0NDFvanR6eXhCYlBoZUJCNXZWcHI2Yk9na3FxaHVaZlR6cnNvUVFFaTRpbmM0aWsvM01nMkZicEVTeDVxTVNnKzRRaFBFbEYzRUw1cHdRakhJNzdhZldRdlVSQkhrdFdBSnhZTTNvWmJ0bHR4VjBLSUNqeTE5T0lWM0M4SWpOYyt3RjdUZys3SHVFMUlyQUdoaWlvVW9PUW50MjEyWUhLdHkwSkcySGpQUHdCZUxaNmZieVk2dEMyYlhDbk94MjNzS1JIRnppbUs3TXgvbmxUcjlJbUhSRFMrYytHaUFrUFBxUU03cmJGSFRKeE5kY1p3em9US3FVcUFmRVBEdEQyM0d4a29xZk9SRnVBeG5Ma1VvQUYzZGJ1NEZCb09FMGFsU3VIR04xSmlLdnBUWHZuZ1gva2JEWURSbHZKUE9FenBmMEtZb2UwQ1lONHpPdVMxSXhNMzFYSTc4OWhYQVplTk5ocTRtOWJ1clJQMy9nOG03UXVUZzlSaVFrT3VDcEtTNkRnS0lPN1NKRU1TMG14dXZrSkhoRkliTTlUczNaT01oWFovZDJQdVl1UGlMc1BmUUQ2OHF4QU9CMnliSXpKVHFXbmwyOXQ5azlBL1pqb09oSGQ2WWdBNXVMdlJYc1ZOUCt3ZVZDaDY3L3lCZWNrdHVpTjhJTTdGTXA3L2l2bFpKWEgzV0I0WFY3WTYzdUxPTHR1YVR1UmlkQTJ2Nm1zdmlvWFV1ZWNJNHpMZVMxN2F1QURGdlBzV0MxZWZtbkM5VjBpSzVQeDd3enpZYU53N0RSRXpqTHNaUmpocUNLelM1cTUrZ1hOQXlWRWtKeGp6Ry9aQkFWZmpRaERXT1NoNXNFenh2QXVOdEhTbldhdTBaQzk1MzViczdmWTcxNjlTRG1hcUFZQlJETlhYMVY4Q2M4VkgxN2Z5ZmtPZGtZMXA3ejVNVWlyZEt2UGcwWit3KzU5QXptMVpZaWtkWnpWRFFORUFIYUprS2pjMGRnRlk5S1dxdi91RGtudFRzc3E4R2xPcW04L090NFp6NkhkS1pZNGx0d3FRTEFWck05eUcxMngybDh1OWI0dWIzSDdxSDZlRXRqSFBvYThUc0dEZVBCdUxta3VaY0JNNWd0MTNGMDhhdU9JZWQyMVlsdzBFZTZvQnByMjlCaSswbS9Ta3pYMkpGYmNYdFVEelRrNGg2U1ZYNlhxY0RVM3B1OTRmc2JQUDkzMjdxNm5vMXh1OWx2QTM5bkZRaER4MVFiVVY2NHI0YXhmN29EZHRWc1NGMEhCS1VSTTNqc2xOWnJoeCs4WGpaeUxmbzNzeElaVGxyWkNzOEl1anIwZE5YeFZBRVlOTVdQbGZOQVpvWXlDVnlqQ1JGNGR6TTZ6TXV3c2tudlMvN25FOHY2Q2FINW4xM1BWaGM3elZicGRrQTdycm5wOFpHbzlEbkN6TXN3VWQyUzYxWnNFWjJGeEIxL0FtZlZSTW1OTTRId1M0SmU0RG1sVG1CaHl5L045dStkZnh2WFQxRHRvYyszUHZoeml2cFU1UTNmVERDRllLQWVMUnpIUEw3dFhQR2RUQzdpVFdTUGlJWUVoVVhHSmpUWjZ3dGhrQ2ZRb2lCakNGcGpVQnJiVDQySFQyQVNzMHVuZnZUMmhXRzBKcGJyUi9QamZVMWQ0MnJIVWJiKzBnUDBZZWJURUN2UUZMSWpvTks3TEthSUE2bmFnZFJIQTFGT2tIcFNEQWR3N1V2SEg2SlNNYUJyYUs5UjJIaVRUc2NvRzlabDZyZnZZWVl5TGYyb1loUWVRamgzVzB1eUhXODlMRW9VSEFkZGovRUlRMzUyMjNITnQ3S05FYnluRVRVemppaVAxU3UrdGRheHU1MnNpUWF2bE8vbmhzZVRpc2U5Znd6T1NmelhvOFpxSWpmMWNPRVRsV1dUOEExZDlIUkpIcSsrY1ZNbjc0QXMrVGZucEgwQjVoZzhMd3lmTmROd0ZUdHVEZVAvQlVlOXUyTTk5dGViRzlBcWtaRW9MdXlrQisyb1RhNTlhL2ZnWksrQXk2Nzk1dlpqY0ppSEZPR1l1TzZvaWg4c3VEaVpROE1xLzQ5TVlsL1VMaXJvdm44bUx1cG5UTUs1ZjJUZ1p2U2RvNUE2NHdmUWRRVndBWktPaWJEOXhyUmZPSkZlMWcwV3ZQYW9lWHZ0YUR6WlZsWGZVWDRQaDlOdVRVWUZsODRZbmdBWTdyTlpLNGs2b09UZEZCcVlDeDJuRklRN01TQnRhdlFOcW9ua2IraUNkZ2pWVEtTMXBBRzFTZzJYdHlDalQ5aWxvNjZEYXpoOFZTM3FvYnBuekNtU2pmVGRPZ1YycFRWRWNOWmF2c084UW55YXRUMWVoVjBsVVJLSndPZVMvQXg3ZkhvZWFyZU5QNVNHemlxaS9wZjBMVlh4NzZGQlQwaCtuT2tKbTNYRkI3MkZhNUs2RS84WDFtYVJUcXRFYkVaa0JETzd4SXQvZXRPb2hEcWlzRTFOYW15ZTNtNVk5cVJMdDA3bFhWQmtiK2lRbytNQkVHWFFldTFkbFFCbGJucDhUUHVQTW5DSzlqS3ZURStSaTd2Q2c5Y1RrUjJQeXJ2WG9BTXc3Z3FzbEtjMmNKNTlNbEhIR2syTHNoVG9KWDZBRkI5Vks1WHllM2ZCVVBZSHJlczNMMGdKekYrNU5ZbTVSWW93dFVEbkZ5cE9GME1oY0tNTTRBNVlBamlIUjc5T1hGbThhNXZUaVgzZGlXdUZZK3NNV1B5MWptQm1hSkFyM3pnYnk4aFRSQk8wbTdNQjJLRkNXRDVTNU1OSmh4UjJ1N2JVYWtjMmNUZ1lGQUpybm1wM3FqYlVrVG5aYWhUSFpxOTlvWUNzdzIwNVlneHc3V0F3eXloSFRNV0NMQUJBYXZMYlZrOHBEZXFRdWJjWkMwNlowUUt6aU1oSTM0ZGo0RnpPSFJIRkIyRFhnNm10NUsyZzYzWSs5RVc0d3BvbU9XSmxEVGZuZnE2Zi8ySmd5djZnVnU5c3N1bmt6ZkNHOWlZTEpINllrRVRYQnd3UlFGNFNlM2tJbyt2VG4xWEcrcW1jUTU3a2RSLzFJSHZqUGJHVkRVUjUrUms5WGRQNC9TNm90N1BYT1MvTHUwbGtzQlFONC8yMDJEcXlJSGVrdkRabEF6TkF3RDVXWitzK21KSHU1RkRQTjZYeDNFTk5na3ZIZXN4OXRTaGs1NjVtWXl5NGtTTDRoSHJHcldrVDBSZXlJWm84a2pWMS9xdlNncjFrUUI3cDJzc1A3Y2JMRHcxYzR2MDhOcGYvRGJ2cWZsUXFLZ2RkbngxdDkyMjFMMUliL1F4VEtZekk3QnV5WnNJb2wxZTJFZTZWRXNqbmR5bWoxQzdvaXl5SFhQSFc4dFVjMHJlc0hpWjRQWGdXOUFSc1lESGJDMGtqcGVlZVN3cEpuMGtJaVRJWUN1aENDOFlJSnZaZkkvSDZLd1ZwNy93bUIrMGJCSW1DbEI1MGFiUjF3MUVQcEdqSDFSNmYzRW5CV28xbXUrRkVJZHo1SVpoVHE2bnFlZWppaWErdGpLRG1HQWNpdkxwWmlzZUZRREkvNEpDdWNua0FUTFhpb21HOGNzTXVMVFFJZzU3eko5ZjAxcythdzFyVXhPMm1ISlUvdzBQczY3M1Bwa3NMazhxbWNEdDBhbXgzZUdIc05Pek1rM29IVVpUU0l3ZVE1SkZMclVzaXlUcDN2d1pvdnNnTEsrbUpmNUZaMHJqUTcrM2Yva3lraVRsTncwSEd5eVh5czZPOEpyb3ZZS3pVRC9Ia1M5R1N6cHoyL0R0YndMT0NSajZWd1RORDlQNkFqTWNaZUVTMGhvN0M3UEp3WXdVZnEzU0Z1Qzk1NTFTOWJmNEs1aVIwT0dmUnRLdjNnVGtFREZvdXozakhzMkUzVU5IQUxweUhpd2F1bEtmazlMVUZZaHc1SkordHplaWxVWFFuUHJ1Z211VmpyeEs2UCsvU05aeUsxMlBJUFFSb004ZnVJM21Bd21WQndQRUpIZmc4TGFvN2sxc1ZSMGx1SzM3VkN4UlBpeDQwd0V4c0lHd1htZlBhOWdoMm55V3k3VjRjaTRMZGhDRWNuQytCNjZRUzh6VDJGOTJ4cUNZbE5oZTd3ek1xb0ZheDlSUkpsQ2k3cFFFOE91ZnF4cEQ3NTA2aUx3NTR1N2lmZ1Fpc0hzQ3ZrcUJDbjE0WHJVZGQwRm5iQ2V1Q1F4RG1BZHQxaUl1K0xkQ1Z0anBjQlVUR2Q5elFTOHcyeWhrVkFQOThiTU85NFNMUWU3c3ZPOW5BdlBnSmZzdXc1T2NyQVIrYnRQV0JjVlVreVZpQXVaeVQyeWx1bFFuOXIwNnZ6MGV5aEtGczg1R0JWc2E5TWJGa05OSFN5ZHJaeDBoNUtMYXFBTE11ZXhONWJ0bXEvNEdvQTVJeUtSZE5LWjRZMTU2UWdwNk1LeCt4ZmRZcHBqTjNuUm1TbGJBYm8rOXVPNjZRYm9DZVNpakhkZHg5bEUvSndPa3h1OWdJZDFJcHE3MmtZMkM5aW1pTHNiNFhqUU94OGlLY0Q5dE5aWUprdmk3ZkdjVSt0Y3B4MHpOTWFUNjczTk42MU5xVERrQjBUaEJPQlhSVE85dXErRGlrM0labGdLRnEvZ1FQUURvd2R0S2ZZQnpJNUZod2UxaTlkWmRBcUhMeWtBaEQrYmtuOFBva1dzM2YzNVVWamNBaUlJL0QyZnNvOEFTQlBvdnBIeU1JdkpBL2R2Y2l5dnhrWk5CZ25yVk9ac0tPVTdKVjFJUVQ5cVdKenpsaUtLdVRaeE5lZEIxVWVVSHh1Mkl1ZlhlSkRuT0dGdVFiWjNnZGhXbkt6allLcGc2ZUNGMjQ4STdiTHV3Zm9ZdzVPMUxXRG9zUXByVFdUT215eW44ODc4MUNjcG9CMXArMU9YdjVHQWtNNXNpWWdaZWJpaGVJdEM3ZXZjcXVmcGtRTWxmaXVsSkJhY0FSU0g5YWZPdklSenFpbU5PZGNJTWtXeUJCa3pxeTlGRTV3VGRORXdiRTFZWDZFMlNxdkxIajZMaTh2UG1tdTV0Zjd0bjJiT0dCRzFtZlhwV29ueU5kRWlsVmgwWnFQNW1RQmhOQVdkQnpIQ293MHhaeE9HNXpVUGo1eFFLN1VaVVJZbVgwWGNjVUVRMWd1ZDZSMGR5cFpqbTNseTJtUUkvSi9LQzhUcVZ2MkR4aXk1M0ZqczdLQ3Z5R25IZ2VoQzdxN0tHK1BPVUVEQ1IyQktNWGFJT1dsRDFWYUN0bGxzYmhxTSswMTNaN1Q3eHRTQUNCMGIvV3J3S3hyRzgxMTNMNmxBZEVvUlN1SFJaTWYvYnZ0SGdLanFUdXR3Znk5ZjdqTlduRFV0dUZ4NVlINWlzZUxmaVZNNHcwQXZpc2ZEODFsRHNDcDlUS0I3dGtLTkVyYlBXa0FqVXVTS1lmRUM4bkFtZ0lIUzJQbzBFbEQwV255Wm1nUGx5WSt4MkhaanpzcmVDQmswakZ5Z21yYVoyQUE3d01aWGthWjRKMy8xenRQU2dTMi9LdkhJYzYvZ3d4VmhoL0pVMFhLcTFuTlpTcXJQRm45aTF6bEVuOWVZQTN0SUxrdDZ0a2NpNHU4aUdyY3I5UVk0TUhlNGlhcXdGdHlSUFVXZWpWUVVtVmtGcWpHeEZqQnJmRDdEdXV5U0psWnFobkxHZDRjTktIaTFybi93UStPN3JheHVQYmEybzdReEVhR0VyZ0pzZ3dpc05pd0xmRFlMTGVJd0h5S05pYUN1ckRpTTY1bmtyNDZZSzRjWTZBK2lOOWtnVjdXYUxqYjk1VmgvemxiV2RTSVdkeERPMnhxZEx5OEpvTjZMTlNEUUtvMk9ueHE4U1pqMnFOcEp1YUJwc0QrTUpNcGNvRC92NG9RZDN4QVlCRXhWSk9aa0NqTm1WK3BBdGRUTXpXOE1TbXBMMEhlWWN5QklEOHdWek82bjdLUVNjMnJqY2FWWUtKamduMUdPdW5HTWJQK2ptNTdmdms3aW9vRW9CK1BkUnJVMnViVWlrMXpkSkNYWm9zREZXKzZIQUkvUnR1azlGN3JnR0pZaGZWak11a1V6eTJySEhuZ1dwbTZnSzhGMXlvcXJWUUpxQ3hBRUNvclJCcTNSQmRqaWhBY2JsY1Jzb0NiRm5uM1U5V2U2d25JUG9GOENtWHBzdkVWSGVCS3NwcGErVE1rVmhSTlR4dXF2bGpMZjczSG92aHFHN2tBRkdsWWdLT1dwbW50ckFxajhKdy9sNW1tY2tqQ0NpM0k5b2xkZ0hkWVp3SzBCT3RhMUVYVUlSeWpYSnZsZHNDVXh6cStlWVpKMjdvTGJMT1FwaWwwNWxtTzFaOGNkcndBZ0JDMGowTWRXV3VpR0hCa29DdkdTQ3IvYVRCYVViY0ZVZVJaMjVGRlFZTXJnTk90QjFMbksxZUdudnlYYTRSNTg1MVdqY3hsVzJIdzJuUlBscWhDTnJNbEVZQ3dRMzJUVzR3aEpIUjAvMktZZVJWYU5hdlRhZUFtY0h3ZnNJc0VVRE9vQ1prRTgxbkQrdVpzdHorbjlwbDlNektBNzU1T1UyR1pJQlpQWVduTmdvNkYwTms1bDVHZGxzakJpbEVDR3dUMm0rb05QVDBuZ05EY0JFOVExTFFDTE1vQ3BtNVVKZEVmWXBVU0xxVS9xRVRhN2I1NjNXVVdJbkNmY1ZIRTVXcGErTjFXdzZwYS9KQXN6TmtuWlZUR3oxM1ZIeUs5ZUdzZ3V2T05uWlZoRjFMcTRtQ1J6eFZ4bkpmSlVSa3ZPWm9HZnZhWlpYSG9KZlpoSXNDNXlqdzZLSHhoMkxRbmhGMEM1UHp0RHFHRzJDeG9XNWsvdHc5WGNFN2RaTHdxOEsvR3lTT3o2RHJLTXE0UXUyUFRrbWtFZlV2ZzVOVDRHOXR6R1FhYTZZTlVCR0hKbWUzY3NmRm1VUVhpN0QxeStnWE5tc2sycFFQMThSeUVVT2RPSWQ0dlo5UnRRS3pyQkVIajBOWlBnejd1N29pQ2kyWFNKMnlFVFU1NWlHRkpCb2JiV0hzb1dxcVZ5bzJNQm4xeS9IelR2bHhoM2ppemhXWUhRakNiNVpQUWxQaDJCeTFIa2pLa0NvcTNNays1Q3NKenlRTzh1aTExVWowWGQyOE00bnZmaUFFdlh6MENUZExhbzdUK2s5SkhtQzhjT09aWDFsNUtpQkxaTDVTS2tLY1Q1ek9jMW1QMUhmbFpFV3VTa0FCdUtyU0xzTDFZVkZObEZ6RHZlZG9WWDNycHFlemZkY0NhdlhnZksxNVN6RFBSM3Fid1dnSnJLM0dVaDczUnUzK29hQXl5NkpqcW5WYWVpSklGWEk1RnlBdEdFSUFqZENobkdQS2hwQzVDZ21rTXNNbm5OcStLbkFTYVlvYjJmM3Jkb2dPSGlZTUZ5OTJCNk9sOTNxL0Nlc0RjSFpILzU1MC9XaStreTJhQjlINFNTWFpaaXpNbHNWN1poNFJIL2FJUkorZlVPVmNQbEVBWG1mRVhzTUJLaFIyM01DMkRLMFdUTXBxakg3SUdzWW43T1B2MDhsMGI3ZVhMdG9sWHNsNnBQbi9lamYyTWZkL1cvczhKNFVCSW1tcmMzRE4rMnNvS3FvRFVlYXBsemFIc21lclZoVTF1eEtzN0h6RG9wMlB6VXBlWDVRZmFCOEtEMmJBUWROQnFZN0ZIVVNOcUhqczFvOUNub05nVEFLMjNHTTRGUk55Y1c0ZkU5N0JqdHJYQ2xGNWE1MU8wWmFDcjhKR1FWOFFhbWNRT2p3ZXorWDF4U3VYTzcwUGt4elJrQ1d0MjdWcm01V0kwc3pHeXVWM0U4TWVSaEVhMGg5dXhodGk1UnhSbjZCcjhIaW5oVnVQUFI5dlp6T0pJMlV3VDBRcUhlUEFXU3Nsd2hNOTU0OTVzTldtZEE5WnhxdkIxYmRlb0F6S0hqaXpsZ2ZjZXBKaHJiMDluQ21HbzVJWGZna05PTDYrV0ViUVp3WGcvRmIwRmRxZ1gvRTRtWHVReWZQTGRhbmhKVW9jYnd6RG9rQXF5NElJWTlKNmMzMzlXNXlTK2k0bU1wR3RWckVoK2ZjaW5TOUtObnVUT3dBblVFZzJ4WThHcHJoWHhUOWFja3hJc1VPMi9jMy9wVUIyQy8xQ2dCV1hNR0Z4U3VLclgyMGZHVFBkU1ZROFpCa1FQUVhxa21WVE5XRWs4OEFvY0dYN0tkeGp2OHVOam5YK2RuRFFpMTBaUzhVOHdqa1VTUW1nbWxhQUdVeXp4M0R4T0RURTNwTGtRcW9lZThZU0VVNHlqaVg4bU9ka1AvVVQwdGZBSC82WGhVSjc2Mnd0YzEyc1VzQ2RoOXpOUHozNkdwNUc5SFlpb1hMaE9iQ3dBOUJCbE9DNmlkSTRWTlVQRUJGNFdpY0tRNW1WQ2VTOHVGcStxdTNQZkpkdTEvdDBvQ0lCLzVkSXB3WGlidzRyRnN0VzVNUnJMU29mZlVlSjZGR1dlNVhGd3BvN2pDUmRtWnVuU1NlT2JVb3pBNzFSd3AwWTlYd0dKaW5HcStySHoveEtXeVFwdUpCWUFoTjVNeC9EaVZxTVVQRDh2OWVkTVdXazRZdzdpK09PZ1RFZjNLZytKMTBoeElEdzhHSDJmbDNmSmJUMDQrUXRDcHF2UEFEQ1Y4MkFZM3lXcVBWMDJMOGhHRkl3V3VVUXczdXdvSTRTdzdiR0VsamRnaXk3UTRxUUFpaFk2OWpuVW1tS09mdG5GZTFXTVJjaG5YaXgwdnR1MUlkcDZNSmJyazA5SXlsYUZmOFp5cjl6eVcyQzRHVEYvZnJwbXhJcko3SWZNQi81aTg2SVFpSWwwK0xrK3E2RXU4bmRtWkdNZEs3RE0ra2FtV3hJbzJtL2UzVmJZeS9WNndDbFpHbHRsNmZEbW1TczZ2VDJuQlNlZkFrVmpkZktvS2dweEhQbWFrSWxEVk5DbXFKTFB6UVU5N2NTWlh6ZkdqWkUrMWJYeDBDMlVObzFNUVRLS2lzWXB0eU5STzR5Rjk4WUYwdHp1YW1lNko1cWFlUm0vZitjU2R4NzFGNTU0VW5WbXhoMk5GRW1ZcWZNYlFqSU5vaFhhWDdDRktrYW5RZzJEWERKNTNmTHl4RmV1eXY3bnpWeHB2N0p5UEpvNHdRU2VPd0FVMHdWYmdWTjBwY1ZTZG9GbkdBeWpnVzdFNlFBd1pISlBkOWwxYVRJY1V4S3d0VmE3SDN4MzhpYzl6bFErdVNPdlJSWGVuZDR4RWVoZEpBTTUrdldWSDdNOEFRamlQMFFSb2NGWTQ1cG92R0dDcnhyTFUzT3A2aHJHZU1KYWQ2cUVXZlpiRWgrK1V6a0RFcjlqcTVGeEtyeEtSdENKMFRncnQ2STBaUHJ3bjQ1SjBrY1g0NTNSVlFXRlZGYk42WTFkY1dPU25CNThCSDRqTXFOV015dXNKR3lmcVA1U2E1K2VTZFJvclBoMWxNSW5VdDJGU1N2bUx6Smx5ZFgwV2NkQnRDRFhFNG4ycnNTeWNSY1MwcXpJUVkrODNZRGZwdWdJWExLN0QvTjA5NHRWTk5WR0JKUHUvKzFtM1RhVGZtTzVCQVNOQ3d1SkNZK1krbmFMK0FUbmlXS1BieFRMUS94aDlWSlo0TzdNaGJSVEkwS2lhRVRmNlRzMmtnb09rS3JYQmxKSUdhdndTRDU4ZityanArU2RCZ1ZZMllUV09qRFprdWs5MVJQNEtTdVFrK0c3dW4rZTVtTndqUXozSkljdjloTTVsRjN2Zkt2Z3U5ZG10MzA3WTZ3d0x0Q1BCQXE4WXo4aEdzLzV4a1RLUlF5SDQ2Y1ZSSmdwZ25PdVFaU0RLRDRselFEaTNiaGVEMmwxSUpGTithdU5xdVcvaGp0QzVXaTNHUWVHWmJrSHJxYkZGVTlTanFDQU5jZzgreVA4V2lVQUV0QnZQUFpsYUltS3NoV3Vvd0tqTXBpK2s5NDB0Z3Bnb0dXUlB2NlM2em05U2d0Qm04WGYyZDZMcWJCaDRsMVorbjF4UitCUGdRUVlvQ05JTDZiWHhweEp1UERCMlNMVU9HekpYTDJPVyt2dGxWWGxoY3ZOTXVQczRnUm5lVlg3UGlNZFJjelJDQVM4VUFqL1dyNVV1OTlwS2p1RWpzeElhc1VUMWRYa3l4b2FLZ3hrZ1BtV0IxM0dQb3lGZExBbVNmYjBDTXQvMGdyZGtMZmZmbnVDOEhsSG81TEhBMWg4Y3FJZ1hBYW5abVVleWFDRk1Rb2hFNUIxK0tMVmozKzNaaFBnWjRLTVhMcUtwRW4wZFQzbTdmbkhadHg2RnhTTThsb2dUN0ljN3B1SnpBMVpmQkIxSjdwdHdXZTFVeTdkQ3FGWktYUHNGQzZTMUlvdURhOHE2NzdKNzdzUWltNEZHQ2RNU1hzb3EyWjRCSEJGcHphL3JmZys3WG9hWlFyekZDZU0va1QzQ3BPSS81Z3l5NDFSVm9DWkpEbHJEdFJJZkFzOE9INDV4dnNkNDR6Q2g1RUpXdWJZeXBnTXlabHBRZE16ZE5hRkh2YkZDOVNjOFRUNGNwZHZ4QVZ6THpYUG01eCtoc2YrL0VrN3AyaUFxaFFlWGJrODZyQllpV1RRenYwT0N3WGxUVHJML3JMQlBzOGk4RW9FSERwb3F4QU1YdW1aZlVoOE5VTm5DaHpaZEZNU1JBTFNWaDZob05LaHdzN1lDT3JWYngzUHJLbkNDVUdySGtMb1FHYjdNM3ZrOFhoT2lYck84KzRtcUoyQkVOQTZwRmJZODRtTkhTSW5hczlSNkVLWU4vT1dhek5TbVR1NmdyTVFnQTErK0ViRWZQdWxodDNZaWRtd2Q5YWF2eGJsZWh5bmRQZDFROUg4NzNnUUVva0oxWUt4RC9vVCtORnFoMEdoWDRLbWFYMkhFTDdpdzdMbFN3M1g3Yml2UGxSRmdQcnJldm1rZXNIK0d4bzFxUHJiaHpnU3dQSHZzeUFVc1gzQXp2TTdlMGdUVEdockVwMC9EM0hqeXJ1RzVqUEtGUzNUMmpGOXlYWkF4RTdoQmdadlpLUkJJWTRYUE8zc0M2ckgraFdqT213cXFDSlFjR0h5UFBiOGg0ZnNFSzBFYXNXNFhEQzJpLzVBMjUyUEFFTUM4bDh0TUV0cGppVUVtKzhKQ1R5cUJoVFZGS2luWDZYZVkvclVlTUF1Q2IvaGdaOEJ1ZFU4T1ZMRzlWVHArOHN4K0VMVHNacDBRakNhQ0xwM0g0THY4ZFM2aHFtd05zVm1UMllTdnJLZlNoeEVTNkpZZUhISmR2Z0VtT2tOL3puckFzWE1BU3FZN1VCMnNITVM0VEZ4WEdXdFordWxuTWtKeWdSVEVDUzJGcnBlUUZNVGtaL2hmSTdmamZ3dDlPRUtEdUV6TWs0ZEhhOExoeU5TRnNoMEhpSUJGaHRmV1VpRlI4dEUxYi9SRDE5VjFIRlAxTUx5RDY4SlRYamRjQ1FxTDVOckdDZzd5eVFYVURZNFE4Vzh0Mm91WVNCVTNsbEk1OXAxSUNHaDZFeTNOVnR1NTlEZnlUQ09ZQ0tvdjUvRFAyWlpaZlNUYnpyUjdVOTdsczRFeFFLK2thSXQyZHNHNzg0MHNONjRSSm9HNjlKSHhVVWRnM1BhRE9XQndYVklTQ1RCdWt1cDJxRTJnNDFzM1ZIc2x1QU1UbTZHeU5sSytSelpoUVVDZ0J0K1NVTU9XZDNRdXdBV0t1RW1NV0hKZCtmRmFYK1N6bzlzMWhzRFpNNklCRU9TWjFLZlJSZ3hFMm00UlUraXJIbk4zRGlMSHIwOFVpemMvbWNQYW9TbzZjYzVRanlFU0QxU0FXUlVETjNVNU90akV6bTNkY3NOTnNMWlRra21UVmxUZXNpZlQ3ZzhGZXVvV1VCZEFEdHorc3BSTGJRQmlsbU44MXpKeEkwaXBuM1JoWXVMVStJL1VjTkpqV1dRaHJyVjBNRXZQY0RaZGJ4bWs2OHJIVkVGT3I1TWRjaWsrU2FNQnJNaDBvdEZqZ1dwcHNHVys2b05FNXVpcWlOQ09vTDRLQkNaazFxcjZhNXlOR0VDM0p6Vkd6c3l3a1luUW96WDhBVHRFQ1lQSy83c1A5MUt6YUdzTDk2SjZoWU5EYW80OTMvS29KejU1Q2xUZ2ZxZVBXVE1NR3UvYTdXZy8zb2FGM2xXYk8xaXZoYzkvWXZDMmtldUwwNVZ2WDFPRFg2RzhTeU52bWlvSGZSeFZXK0JFSjF0eEg4K1hKM1pXS1I1ZGxIcW40R3JxNXhEakdwOFJ2ckdQUGx4aTloY2RTdjE5eXBGREVjRUlNbGRkYU5aUkMybGd5ZFdTMVovYlJOME1pMmdKa2RET2VMOGhUU1g2UG9FRkNuNGpvMVlndjlpT2thV2hXTWpIek9hdkZtR0JLU2xsZDRybUt0c3lUS2RzR1pzQmtkelp4eDUvL2l5UWs1UHFVNC9xZDg5ejNvQ0FYcmJXaUlCOVVvM3ovREVRQTBJNXBMYktPM3lIVFJRSTEvMlhMSUc4WkxRQkV6Y1Z6SnI1cTVaU1VjcDBWYnZSaWpRcHpkRkxXcG9FcTNsZ2kzZXJlK25xNmR2WDdlempCMWlPK2ZnMGpvOU9Ld3gxb2cwa1hEa094SGdsTFpNT1paU3ZaY1pYNEc2N2gyNFFBUVA4cDFIdXliVTcyek1JMFU0NzE0TkpFSVJjNUZyZC90RmpjQ2ZZcElhOWdtYk9ocmFUdVgvbENOaHlQRkZNdHA0bVBEcUNOTHp5ZEVXTmlTZEluWWNhNjRRc2tGRUY3Q2wydjFpL0VEU0VxTmR2R3JpS3FOYWs5YVVROXkxZVZDWVFmSExkYWVJdTZIUVNFclZmRm9nOUFEc2hRRDh1bE5FSm1OSXcyZkp5a1hKbVV5T3h0bkg2SVVERUdZZDB3QWE5aTRBTGZnTGdqcFRkQ2NhSDA5T1pDS2RPZ1J6UEd6NncvMEJmanRaN1BEVDR1YmZDVFhCaDdYSmVaZWxGSzYrckNPUXU1YzZhN3JNdE14V3ZURlJiY0hSSXVzRFNFbkxuOUVMTHp3TW0xdEJkNkk3ZmtlSzZaclBCWjNoaVIybmV0aXAySjYwZXJ2cmF3Z3dxZkVKUkJ5L1dMVEs1bjJXY1QrMDVTdXhZRWdHQzJ5am5LYXVRbEJMMlBvZ3p2azhDR05PNGs0bDFYQkxsL0hjYzl0Vm42NlVNeFZQcVRjSDF1NHRYam5ycktBWVNvTUc0UnlvMXBZa0h2czY0MzRWdHVGeEtaRUVaVXA5Z3RZb0hsQ3RuYmdsU2JFMkgxNHlhUnNCaXp4Nk5FclVaRGx0U01GZUZIaC9WVXg4ek1zbGFXK09VNGdUVXQrN2ZFeU5WZ2M5Y3lGWkhDYXh0VHAzRFFjYjdkeEZFUER0NzdObFZBN2hJeFZEOEhTTVpDdHlsZCs4MlF2dTRpMXNZOG1MR2FodHJMVXdGbUVXZ014dUphUHN3SXV4T0JSeXhqQUg3bk5JU0xVMFlnbEw4Z0EwRXpiR3BFZExWeWFBZUpkaFl6Y0RMR3AyTGFhVno2SEh5SUtPUENSV2RnK1RDWXBJaHY2cTNITzJLNlpOYkQ3YzJhc1VLY29RRm1YRmFVZ3d1ZUxLVmpIOTJ4dExKbmJWa1dGN2JjODdTaEVMc2x0VjgvWVd3YjZoVmRwTXV6eG50WXFIdlRyZ2RKd1VhN0RhNEhJaXRKREFhWFJHTjM3T01nNURZZHRUeTBuc3FZZi82N0xmS0ZNVitTalVjbWpkajdhSElZQldsOC9jbE85SVJqWmF2Z0x6ZEVlSTRaSmVDSEl0SzBxSTBUa1lJZ0N5RU8ybzBvTDJBMHRGQWdKRTNtb0p5WjZLYmZkcUNDaWhlS28rMmhOZHZTSEpidnZPZzRnMUs4SzVCa1VxK3VueFBERUZLVkRpWGZ4L3oxdE5mV3NlQXZNOVVUSFNudTNON3VKbHlFaXhNanZ2WTJmTHhiSTg1a0VCRFEwbFJ5YStFWFA0Y3h5WXBSSS9SZC9DNkN6Kys5bGd1d1RYSkFtakphMVE2OTMxVzRTSHNFWW41dWgybUhwYkIzcnBkeTR5YW9XOXgvYVpPdUJ3WXhGZFQ2QzFkbGplSk5qNjYvZnNObjFrY2J4cElzczE4L2xwaXEvcTBJNHhTQ1hkbUcrNnRVaUtuSElHNHJmM0VSMkZYeWx3MTllVVVZR0JXQVAxVDJPZmQ1VlAyZFVyWmhWb1RYeFFmVW9PNm1ETDBDUlc2N2I5ZUx1T2lackFZSnMzNlJjM3drakxtSTNRWUhqcW9Lc3dDZDZoSXlNUzRoZ3lPampQRXFLTFhSOXNtOHVmN2l6U2YzSWtFeGI4QWF1WnVqVnJCTWJxS0JyQUhnQ0QyemVDQW5rRjJXRzlsVi9KQzVKOXFhcXZxb21WMFp1N0JMbWxYNUlMTnZFbUxrT0lhbGhZQlg0QXZRMlY0VDF6UEYvTXg5MGlXRzArSk1kNjY1dk5lRjMwMW1yYjRSRmdzcmNjV05iRERpNnlJMy9BUTk4NVJ0THZYVkRWbVI2SGpSc3Z3aUMrT1ZFL2VtK3lZUVVvQlg5Mi9Rc1JiRlZRbHlnRElYVXF1NjVpeXd6L3cyWnNWWU8xU3hXcUNPZlU5bVJVc2pWU0g1amNKc3A2L2ZEZjNrVmRVcEI0blVTaEZGZ2IvaDZrOGdHOHZGUDVNUzI3Qm9vTTI3NVJzRzVhZFlQL3dpNW5UeWpDQUhqOEg1ZUtid3FUMmNTdWtZTWFJQyttZlhSZlN1Vlc1SzZ4YkVERlJHOG5BZWtpZ2pnTHdxRzMrZzhXZGxaUVZRMXJvTWxCaXlyaWlialMySDhleVBnSWN3TUx0YnV5bzZ3Uy9ZeEtYSysxUm5obTg3ckZJcm5Bd1hneFhaMmhETkdoTHBZS200dEpZcUFCNUVsQlJlSWs4S1p3OWFJVkZMYStTZ0RZcUpkRnA5c3pWektzekYvSTBUM3hMZmRlcGVSUDhNRjJtSUowRHljZnZpNkNtdU5QTjBqZ3ErdkV2SFhoVnhHTmh6c2dZOWEzRFFHc2RBRmhuQWVzeWNka0VzZGtCUzFCQ1B5NmJkeHZlMDZRSkRYMkgweHJlbE0zcG05bFp2WVQ0bGVuVFI0RmY1SGpJUXpMckFEUmtrY2JQaVJabm5TUzJ3R0V4K0RLTnQ3VHcrbzFoTWk1clhmV1ZKUGdXUjR0d1AzRXN1SmEzSlFZN1YrRXFBb2ZEa0JnTzhyd1pzb0prekNYVDd5TGlzOXRtUjFkQnF1RXhZRkcxdGZiVzJBRnhJM2ZZcEN3WVZYamJzZ2Y4NnZTcHBPTkVaMkhyeEhZUm5VcjhqOEVndHFzVGRMdFdOSzNDbDNrcnprRlVSRmFPeGRRWVovSXQ5R2JTVjRqbWlwWm52Z0Z4ZFNoWEF4V3ZHZGs4MmQ5WmJpVFJvUDBzQlpVYU9mY0hHVjhXODlYY2FPcWpWcmlKbTZ6dU92QVRuTUx3cGVrLzV5VDI4QjJhdHRHdjJIbWhleDRqRWJpSXlLZzNVV1o1cjdHYkNmMTQ4ZlNqTWk4U0crQzhaM2t6Q1ozNnFSVzZyeHhLSEhCSTJlZ1dGb0d1QSt1RUNRQko0d0krREhScCtOUWFMM2xGZWVaN1dzNFpUTVVFYWpRdWxyMlVqM3VlWHJMMSthWXk3Mnh6T1gxV2EySUc0bmFQL2VBanBkd2xiblNaTWdPUnl1NkNrS0Y5QXJkeVJob1FoeWc0UGMzeUExOCtMcjZVWC9nMGlvS3JrU2o0YWFaaVRDcEx4Sm0rV2pWZU5UQ2gzdmRSWEpPaEN5RXFDWkJxR2xHV2kxOWdsa3FWN1IrOWNSQ09XbW9JT2QrOUx4SkxIZFcrUUY5aGtuLzNUdkRVbTA2SCtBblEzQ1MycUtVRWVhOHBUdDA2QzlSRENNU0VpUHNvb2ZXZ1dKUUswdjhVVWh3c2N6NDlkMERwS1JUamV3OWF1VEVsbDRkUjRYNkxHNUdFQ0x6RjBSRGdPUDMxQzRPOUtNSDBESW5qQVVPY2t3clc2V0U3eVRiRjZsRWJ3bVpVdm9YMXE3ZnI1MFM1UXdIU3BHZTY4RGc2eGswZkNqTDErRWY3bm1pS2tLcEMyN2xNdFlnYWFQcmkzNmlTTlJqRy9NbTNPQTVBNEsxY3JWTlZqTUp5blcrOTZwWFFDTE03dzRobDNxSGtKTUxOUlEybU40SUFTUmJkelpxMHRDa3RPWWw5M2F2Z0VKd0ZTNTB2TGhyTUZRYjhmenVjVGlRVEIySHZMRGxJa3RaMXNqKzVMOVhqYjAyMjhDcTd3N21FVWxLbU0rZ0tGQTFtdS9URWhjYXJsaE5qdnZzNGlWODhZdk14cDJydG1GbXhEcER1WUVDYkxRT3htWG52WXdPb3RPWXVyd1ZNL2xkaVgvdzdXT2psWC9RVzlKTW54MXorM1lsTlNwbjNvRmVhbkdSTWZoY2JhREtsOUhKRTVQVTFSUkR5ZDlGYVhwQnYzL2IwTUFHcHJQblduRXUxR01qVWY3N0NiWDhmbHdNeU1NNll2K0lXaGZ6Q1c5aUNNWklMK21EQWNocSs1REFmbjhqMnRGaDhsTVozaERWV1FISWxwaTkzU21LamhhZXovNTNyVEMyaVVRZmpWL1FMdXd3VDVTVzc0cGgzMTJwQnBOYUswVmh5Wmc3QmRiQ1dDdGxYUUc5QnBwcG9rc0NSU28yVzZwYndIREcrV3Z3OVFIbWV3YjRZLzcvYnJLTGRVK3JGNlpXZ2E2ODcybGNzWVJJVUJqaGJycmRucGJxcmM3cjVvajRoNFpLa2s2TjNxRTl4R3hCaXhORExCWmFrdi9Za0xCSldESTFUY21LSnV2cHhkbWZXWXlDZXlXUXozS1hPZ3ZVVHYzdXVNYWUvK3F3NjFxL2lQMXMwbjNFT2U4QUFpa1E0Y3hoUG5WZWVaZENJVkJHSXhpK2JEaEovbFBMZ2J4OVNZT2ZUZURSVnl0enBWYk1nQkRrUXVuRC9IZWdyT0Z1d1J2aFVSZjdlVC9YQXJWazhwRjZ6SWpXNTQ2dmtKZk8zZVNRZHdNanBCU1IvVWgveENXTGwyLzJPVUR2ZWVlYkdZMk1pU2QxZVNmRzlOcmQ0amtjRno2Y04ra25Sall6VDdVUE1NcEo5alFjL1krSm9Pd0w4MFYrdis5OFBHTFZvNFB3SWRLS1RQcjkzeVlkdXR5REYrWk54M2dXbTVjT3N5NjlycUFoMG5uL0FDaWV0QmhSVW15Q3FKUk4wdHZoM0NwbmhkbmJpQVFyNmtBNG85Q0FpUFphTUhMakJlbkc5N0RGb1NQSEE9IiwgIm1hYyI6ICJmOGNhN2U5N2ZjYWE2OGJiNjdmZjA1NTNjNDlmM2ZkZDllZDY2ZTRmMjcyNjgzMDRkYjEyNzU1NDY5M2QwMWFjIn0="
        # Round-trip with our own decryptor (should match the original string)
        roundtrip = encryptor_laravel.decrypt_text(encrypted_laravel_test)
        print(f"🔁 Round-trip decrypted: {roundtrip}")
        # Basic assertion for local round-trip
        # if roundtrip != TEST_STRING:
        #     raise AssertionError("Round-trip encryption/decryption mismatch.")

        # decrypted_laravel_test = encryptor_laravel.decrypt_text(LARAVEL_ENCRYPTED_SAMPLE)
        # print(f"🔓 Decrypted Laravel text: {decrypted_laravel_test}")
        # We don't assert equality with TEST_STRING here, since LARAVEL_ENCRYPTED_SAMPLE
        # may represent some other plaintext.

        # Test with the provided Laravel encrypted sample
        # We will decrypt the sample, but not assert its content against TEST_STRING
        # The decryption of LARAVEL_ENCRYPTED_SAMPLE is already handled above.
        
        print("\n" + "="*50 + "\n")

    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
