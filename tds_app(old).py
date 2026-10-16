import streamlit as st
from bfs_backtrack import bfs_search
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import io
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import docx
import tempfile
import traceback

# Load and verify environment variables
load_dotenv()
required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT_NAME']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Your persona prompt (with all curly braces in JSON/table example escaped as {{ ... }})
TDS_PERSONA_PROMPT = """
Role: You are a tax professional who is an expert in Tax Deducted at Source i.e. TDS provisions under Indian Income Tax Act.
Your capabilities include explaining TDS provisions, analyzing the transactions and suggesting the most appropriate TDS Section applicable to that transaction.
You serve individual taxpayers, small business owners, and tax professionals, offering a quick reference to TDS provisions and will solve all queries pertaining to TDS provisions
You address misunderstandings by referencing the updated TDS provisions, offering examples for clarity, and directing to professional advice for intricate matters.
If any question falls outside your domain, you will respond with "I am sorry. This question is not my expertise. Do you have any other queries pertaining to TDS provisions?"
If the query of the user is confusing or not clear, ask for more clarity by saying "Could you please rephrase that or provide more details? I'm not sure I understand what you're asking."
Your knowledge is based on the "Income Tax Act 1961 - TDS Provisions - 2024 Amendment.pdf" file uploaded in the 'Knowledge Centre'
While giving answers, please provide Confidence Score in the end.
Also provide the details about the source basis which the answer is derived from. If there are multiple sources, show all sources as a list. Also, if the user has uploaded any document or image, then please refer that document/image as one of the sources.
At the end, ask the user “Do you need any further information on the above response?”
 
The detailed tasks required to be done while answering any query and question asked by the user are as under:
Task 1 - If the User asks any question pertaining to applicability or non-applicability of TDS, you will use the "Income Tax Act 1961 - TDS Provisions - 2024 Amendment.pdf" file available in the Knowledge Centre to analyze and identify the initial TDS Section.
Task 2 - If the user has uploaded any document (such as Invoice, Purchase Order or Contract/Agreement), you will perform the following tasks:
​- Go through the entire document and identify “Description of Goods / Services” i.e. the goods or services for which the invoice is created
​- Analyze the “Description of Good/Services” and use the "Income Tax Act 1961 - TDS Provisions - 2024 Amendment" file available in the Knowledge Centre to analyze and identify the initial TDS Section.
Task 3 - If the Initial TDS Section is either Section 194J or Section 194C or Section 194H or Section 194I or No TDS, then please refer to these data sources for additional information about taxability:
​- For Section 194J, refer file ‘TDS Paper - Section 194J - v1 - 27.03.2025.docx’ uploaded in the ‘Knowledge Centre’
​- For Section 194C, refer file ‘TDS Paper - Section 194C- v1 - 0402.2025.docx’ uploaded in the ‘Knowledge Centre’
​- For Section 194H, refer file ‘TDS Paper - Section 194H - v1 - 0603.2025.docx’ uploaded in the ‘Knowledge Centre’
​- For Section 194I, refer file ‘TDS Paper - Section 194I - v1 - 27.03.2025.docx’ uploaded in the ‘Knowledge Centre’
​- For No TDS, refer file ‘TDS Paper - NO TDS - v1 - 28.03.2025.docx’ uploaded in the ‘Knowledge Centre’
 
Some KEY CONSIDERATIONS while answering the query:
1. CONTEXT-DRIVEN RESPONSES: You must base your responses strictly on the provided context and the documents without relying on pre-existing knowledge or external open-source/online information.
2. THRESHOLD ASSUMPTION: Assume that all applicable threshold limits for deductors and deductees are met, making TDS provisions applicable.
3. SUB-SECTION BIFURCATION:
- If Section 194J is applicable, specify the correct sub-section based on the nature of the service:
- 194J(1)(a): Fees for professional services
- 194J(1)(b): Fees for technical services
- 194J(1)(ba): Any remuneration or fees or commission by whatever name called, other than those on which tax is deductible under section 192 (TDS on Salary), to a director of a company
- 194J(1)(c): Payment of royalty for sale, distribution or exhibition of cinematographic films.
- 194J(1)(d): Non-compete fees and other compensation in connection with business or profession.
- If Section 194I is applicable, specify the correct sub-section based on the type of asset rented:
- 194I(1)(a): Rent on plant, machinery, or equipment.
- 194I(1)(b): Rent on land, building, furniture.
4. GOODS-BASED IDENTIFICATION (SECTION 194Q):
- If the description explicitly contains any goods name, the term "goods," or "purchase of goods" or “Supply of goods” (without any services included), Section 194Q should be applied directly without evaluating other TDS sections.
5. ALTERNATIVE TDS SECTIONS: If another TDS section is potentially applicable, suggest it along with justification, enabling users to make an informed decision.
6. ACCURACY & VERIFICATION: Cross-verify all responses against the given context to ensure correctness, completeness, and freedom from hallucinations, errors, or omissions.
 
DESIRED OUTPUT FORMAT
1.​STRICTLY SHOW THE OUTPUT IN THE FOLLOWING FORMAT:
First visualize the answer in the following JSON Format and convert it into a table markdown format for the user and give definite answers only
[
{{
"Sr. No.": "1",
"Goods or Services": "Contract Services",
"Brief Explanation of Goods or services": "Services provided under contract agreement",
"TDS Section": "194C(1)",
"TDS Rate": ["1% for Individual/HUF", "2% for Others"],  # Modified to handle multiple rates
"Rate Details": "1% where payment is to individual/HUF; 2% where payment is to others",  # New field
"Probability TDS Section": "Probability percentage of this TDS section being applicable",
"Reason TDS Section": "Justification/conditions for the TDS section's applicability...",
"Alt TDS Section": "Alternate applicable TDS section (if any)",
"Alt TDS Rate": "TDS rate under the alternate section",
"Prob Alt TDS Section": "Probability percentage of the alternate TDS section being applicable",
"Reason Alt TDS Section": "Justification/conditions for the alternate TDS section's applicability..."
}}
]

# Special handling instructions for Section 194C:
When the applicable section is 194C(1), always show both rates:
- 1% for payments to individual/HUF
- 2% for payments to other entities
Make sure to explain the rate differentiation in the "Rate Details" field.

2. Do not show the JSON output to the user. Only populate the Final Table format
3. ONLY IN AN EXTREME SITUATION WHERE THE QUERY OR QUESTION ASKED BY THE USER IS SUCH THAT IT CANNOT BE SHOWN IN THE ABOVE FORMAT, YOU CAN ANSWER IN ANY FORMAT

User Query:
{query}

Context Chunks:
{context}

{uploaded_info}
"""

# Azure OpenAI config
llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-05-15"),
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.2,
)

def get_context_for_query(query, max_docs=20, threshold=0.5):
    docs = bfs_search(query, threshold=threshold, max_docs=max_docs)
    context = []
    for doc in docs:
        meta = doc.metadata
        ref = f"{meta.get('source_file', 'unknown')} (chunk {meta.get('chunk_index', meta.get('cluster', '-') )})"
        context.append(f"[{ref}]: {doc.page_content[:500]}")  # Truncate for prompt size
    return "\n\n".join(context)

def run_persona(query, uploaded_text=None, uploaded_image=None):
    context = get_context_for_query(query)
    uploaded_info = f"Uploaded Document Content:\n{uploaded_text}" if uploaded_text else ""
    if uploaded_image:
        uploaded_info += "\nUploaded Image Content: Image provided"
    prompt = TDS_PERSONA_PROMPT.format(query=query, context=context, uploaded_info=uploaded_info)
    chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
    return chain.invoke({"prompt": prompt})

# --- Payment Type Categories (TDS-specific, 17 types) ---
TDS_PAYMENT_TYPES = [
    "Salary", "Interest other than securities", "Interest on securities", "Dividends", "Winnings from lottery or crossword puzzle",
    "Winnings from horse race", "Insurance commission", "Payment to contractors/sub-contractors", "Brokerage or commission",
    "Rent", "Fees for professional or technical services", "Royalty", "Commission or brokerage", "Transfer of immovable property",
    "Payment of compensation on acquisition of certain immovable property", "Payment to non-resident sportsmen or sports associations",
    "Other"
]

# --- Helper: Extract text from uploaded invoice (multi-format) ---
def extract_invoice_text(uploaded_file):
    try:
        file_type = uploaded_file.type
        if file_type.startswith('image'):
            try:
                import pytesseract
                from PIL import Image
                image = Image.open(uploaded_file)
                text = pytesseract.image_to_string(image)
                return text, 'image'
            except Exception as e:
                return f"[OCR Error: {e}]", 'image'
        elif file_type == 'application/pdf':
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp.flush()
                    doc = fitz.open(tmp.name)
                    text = "\n".join(page.get_text() for page in doc)
                    doc.close()
                return text, 'pdf'
            except Exception as e:
                return f"[PDF Extraction Error: {e}]", 'pdf'
        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp.flush()
                    doc = docx.Document(tmp.name)
                    text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
                return text, 'docx'
            except Exception as e:
                return f"[DOCX Extraction Error: {e}]", 'docx'
        elif file_type == 'text/plain':
            try:
                text = uploaded_file.read().decode(errors="ignore")
                return text, 'txt'
            except Exception as e:
                return f"[TXT Extraction Error: {e}]", 'txt'
        else:
            return "[Unsupported file type]", 'unknown'
    except Exception as e:
        return f"[General Extraction Error: {e}]", 'unknown'

# --- Helper: LLM-based payment type extraction ---
def classify_payment_type_llm(invoice_text, llm, max_tokens=2000):
    # Truncate to avoid token overflow
    truncated_text = invoice_text[:8000] if len(invoice_text) > 8000 else invoice_text
    prompt = f"""
You are a tax expert AI. Given the following invoice or bill text, classify the payment type into one of the following 17 TDS-specific categories (output ONLY the category name):
{TDS_PAYMENT_TYPES}

Also provide a confidence score (0-100) for your classification.

If the text is not an invoice or is ambiguous, return 'Other' with low confidence.

Respond in this JSON format:
{{
  "payment_type": <one of the 17 categories>,
  "confidence": <integer 0-100>,
  "explanation": <short explanation>
}}

Invoice Text:
"""
    prompt += truncated_text
    chain = ChatPromptTemplate.from_template("{prompt}") | llm | StrOutputParser()
    try:
        response = chain.invoke({"prompt": prompt})
        import json
        # Defensive: try to parse JSON, fallback to string search
        try:
            result = json.loads(response)
            payment_type = result.get("payment_type", "Other")
            confidence = result.get("confidence", 0)
            explanation = str(result.get("explanation", ""))
        except Exception:
            # Fallback: try to extract from string
            import re
            payment_type = next((pt for pt in TDS_PAYMENT_TYPES if pt.lower() in response.lower()), "Other")
            conf_match = re.search(r"confidence[\":\s]*(\d+)", response, re.I)
            confidence = int(conf_match.group(1)) if conf_match else 0
            explanation = response
        # Validate payment type
        if payment_type not in TDS_PAYMENT_TYPES:
            payment_type = "Other"
        return payment_type, confidence, explanation, response
    except Exception as e:
        return "Other", 0, f"[LLM Error: {e}]", ""

# --- Enhanced Streamlit UI ---
st.set_page_config(page_title="TDS Provisions Expert (RAG System)", layout="wide")

with st.sidebar:
    st.header("TDS RAG System: Invoice Analysis")
    st.markdown("""
    **Features:**
    - Upload invoices (PDF, DOCX, TXT, Image)
    - AI-powered payment type classification (17 TDS categories)
    - Confidence scoring and explanation
    - Auto-suggested TDS queries
    - Visualize extracted text and classification
    - Progress indicators and error handling
    """)
    st.info("Install Tesseract for OCR: `brew install tesseract` (Mac) or see https://github.com/tesseract-ocr/tesseract")

st.title("TDS Provisions Expert (RAG System)")

query = st.text_area("Enter your TDS query:")
uploaded_file = st.file_uploader(
    "Upload a document (optional)", 
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg"]
)

uploaded_text = None
uploaded_image = None
invoice_text = None
payment_type = None
confidence = None
explanation = None
llm_response = None
file_format = None

if uploaded_file is not None:
    with st.spinner("Extracting text from uploaded file..."):
        invoice_text, file_format = extract_invoice_text(uploaded_file)
        if file_format == 'image':
            try:
                from PIL import Image
                image = Image.open(uploaded_file)
                uploaded_image = image
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        elif file_format in ['pdf', 'docx', 'txt']:
            uploaded_text = invoice_text
        else:
            st.warning("Unsupported or failed file extraction.")

    # --- LLM-based payment type extraction ---
    with st.spinner("Classifying payment type using AI..."):
        payment_type, confidence, explanation, llm_response = classify_payment_type_llm(invoice_text, llm)

    # --- Show extracted text and classification ---
    st.subheader("Extracted Invoice Text")
    st.code(invoice_text[:2000] + ("..." if len(invoice_text) > 2000 else ""), language="text")
    st.subheader("AI Payment Type Classification")
    st.markdown(f"**Payment Type:** `{payment_type}`")
    st.markdown(f"**Confidence:** `{confidence}`")
    # Try to pretty-print explanation if it's JSON, else show as plain text
    import json
    try:
        explanation_obj = json.loads(explanation)
        st.markdown("**Explanation:**")
        st.json(explanation_obj)
    except Exception:
        st.markdown(f"**Explanation:** {explanation}")

    # --- Auto-suggested TDS query ---
    st.subheader("Suggested TDS Query")
    st.markdown(f"_What is the TDS applicability for payment type: **{payment_type}**?_")

if st.button("Get TDS Analysis"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing..."):
            # If invoice was uploaded, pass extracted text as context
            result = run_persona(query, uploaded_text or invoice_text, uploaded_image)
            st.markdown(result)


