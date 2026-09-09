import streamlit as st
import fitz
import pytesseract
import json
import os
import re
import io
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocTruth AI",
    page_icon="🛡️",
    layout="wide"
)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY not found. Please check your .env file."
    )
    st.stop()

# Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.5-flash-lite"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🛡️ DocTruth AI")
st.caption(
    "AI-powered document intelligence with extraction, validation and evidence verification."
)

st.divider()


# ============================================================
# FUNCTIONS
# ============================================================

def extract_text_from_image(uploaded_file):
    """
    Extract text from an uploaded image using Tesseract OCR.
    """

    image = Image.open(uploaded_file)

    text = pytesseract.image_to_string(
        image
    )

    return text


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF.
    First tries normal PDF text extraction.
    If a page has no text, OCR is used.
    """

    pdf_bytes = uploaded_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    all_text = []

    for page_number, page in enumerate(document):

        # Try normal PDF text extraction
        page_text = page.get_text(
            "text"
        ).strip()

        if page_text:
            all_text.append(
                f"\n--- Page {page_number + 1} ---\n"
            )
            all_text.append(
                page_text
            )

        else:
            # OCR fallback for scanned pages
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2)
            )

            image_bytes = pix.tobytes(
                "png"
            )

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            ocr_text = pytesseract.image_to_string(
                image
            )

            all_text.append(
                f"\n--- Page {page_number + 1} ---\n"
            )

            all_text.append(
                ocr_text
            )

    document.close()

    return "\n".join(
        all_text
    )


def extract_document_text(uploaded_file):
    """
    Decide whether the uploaded file is a PDF or image.
    """

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(
            uploaded_file
        )

    else:
        return extract_text_from_image(
            uploaded_file
        )


def extract_structured_data(document_text):
    """
    Send extracted document text to Gemini
    and return structured JSON.
    """

    prompt = f"""
You are a document intelligence system.

Your job is to extract structured information from the
provided document text.

IMPORTANT RULES:

1. Never invent information.
2. Only use information explicitly present in the document.
3. If a field is missing, return null.
4. Every extracted field must include evidence.
5. Evidence must be an exact short quote copied from the document text.
6. Do not create evidence that does not exist.
7. Preserve values as accurately as possible.
8. For vendor_name, only identify a vendor/company/service provider
   when the document clearly supports it.
9. total_amount should contain the numeric total amount if available.
10. Return ONLY valid JSON.

Return exactly this structure:

{{
    "document_type": {{
        "value": "...",
        "evidence": "..."
    }},
    "invoice_number": {{
        "value": "...",
        "evidence": "..."
    }},
    "date": {{
        "value": "...",
        "evidence": "..."
    }},
    "customer_name": {{
        "value": "...",
        "evidence": "..."
    }},
    "vendor_name": {{
        "value": "...",
        "evidence": "..."
    }},
    "total_amount": {{
        "value": "...",
        "evidence": "..."
    }},
    "currency": {{
        "value": "...",
        "evidence": "..."
    }},
    "phone": {{
        "value": "...",
        "evidence": "..."
    }},
    "email": {{
        "value": "...",
        "evidence": "..."
    }}
}}

If a field does not exist:

{{
    "value": null,
    "evidence": null
}}

DOCUMENT TEXT:

-------------------------
{document_text}
-------------------------
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    return response.text


def clean_json_response(text):
    """
    Remove accidental markdown JSON fences.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text[
            7:
        ]

    elif text.startswith("```"):
        text = text[
            3:
        ]

    if text.endswith("```"):
        text = text[
            :-3
        ]

    return text.strip()


def normalize_text(text):
    """
    Normalize text for evidence comparison.
    """

    if not text:
        return ""

    return " ".join(
        str(text).lower().split()
    )


def simplified_text(text):
    """
    Remove punctuation and spaces for
    a more tolerant evidence comparison.
    """

    if not text:
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower()
    )


def verify_evidence(
    evidence,
    original_text
):
    """
    Check whether the evidence actually exists
    in the source document.
    """

    if not evidence:
        return False

    normalized_evidence = normalize_text(
        evidence
    )

    normalized_document = normalize_text(
        original_text
    )

    if (
        normalized_evidence
        in normalized_document
    ):
        return True

    simplified_evidence = simplified_text(
        evidence
    )

    simplified_document = simplified_text(
        original_text
    )

    if (
        simplified_evidence
        and simplified_evidence
        in simplified_document
    ):
        return True

    return False


def validate_field(
    field,
    value
):
    """
    Basic rule-based validation.
    """

    if value is None:
        return "Missing"

    value_string = str(
        value
    ).strip()

    if not value_string:
        return "Missing"

    # Email validation
    if field == "email":

        email_pattern = (
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        )

        if re.match(
            email_pattern,
            value_string
        ):
            return "Valid"

        return "Invalid"

    # Phone validation
    if field == "phone":

        digits = re.sub(
            r"\D",
            "",
            value_string
        )

        if 10 <= len(digits) <= 15:
            return "Valid"

        return "Invalid"

    # Amount validation
    if field == "total_amount":

        amount = re.sub(
            r"[^\d.]",
            "",
            value_string
        )

        try:
            float(amount)
            return "Valid"

        except ValueError:
            return "Invalid"

    # Date validation
    if field == "date":

        date_formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y-%m-%d",
            "%d/%m/%y",
            "%d-%m-%y"
        ]

        for fmt in date_formats:

            try:
                datetime.strptime(
                    value_string,
                    fmt
                )

                return "Valid"

            except ValueError:
                continue

        return "Invalid"

    # Document type
    if field == "document_type":

        if len(value_string) > 1:
            return "Valid"

        return "Invalid"

    # Other fields
    return "Valid"


def calculate_confidence(
    value,
    status,
    evidence_verified
):
    """
    Calculate field-level confidence.
    """

    if value is None:
        return 0

    score = 50

    if evidence_verified:
        score += 30

    if status == "Valid":
        score += 20

    elif status == "Invalid":
        score -= 30

    if not evidence_verified:
        score -= 20

    score = max(
        0,
        min(
            100,
            score
        )
    )

    return score


def confidence_label(score):

    if score >= 80:
        return "🟢 High"

    elif score >= 50:
        return "🟡 Medium"

    return "🔴 Low"


def process_validation(
    structured_data,
    original_text
):
    """
    Validate all extracted fields and
    calculate confidence.
    """

    results = {}

    for field, details in structured_data.items():

        if isinstance(
            details,
            dict
        ):

            value = details.get(
                "value"
            )

            evidence = details.get(
                "evidence"
            )

        else:

            value = details
            evidence = None

        status = validate_field(
            field,
            value
        )

        evidence_verified = verify_evidence(
            evidence,
            original_text
        )

        confidence = calculate_confidence(
            value,
            status,
            evidence_verified
        )

        results[field] = {
            "value": value,
            "evidence": evidence,
            "status": status,
            "evidence_verified": evidence_verified,
            "confidence": confidence
        }

    return results


def calculate_overall_quality(
    validation_results
):
    """
    Calculate average confidence
    of fields that were extracted.
    """

    scores = []

    for field, result in validation_results.items():

        if result["value"] is not None:
            scores.append(
                result["confidence"]
            )

    if not scores:
        return 0

    return round(
        sum(scores) / len(scores)
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF or image document",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file:

    st.info(
        f"Selected document: **{uploaded_file.name}**"
    )

    if st.button(
        "🚀 Analyze Document",
        type="primary"
    ):

        # ----------------------------------------------------
        # STEP 1 — TEXT EXTRACTION
        # ----------------------------------------------------

        with st.spinner(
            "Extracting text from document..."
        ):

            try:

                full_text = extract_document_text(
                    uploaded_file
                )

            except Exception as e:

                st.error(
                    f"Text extraction failed: {e}"
                )

                st.stop()

        if not full_text.strip():

            st.error(
                "No text could be extracted from the document."
            )

            st.stop()

        # Save extracted text
        st.session_state[
            "document_text"
        ] = full_text

        # ----------------------------------------------------
        # SHOW EXTRACTED TEXT
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Extracted Text"
        ):

            st.text_area(
                "Document text",
                full_text,
                height=300
            )

        # ----------------------------------------------------
        # STEP 2 — AI EXTRACTION
        # ----------------------------------------------------

        with st.spinner(
            "AI is analyzing the document..."
        ):

            try:

                structured_json = (
                    extract_structured_data(
                        full_text
                    )
                )

                structured_json = (
                    clean_json_response(
                        structured_json
                    )
                )

                structured_data = json.loads(
                    structured_json
                )

                # Save extracted data
                st.session_state[
                    "structured_data"
                ] = structured_data

                st.success(
                    "Structured extraction completed!"
                )

            except Exception as e:

                st.error(
                    f"AI extraction failed: {e}"
                )

                st.stop()

        # ----------------------------------------------------
        # STEP 3 — VALIDATION
        # ----------------------------------------------------

        validation_results = (
            process_validation(
                structured_data,
                full_text
            )
        )

        overall_quality = (
            calculate_overall_quality(
                validation_results
            )
        )

        st.session_state[
            "validation_results"
        ] = validation_results

        st.session_state[
            "quality_score"
        ] = overall_quality

        # ----------------------------------------------------
        # STEP 4 — RESULTS DASHBOARD
        # ----------------------------------------------------

        st.divider()

        st.header(
            "🛡️ Verification & Validation"
        )

        # Overall score
        st.metric(
            "Overall Extraction Quality",
            f"{overall_quality}%"
        )

        st.markdown(
            "### 📊 Field-Level Confidence"
        )

        # ----------------------------------------------------
        # FIELD RESULTS
        # ----------------------------------------------------

        for field, result in validation_results.items():

            value = result[
                "value"
            ]

            evidence = result[
                "evidence"
            ]

            status = result[
                "status"
            ]

            evidence_verified = result[
                "evidence_verified"
            ]

            confidence = result[
                "confidence"
            ]

            label = confidence_label(
                confidence
            )

            field_name = (
                field
                .replace(
                    "_",
                    " "
                )
                .title()
            )

            # Field heading
            st.markdown(
                f"**{field_name}** — "
                f"{label} ({confidence}%)"
            )

            # Value
            if value is not None:

                st.write(
                    f"**Value:** {value}"
                )

            else:

                st.warning(
                    f"{field_name}: No value found."
                )

            # Validation
            if value is not None:

                if status == "Valid":

                    st.success(
                        f"{field_name}: Valid"
                    )

                elif status == "Invalid":

                    st.error(
                        f"{field_name}: Invalid value"
                    )

            # Evidence
            if evidence:

                if evidence_verified:

                    st.info(
                        f'📌 Evidence verified: "{evidence}"'
                    )

                else:

                    st.warning(
                        f'⚠️ Evidence needs review: "{evidence}"'
                    )

            st.divider()

        # ----------------------------------------------------
        # CLEAN STRUCTURED DATA TABLE
        # ----------------------------------------------------

        st.markdown(
            "### 📋 Structured Results"
        )

        table_data = []

        for field, result in validation_results.items():

            table_data.append(
                {
                    "Field": field.replace(
                        "_",
                        " "
                    ).title(),

                    "Value": (
                        result["value"]
                        if result["value"] is not None
                        else "Not found"
                    ),

                    "Validation": result[
                        "status"
                    ],

                    "Confidence": (
                        f'{result["confidence"]}%'
                    ),

                    "Evidence Verified": (
                        "Yes"
                        if result[
                            "evidence_verified"
                        ]
                        else "No"
                    )
                }
            )

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # RAW JSON
        # ----------------------------------------------------

        with st.expander(
            "🧩 View Raw JSON"
        ):

            st.json(
                structured_data
            )

        # ----------------------------------------------------
        # DOWNLOAD JSON
        # ----------------------------------------------------

        json_download = json.dumps(
            structured_data,
            indent=4,
            ensure_ascii=False
        )

        st.download_button(
            label="📥 Download Structured JSON",
            data=json_download,
            file_name="doctruth_extraction.json",
            mime="application/json"
        )