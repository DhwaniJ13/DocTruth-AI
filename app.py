import streamlit as st
import fitz
import pytesseract
from PIL import Image
import io
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# AI EXTRACTION WITH EVIDENCE
# ============================================================

def extract_structured_data(text):

    prompt = """
You are a document intelligence assistant.

Extract important information from the document.

For every extracted field, return:
- value
- evidence

The evidence MUST be an exact short quote copied from the
provided document text that supports the value.

If a field is not present:
- value = null
- evidence = null

Return JSON only.

Use these fields:

- document_type
- invoice_number
- date
- customer_name
- vendor_name
- total_amount
- currency
- phone
- email

Rules:

1. Never invent information.
2. Evidence must come directly from the document.
3. Keep evidence short.
4. Do not create evidence for a missing field.
5. Only classify something as vendor_name if the document
   clearly identifies it as a vendor/company/service provider.
6. For total_amount, return the numeric value.
7. Preserve original information when possible.

Example:

{
    "document_type": {
        "value": "invoice",
        "evidence": "Invoice"
    },
    "invoice_number": {
        "value": "INV-1024",
        "evidence": "Invoice No: INV-1024"
    },
    "date": {
        "value": "15/09/2026",
        "evidence": "Date: 15/09/2026"
    },
    "customer_name": {
        "value": null,
        "evidence": null
    }
}

DOCUMENT:

""" + text

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    return response.text


# ============================================================
# EVIDENCE VERIFICATION
# ============================================================

def verify_evidence(value, evidence, original_text):

    if not value or not evidence:
        return False

    normalized_document = re.sub(
        r"\s+",
        " ",
        original_text
    ).strip().lower()

    normalized_evidence = re.sub(
        r"\s+",
        " ",
        str(evidence)
    ).strip().lower()

    # Exact evidence match
    if normalized_evidence in normalized_document:
        return True

    # Simplified comparison for OCR punctuation differences
    simplified_document = re.sub(
        r"[^a-z0-9@.\s]",
        "",
        normalized_document
    )

    simplified_evidence = re.sub(
        r"[^a-z0-9@.\s]",
        "",
        normalized_evidence
    )

    return simplified_evidence in simplified_document


# ============================================================
# VALIDATION + CONFIDENCE
# ============================================================

def validate_data(data, original_text):

    results = []

    overall_score = 0
    scored_fields = 0

    for field_name, field_data in data.items():

        # ----------------------------------------------------
        # Extract value and evidence
        # ----------------------------------------------------

        if isinstance(field_data, dict):

            value = field_data.get("value")
            evidence = field_data.get("evidence")

        else:

            value = field_data
            evidence = None

        display_name = field_name.replace(
            "_",
            " "
        ).title()


        # ----------------------------------------------------
        # Missing field
        # ----------------------------------------------------

        if value is None or str(value).strip() == "":

            results.append(
                {
                    "field": display_name,
                    "status": "Missing",
                    "message": "No value found.",
                    "evidence_verified": False,
                    "value": None,
                    "evidence": None,
                    "confidence": 0
                }
            )

            continue


        # ----------------------------------------------------
        # Evidence verification
        # ----------------------------------------------------

        evidence_verified = verify_evidence(
            value,
            evidence,
            original_text
        )


        # ----------------------------------------------------
        # Base confidence
        # ----------------------------------------------------

        field_confidence = 50


        # Evidence verified
        if evidence_verified:
            field_confidence += 30


        # ----------------------------------------------------
        # Email validation
        # ----------------------------------------------------

        if field_name == "email":

            email_pattern = (
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
            )

            if re.match(
                email_pattern,
                str(value)
            ):

                status = "Valid"

                message = (
                    "Email format is valid."
                )

                field_confidence += 20

            else:

                status = "Invalid"

                message = (
                    "Email format appears incorrect."
                )

                field_confidence -= 30


        # ----------------------------------------------------
        # Phone validation
        # ----------------------------------------------------

        elif field_name == "phone":

            digits = re.sub(
                r"\D",
                "",
                str(value)
            )

            if 10 <= len(digits) <= 15:

                status = "Valid"

                message = (
                    "Phone number has a plausible length."
                )

                field_confidence += 20

            else:

                status = "Invalid"

                message = (
                    "Phone number length looks unusual."
                )

                field_confidence -= 30


        # ----------------------------------------------------
        # Date validation
        # ----------------------------------------------------

        elif field_name == "date":

            valid_date = False

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
                        str(value),
                        fmt
                    )

                    valid_date = True
                    break

                except ValueError:
                    pass

            if valid_date:

                status = "Valid"

                message = (
                    "Date format is valid."
                )

                field_confidence += 20

            else:

                status = "Needs review"

                message = (
                    "Date was extracted but could "
                    "not be verified."
                )

                field_confidence -= 10


        # ----------------------------------------------------
        # Amount validation
        # ----------------------------------------------------

        elif field_name == "total_amount":

            try:

                float(
                    str(value).replace(",", "")
                )

                status = "Valid"

                message = (
                    "Amount is numeric."
                )

                field_confidence += 20

            except ValueError:

                status = "Invalid"

                message = (
                    "Amount is not numeric."
                )

                field_confidence -= 30


        # ----------------------------------------------------
        # Document type
        # ----------------------------------------------------

        elif field_name == "document_type":

            status = "Detected"

            message = (
                "Document type identified."
            )


        # ----------------------------------------------------
        # Other fields
        # ----------------------------------------------------

        else:

            status = "Extracted"

            message = (
                "Value extracted from document."
            )


        # ----------------------------------------------------
        # Evidence failure
        # ----------------------------------------------------

        if not evidence_verified:

            field_confidence -= 20

            message += (
                " Evidence could not be verified "
                "against the original text."
            )


        # ----------------------------------------------------
        # Keep confidence between 0 and 100
        # ----------------------------------------------------

        field_confidence = max(
            0,
            min(100, field_confidence)
        )


        # Add to overall score
        overall_score += field_confidence
        scored_fields += 1


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "field": display_name,
                "status": status,
                "message": message,
                "evidence_verified": evidence_verified,
                "value": value,
                "evidence": evidence,
                "confidence": field_confidence
            }
        )


    # --------------------------------------------------------
    # Overall confidence
    # --------------------------------------------------------

    if scored_fields > 0:

        overall_score = round(
            overall_score / scored_fields
        )

    else:

        overall_score = 0


    return results, overall_score


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="DocTruth AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocTruth AI")

st.write(
    "From unstructured documents to structured, verified data."
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.subheader("📤 Upload your document")

uploaded_file = st.file_uploader(
    "Upload a PDF or image",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    st.success(
        "Document uploaded successfully!"
    )

    st.write(
        "**File name:**",
        uploaded_file.name
    )

    st.write(
        "**File type:**",
        uploaded_file.type
    )

    st.write(
        "**File size:**",
        uploaded_file.size,
        "bytes"
    )

    st.divider()


    # ========================================================
    # IMAGE OCR
    # ========================================================

    if uploaded_file.type.startswith("image/"):

        st.subheader("🔍 OCR Processing")

        image = Image.open(
            io.BytesIO(file_bytes)
        )

        st.image(
            image,
            caption="Uploaded document",
            width=500
        )

        with st.spinner(
            "Reading document with OCR..."
        ):

            full_text = pytesseract.image_to_string(
                image
            )

        st.success("OCR completed!")

        st.subheader("📝 Extracted Text")

        st.text_area(
            "OCR Result",
            full_text,
            height=400
        )


    # ========================================================
    # PDF PROCESSING
    # ========================================================

    elif uploaded_file.type == "application/pdf":

        document = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        st.write(
            "**Number of pages:**",
            len(document)
        )

        full_text = ""

        for page_number, page in enumerate(document):

            text = page.get_text()

            if text.strip():

                full_text += text + "\n"

                with st.expander(
                    f"Page {page_number + 1} — Text extracted"
                ):

                    st.text(text)

            else:

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2)
                )

                image = Image.frombytes(
                    "RGB",
                    [
                        pix.width,
                        pix.height
                    ],
                    pix.samples
                )

                with st.spinner(
                    f"Running OCR on page {page_number + 1}..."
                ):

                    ocr_text = pytesseract.image_to_string(
                        image
                    )

                full_text += ocr_text + "\n"

                with st.expander(
                    f"Page {page_number + 1} — OCR"
                ):

                    st.text(ocr_text)


        st.divider()

        st.subheader(
            "📝 Complete Document Text"
        )

        st.text_area(
            "Extracted text",
            full_text,
            height=400
        )


    # ========================================================
    # AI EXTRACTION
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 AI Structured Extraction"
    )

    if st.button(
        "Extract structured data",
        type="primary"
    ):

        if full_text.strip():

            with st.spinner(
                "AI is analyzing the document..."
            ):

                try:

                    structured_json = (
                        extract_structured_data(
                            full_text
                        )
                    )

                    structured_data = json.loads(
                        structured_json
                    )

                    st.session_state[
                        "structured_data"
                    ] = structured_data

                    st.session_state[
                        "document_text"
                    ] = full_text

                    st.success(
                        "Structured extraction completed!"
                    )

                    st.json(
                        structured_data
                    )

                except Exception as e:

                    st.error(
                        f"AI extraction failed: {e}"
                    )

        else:

            st.warning(
                "No text was found in the document."
            )


    # ========================================================
    # VALIDATION + EVIDENCE + CONFIDENCE
    # ========================================================

    if "structured_data" in st.session_state:

        st.divider()

        st.subheader(
            "🛡️ Verification & Validation"
        )

        validation_results, overall_score = (
            validate_data(
                st.session_state[
                    "structured_data"
                ],
                st.session_state[
                    "document_text"
                ]
            )
        )


        # ====================================================
        # OVERALL SCORE
        # ====================================================

        st.metric(
            "Overall Extraction Quality",
            f"{overall_score}%"
        )


        # ====================================================
        # FIELD-LEVEL RESULTS
        # ====================================================

        st.subheader(
            "📊 Field-Level Confidence"
        )

        for result in validation_results:

            field = result["field"]
            status = result["status"]
            message = result["message"]
            evidence = result.get(
                "evidence"
            )
            evidence_verified = result.get(
                "evidence_verified",
                False
            )
            confidence = result.get(
                "confidence",
                0
            )


            # ------------------------------------------------
            # Confidence label
            # ------------------------------------------------

            if confidence >= 80:

                confidence_label = "🟢 High"

            elif confidence >= 50:

                confidence_label = "🟡 Medium"

            else:

                confidence_label = "🔴 Low"


            st.write(
                f"**{field}** — "
                f"{confidence_label} "
                f"({confidence}%)"
            )


            # ------------------------------------------------
            # Validation status
            # ------------------------------------------------

            if status == "Valid":

                st.success(
                    f"✓ {field}: {message}"
                )

            elif status == "Missing":

                st.warning(
                    f"⚠ {field}: {message}"
                )

            elif status == "Invalid":

                st.error(
                    f"✗ {field}: {message}"
                )

            else:

                st.info(
                    f"ℹ {field}: {message}"
                )


            # ------------------------------------------------
            # Evidence
            # ------------------------------------------------

            if evidence:

                if evidence_verified:

                    st.caption(
                        f'📌 Evidence verified: '
                        f'"{evidence}"'
                    )

                else:

                    st.caption(
                        f'⚠ Evidence needs review: '
                        f'"{evidence}"'
                    )


        st.divider()

        st.caption(
            "The Extraction Quality Score is a "
            "rule-based quality indicator. "
            "Field confidence combines validation "
            "and evidence verification. It is not "
            "a guarantee of correctness."
        )