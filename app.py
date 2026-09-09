import streamlit as st
import fitz

st.set_page_config(
    page_title="DocTruth AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocTruth AI")
st.write("From unstructured documents to structured, verified data.")

st.divider()

st.subheader("Upload your document")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("Document uploaded successfully!")

    st.write("**File name:**", uploaded_file.name)
    st.write("**File type:**", uploaded_file.type)
    st.write("**File size:**", uploaded_file.size, "bytes")

    # Read the uploaded PDF
    pdf_bytes = uploaded_file.read()

    # Open the PDF
    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    st.write("**Number of pages:**", len(document))

    st.divider()

    st.subheader("Extracted text")

    full_text = ""

    for page_number, page in enumerate(document):

        text = page.get_text()

        full_text += text + "\n"

        with st.expander(f"Page {page_number + 1}"):
            st.text(text)

    st.divider()

    st.subheader("Complete document text")

    st.text_area(
        "Extracted text",
        full_text,
        height=300
    )