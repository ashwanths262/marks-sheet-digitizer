import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

st.set_page_config(page_title="Lab & Quiz Marks Sheet Digitizer", layout="wide")
st.title("Lab & Quiz Marks Sheet Digitizer")
st.caption("Upload one or more handwritten/printed marks sheets to extract roll numbers, names, and marks into CSV.")

# --- API Key Setup ---
api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if not api_key:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("Please provide your GEMINI_API_KEY in Streamlit secrets or the sidebar to continue.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- Sidebar Settings ---
st.sidebar.header("Validation Settings")
max_marks = st.sidebar.number_input("Maximum Marks", min_value=1, max_value=1000, value=100)
min_marks = st.sidebar.number_input("Minimum Marks", min_value=0, max_value=100, value=0)

PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash-lite"

OUTPUT_COLUMNS = ["roll_no", "name", "marks"]


def prepare_image_bytes(uploaded_file) -> bytes:
    """Resize image cleanly to speed up Gemini Vision upload & inference while keeping handwriting sharp."""
    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((1400, 1400))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def extract_from_single_image(file_name: str, img_bytes: bytes) -> list[dict]:
    """Extract roll_no, name, and marks from a single sheet image with retry and model fallback."""
    prompt = (
        "Extract every student row from this marks sheet table. "
        "Return ONLY a valid JSON array of objects with compact keys: "
        '{"r": "<roll_no or S.No>", "n": "<student_name>", "m": "<marks_or_grade>"}. '
        "Rules:\n"
        "1. Always extract the student's full name into 'n' (if no name column exists on the sheet, set 'n' to '').\n"
        "2. Map S.No or Roll Number to 'r'.\n"
        "3. Map Marks (or Grade / AB) to 'm'.\n"
        "4. Do not skip any rows. Output pure JSON only without markdown fences."
    )

    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw_text = (response.text or "").strip()
                raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text)
                records = json.loads(raw_text)

                cleaned_rows = []
                for item in records:
                    if not isinstance(item, dict):
                        continue
                    roll_val = str(item.get("r", item.get("roll_no", ""))).strip()
                    name_val = str(item.get("n", item.get("name", item.get("student_name", "")))).strip()
                    marks_val = str(item.get("m", item.get("marks", ""))).strip()
                    cleaned_rows.append(
                        {
                            "roll_no": roll_val,
                            "name": name_val,
                            "marks": marks_val,
                        }
                    )
                return cleaned_rows
            except Exception as exc:
                last_error = exc
                time.sleep(0.8)

    raise RuntimeError(f"Failed to process {file_name}: {last_error}")


def check_errors(df: pd.DataFrame, min_val: float, max_val: float) -> list[str]:
    """Validate rows for duplicate roll numbers, missing names, and out-of-range/invalid marks."""
    work = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in work.columns:
            work[col] = ""

    work["roll_no"] = work["roll_no"].fillna("").astype(str).str.strip()
    work["name"] = work["name"].fillna("").astype(str).str.strip()
    marks_raw = work["marks"].fillna("").astype(str).str.strip()
    numeric_marks = pd.to_numeric(marks_raw, errors="coerce")

    non_empty_rolls = work["roll_no"] != ""
    dup_mask = non_empty_rolls & work.duplicated(subset=["roll_no"], keep=False)

    warnings = []
    for idx in range(len(work)):
        issues = []
        r_val = work.loc[idx, "roll_no"]
        n_val = work.loc[idx, "name"]
        m_raw = marks_raw.iloc[idx]
        m_num = numeric_marks.iloc[idx]

        if not r_val:
            issues.append("missing roll_no")
        if dup_mask.iloc[idx]:
            issues.append("duplicate roll_no")
        if not n_val:
            issues.append("missing name")

        if m_raw.upper() in {"AB", "ABSENT", "NA", "-"}:
            issues.append("absent")
        elif pd.isna(m_num):
            issues.append("non-numeric marks")
        elif m_num < min_val or m_num > max_val:
            issues.append(f"marks outside {min_val}-{max_val}")

        if issues:
            warnings.append(f"Row {idx + 1} (roll_no: '{r_val or '?'}'): {', '.join(issues)}")

    return warnings


# --- File Uploader & Session State ---
uploaded_files = st.file_uploader(
    "Upload Marks Sheet Images (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    current_file_keys = [f"{f.name}_{f.size}" for f in uploaded_files]
    if st.session_state.get("last_uploaded_keys") != current_file_keys:
        st.session_state["last_uploaded_keys"] = current_file_keys
        st.session_state.pop("extracted_df", None)

    with st.expander("Preview Uploaded Sheets", expanded=False):
        cols = st.columns(min(len(uploaded_files), 3))
        for i, f in enumerate(uploaded_files):
            cols[i % 3].image(f, caption=f.name, use_container_width=True)

    if st.button("Extract & Digitize Marks", type="primary") or "extracted_df" in st.session_state:
        if "extracted_df" not in st.session_state:
            with st.spinner(f"Extracting data from {len(uploaded_files)} sheet(s)..."):
                prepared_items = [(f.name, prepare_image_bytes(f)) for f in uploaded_files]
                all_rows = []
                errors = []

                with ThreadPoolExecutor(max_workers=min(4, len(prepared_items))) as executor:
                    futures = {
                        executor.submit(extract_from_single_image, fname, fbytes): fname
                        for fname, fbytes in prepared_items
                    }
                    for fut in futures:
                        try:
                            all_rows.extend(fut.result())
                        except Exception as e:
                            errors.append(str(e))

                for err in errors:
                    st.error(err)

                if all_rows:
                    st.session_state["extracted_df"] = pd.DataFrame(all_rows)[OUTPUT_COLUMNS]

        if "extracted_df" in st.session_state and not st.session_state["extracted_df"].empty:
            base_df = st.session_state["extracted_df"][OUTPUT_COLUMNS].copy()
            row_warnings = check_errors(base_df, min_marks, max_marks)

            c1, c2 = st.columns(2)
            c1.metric("Total Rows Extracted", len(base_df))
            c2.metric("Validation Flags", len(row_warnings))

            if row_warnings:
                with st.expander("Validation Warnings", expanded=False):
                    for w in row_warnings:
                        st.write(f"- {w}")

            st.subheader("Verify & Edit Extracted Data")
            edited_df = st.data_editor(
                base_df,
                num_rows="dynamic",
                use_container_width=True,
                column_order=OUTPUT_COLUMNS,
            )[OUTPUT_COLUMNS]

            csv_data = edited_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Final CSV",
                data=csv_data,
                file_name="digitized_marks.csv",
                mime="text/csv",
                type="primary",
            )
