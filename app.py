import json
import time
import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors
from PIL import Image

st.set_page_config(layout="wide")
st.title("📝 Marks Sheet Digitizer")

api_key = st.sidebar.text_input("Paste Gemini API Key", type="password")
max_marks = st.sidebar.number_input("Max Marks", value=20)
uploaded_files = st.file_uploader(
    "Upload Marks Sheet Photos",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True,
)

current_file_names = [f.name for f in uploaded_files] if uploaded_files else []
if st.session_state.get("uploaded_names") != current_file_names:
    st.session_state.uploaded_names = current_file_names
    st.session_state.pop("data", None)


def extract_with_retry(client, img, prompt):
    models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    last_err = None

    for model_name in models_to_try:
        for attempt in range(3):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=[img, prompt],
                )
                clean_json = (
                    res.text.replace("```json", "").replace("```", "").strip()
                )
                return json.loads(clean_json)
            except errors.ServerError as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
    raise last_err


if uploaded_files and api_key:
    col1, col2 = st.columns(2)
    with col1:
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)

    with col2:
        if st.button("Extract Marks") or "data" in st.session_state:
            if "data" not in st.session_state:
                client = genai.Client(api_key=api_key)
                prompt = """
                Read this marks table. Return ONLY a valid JSON list of objects.
                Format: [{"roll_no": "101", "marks": 18, "low_confidence": false}]
                Set low_confidence to true if handwriting is messy/unclear.
                """
                all_rows = []
                with st.spinner(f"Reading {len(uploaded_files)} sheet(s)..."):
                    for file in uploaded_files:
                        img = Image.open(file)
                        img.thumbnail((1600, 1600))
                        try:
                            rows = extract_with_retry(client, img, prompt)
                            for row in rows:
                                row["sheet"] = file.name
                            all_rows.extend(rows)
                        except errors.ServerError:
                            st.error(
                                f"Gemini server is temporarily overloaded while reading `{file.name}`. Please wait a few seconds and click Extract Marks again."
                            )
                            st.stop()

                st.session_state.data = pd.DataFrame(all_rows)

            df = st.session_state.data.copy()

            if df.empty:
                st.warning("No rows were extracted from the uploaded sheets.")
                st.stop()

            if "roll_no" not in df.columns:
                df["roll_no"] = ""
            if "marks" not in df.columns:
                df["marks"] = None
            if "low_confidence" not in df.columns:
                df["low_confidence"] = False
            else:
                df["low_confidence"] = df["low_confidence"].fillna(False).astype(bool)

            df["marks"] = pd.to_numeric(df["marks"], errors="coerce")

            def check_errors(row):
                errors_list = []
                marks_val = row["marks"]
                if pd.isna(marks_val):
                    errors_list.append("⚠️ Non-Numeric / Missing Marks")
                elif marks_val > max_marks or marks_val < 0:
                    errors_list.append("❌ Out of Range")
                if row["low_confidence"]:
                    errors_list.append("🔍 Check Handwriting")
                return " | ".join(errors_list) if errors_list else "✅ Good"

            df["Status"] = df.apply(check_errors, axis=1)
            df.loc[df.duplicated("roll_no", keep=False), "Status"] += " ⚠️ Duplicate"

            st.subheader("Edit & Verify")
            edited_df = st.data_editor(
                df[["sheet", "roll_no", "marks", "Status"]],
                use_container_width=True,
            )

            csv = edited_df[["roll_no", "marks"]].to_csv(index=False)
            st.download_button("⬇️ Download Clean CSV", csv, "marks.csv", "text/csv")
