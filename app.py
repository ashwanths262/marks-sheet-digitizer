import json
import pandas as pd
import streamlit as st
from google import genai
from PIL import Image

st.set_page_config(layout="wide")
st.title("📝 Marks Sheet Digitizer")

api_key = st.sidebar.text_input("Paste Gemini API Key", type="password")
max_marks = st.sidebar.number_input("Max Marks", value=20)
uploaded_file = st.file_uploader("Upload Marks Sheet Photo", type=["jpg", "png", "jpeg"])

if uploaded_file and api_key:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Your Sheet", use_container_width=True)
    with col2:
        if st.button("Extract Marks") or "data" in st.session_state:
            if "data" not in st.session_state:
                client = genai.Client(api_key=api_key)
                img = Image.open(uploaded_file)
                prompt = """
                Read this marks table. Return ONLY a valid JSON list of objects.
                Format: [{"roll_no": "101", "marks": 18, "low_confidence": false}]
                Set low_confidence to true if handwriting is messy/unclear.
                """
                with st.spinner("Reading sheet..."):
                    res = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=[img, prompt]
                    )
                    clean_json = res.text.replace("```json", "").replace("```", "").strip()
                    st.session_state.data = pd.DataFrame(json.loads(clean_json))

            df = st.session_state.data.copy()

            def check_errors(row):
                errors = []
                if row["marks"] > max_marks or row["marks"] < 0:
                    errors.append("❌ Out of Range")
                if row["low_confidence"]:
                    errors.append("🔍 Check Handwriting")
                return " | ".join(errors) if errors else "✅ Good"

            df["Status"] = df.apply(check_errors, axis=1)
            df.loc[df.duplicated("roll_no", keep=False), "Status"] += " ⚠️ Duplicate"

            st.subheader("Edit & Verify")
            edited_df = st.data_editor(df[["roll_no", "marks", "Status"]], use_container_width=True)

            csv = edited_df[["roll_no", "marks"]].to_csv(index=False)
            st.download_button("⬇️ Download Clean CSV", csv, "marks.csv", "text/csv")