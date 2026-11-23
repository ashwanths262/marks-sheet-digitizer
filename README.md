# Lab & Quiz Marks Sheet Digitizer

## Overview
The **Lab & Quiz Marks Sheet Digitizer** is an AI-powered software utility designed to automate the digitization of tabular student assessment records from scanned or photographed marks sheets. By leveraging multimodal vision-language models and structured validation pipelines, the system eliminates manual data entry bottlenecks, reduces transcription errors, and provides an interactive human-in-the-loop verification workflow prior to exporting clean spreadsheet-ready CSV files.

---

## Problem Statement
Faculty members, teaching assistants, and academic coordinators routinely spend significant time manually transcribing handwritten or printed lab and quiz scores into digital spreadsheets. This repetitive process frequently leads to:
* Operational fatigue and delayed grade publication.
* Row misalignment and off-by-one entry errors.
* Transcription inaccuracies caused by illegible handwriting or skewed document scans.
* Unnoticed duplicate entries or out-of-range score inputs.

---

## Key Features
* **Multimodal OCR & Tabular Extraction:** Extracts structured student roll numbers and assessment scores directly from photographed or scanned marks sheets (`JPG`, `JPEG`, `PNG`), automatically compensating for minor page tilt and alignment variations.
* **Automated Anomaly & Confidence Flagging:**
  * **Out-of-Range Validation:** Flags scores below `0` or exceeding the user-configured maximum marks threshold (`❌ Out of Range`).
  * **Confidence Estimation:** Identifies ambiguous or illegible handwriting for targeted human review (`🔍 Check Handwriting`).
  * **Duplicate Detection:** Highlights repeated student roll numbers across rows (`⚠️ Duplicate`).
* **Interactive Human-in-the-Loop Review:** Displays the original marks sheet alongside an editable data grid (`st.data_editor`), enabling rapid inline corrections.
* **Resilient Model Fallback:** Automatically cycles through available Gemini Flash endpoints to ensure high availability during transient API load spikes.
* **Clean CSV Export:** Exports verified, sanitized records (`roll_no`, `marks`) into a standardized `marks.csv` file ready for LMS or spreadsheet import.

---

## System Architecture & Workflow
1. **Input Ingestion:** User configures the maximum permissible score, supplies a Gemini API key, and uploads a marks sheet image.
2. **Vision-Language Extraction:** The image is processed via the `google-genai` SDK using structured prompting to return normalized JSON records with confidence flags.
3. **Rule-Based Validation:** A `pandas` validation pipeline evaluates every record against range constraints, duplicate keys, and legibility flags.
4. **Verification & Export:** The reviewer inspects flagged rows side-by-side against the source image, edits discrepancies inline, and downloads the final CSV output.

---

## Technology Stack
* **Frontend & Interactive UI:** Streamlit
* **AI / Vision-Language Engine:** Google Gemini API (`google-genai` SDK)
* **Data Processing & Validation:** Pandas
* **Image Handling:** Pillow (PIL)
* **Language:** Python 3.10+

---

## Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ashwanths262/marks-sheet-digitizer.git
cd marks-sheet-digitizer
```

### 2. Install Dependencies
```bash
python -m pip install streamlit google-genai pandas pillow
```

### 3. Obtain a Gemini API Key
Generate an API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Run the Application
```bash
python -m streamlit run app.py
```

---

## Usage Guide
1. Open the local Streamlit URL displayed in your terminal (typically `http://localhost:8501`).
2. Enter your **Gemini API Key** and specify the **Max Marks** limit in the left sidebar.
3. Upload a clear image (`JPG`, `JPEG`, or `PNG`) of a lab or quiz marks sheet.
4. Click **Extract Marks** to run optical extraction and automated validation.
5. Review any rows marked with warning flags in the **Edit & Verify** table and click directly on cells to correct values if needed.
6. Click **Download Clean CSV** to export the finalized `marks.csv` file.

---

## Project Structure
```text
marks-sheet-digitizer/
├── app.py          # Main Streamlit application, AI extraction logic, validation, and CSV export
└── README.md       # Project documentation
```

---

## Security & Best Practices
* API keys are entered dynamically at runtime via a masked password input field and are never hardcoded or stored in version control.
* Always review flagged entries before finalizing academic records.

---

## Future Enhancements
* Batch processing for multi-page lab assessment sheets.
* Missing roll number sequence detection against an uploaded master class roster.
* Direct export support for `.xlsx` (Excel) workbooks with summary statistics.
