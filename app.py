# app.py

import streamlit as st
import pdfplumber
import re
import pandas as pd
from datetime import datetime
from io import BytesIO



st.set_page_config(
    page_title="Appointments to Daily Sheet",
    page_icon="📅",
    initial_sidebar_state="expanded"
)

# Display a header
st.title("Appointments to Daily Sheet")
st.subheader("Convert PDF appointment lists into schedules")



# ------------------------
# --- 0. Config & Setup ---
# ------------------------

time_pattern = re.compile(r"\d{1,2}:\d{2}\s[AP]M\s-\s\d{1,2}:\d{2}\s[AP]M")
input_pdf = "PET.pdf"
base_name = "PET"
appt_date_input = st.date_input("Select appointment date")
appt_date = datetime.combine(appt_date_input, datetime.min.time())
EXCEPTIONS = {"EKG", "RPMC", "PMC", "CUS", "SE", "TFU", "U/S", "PET"}  # multi-word exceptions
appt_map = {
    "Remote Pacemaker Check": "RPMC",
    "Remote Pacemaker  Check":"RPMC",
    "Pacemaker check": "PMC",
    "Follow Up": "FU",
    "Echo": "Echo",
    "EKG": "EKG",
    "Carotid U/S": "CUS",
    "Carotid /": "CUS",
    "Stress Echo": "SE",
    "Telehealth Follow-Up": "TFU",
    "New Patient": "NP",
    "Lexiscan PET": "LexiPET",
    "Lexiscan": "LexiPET"
}

cpt_rules = {
    "NP":   [["99203"], ["99204"], ["99205"]],
    "SE":   [["93320","93325","93351"], ["93351","93306"]],
    "Echo": [["93306"]],
    "PMC":  [["93283","93289"], ["93282","93289"], ["93288","93280"]],
    "EKG":  [["93000"]],
    "FU":   [["99213"], ["99214"], ["99215"]],
    "TFU":  [["99213"], ["99214"], ["99215"]],
    "LexiPET": [["78431"], ["78434"], ["93015"]],
    "RPMC": [],
    "CUS": [["93880"]]
    
}


record_keywords = [
    " SE ", "Echo", "ETT ", " CUS ", "OSH ", "Results", "Result", "Report","Records", "lab", "blood work",
    "SPECT", "MR", " Lexi", "Hospital", "XR ", "Review", "Imaging",
    "Monitor", "Bardy", "HM ", "Cardiac Solo", "solo"
]



# ------------------------
# --- PDF Upload ---
# ------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file is not None:
    input_pdf = uploaded_file  # Use uploaded PDF


    
    # ------------------------
    # --- 0. PDF Parsing ---
    # ------------------------


    def split_appointments(text):
        parts = time_pattern.split(text)
        times = time_pattern.findall(text)
        blocks = []
        for t, p in zip(times, parts[1:]):
            blocks.append(t + "\n" + p.strip())
        return blocks


    def parse_block(block):
        d = {}

        # --- Time Slot ---
        m = re.search(r"(?P<time>\d{1,2}:\d{2}\s[AP]M)", block)
        d["Time Slot"] = m.group("time") if m else ""

        # --- Raw Appointment Type ---
        m = re.search(
            r"\d{1,2}:\d{2}.*?(AM|PM)\s-\s.*?(AM|PM)\s(?P<appt>.+?)\s[A-Z, ']+?\s\(\d+yo",
            block,
            flags=re.DOTALL
        )
        appt_raw = m.group("appt").strip() if m else ""

        # --- Extract embedded names ---
        all_caps_words = re.findall(r"\b[A-Z]+(?:-[A-Z]+)?(?:/[A-Z]+)?\b", appt_raw)
        name_in_appt = [w for w in all_caps_words if w not in EXCEPTIONS]

        if name_in_appt:
            embedded_name = " ".join(name_in_appt)
            # Remove embedded name from Appt Type
            appt_clean = appt_raw
            for n in name_in_appt:
                appt_clean = re.sub(r"\b{}\b".format(re.escape(n)), "", appt_clean)
            appt_clean = appt_clean.replace(",", "").strip()
        else:
            embedded_name = ""
            appt_clean = appt_raw

        if not appt_clean and any(word in appt_raw for word in EXCEPTIONS):
            appt_clean = next((word for word in EXCEPTIONS if word in appt_raw), "Unknown")

        appt_clean = appt_clean.replace("\n", "").replace("  ", " ").strip()
        d["Appt. Type"] = appt_clean

        # --- Patient Name + Athena ID ---
        m = re.search(r"(?P<name>[A-Z ,'-]+)\s\(\d+yo\s[MF]\)\s#(?P<athena>\d+)", block)
        if not m:
            return None
        last_name_part = m.group("name").strip()
        d["Athena"] = m.group("athena")
        full_name = f"{last_name_part} {embedded_name}" if embedded_name else last_name_part

        # Clean patient name
        full_name_parts = full_name.split()
        while full_name_parts and full_name_parts[0] in EXCEPTIONS:
            full_name_parts = full_name_parts[1:]
        while full_name_parts and len(full_name_parts[0]) == 1:
            full_name_parts = full_name_parts[1:]

        cleaned_parts = []
        i = 0
        while i < len(full_name_parts):
            if full_name_parts[i] in {"U", "S"} and i+2 <= len(full_name_parts) and full_name_parts[i+1] == "/":
                cleaned_parts.append(f"{full_name_parts[i]}/{full_name_parts[i+2]}")
                i += 3
            elif full_name_parts[i] not in EXCEPTIONS:
                cleaned_parts.append(full_name_parts[i])
                i += 1
            else:
                i += 1
        d["Patient Name"] = " ".join(cleaned_parts)

        # --- Insurance ---
        insurance_match = re.search(r"INS\s*:\s*(.+?)(?=\n(?:PRIOR AUTH|FU|elig|#|\Z))", block, flags=re.DOTALL)
        if insurance_match:
            insurance = re.sub(r"\s+", " ", insurance_match.group(1)).strip()
        else:
            # Medicare fallback
            med_match = re.search(r"(MEDICARE-CA SOUTHERN[^\n]*)", block, flags=re.IGNORECASE)
            ppo_match = re.search(r"(PPO[^\n]*)", block, flags=re.IGNORECASE)
            epo_match = re.search(r"(EPO[^\n]*)", block, flags=re.IGNORECASE)

            if med_match:
                id_match = re.search(r"#\w+", med_match.group(1))
                medicare_id = id_match.group(0) if id_match else ""
                insurance = f"MEDICARE-CA SOUTHERN (MEDICARE) {medicare_id}"

            elif ppo_match:
                id_match = re.search(r"#\w+", ppo_match.group(1))
                ppo_id = id_match.group(0) if id_match else ""
                insurance = f"PPO {ppo_id}".strip()

            elif epo_match:
                id_match = re.search(r"#\w+", epo_match.group(1))
                epo_id = id_match.group(0) if id_match else ""
                insurance = f"EPO {epo_id}".strip()

            else:
                insurance = ""

        d["Insurance"] = insurance


        # --- Prior Auth ---
        m = re.search(r"PRIOR AUTH\s*:\s*(.+?approved)", block, flags=re.DOTALL)
        d["Prior Auth"] = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

        # --- Notes ---
        notes_start = re.search(r"\)\s#\d+", block)
        notes_section = block[notes_start.end():] if notes_start else block
        notes_lines = [line.strip() for line in notes_section.strip().splitlines() if line.strip()]
        d["Notes"] = " ".join(notes_lines)

        return d

    # --- extract text ---
    with pdfplumber.open(input_pdf) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)

    blocks = split_appointments(text)
    rows = [parse_block(b) for b in blocks if parse_block(b)]
    df = pd.DataFrame(rows)

    # ------------------------
    # --- 1. Setup & Rename ---
    # ------------------------
    df = df.rename(columns={
        'Time Slot': 'Appt. Time',
        'Appt Type': 'Appt. Type',
        'Prior Auth': 'Prior_Auth_Raw'
    })

    df['Patient'] = df['Patient Name'].apply(lambda x: x.split(",")[0].strip().title())

    # Appointment type mapping
    df['Appt. Type'] = df['Appt. Type'].map(appt_map).fillna(df['Appt. Type'])

    # ------------------------
    # --- 2. CPT Rules ---
    # ------------------------

    def has_cpt_combo(found, rule_set):
        return any(all(code in found for code in combo) for combo in rule_set)

    def validate_cpt(appt_types, cpt_list):
        appts = [a.strip() for a in appt_types.split(",")]
        if len(cpt_list)==0:
            return False
        for a in appts:
            if a in cpt_rules:
                if not has_cpt_combo(cpt_list, cpt_rules[a]):
                    return False
        return True

    # ------------------------
    # --- 3. Extract CPTs from all notes ---
    # ------------------------
    df['Current_CPTs'] = df['Notes'].apply(lambda x: sorted(list(set(re.findall(r'\b\d{5}\b', str(x))))))

    def check_cpt_valid(row):
        if str(row['Insurance']).upper().startswith(("MEDICARE", "M")):
            return True
        if any(x in str(row['Insurance']).upper() for x in ["PPO", "EPO"]):
            return True
        return validate_cpt(row['Appt. Type'], row['Current_CPTs'])

    df['CPT_Valid'] = df.apply(check_cpt_valid, axis=1)

    # ------------------------
    # --- 4. Prior Auth with Fail Reason ---
    # ------------------------
    def check_prior_auth_with_reason(row):
        insurance = str(row['Insurance']).upper()
        prior_auth = str(row['Prior_Auth_Raw'])

        if "MEDICARE-CA SOUTHERN" in insurance:
            return "M", ""
        if "PPO" in insurance:
            return "PPO",""
        if "EPO" in insurance:
            return "EPO",""


        reasons = []
        if not prior_auth or "Valid" not in prior_auth:
            reasons.append("No auth linked")
            return False, ", ".join(reasons)

        try:
            section = prior_auth.split("Valid")[-1].strip()
            exp_date_str = section.split("-")[-1].split(",")[0].strip()
            exp_date = datetime.strptime(exp_date_str, "%m/%d/%Y")
            if exp_date < appt_date:
                reasons.append(f"Authorization expired ({exp_date_str})")
            visits = int(section.split(",")[-1].split("visit")[0].strip())
            if visits < 1:
                reasons.append(f"Insufficient visits approved ({visits})")
        except:
            reasons.append("Invalid authorization format")

        a_value = True if len(reasons) == 0 else False
        reason_str = "" if a_value else ", ".join(reasons)
        return a_value, reason_str

    df[['A','A_Fail_Reason']] = df.apply(check_prior_auth_with_reason, axis=1, result_type="expand")


    # ------------------------
    # --- 5. Detect if Medical Record Needed ---
    # ------------------------

    def detect_record_needed(note):
        note_lower = str(note).lower()
        found_keywords = [word for word in record_keywords if word.lower() in note_lower]
        if len(found_keywords) == 1:
            return f"{found_keywords[0]} Required"
        elif len(found_keywords) > 1:
            return "Required"
        else:
            return ""

    df["M"] = df.apply(
        lambda row: detect_record_needed(row["Notes"])
            if row["Appt. Type"] in ["FU", "NP", "TFU"]
            else "",
        axis=1
    )



    # ------------------------
    # --- 6. Merge multiple appointments per patient ---
    # ------------------------
    def merge_appt_types(appt_list):
        appt_set = set(appt_list)
        ordered = []

        for x in ["FU", "NP"]:
            if x in appt_set:
                ordered.append(x)
                appt_set.remove(x)

        ekg_present = "EKG" in appt_set
        if ekg_present:
            appt_set.remove("EKG")

        ordered.extend(sorted(appt_set))

        if ekg_present:
            ordered.append("EKG")

        return ", ".join(ordered)


    merged = df.groupby('Athena').agg({
        'Appt. Time': 'min',
        'Patient': lambda x: x.iloc[0],  # keep one representative patient name
        'Appt. Type': lambda x: merge_appt_types(x),
        'Insurance': lambda x: max(x, key=len) if len(x) > 0 else "",
        'Notes': lambda x: " ".join(str(n) for n in x if str(n).strip() != ""),
        'A': lambda x: next((v for v in x if v in {"M","PPO","EPO"}), all(x.values)),
        'A_Fail_Reason': lambda x: ", ".join([r for r in x if r]),
        'Current_CPTs': lambda x: sorted(set(sum(x, []))),
        'CPT_Valid': lambda x: all(x.values),
        'M': lambda x: ", ".join([r for r in x if r])  # Keep all detected keywords
        }).reset_index()



    # ------------------------
    # --- 7. Filter Notes to TFU only ---
    # ------------------------
    merged['Notes'] = merged['Notes'].apply(
        lambda x: " ".join([line for line in str(x).splitlines() if line.strip().startswith("TFU")])
    )

    # ------------------------
    # --- 8. Drop notes after a CPT code ---
    # ------------------------
    def drop_after_cpt(notes):
        if not isinstance(notes, str):
            return notes
        split_notes = re.split(r"\b\d{5}\b", notes)
        return split_notes[0].strip() if split_notes else notes

    merged["Notes"] = merged["Notes"].apply(drop_after_cpt)


    # ------------------------
    # --- 9. Update A for invalid CPTs ---
    # ------------------------
    def append_cpt_fail_reason(row):
        if row['CPT_Valid'] is False and row['A'] not in {"M","PPO","EPO"}:
            row['A'] = False
            existing_reasons = str(row['A_Fail_Reason']).strip()
            if "Check CPT code(s)" not in existing_reasons:
                if existing_reasons:
                    row['A_Fail_Reason'] = existing_reasons + "; Check CPT code(s)"
                else:
                    row['A_Fail_Reason'] = "Check CPT code(s)"
        return row

    merged = merged.apply(append_cpt_fail_reason, axis=1)


    # ------------------------
    # --- 10. Sort & output ---
    # ------------------------
    merged['Time Sort'] = pd.to_datetime(merged['Appt. Time'], format='%I:%M %p')
    merged = merged.sort_values('Time Sort').reset_index(drop=True)
    merged.index = merged.index + 1
    # Drop helper column
    # merged = merged.drop(columns=['Athena'], errors='ignore') #Athena
    merged = merged.drop(columns=['Time Sort']) #Time helper
    # merged = merged[['Appt. Time','Patient','Appt. Type','Insurance','A','A_Fail_Reason',
    #                  'Current_CPTs','CPT_Valid','Notes','M']]

    recorder_cols = [
        'Appt. Time',
        'Patient',
        'Appt. Type',
        'A',
        'A_Fail_Reason',
        'M', 
        'Notes',
        'Current_CPTs',
        'CPT_Valid',
        'Athena',
        'Insurance'
    ]

    recorder_df = merged[recorder_cols].copy()

    # ------------------------
    # --- Show DataFrame & Download ---
    # ------------------------
    st.subheader("Parsed Schedule")
    st.dataframe(recorder_df)

    output = BytesIO()
    recorder_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download Excel",
        data=output,
        file_name=f"{base_name}_schedule.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    
    


