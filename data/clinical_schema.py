"""
Clinical schema for synthetic MIMIC-like data.
Maps itemids to names and defines valid ranges for chart/lab events.
"""

# Chart events (vitals) - MIMIC-IV style itemids (simplified set)
CHART_ITEMS = {
    220045: ("Heart Rate", "bpm", 20, 250),
    220050: ("Arterial Blood Pressure systolic", "mmHg", 40, 300),
    220051: ("Arterial Blood Pressure diastolic", "mmHg", 20, 200),
    223761: ("Temperature Fahrenheit", "°F", 80, 115),
    220210: ("Respiratory Rate", "breaths/min", 0, 80),
    220277: ("O2 saturation pulseoxymetry", "%", 0, 100),
}

# Lab items (common ICU labs)
LAB_ITEMS = {
    50813: ("Lactate", "mmol/L", 0.5, 20.0),
    50809: ("Glucose", "mg/dL", 20, 600),
    50912: ("Creatinine", "mg/dL", 0.2, 15.0),
    50822: ("Hemoglobin", "g/dL", 3, 25),
    50971: ("Potassium", "mEq/L", 2.0, 9.0),
    50824: ("Hemoglobin A1C", "%", 4, 20),
    50983: ("Sodium", "mEq/L", 100, 180),
    50804: ("Bilirubin, total", "mg/dL", 0, 50),
    50963: ("BUN", "mg/dL", 5, 250),
    50820: ("pH", "pH", 6.8, 7.8),
    50821: ("PO2", "mmHg", 20, 600),
    50818: ("PCO2", "mmHg", 10, 150),
}

# Admission types
ADMISSION_TYPES = ("EMERGENCY", "ELECTIVE", "NEWBORN", "OBSERVATION")

# ICU care units
CAREUNITS = ("MICU", "SICU", "CCU", "CSRU", "NICU", "TSICU")

# Sample ICD-10 codes (simplified) for common conditions
ICD_SAMPLES = [
    "I10",   # Essential hypertension
    "E11.9", # Type 2 diabetes
    "J18.9", # Pneumonia, unspecified
    "I50.9", # Heart failure
    "N17.9", # Acute kidney failure
    "K21.9", # GERD
    "J44.9", # COPD
    "I25.10", # Coronary artery disease
    "R06.00", # Dyspnea
    "A41.9", # Sepsis, unspecified
    "R65.20", # Severe sepsis without shock
]

# Demographics
GENDERS = ("M", "F")
INSURANCE = ("Medicare", "Medicaid", "Private", "Self Pay", "Government")
RACE = ("White", "Black", "Asian", "Hispanic", "Other", "Unknown")
