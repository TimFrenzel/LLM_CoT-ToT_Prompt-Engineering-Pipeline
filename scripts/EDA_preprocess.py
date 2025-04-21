#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Tim Frenzel
Version: 1.8
Usage:  python EDA_preprocess.py

Description:
------------
Performs comprehensive data ingestion, preprocessing, Exploratory Data Analysis (EDA), and severity labeling for Synthea COVID-19 patient data, generating enhanced datasets and visualizations.

Key Features:
-----------
1. Data Loading & Filtering
   - Reads multiple Synthea CSVs and filters records for COVID-19 diagnosed patients.

2. Feature Engineering
   - Derives age, flags key comorbidities (hypertension, diabetes), medication usage (antibiotics, antivirals, top N), allergies, care plans, encounter counts, and recent flu vaccination status.

3. Observation Pivoting
   - Extracts and pivots the latest recorded values for target vital signs and laboratory results (e.g., Blood Pressure, SpO2, Temperature, CRP, Glucose).

4. Corrected Calculations
   - Aggregates facemask usage accurately from `procedures.csv` based on procedure descriptions.

5. Severity Labeling
   - Assigns patient severity labels ('Mild', 'Moderate', 'Severe', 'Critical') based on clinical outcomes like hospitalization, ventilation, and mortality.

6. Exploratory Data Analysis (EDA)
   - Generates and saves multiple plots visualizing distributions and correlations related to severity, demographics, comorbidities, and key lab values.

7. Patient Journey Visualization
   - Creates detailed clinical timelines for representative patients from each severity class, illustrating key events and lab trends relative to diagnosis date.

8. Output Generation
   - Exports the final processed dataset (`final_covid19_labeled.csv`) and all generated EDA plots to structured results directories.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import re
import logging # Added logging
from datetime import timedelta
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. Configuration & Setup
# ------------------------------------------------------------------------------
# region Configuration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Derive base directory relative to the script location
# Assumes script is in 'scripts/' and project root is one level up
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

DATA_DIR = BASE_DIR / "data" / "10k_synthea_covid19_csv"
RESULTS_DIR = BASE_DIR / "results" / "EDA"
PATIENT_JOURNEY_DIR = RESULTS_DIR / "patient_journeys"

# Create directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PATIENT_JOURNEY_DIR.mkdir(parents=True, exist_ok=True)

# Define constants (consider moving to constants.py in a larger project)
RELEVANT_FILES = {
    "patients": "patients.csv",
    "conditions": "conditions.csv",
    "observations": "observations.csv",
    "encounters": "encounters.csv",
    "medications": "medications.csv",
    "devices": "devices.csv",
    "supplies": "supplies.csv",
    "procedures": "procedures.csv",
    "careplans": "careplans.csv",
    "allergies": "allergies.csv",
    "immunizations": "immunizations.csv"
}

COVID_CODES = [840539006, 840544004]  # Confirmed COVID-19, Suspected COVID-19

TARGET_OBSERVATIONS = [
    "systolic blood pressure",
    "diastolic blood pressure",
    "oxygen saturation in arterial blood",
    "body temperature",
    "c reactive protein",
    "glucose"
]

KEY_LAB_VALUES = [
    "oxygen saturation",
    "body temperature",
    "c reactive protein",
    "respiratory rate"
]

JOURNEY_COLORS = {
    "DiagnosedCOVID": "#2E86C1",
    "Inpatient": "#F39C12",
    "VentProc": "#E74C3C",
    "Death": "#7D3C98",
    "Discharge": "#27AE60",
    "Medication": "#16A085",
    "O2Line": "#3498DB",
    "TempLine": "#E67E22",
    "CRPLine": "#C0392B",
    "RespRateLine": "#8E44AD"
}

# endregion Configuration

# region Utility Functions

def safe_fillna(series, fill_value, dtype=None):
    """Safely handle fillna operation to avoid FutureWarning."""
    result = series.copy()
    mask = result.isna()
    result = result.astype(object)
    result[mask] = fill_value
    if dtype is not None:
        result = result.astype(dtype)
    return result

def read_csv_or_exit(filename: str) -> pd.DataFrame:
    """Reads a CSV from DATA_DIR; exits script if file not found."""
    fpath = DATA_DIR / filename # Use pathlib object
    if not fpath.is_file():
        logging.error(f"CSV file not found: {fpath}")
        sys.exit(1)
    return pd.read_csv(fpath)

def get_age(birthdate_str, ref_year=2025):
    """Approximate patient age using January 1 of ref_year as a reference date."""
    try:
        import datetime as dt
        birthdate = pd.to_datetime(birthdate_str, errors="coerce")
        # Check if conversion failed
        if pd.isna(birthdate):
            return np.nan
        ref_date = pd.Timestamp(f"{ref_year}-01-01")
        # Ensure birthdate is not in the future relative to ref_date
        if birthdate > ref_date:
            return np.nan
        return (ref_date - birthdate).days // 365
    except (ValueError, TypeError) as e:
        logging.warning(f"Could not parse date '{birthdate_str}': {e}")
        return np.nan
    except Exception as e:
        logging.warning(f"Unexpected error calculating age for '{birthdate_str}': {e}")
        return np.nan

# endregion Utility Functions

# ------------------------------------------------------------------------------
# 2. Data Import & Filtering
# ------------------------------------------------------------------------------
# region Filtering Logic

def filter_covid_patients(conditions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of conditions containing only rows where the CODE
    is in COVID_CODES (e.g. 840539006 or 840544004).
    """
    covid_mask = conditions_df["CODE"].isin(COVID_CODES)
    return conditions_df.loc[covid_mask].copy()

# endregion Filtering Logic

# ------------------------------------------------------------------------------
# 3. Merging & Structuring
# ------------------------------------------------------------------------------
# region Data Merging & Feature Engineering

def build_patient_level_dataset(
    covid_condition_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    procedures_df: pd.DataFrame,
    devices_df: pd.DataFrame,
    supplies_df: pd.DataFrame,
    medications_df: pd.DataFrame,
    allergies_df: pd.DataFrame,
    careplans_df: pd.DataFrame,
    immunizations_df: pd.DataFrame,
    target_obs: list[str]
) -> pd.DataFrame:
    """
    Builds a single patient-level table with relevant features for all 
    patients who appear in the `covid_condition_df`.

    Incorporates:
      - Basic demographics
      - Hospitalization & ventilation flags
      - Comorbidities (hypertension, diabetes)
      - More medication details:
          * top N meds usage
          * antibiotic/antiviral flags
      - Observations pivot (selected labs/vitals)
      - Face mask usage from procedures (fixing the prior 0-issue)
      - Additional supplies usage if relevant
      - NEW: Allergy flags (nut, pollen, drug)
      - NEW: Encounter counts (emergency, wellness)
      - NEW: Recent flu vaccine flag
      - NEW: Care plan flags (hypertension, diabetes)
    """

    logging.info("Building patient-level dataset...")

    # --------------------------------------------------------------------------
    # 3A. Filter dataframes to COVID patient set
    # --------------------------------------------------------------------------
    covid_patient_ids = covid_condition_df["PATIENT"].unique()

    pat_df = patients_df[patients_df["Id"].isin(covid_patient_ids)].copy()
    enc_df = encounters_df[encounters_df["PATIENT"].isin(covid_patient_ids)].copy()
    obs_df = observations_df[observations_df["PATIENT"].isin(covid_patient_ids)].copy()
    proc_df = procedures_df[procedures_df["PATIENT"].isin(covid_patient_ids)].copy()
    dev_df = devices_df[devices_df["PATIENT"].isin(covid_patient_ids)].copy()
    sup_df = supplies_df[supplies_df["PATIENT"].isin(covid_patient_ids)].copy()
    med_df = medications_df[medications_df["PATIENT"].isin(covid_patient_ids)].copy()
    alg_df = allergies_df[allergies_df["PATIENT"].isin(covid_patient_ids)].copy()
    cp_df = careplans_df[careplans_df["PATIENT"].isin(covid_patient_ids)].copy()
    imm_df = immunizations_df[immunizations_df["PATIENT"].isin(covid_patient_ids)].copy()

    # --------------------------------------------------------------------------
    # 3B. Basic demographics, DEATHDATE, and First COVID Date
    # --------------------------------------------------------------------------
    pat_df["AGE"] = pat_df["BIRTHDATE"].apply(lambda x: get_age(x, ref_year=2025))
    pat_df["IS_DECEASED"] = pat_df["DEATHDATE"].notnull().astype(int)

    master_df = pat_df[["Id", "AGE", "GENDER", "DEATHDATE"]].copy()
    master_df = master_df.rename(columns={"Id": "PATIENT"})

    # --- Get First COVID Condition Date for Immunization Timing ---
    covid_condition_df['START_TS'] = pd.to_datetime(covid_condition_df['START'], errors='coerce')
    first_covid_date = covid_condition_df.groupby('PATIENT')['START_TS'].min().rename('FIRST_COVID_DATE')
    master_df = master_df.merge(first_covid_date, on='PATIENT', how='left')
    # Ensure the date is tz-naive for comparisons later
    if pd.api.types.is_datetime64_any_dtype(master_df['FIRST_COVID_DATE']) and master_df['FIRST_COVID_DATE'].dt.tz is not None:
        master_df['FIRST_COVID_DATE'] = master_df['FIRST_COVID_DATE'].dt.tz_localize(None)

    # --------------------------------------------------------------------------
    # 3C. Hospitalization & ventilation flags
    # --------------------------------------------------------------------------
    if "ENCOUNTERCLASS" in enc_df.columns:
        enc_df["INPATIENT_FLAG"] = enc_df["ENCOUNTERCLASS"].eq("inpatient").astype(int)
    else:
        enc_df["INPATIENT_FLAG"] = 0

    dev_df["IS_VENTILATOR"] = dev_df["DESCRIPTION"].str.contains(
        r"ventilator|vent", case=False, na=False, regex=True
    ).astype(int)
    proc_df["IS_VENT_PROC"] = proc_df["DESCRIPTION"].str.contains(
        r"ventilation|intubation|intubat", case=False, na=False, regex=True
    ).astype(int)

    inpatient_agg = enc_df.groupby("PATIENT")["INPATIENT_FLAG"].max().rename("ANY_INPATIENT")
    vent_dev_agg   = dev_df.groupby("PATIENT")["IS_VENTILATOR"].max().rename("USED_MECH_VENT")
    vent_proc_agg  = proc_df.groupby("PATIENT")["IS_VENT_PROC"].max().rename("VENT_PROCEDURE")

    master_df = master_df.merge(inpatient_agg, on="PATIENT", how="left")
    master_df = master_df.merge(vent_dev_agg, on="PATIENT", how="left")
    master_df = master_df.merge(vent_proc_agg, on="PATIENT", how="left")

    # --------------------------------------------------------------------------
    # 3D. Comorbidities (e.g., hypertension, diabetes) - Efficient Method
    # --------------------------------------------------------------------------
    full_cond_sub = conditions_df[conditions_df["PATIENT"].isin(covid_patient_ids)].copy()
    full_cond_sub["LOWER_DESC"] = full_cond_sub["DESCRIPTION"].str.lower().fillna('')

    # Use regex for broader matching (e.g., "high blood pressure")
    hypertension_pattern = r"hypertension|high blood pressure"
    diabetes_pattern = r"diabetes|diabetic"

    # Define aggregation functions
    def check_hypertension(series):
        return series.str.contains(hypertension_pattern, regex=True, na=False).any()

    def check_diabetes(series):
        return series.str.contains(diabetes_pattern, regex=True, na=False).any()

    # Group by patient and aggregate checks using .agg()
    comorbidities = full_cond_sub.groupby("PATIENT")["LOWER_DESC"].agg(
        HAS_HYPERTENSION=check_hypertension,
        HAS_DIABETES=check_diabetes
    ).astype(int).reset_index()

    # Merge the flags into the master dataframe
    master_df = master_df.merge(comorbidities, on="PATIENT", how="left")
    # Fill NaN for patients with no conditions listed in full_cond_sub
    master_df["HAS_HYPERTENSION"] = master_df["HAS_HYPERTENSION"].fillna(0).astype(int)
    master_df["HAS_DIABETES"] = master_df["HAS_DIABETES"].fillna(0).astype(int)

    # --------------------------------------------------------------------------
    # 3E. Medications: - Efficient Method
    #     1) Antibiotic/antiviral flags
    #     2) Count distinct medications
    #     3) Top N medication names as separate flags
    # --------------------------------------------------------------------------
    med_df["LOWER_DESC"] = med_df["DESCRIPTION"].str.lower().fillna('')

    # Use broader regex patterns for antibiotic/antiviral detection
    antibiotic_pattern = r"cillin|ceph|micin|mycin|cycline|penem|floxacin|sulfa|nitrofurantoin|azithromycin|antibiotic"
    antiviral_pattern = r"oseltamivir|zanamivir|peramivir|baloxavir|antiviral|favipiravir|remdesivir|vir$|vir\b" # Added common antiviral suffixes

    med_df["IS_ANTIBIOTIC"] = med_df["LOWER_DESC"].str.contains(antibiotic_pattern, regex=True, na=False)
    med_df["IS_ANTIVIRAL"] = med_df["LOWER_DESC"].str.contains(antiviral_pattern, regex=True, na=False)

    med_agg_df = med_df.groupby("PATIENT").agg(
        DISTINCT_MED_COUNT = ("DESCRIPTION", "nunique"),
        ANY_ANTIBIOTIC     = ("IS_ANTIBIOTIC", "max"),
        ANY_ANTIVIRAL      = ("IS_ANTIVIRAL", "max")
    ).reset_index()

    master_df = master_df.merge(med_agg_df, on="PATIENT", how="left")

    # Convert booleans/NaNs to int (0 or 1) - Fix for FutureWarning
    master_df["ANY_ANTIBIOTIC"] = safe_fillna(master_df["ANY_ANTIBIOTIC"], False, dtype=int)
    master_df["ANY_ANTIVIRAL"] = safe_fillna(master_df["ANY_ANTIVIRAL"], False, dtype=int)
    master_df["DISTINCT_MED_COUNT"] = safe_fillna(master_df["DISTINCT_MED_COUNT"], 0, dtype=int)

    # For top N medication descriptions, let's pick top 5 by overall frequency
    top5_meds = (
        med_df["DESCRIPTION"]
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    # Efficiently create flags for top 5 meds
    # Create a mapping for patients and their medications
    patient_meds_set = med_df.groupby('PATIENT')['DESCRIPTION'].apply(set)

    for med_name in top5_meds:
        # Sanitize med_name for column name
        safe_med_name = re.sub(r'\W+', '_', med_name.lower()) # Replace non-alphanumeric with _
        col_flag = f"MED_FLAG_{safe_med_name[:20]}" # Keep it reasonably short

        # Map patient IDs to a boolean indicating if they took the med
        master_df[col_flag] = master_df["PATIENT"].map(
            lambda pid: int(med_name in patient_meds_set.get(pid, set()))
        )
        # Alternative using merge (might be faster on very large dataframes):
        # med_flag_df = med_df[med_df['DESCRIPTION'] == med_name][['PATIENT']].drop_duplicates()
        # med_flag_df[col_flag] = 1
        # master_df = master_df.merge(med_flag_df, on='PATIENT', how='left')
        # master_df[col_flag] = master_df[col_flag].fillna(0).astype(int)

    # --------------------------------------------------------------------------
    # 3F. Observations pivot for selected labs/vitals
    #     We'll pick a few columns to pivot based on `target_obs` list.
    #     ... (rest of description) ...
    # --------------------------------------------------------------------------
    obs_df["DATE"] = pd.to_datetime(obs_df["DATE"], errors="coerce")
    obs_df["LOW_DESC"] = obs_df["DESCRIPTION"].str.lower().fillna('')

    # Filter to target descriptions using regex for partial matches (case-insensitive)
    # Example: "glucose" in target_obs should match "glucose [mass/volume] in serum or plasma"
    target_pattern = "|".join([re.escape(t) for t in target_obs]) # Create regex pattern like 'systolic blood pressure|diastolic blood pressure|...'
    sub_obs_df = obs_df[obs_df["LOW_DESC"].str.contains(target_pattern, case=False, regex=True, na=False)].copy()

    # Pivot approach: for each patient, for each "DESCRIPTION", keep the last "VALUE"
    # We'll keep the original "DESCRIPTION" as columns.
    # Then rename them to a standardized short column.
    # Ensure VALUE is numeric if possible for aggregation, coerce errors
    sub_obs_df['VALUE_numeric'] = pd.to_numeric(sub_obs_df['VALUE'], errors='coerce')

    pivot_df = sub_obs_df.dropna(subset=['VALUE_numeric']).sort_values(["PATIENT", "DATE"]).groupby(
        ["PATIENT", "DESCRIPTION"]
    )["VALUE_numeric"].last().unstack()  # pivot wide

    if pivot_df is not None and not pivot_df.empty:
        # rename columns => "LAST_<desc>"
        # Strategy: replace common special chars, limit length.
        new_cols = {}
        for c in pivot_df.columns:
            clean_col = re.sub(r'[\s\[\](),./]+', '_', c) # Replace space and special chars with _
            clean_col = re.sub(r'_+', '_', clean_col).strip('_') # Consolidate underscores
            clean_col = "LAST_" + clean_col[:50]  # keep it reasonably short
            new_cols[c] = clean_col
        pivot_df.rename(columns=new_cols, inplace=True)
        pivot_df.reset_index(inplace=True)
        # Merge to master
        master_df = master_df.merge(pivot_df, on="PATIENT", how="left")

    # --------------------------------------------------------------------------
    # 3G. Face mask usage from `procedures.csv`
    #     "Face mask (physical object)" is there, so let's sum how many times
    #     each patient had a "Face mask (physical object)" procedure.
    # --------------------------------------------------------------------------
    mask_proc = proc_df[
        proc_df["DESCRIPTION"].str.contains(r"face mask \(physical object\)|face mask", case=False, na=False, regex=True)
    ].copy()
    face_mask_count = mask_proc.groupby("PATIENT")["DESCRIPTION"].count().rename("TOTAL_FACEMASK_USED")
    master_df = master_df.merge(face_mask_count, on="PATIENT", how="left")

    # Some patients might not have face mask usage => fill with 0
    master_df["TOTAL_FACEMASK_USED"] = master_df["TOTAL_FACEMASK_USED"].fillna(0).astype(int)

    # --------------------------------------------------------------------------
    # 3H. Supplies usage (optional)
    #     If your dataset actually has relevant supplies. We keep it as before.
    # --------------------------------------------------------------------------
    sup_df["LOWER_DESC"] = sup_df["DESCRIPTION"].str.lower()
    sup_agg_df = sup_df.groupby("PATIENT")["DESCRIPTION"].count().rename("TOTAL_SUPPLIES").reset_index()
    master_df = master_df.merge(sup_agg_df, on="PATIENT", how="left")
    master_df["TOTAL_SUPPLIES"] = master_df["TOTAL_SUPPLIES"].fillna(0).astype(int)

    # --------------------------------------------------------------------------
    # 3I. Allergies - NEW Section
    # --------------------------------------------------------------------------
    if not alg_df.empty and 'DESCRIPTION' in alg_df.columns:
        alg_df["LOWER_DESC"] = alg_df["DESCRIPTION"].str.lower().fillna('')

        # Define allergy patterns (add more as needed)
        nut_pattern = r"nut"
        pollen_pattern = r"pollen|grass"
        drug_pattern = r"drug|penicillin|sulfonamide|sulfa"

        def check_nut(s): return s.str.contains(nut_pattern, regex=True).any()
        def check_pollen(s): return s.str.contains(pollen_pattern, regex=True).any()
        def check_drug(s): return s.str.contains(drug_pattern, regex=True).any()

        allergy_flags = alg_df.groupby("PATIENT")["LOWER_DESC"].agg(
            HAS_NUT_ALLERGY=check_nut,
            HAS_POLLEN_ALLERGY=check_pollen,
            HAS_DRUG_ALLERGY=check_drug
        ).astype(int).reset_index()

        master_df = master_df.merge(allergy_flags, on="PATIENT", how="left")
        # Fill NaN for patients with no allergy entries
        for col in ["HAS_NUT_ALLERGY", "HAS_POLLEN_ALLERGY", "HAS_DRUG_ALLERGY"]:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(0).astype(int)
            else:
                master_df[col] = 0 # Add column as 0 if no allergies found at all
    else:
        logging.warning("Allergies data frame is empty or missing 'DESCRIPTION'. Skipping allergy features.")
        master_df["HAS_NUT_ALLERGY"] = 0
        master_df["HAS_POLLEN_ALLERGY"] = 0
        master_df["HAS_DRUG_ALLERGY"] = 0

    # --------------------------------------------------------------------------
    # 3J. Encounter History - NEW Section
    # --------------------------------------------------------------------------
    if not enc_df.empty and 'ENCOUNTERCLASS' in enc_df.columns:
        # Count specific encounter types per patient
        encounter_counts = enc_df.groupby('PATIENT')['ENCOUNTERCLASS'].value_counts().unstack(fill_value=0)

        # Select and rename relevant columns (adjust names if needed)
        cols_to_add = {}
        if 'emergency' in encounter_counts.columns: cols_to_add['emergency'] = 'TOTAL_EMERGENCY'
        if 'wellness' in encounter_counts.columns: cols_to_add['wellness'] = 'TOTAL_WELLNESS'
        if 'ambulatory' in encounter_counts.columns: cols_to_add['ambulatory'] = 'TOTAL_AMBULATORY'
        # Add total encounters
        encounter_counts['TOTAL_ENCOUNTERS'] = encounter_counts.sum(axis=1)
        cols_to_add['TOTAL_ENCOUNTERS'] = 'TOTAL_ENCOUNTERS'

        encounter_counts = encounter_counts[list(cols_to_add.keys())].rename(columns=cols_to_add)

        master_df = master_df.merge(encounter_counts, on='PATIENT', how='left')

        # Fill NaNs for patients with no encounters in the filtered set
        for col in cols_to_add.values():
            if col in master_df.columns:
                 master_df[col] = master_df[col].fillna(0).astype(int)
            else:
                 master_df[col] = 0
    else:
        logging.warning("Encounters data frame is empty or missing 'ENCOUNTERCLASS'. Skipping encounter history features.")
        master_df['TOTAL_EMERGENCY'] = 0
        master_df['TOTAL_WELLNESS'] = 0
        master_df['TOTAL_AMBULATORY'] = 0
        master_df['TOTAL_ENCOUNTERS'] = 0

    # --------------------------------------------------------------------------
    # 3K. Immunization History (Recent Flu Vax) - NEW Section
    # --------------------------------------------------------------------------
    master_df['HAD_RECENT_FLU_VAX'] = 0 # Initialize column
    if not imm_df.empty and 'CODE' in imm_df.columns and 'DATE' in imm_df.columns and 'FIRST_COVID_DATE' in master_df.columns:
        imm_df['DATE'] = pd.to_datetime(imm_df['DATE'], errors='coerce')
        # Ensure date is tz-naive
        if pd.api.types.is_datetime64_any_dtype(imm_df['DATE']) and imm_df['DATE'].dt.tz is not None:
            imm_df['DATE'] = imm_df['DATE'].dt.tz_localize(None)

        # Identify flu vaccines (CODE 140 is common in Synthea for seasonal flu)
        # Add description check if needed: | imm_df['DESCRIPTION'].str.contains('influenza', case=False, na=False)
        flu_vax = imm_df[imm_df['CODE'] == 140].copy()

        if not flu_vax.empty:
            # Get the latest flu vaccine date per patient
            latest_flu_vax_date = flu_vax.groupby('PATIENT')['DATE'].max()

            # Merge latest flu vax date with master_df (which has FIRST_COVID_DATE)
            temp_df = master_df.merge(latest_flu_vax_date.rename('LATEST_FLU_VAX'), on='PATIENT', how='left')

            # Check if LATEST_FLU_VAX is within 1 year BEFORE FIRST_COVID_DATE
            # Define the time window (e.g., 365 days)
            one_year = pd.Timedelta(days=365)
            mask = (
                pd.notnull(temp_df['LATEST_FLU_VAX']) &
                pd.notnull(temp_df['FIRST_COVID_DATE']) &
                (temp_df['FIRST_COVID_DATE'] > temp_df['LATEST_FLU_VAX']) & # Vax must be before COVID
                ((temp_df['FIRST_COVID_DATE'] - temp_df['LATEST_FLU_VAX']) <= one_year)
            )
            # Update the flag in the original master_df based on the mask
            master_df.loc[mask[mask].index, 'HAD_RECENT_FLU_VAX'] = 1
        else:
            logging.info("No flu vaccine (CODE=140) entries found.")

    else:
        logging.warning("Immunizations data frame is empty or missing columns. Skipping recent flu vax feature.")

    # --------------------------------------------------------------------------
    # 3L. Care Plan Information - NEW Section
    # --------------------------------------------------------------------------
    if not cp_df.empty and ('CODE' in cp_df.columns or 'DESCRIPTION' in cp_df.columns):
        cp_df["DESC_LOWER"] = cp_df["DESCRIPTION"].str.lower().fillna('') if 'DESCRIPTION' in cp_df.columns else ''
        # Using CODE if available, otherwise DESCRIPTION
        # Synthea careplan codes: 443402002 (HTN Lifestype), 736376001 (Infectious Disease)
        # Diabetes care plans might vary, check description.
        htn_cp_pattern = r"hypertension"
        diab_cp_pattern = r"diabetes|diabetic"
        infect_cp_pattern = r"infectious disease"

        def check_htn_cp(s): return s.str.contains(htn_cp_pattern, regex=True).any()
        def check_diab_cp(s): return s.str.contains(diab_cp_pattern, regex=True).any()
        def check_infect_cp(s): return s.str.contains(infect_cp_pattern, regex=True).any()

        careplan_flags = cp_df.groupby("PATIENT")["DESC_LOWER"].agg(
            HAS_HYPERTENSION_CAREPLAN=check_htn_cp,
            HAS_DIABETES_CAREPLAN=check_diab_cp,
            HAS_INFECT_DISEASE_CAREPLAN=check_infect_cp # Note: This might overlap heavily with COVID itself
        ).astype(int).reset_index()

        master_df = master_df.merge(careplan_flags, on="PATIENT", how="left")
        # Fill NaN for patients with no careplan entries
        for col in ["HAS_HYPERTENSION_CAREPLAN", "HAS_DIABETES_CAREPLAN", "HAS_INFECT_DISEASE_CAREPLAN"]:
             if col in master_df.columns:
                master_df[col] = master_df[col].fillna(0).astype(int)
             else:
                 master_df[col] = 0 # Add column as 0 if no careplans found
    else:
        logging.warning("Careplans data frame is empty or missing columns. Skipping care plan features.")
        master_df["HAS_HYPERTENSION_CAREPLAN"] = 0
        master_df["HAS_DIABETES_CAREPLAN"] = 0
        master_df["HAS_INFECT_DISEASE_CAREPLAN"] = 0

    logging.info("Patient-level dataset build complete.")
    return master_df

# endregion Data Merging & Feature Engineering

# ------------------------------------------------------------------------------
# 4. Labeling (Defining the Y-Variable)
# ------------------------------------------------------------------------------
# region Severity Labeling

def assign_severity_labels(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives a final COVID-19 severity label for each patient row based on:
      - DEATHDATE => 'Critical'
      - (Hospitalized + Mechanical Vent) => 'Severe'
      - (Hospitalized only) => 'Moderate'
      - Else => 'Mild'
    """
    def severity_logic(row):
        if pd.notnull(row["DEATHDATE"]):
            return "Critical"
        elif (row.get("ANY_INPATIENT", 0) == 1) and (
            (row.get("USED_MECH_VENT", 0) == 1) or (row.get("VENT_PROCEDURE", 0) == 1)
        ):
            return "Severe"
        elif row.get("ANY_INPATIENT", 0) == 1:
            return "Moderate"
        else:
            return "Mild"
    
    master_df["COVID19_SEVERITY"] = master_df.apply(severity_logic, axis=1)
    return master_df

# endregion Severity Labeling

# ------------------------------------------------------------------------------
# 5. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
# region EDA Plotting

def run_eda_and_save_plots(master_df: pd.DataFrame) -> None:
    """
    Creates enhanced EDA plots, including:
      - standard severity distribution
      - new lab-based columns (e.g., last c-reactive protein) 
      - medication usage 
      - correlation with severity
      - NEW: Added specific lab value plots (body temperature, diastolic BP, oxygen saturation)
    """
    logging.info("Generating EDA plots...")
    ordered_sev = ["Mild", "Moderate", "Severe", "Critical"]
    sns.set_theme(style="whitegrid")

    # --- Helper function for Pie Chart Generation ---
    def plot_binary_feature_pie_comparison(master_df, feature_col, feature_desc, palette_name="viridis", ordered_sev=None):
        """Generates a side-by-side pie chart comparison for a binary feature vs. severity.

        Args:
            master_df (pd.DataFrame): The main dataframe.
            feature_col (str): The binary (0/1) feature column name.
            feature_desc (str): A human-readable description of the feature for titles.
            palette_name (str, optional): Color palette name for seaborn. Defaults to "viridis".
            ordered_sev (list, optional): Order of severity labels. Defaults to None.
        """
        if ordered_sev is None:
            ordered_sev = ["Mild", "Moderate", "Severe", "Critical"]

        if feature_col not in master_df.columns:
            logging.warning(f"Column '{feature_col}' not found for pie chart plotting.")
            return

        # Ensure the feature column is integer type (0 or 1)
        master_df[feature_col] = master_df[feature_col].astype(int)

        # Calculate severity counts for each group (feature=0 and feature=1)
        counts_no = master_df[master_df[feature_col] == 0]["COVID19_SEVERITY"].value_counts().reindex(ordered_sev, fill_value=0)
        counts_yes = master_df[master_df[feature_col] == 1]["COVID19_SEVERITY"].value_counts().reindex(ordered_sev, fill_value=0)

        # Check if either group is empty
        if counts_no.sum() == 0 and counts_yes.sum() == 0:
            logging.info(f"[INFO] No data for feature '{feature_col}' to plot pie charts.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"COVID-19 Severity Distribution by {feature_desc}", fontsize=14, fontweight='bold')

        # Define colors based on severity order
        colors = sns.color_palette(palette_name, n_colors=len(ordered_sev))

        # Plot Pie for "No" group
        ax = axes[0]
        if counts_no.sum() > 0:
            wedges, texts, autotexts = ax.pie(
                counts_no,
                labels=counts_no.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                wedgeprops={'edgecolor': 'white'}
            )
            # Adjust title size
            ax.set_title(f"Without {feature_desc} (N={counts_no.sum()})", fontsize=12)
            # Adjust autopct (percentage) size and color
            plt.setp(autotexts, size=11, weight="bold", color="black")
            # Adjust label (category) size
            plt.setp(texts, size=10)
        else:
            ax.text(0.5, 0.5, 'No Patients', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"Without {feature_desc} (N=0)", fontsize=12)
        ax.axis('equal') # Equal aspect ratio ensures a circular pie chart.

        # Plot Pie for "Yes" group
        ax = axes[1]
        if counts_yes.sum() > 0:
            wedges, texts, autotexts = ax.pie(
                counts_yes,
                labels=counts_yes.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                wedgeprops={'edgecolor': 'white'}
            )
            # Adjust title size
            ax.set_title(f"With {feature_desc} (N={counts_yes.sum()})", fontsize=12)
            # Adjust autopct (percentage) size and color
            plt.setp(autotexts, size=11, weight="bold", color="black")
            # Adjust label (category) size
            plt.setp(texts, size=10)
        else:
            ax.text(0.5, 0.5, 'No Patients', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"With {feature_desc} (N=0)", fontsize=12)
        ax.axis('equal')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        outpath = RESULTS_DIR / f"{feature_col.lower()}_severity_pie_comparison.png"
        plt.savefig(outpath)
        plt.close(fig) # Close the figure to free memory
    # --- End Helper function ---

    # A) Severity distribution (using original countplot, fixing FutureWarning)
    plt.figure(figsize=(6, 4))
    # Fix seaborn FutureWarning by explicitly setting hue parameter
    sns.countplot(data=master_df, x="COVID19_SEVERITY", order=ordered_sev, 
                 hue="COVID19_SEVERITY", legend=False)  # Updated approach
    plt.title("COVID-19 Severity Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Severity")
    plt.ylabel("Count")
    plt.tight_layout()
    outpath = RESULTS_DIR / "covid19_severity_distribution.png"
    plt.savefig(outpath)
    plt.close()

    # B) Age distribution by severity (fixing FutureWarning)
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=master_df, x="COVID19_SEVERITY", y="AGE", order=ordered_sev, 
               hue="COVID19_SEVERITY", legend=False)  # Fixed warning by setting hue and legend=False
    plt.title("Age by Severity", fontsize=12, fontweight='bold')
    plt.xlabel("Severity")
    plt.ylabel("Age")
    plt.tight_layout()
    outpath = RESULTS_DIR / "age_by_severity.png"
    plt.savefig(outpath)
    plt.close()

    # --- NEW PLOTS PER USER REQUEST ---
    
    # 1. Body Temperature vs Severity
    body_temp_col = None
    for c in master_df.columns:
        if re.search(r"last.*body.*temp", c.lower()):
            body_temp_col = c
            break
    
    if body_temp_col and master_df[body_temp_col].notnull().sum() > 0:
        plt.figure(figsize=(8, 6))
        master_df[body_temp_col] = pd.to_numeric(master_df[body_temp_col], errors='coerce')
        sns.boxplot(data=master_df.dropna(subset=[body_temp_col]), 
                   x="COVID19_SEVERITY", y=body_temp_col, 
                   order=ordered_sev, hue="COVID19_SEVERITY", 
                   palette="YlOrRd", legend=False)
        plt.title("Body Temperature by COVID-19 Severity", fontsize=14, fontweight='bold')
        plt.xlabel("Severity", fontsize=12)
        plt.ylabel("Body Temperature", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        temp_outpath = RESULTS_DIR / "body_temperature_by_severity.png"
        plt.savefig(temp_outpath, dpi=120)
        plt.close()
        logging.info(f"Created body temperature by severity chart: {temp_outpath}")
    else:
        logging.warning("Body temperature column not found or has no data for plotting.")
        
    # 2. Diastolic Blood Pressure vs Severity
    dbp_col = None
    for c in master_df.columns:
        if re.search(r"last.*diastolic", c.lower()):
            dbp_col = c
            break
    
    if dbp_col and master_df[dbp_col].notnull().sum() > 0:
        plt.figure(figsize=(8, 6))
        master_df[dbp_col] = pd.to_numeric(master_df[dbp_col], errors='coerce')
        sns.boxplot(data=master_df.dropna(subset=[dbp_col]), 
                   x="COVID19_SEVERITY", y=dbp_col, 
                   order=ordered_sev, hue="COVID19_SEVERITY", 
                   palette="Blues", legend=False)
        plt.title("Diastolic Blood Pressure by COVID-19 Severity", fontsize=14, fontweight='bold')
        plt.xlabel("Severity", fontsize=12)
        plt.ylabel("Diastolic Blood Pressure (mmHg)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        dbp_outpath = RESULTS_DIR / "diastolic_bp_by_severity.png"
        plt.savefig(dbp_outpath, dpi=120)
        plt.close()
        logging.info(f"Created diastolic blood pressure by severity chart: {dbp_outpath}")
    else:
        logging.warning("Diastolic blood pressure column not found or has no data for plotting.")
        
    # 3. Oxygen Saturation vs Severity
    oxy_col = None
    for c in master_df.columns:
        if re.search(r"last.*oxygen.*sat", c.lower()):
            oxy_col = c
            break
    
    if oxy_col and master_df[oxy_col].notnull().sum() > 0:
        plt.figure(figsize=(8, 6))
        master_df[oxy_col] = pd.to_numeric(master_df[oxy_col], errors='coerce')
        sns.boxplot(data=master_df.dropna(subset=[oxy_col]), 
                   x="COVID19_SEVERITY", y=oxy_col, 
                   order=ordered_sev, hue="COVID19_SEVERITY", 
                   palette="Greens", legend=False)
        plt.title("Oxygen Saturation by COVID-19 Severity", fontsize=14, fontweight='bold')
        plt.xlabel("Severity", fontsize=12)
        plt.ylabel("Oxygen Saturation (%)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        oxy_outpath = RESULTS_DIR / "oxygen_saturation_by_severity.png"
        plt.savefig(oxy_outpath, dpi=120)
        plt.close()
        logging.info(f"Created oxygen saturation by severity chart: {oxy_outpath}")
    else:
        logging.warning("Oxygen saturation column not found or has no data for plotting.")
    
    # --- CONTINUE WITH OTHER EXISTING PLOTS ---
    
    # C) Gender distribution by severity
    plt.figure(figsize=(7, 5))
    sns.countplot(data=master_df, x="GENDER", hue="COVID19_SEVERITY", hue_order=ordered_sev, palette="husl")
    plt.title("Gender vs. Severity", fontsize=14, fontweight='bold')
    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title='Severity', title_fontsize='11', fontsize='10')
    plt.tight_layout()
    outpath = RESULTS_DIR / "gender_by_severity.png"
    plt.savefig(outpath)
    plt.close()

    # D) Hypertension vs severity (Using Pie Chart Comparison)
    plot_binary_feature_pie_comparison(master_df, 'HAS_HYPERTENSION', 'Hypertension', palette_name='plasma', ordered_sev=ordered_sev)

    # E) Diabetes vs severity (Using Pie Chart Comparison)
    plot_binary_feature_pie_comparison(master_df, 'HAS_DIABETES', 'Diabetes', palette_name='cividis', ordered_sev=ordered_sev)

    # F) Any Antibiotic usage vs severity (Using Pie Chart Comparison)
    plot_binary_feature_pie_comparison(master_df, 'ANY_ANTIBIOTIC', 'Antibiotic Usage', palette_name='Accent', ordered_sev=ordered_sev)

    # G) Distinct medication count by severity
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=master_df, x="COVID19_SEVERITY", y="DISTINCT_MED_COUNT", order=ordered_sev,
                hue="COVID19_SEVERITY", legend=False)  # Fixed warning
    plt.title("Distinct Med Count by Severity", fontsize=12, fontweight='bold')
    plt.xlabel("Severity")
    plt.ylabel("# Distinct Medications")
    plt.tight_layout()
    outpath = RESULTS_DIR / "med_count_by_severity.png"
    plt.savefig(outpath)
    plt.close()

    # Additional plots for newly pivoted labs/vitals

    # H) C-reactive protein vs Severity
    crp_col = None
    for c in master_df.columns:
        # Use regex to find the column more reliably
        if re.search(r"last_c_reactive_protein", c.lower()):
            crp_col = c
            break

    if crp_col and master_df[crp_col].notnull().sum() > 0:
        # Convert to numeric if it's not already, coercing errors
        master_df[crp_col] = pd.to_numeric(master_df[crp_col], errors='coerce')
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=master_df.dropna(subset=[crp_col]),
                    x="COVID19_SEVERITY", y=crp_col, order=ordered_sev,
                    hue="COVID19_SEVERITY", legend=False)  # Fixed warning
        plt.title("Last C-reactive protein vs Severity", fontsize=12, fontweight='bold')
        plt.xlabel("Severity")
        plt.ylabel("CRP Value") # Removed uncertain units
        plt.tight_layout()
        outpath = RESULTS_DIR / "crp_by_severity.png"
        plt.savefig(outpath)
        plt.close()

    # I) Oxygen saturation in arterial blood vs Severity
    oxy_col = None
    for c in master_df.columns:
        if re.search(r"last_oxygen_saturation.*art", c.lower()): # Find O2 sat in arterial blood
            oxy_col = c
            break

    if oxy_col and master_df[oxy_col].notnull().sum() > 0:
        master_df[oxy_col] = pd.to_numeric(master_df[oxy_col], errors='coerce')
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=master_df.dropna(subset=[oxy_col]),
                    x="COVID19_SEVERITY", y=oxy_col, order=ordered_sev,
                    hue="COVID19_SEVERITY", legend=False)  # Fixed warning
        plt.title("Last O2 Saturation (Arterial) vs Severity", fontsize=12, fontweight='bold')
        plt.xlabel("Severity")
        plt.ylabel("Oxygen Saturation (%)") # Assuming % unit is standard
        plt.tight_layout()
        outpath = RESULTS_DIR / "oxygen_saturation_by_severity.png"
        plt.savefig(outpath)
        plt.close()

    # J) Serum Glucose vs Severity
    glu_col = None
    for c in master_df.columns:
        if re.search(r"last_glucose.*serum", c.lower()): # Find glucose in serum/plasma
            glu_col = c
            break

    if glu_col and master_df[glu_col].notnull().sum() > 0:
        master_df[glu_col] = pd.to_numeric(master_df[glu_col], errors='coerce')
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=master_df.dropna(subset=[glu_col]),
                    x="COVID19_SEVERITY", y=glu_col, order=ordered_sev,
                    hue="COVID19_SEVERITY", legend=False)  # Fixed warning
        plt.title("Last Serum Glucose vs Severity", fontsize=12, fontweight='bold')
        plt.xlabel("Severity")
        plt.ylabel("Glucose Value") # Removed uncertain units
        plt.tight_layout()
        outpath = RESULTS_DIR / "serum_glucose_by_severity.png"
        plt.savefig(outpath)
        plt.close()

    # K) Systolic Blood Pressure vs Severity
    sbp_col = None
    for c in master_df.columns:
        # Use regex to find the systolic BP column more reliably
        if re.search(r"last_systolic_blood_pressure", c.lower()):
            sbp_col = c
            break

    if sbp_col and master_df[sbp_col].notnull().sum() > 0:
        master_df[sbp_col] = pd.to_numeric(master_df[sbp_col], errors='coerce')
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=master_df.dropna(subset=[sbp_col]),
                    x="COVID19_SEVERITY", y=sbp_col, order=ordered_sev,
                    hue="COVID19_SEVERITY", legend=False)  # Fixed warning
        plt.title("Last Systolic BP vs Severity", fontsize=12, fontweight='bold')
        plt.xlabel("Severity")
        plt.ylabel("Systolic Blood Pressure (mmHg?)") # Add likely unit
        plt.tight_layout()
        outpath = RESULTS_DIR / "systolic_bp_by_severity.png"
        plt.savefig(outpath)
        plt.close()

    # --- NEW Plots for Added Features ---

    # L) Allergy Flags vs Severity (Using Pie Chart Comparison)
    allergy_cols = {
        "HAS_NUT_ALLERGY": "Nut Allergy",
        "HAS_POLLEN_ALLERGY": "Pollen/Grass Allergy",
        "HAS_DRUG_ALLERGY": "Drug Allergy"
    }
    for i, (col, desc) in enumerate(allergy_cols.items()):
        plot_binary_feature_pie_comparison(master_df, col, desc, palette_name=f"Set{i+1}", ordered_sev=ordered_sev)

    # M) Encounter Counts vs Severity
    encounter_cols = {
        'TOTAL_EMERGENCY': "Total Emergency Visits",
        'TOTAL_WELLNESS': "Total Wellness Visits",
        'TOTAL_AMBULATORY': "Total Ambulatory Visits",
        'TOTAL_ENCOUNTERS': "Total Encounters"
    }
    for i, (col, desc) in enumerate(encounter_cols.items()):
         if col in master_df.columns:
            plt.figure(figsize=(7, 5))
            sns.boxplot(data=master_df, x="COVID19_SEVERITY", y=col, order=ordered_sev, 
                       hue="COVID19_SEVERITY", legend=False) # Fixed FutureWarning
            plt.title(f"{desc} vs. Severity", fontsize=12, fontweight='bold')
            plt.xlabel("Severity")
            plt.ylabel(f"Number of {desc}")
            # Optionally set y-limit if counts vary wildly
            # plt.ylim(0, master_df[col].quantile(0.99) * 1.1) # Example: Limit to 99th percentile
            plt.tight_layout()
            outpath = RESULTS_DIR / f"{col.lower()}_by_severity.png"
            plt.savefig(outpath)
            plt.close()
         else:
             logging.warning(f"[WARNING] Column '{col}' not found for plotting.")

    # N) Recent Flu Vaccine vs Severity (Using Pie Chart Comparison)
    plot_binary_feature_pie_comparison(master_df, 'HAD_RECENT_FLU_VAX', 'Recent Flu Vaccine', palette_name='coolwarm', ordered_sev=ordered_sev)

    # O) Care Plan Flags vs Severity (Using Pie Chart Comparison)
    cp_cols = {
        "HAS_HYPERTENSION_CAREPLAN": "Hypertension Care Plan",
        "HAS_DIABETES_CAREPLAN": "Diabetes Care Plan",
        "HAS_INFECT_DISEASE_CAREPLAN": "Infectious Disease Care Plan"
    }
    for i, (col, desc) in enumerate(cp_cols.items()):
        plot_binary_feature_pie_comparison(master_df, col, desc, palette_name=f"cubehelix", ordered_sev=ordered_sev)

    logging.info("EDA plot generation complete.")
# endregion EDA Plotting

# ------------------------------------------------------------------------------
# 6. Enhanced Patient Journey Plots (Version 1.8)
# ------------------------------------------------------------------------------
# region Patient Journey Plotting

def get_patient_lab_trends(patient_id, obs_df, key_lab_values, covid_diagnosis_date):
    """
    Extract lab value trends for a specific patient, indexed relative to COVID diagnosis date.
    Returns a dictionary of lab trends with days since diagnosis as key.
    """
    # Convert date columns
    obs_df["DATE"] = pd.to_datetime(obs_df["DATE"], errors="coerce")
    
    # Filter for the specific patient and key lab values
    patient_obs = obs_df[(obs_df["PATIENT"] == patient_id)].copy()  # Added .copy() to avoid SettingWithCopyWarning
    
    # Skip if patient has no observations
    if patient_obs.empty:
        return {lab: [] for lab in key_lab_values}
    
    # Ensure timezone-naive for comparison
    if not patient_obs["DATE"].empty and hasattr(patient_obs["DATE"].iloc[0], 'tz') and patient_obs["DATE"].iloc[0].tz is not None:
        patient_obs["DATE"] = patient_obs["DATE"].dt.tz_localize(None)
    
    # Standardize lab descriptions to lowercase for matching
    patient_obs["LOW_DESC"] = patient_obs["DESCRIPTION"].str.lower()
    
    # Create a dictionary to store lab trends
    lab_trends = {lab: [] for lab in key_lab_values}
    
    # Process each key lab value
    for lab in key_lab_values:
        # Filter observations for this lab value using partial matching
        lab_obs = patient_obs[patient_obs["LOW_DESC"].str.contains(lab.lower(), na=False)].copy()  # Added .copy()
        
        # Skip if no data for this lab value
        if lab_obs.empty:
            continue
        
        # Convert values to numeric, coercing errors to NaN
        lab_obs["VALUE_NUM"] = pd.to_numeric(lab_obs["VALUE"], errors="coerce")
        
        # Calculate days since COVID diagnosis
        lab_obs["DAYS_SINCE_COVID"] = (lab_obs["DATE"] - covid_diagnosis_date).dt.days
        
        # Skip if no valid days calculation 
        if lab_obs["DAYS_SINCE_COVID"].isna().all():
            continue
        
        # Store the result as (days_since_covid, value) tuples, sorted by date
        filtered_obs = lab_obs.dropna(subset=["DAYS_SINCE_COVID", "VALUE_NUM"])
        data = filtered_obs[["DAYS_SINCE_COVID", "VALUE_NUM"]].sort_values("DAYS_SINCE_COVID").values.tolist()
        
        # Store the data for this lab
        lab_trends[lab] = data
    
    return lab_trends

def get_patient_medications(patient_id, med_df, covid_diagnosis_date):
    """
    Extract medication periods for a patient, indexed relative to COVID diagnosis date.
    Returns a list of medication periods with start/end days and description.
    """
    # Convert date columns
    med_df["START"] = pd.to_datetime(med_df["START"], errors="coerce")
    med_df["STOP"] = pd.to_datetime(med_df["STOP"], errors="coerce")
    
    # Filter for the specific patient
    patient_meds = med_df[med_df["PATIENT"] == patient_id].copy()  # Added .copy()
    
    # Skip if no medications
    if patient_meds.empty:
        return []
    
    # Ensure timezone-naive for comparison
    if not patient_meds["START"].empty and hasattr(patient_meds["START"].iloc[0], 'tz') and patient_meds["START"].iloc[0].tz is not None:
        patient_meds["START"] = patient_meds["START"].dt.tz_localize(None)
    
    if not patient_meds["STOP"].empty and hasattr(patient_meds["STOP"].iloc[0], 'tz') and patient_meds["STOP"].iloc[0].tz is not None:
        patient_meds["STOP"] = patient_meds["STOP"].dt.tz_localize(None)
    
    # Calculate days since COVID diagnosis
    patient_meds["START_DAYS"] = (patient_meds["START"] - covid_diagnosis_date).dt.days
    patient_meds["STOP_DAYS"] = (patient_meds["STOP"] - covid_diagnosis_date).dt.days
    
    # Filter out medications with invalid start/stop days
    patient_meds = patient_meds.dropna(subset=["START_DAYS", "STOP_DAYS"])
    
    # Convert to list of (start_day, end_day, description) tuples
    med_periods = []
    for _, row in patient_meds.iterrows():
        med_periods.append((row["START_DAYS"], row["STOP_DAYS"], row["DESCRIPTION"]))
    
    return med_periods

def get_inpatient_periods(patient_id, enc_df, covid_diagnosis_date):
    """
    Extract inpatient periods for a patient, indexed relative to COVID diagnosis date.
    Returns a list of (start_day, end_day) tuples.
    """
    # Convert date columns
    enc_df["START"] = pd.to_datetime(enc_df["START"], errors="coerce")
    enc_df["STOP"] = pd.to_datetime(enc_df["STOP"], errors="coerce")
    
    # Filter for the specific patient's inpatient encounters
    inpatient_enc = enc_df[
        (enc_df["PATIENT"] == patient_id) &
        (enc_df["ENCOUNTERCLASS"] == "inpatient")
    ].copy()  # Use copy to prevent SettingWithCopyWarning
    
    # Skip if no inpatient encounters
    if inpatient_enc.empty:
        return []
    
    # Ensure timezone-naive for comparison
    if not inpatient_enc["START"].empty and hasattr(inpatient_enc["START"].iloc[0], 'tz') and inpatient_enc["START"].iloc[0].tz is not None:
        inpatient_enc["START"] = inpatient_enc["START"].dt.tz_localize(None)
    
    if not inpatient_enc["STOP"].empty and hasattr(inpatient_enc["STOP"].iloc[0], 'tz') and inpatient_enc["STOP"].iloc[0].tz is not None:
        inpatient_enc["STOP"] = inpatient_enc["STOP"].dt.tz_localize(None)
    
    # Calculate days since COVID diagnosis
    inpatient_enc["START_DAYS"] = (inpatient_enc["START"] - covid_diagnosis_date).dt.days
    inpatient_enc["STOP_DAYS"] = (inpatient_enc["STOP"] - covid_diagnosis_date).dt.days
    
    # Filter out encounters with invalid start/stop days
    inpatient_enc = inpatient_enc.dropna(subset=["START_DAYS", "STOP_DAYS"])
    
    # Convert to list of (start_day, end_day) tuples
    inpatient_periods = []
    for _, row in inpatient_enc.iterrows():
        inpatient_periods.append((row["START_DAYS"], row["STOP_DAYS"]))
    
    return inpatient_periods

def get_patient_events(patient_id, master_df, conditions_df, encounters_df, procedures_df, covid_diagnosis_date):
    """
    Extract key clinical events for a patient, indexed relative to COVID diagnosis date.
    Returns a dictionary of event types and their days since diagnosis.
    """
    events = {}
    
    # Get patient data from master dataset
    patient_row = master_df[master_df["PATIENT"] == patient_id]
    if patient_row.empty:
        return events
    
    # Check for death
    if pd.notnull(patient_row["DEATHDATE"].values[0]):
        death_date = pd.to_datetime(patient_row["DEATHDATE"].values[0])
        # Ensure timezone-naive for comparison
        if hasattr(death_date, 'tz') and death_date.tz is not None:
            death_date = death_date.tz_localize(None)
        death_days = (death_date - covid_diagnosis_date).days
        if not pd.isna(death_days):
            events["Death"] = death_days
    
    # Get first inpatient admission
    pat_enc = encounters_df[
        (encounters_df["PATIENT"] == patient_id) &
        (encounters_df["ENCOUNTERCLASS"] == "inpatient")
    ].copy()  # Use copy to prevent warning
    
    if not pat_enc.empty:
        pat_enc["START"] = pd.to_datetime(pat_enc["START"], errors="coerce")
        # Ensure timezone-naive for comparison
        if hasattr(pat_enc["START"].iloc[0], 'tz') and pat_enc["START"].iloc[0].tz is not None:
            pat_enc["START"] = pat_enc["START"].dt.tz_localize(None)
        
        first_inpatient = pat_enc["START"].min()
        if not pd.isna(first_inpatient):
            # Ensure timezone-naive for comparison
            if hasattr(first_inpatient, 'tz') and first_inpatient.tz is not None:
                first_inpatient = first_inpatient.tz_localize(None)
            inpatient_days = (first_inpatient - covid_diagnosis_date).days
            if not pd.isna(inpatient_days):
                events["Inpatient"] = inpatient_days
    
    # Get last discharge date
    if not pat_enc.empty:
        pat_enc["STOP"] = pd.to_datetime(pat_enc["STOP"], errors="coerce")
        # Ensure timezone-naive for comparison
        if not pat_enc["STOP"].empty and hasattr(pat_enc["STOP"].iloc[0], 'tz') and pat_enc["STOP"].iloc[0].tz is not None:
            pat_enc["STOP"] = pat_enc["STOP"].dt.tz_localize(None)
        
        last_discharge = pat_enc["STOP"].max()
        if not pd.isna(last_discharge):
            # Ensure timezone-naive for comparison
            if hasattr(last_discharge, 'tz') and last_discharge.tz is not None:
                last_discharge = last_discharge.tz_localize(None)
            discharge_days = (last_discharge - covid_diagnosis_date).days
            if not pd.isna(discharge_days):
                events["Discharge"] = discharge_days
    
    # Get ventilation/intubation procedure
    pat_proc = procedures_df[
        (procedures_df["PATIENT"] == patient_id) &
        (procedures_df["DESCRIPTION"].str.contains("ventilation|intubat", case=False, na=False))
    ].copy()  # Use copy to prevent warning
    
    if not pat_proc.empty:
        pat_proc["DATE"] = pd.to_datetime(pat_proc["DATE"], errors="coerce")
        # Ensure timezone-naive for comparison
        if not pat_proc["DATE"].empty and hasattr(pat_proc["DATE"].iloc[0], 'tz') and pat_proc["DATE"].iloc[0].tz is not None:
            pat_proc["DATE"] = pat_proc["DATE"].dt.tz_localize(None)
        
        first_vent = pat_proc["DATE"].min()
        if not pd.isna(first_vent):
            # Ensure timezone-naive for comparison
            if hasattr(first_vent, 'tz') and first_vent.tz is not None:
                first_vent = first_vent.tz_localize(None)
            vent_days = (first_vent - covid_diagnosis_date).days
            if not pd.isna(vent_days):
                events["VentProc"] = vent_days
    
    return events

def create_patient_journey_plots(master_df: pd.DataFrame,
                                encounters_df: pd.DataFrame,
                                conditions_df: pd.DataFrame,
                                procedures_df: pd.DataFrame,
                                observations_df: pd.DataFrame,
                                medications_df: pd.DataFrame) -> None:
    """
    Creates detailed patient journey visualizations, one for each severity level.
    Each visualization includes:
    - Event timeline (diagnosis, hospitalization, ventilation, discharge/death)
    - Lab value trends over time
    SIMPLIFIED: Removed medication timeline and clinical context to focus on key elements
    Only includes adult patients (18+ years old)
    Uses random selection with fixed seed for choosing representative patients
    """
    logging.info("Creating enhanced patient journey visualizations...")
    
    # Set random seed for reproducibility
    np.random.seed(55)
    
    # Convert date fields for key dataframes
    conditions_df["START_TS"] = pd.to_datetime(conditions_df["START"], errors="coerce")
    
    # Filter to only include adult patients (18+ years old)
    adult_patients = master_df[master_df["AGE"] >= 18]["PATIENT"].unique()
    logging.info(f"[INFO] Found {len(adult_patients)} adult patients (18+ years old)")
    
    # Get one representative patient for each severity level (adults only)
    severity_levels = ["Mild", "Moderate", "Severe", "Critical"]
    selected_patients = []
    
    for severity in severity_levels:
        # Filter patients with this severity (adults only)
        severity_patients = master_df[(master_df["COVID19_SEVERITY"] == severity) & 
                                     (master_df["PATIENT"].isin(adult_patients))]
        
        if len(severity_patients) == 0:
            logging.warning(f"[WARNING] No adult patients found with {severity} severity")
            continue
        
        # Try to find patients with good data availability
        suitable_patients = []
        
        # First check if we can find patients with both hospitalization and lab data
        if severity in ["Moderate", "Severe", "Critical"]:
            # These severities should have inpatient stays
            for _, patient in severity_patients.iterrows():
                patient_id = patient["PATIENT"]
                
                # Check if the patient has COVID diagnosis date
                patient_covid = conditions_df[
                    (conditions_df["PATIENT"] == patient_id) & 
                    (conditions_df["CODE"].isin(COVID_CODES))
                ]
                
                if patient_covid.empty:
                    continue
                
                # Check if the patient has inpatient encounters
                patient_enc = encounters_df[
                    (encounters_df["PATIENT"] == patient_id) & 
                    (encounters_df["ENCOUNTERCLASS"] == "inpatient")
                ]
                
                if patient_enc.empty and severity != "Mild":
                    continue
                
                # Check if the patient has some observations
                patient_obs = observations_df[observations_df["PATIENT"] == patient_id]
                has_obs = False
                
                if not patient_obs.empty:
                    # Check for at least some key lab values
                    for lab in KEY_LAB_VALUES:
                        if patient_obs["DESCRIPTION"].str.contains(lab, case=False, na=False).any():
                            has_obs = True
                            break
                
                if has_obs:
                    suitable_patients.append(patient_id)
        
        # If suitable patients are found, randomly select one
        if suitable_patients:
            # Randomly select a patient from the suitable patients list
            random_patient_id = np.random.choice(suitable_patients)
            selected_patients.append((random_patient_id, severity))
            logging.info(f"[INFO] Randomly selected a suitable patient for {severity} severity from {len(suitable_patients)} candidates")
        else:
            # No suitable patients found, randomly select from all patients in this severity
            all_patients_ids = severity_patients["PATIENT"].tolist()
            random_patient_id = np.random.choice(all_patients_ids)
            selected_patients.append((random_patient_id, severity))
            logging.info(f"[INFO] Randomly selected from all {severity} patients ({len(all_patients_ids)} patients)")
    
    logging.info(f"[INFO] Selected {len(selected_patients)} adult patients for journey visualization: {selected_patients}")
    
    # Create detailed visualizations for each selected patient
    for i, (patient_id, severity) in enumerate(selected_patients):
        # Get the patient's COVID diagnosis date
        patient_covid = conditions_df[
            (conditions_df["PATIENT"] == patient_id) & 
            (conditions_df["CODE"].isin(COVID_CODES))
        ]
        
        if patient_covid.empty:
            logging.warning(f"[WARNING] No COVID diagnosis found for patient {patient_id}")
            continue
        
        covid_diagnosis_date = patient_covid["START_TS"].min()
        
        # Get patient demographics
        patient_row = master_df[master_df["PATIENT"] == patient_id].iloc[0]
        age = patient_row.get("AGE", "unknown")
        gender = patient_row.get("GENDER", "unknown")
        
        # Get key events
        events = get_patient_events(patient_id, master_df, conditions_df, 
                                   encounters_df, procedures_df, covid_diagnosis_date)
        
        # Get lab trends
        lab_trends = get_patient_lab_trends(patient_id, observations_df, 
                                          KEY_LAB_VALUES, covid_diagnosis_date)
        
        # Get inpatient periods
        inpatient_periods = get_inpatient_periods(patient_id, encounters_df, covid_diagnosis_date)
        
        # Handle edge case of empty data
        if len(events) == 0 and all(len(data) == 0 for data in lab_trends.values()):
            logging.warning(f"[WARNING] Not enough data to create visualization for patient {patient_id} with {severity} severity. Skipping.")
            continue
        
        # Create simplified multi-panel visualization - ONLY 2 panels
        fig = plt.figure(figsize=(10, 8))
        plt.clf()  # Clear any existing plots
        
        # Use 2 panels with more space between them
        gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        
        # Panel 1: Event Timeline
        ax1 = fig.add_subplot(gs[0])
        
        # Sort events by day
        sorted_events = [(event, day) for event, day in events.items()]
        sorted_events.append(("DiagnosedCOVID", 0))  # Always add diagnosis at day 0
        sorted_events = sorted(sorted_events, key=lambda x: x[1])
        
        # Plot events as a timeline
        event_days = [day for _, day in sorted_events]
        event_y = np.ones(len(event_days))
        
        ax1.plot(event_days, event_y, 'o-', color='steelblue', linewidth=2, markersize=10)
        
        # Add event labels with day numbers - Improved readability
        for i, (event, day) in enumerate(sorted_events):
            color = JOURNEY_COLORS.get(event, 'black')
            ax1.annotate(f"{event}\nDay {day}", 
                       xy=(day, 1), 
                       xytext=(0, 10),
                       textcoords="offset points",
                       ha='center', 
                       va='bottom',
                       color=color,
                       fontweight='bold',
                       fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=color, alpha=0.8))
        
        # Mark inpatient periods with shading
        for start_day, end_day in inpatient_periods:
            rect = patches.Rectangle((start_day, 0.7), end_day - start_day, 0.6, 
                                    linewidth=1, alpha=0.2, facecolor='orange')
            ax1.add_patch(rect)
        
        # Set axis limits and labels
        min_day = min(event_days) - 1 if event_days else -1
        max_day = max(event_days) + 1 if event_days else 30
        
        # Special handling for the specific problematic Critical patient
        if patient_id == 'c70992c9-ff13-467b-9032-1901506edeef':
            logging.info(f"[INFO] Applying special handling for critical patient {patient_id}")
            # Override to only show last 1 month before and after diagnosis
            min_day = -30
            max_day = 30
        # For Mild and Critical, limit the x-axis range to avoid image size issues
        elif severity in ["Mild", "Critical"]:
            # Ensure the range is at most 60 days to prevent size issues
            if max_day - min_day > 60:
                mean_day = (max_day + min_day) / 2
                min_day = max(min_day, mean_day - 30)
                max_day = min(max_day, mean_day + 30)
                logging.info(f"[INFO] Limiting {severity} patient x-axis range to prevent image size issues: [{min_day}, {max_day}]")
        
        ax1.set_xlim(min_day, max_day)
        ax1.set_ylim(0.5, 1.5)
        ax1.set_yticks([])
        ax1.set_xlabel("Days Since COVID-19 Diagnosis", fontsize=12)
        ax1.set_title(f"Patient Timeline: {severity} Severity (Age: {age}, Gender: {gender})", 
                     fontsize=16, fontweight='bold')
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Panel 2: Lab Value Trends
        ax2 = fig.add_subplot(gs[1])
        
        # Plot each lab trend on the same timeline with normalized values for better visualization
        lab_lines = []
        
        # Check if we have any lab data
        has_lab_data = any(len(data) > 0 for lab, data in lab_trends.items())
        
        if has_lab_data:
            # For each lab type, normalize or scale values if necessary for better display
            for lab, data in lab_trends.items():
                if not data:
                    continue
                    
                x_vals = [point[0] for point in data]
                y_vals = [point[1] for point in data]
                
                if len(y_vals) > 0:
                    color = JOURNEY_COLORS.get(f"{lab.split()[0]}Line", 'gray')
                    
                    # Add proper label with units depending on the lab
                    label = lab
                    if "oxygen saturation" in lab.lower():
                        label = f"{lab} (%)"
                    elif "body temperature" in lab.lower():
                        label = f"{lab} (°C or °F)"
                    elif "c reactive protein" in lab.lower():
                        label = f"{lab} (mg/L)"
                    elif "respiratory rate" in lab.lower():
                        label = f"{lab} (breaths/min)"
                    
                    line, = ax2.plot(x_vals, y_vals, 'o-', label=label, linewidth=2, markersize=6, color=color)
                    lab_lines.append(line)
            
            # Add a legend if we have lab data
            if lab_lines:
                ax2.legend(handles=lab_lines, loc='upper right', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'No lab data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            
        # Set axis labels
        ax2.set_xlabel("Days Since COVID-19 Diagnosis", fontsize=12)
        ax2.set_ylabel("Lab Values", fontsize=12)
        ax2.set_title("Lab Value Trends", fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Match x-axis limits with the event timeline
        ax2.set_xlim(min_day, max_day)
        
        # Mark inpatient periods with shading
        if has_lab_data:
            y_min, y_max = ax2.get_ylim()
            for start_day, end_day in inpatient_periods:
                rect = patches.Rectangle((start_day, y_min), end_day - start_day, y_max - y_min, 
                                        linewidth=1, alpha=0.1, facecolor='orange')
                ax2.add_patch(rect)
        
        # Add severity indicator in top right corner
        severity_colors = {
            "Mild": "#2ECC71",      # Green
            "Moderate": "#F39C12",  # Orange
            "Severe": "#E74C3C",    # Red
            "Critical": "#7D3C98"   # Purple
        }
        
        # Add an annotation for severity
        ax1.annotate(f"Severity: {severity}", 
                   xy=(0.98, 0.95),
                   xycoords='axes fraction',
                   ha='right',
                   va='top',
                   fontsize=12,
                   fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=severity_colors.get(severity, "gray"),
                            alpha=0.7,
                            edgecolor='black'))
        
        # Add extra margin space around plots
        plt.subplots_adjust(left=0.12, right=0.92, top=0.90, bottom=0.10)
        
        # Ensure the output directory exists (already done at top, but good practice)
        PATIENT_JOURNEY_DIR.mkdir(parents=True, exist_ok=True)
            
        # Save the figure - PNG format only, with reduced DPI for problematic patients
        outpath = PATIENT_JOURNEY_DIR / f"patient_journey_{severity.lower()}.png"
        try:
            # Use a moderate DPI that balances quality and file size
            dpi_value = 100 if severity in ["Mild", "Critical"] else 120
            plt.savefig(outpath, dpi=dpi_value, bbox_inches='tight', format='png')
            logging.info(f"Created patient journey visualization for {severity} severity: {outpath}")
        except Exception as e:
            logging.error(f"Failed to save patient journey for {severity} severity: {str(e)}")
            # Try with an even more restricted approach if PNG still fails
            try:
                # Further reduce complexity for problematic cases
                if severity in ["Mild", "Critical"]:
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, f"Patient timeline for {severity} severity\n(Age: {age}, Gender: {gender})",
                            ha='center', va='center', fontsize=14, fontweight='bold')
                    plt.axis('off')
                    fallback_path = PATIENT_JOURNEY_DIR / f"patient_journey_{severity.lower()}_fallback.png"
                    plt.savefig(fallback_path, dpi=80, format='png')
                    logging.info(f"Created fallback visualization for {severity} severity: {fallback_path}")
                else:
                    # Try JPEG format with low quality for other severities
                    alt_outpath = PATIENT_JOURNEY_DIR / f"patient_journey_{severity.lower()}.jpg"
                    plt.savefig(alt_outpath, dpi=80, bbox_inches='tight', format='jpg')
                    logging.info(f"Created JPEG format patient journey for {severity} severity: {alt_outpath}")
            except Exception as e2:
                logging.error(f"All save attempts failed for {severity} patient journey: {str(e2)}")
        
        plt.close(fig)
    
    logging.info(f"Patient journey visualizations completed and saved to {PATIENT_JOURNEY_DIR}")

# endregion Patient Journey Plotting

# ------------------------------------------------------------------------------
# 7. Main Script Execution
# ------------------------------------------------------------------------------
# region Main Execution

def main():
    logging.info("=== Starting COVID-19 EDA & Preprocessing Script (v1.8) ===")

    # Step 1: Load data
    logging.info("Reading CSV files...")
    patients_df   = read_csv_or_exit(RELEVANT_FILES["patients"])
    conditions_df = read_csv_or_exit(RELEVANT_FILES["conditions"])
    observations_df = read_csv_or_exit(RELEVANT_FILES["observations"])
    encounters_df   = read_csv_or_exit(RELEVANT_FILES["encounters"])
    medications_df  = read_csv_or_exit(RELEVANT_FILES["medications"])
    devices_df      = read_csv_or_exit(RELEVANT_FILES["devices"])
    supplies_df     = read_csv_or_exit(RELEVANT_FILES["supplies"])
    procedures_df   = read_csv_or_exit(RELEVANT_FILES["procedures"])
    careplans_df    = read_csv_or_exit(RELEVANT_FILES["careplans"])
    allergies_df    = read_csv_or_exit(RELEVANT_FILES["allergies"])
    immunizations_df = read_csv_or_exit(RELEVANT_FILES["immunizations"])

    # Step 2: Filter for COVID-19
    logging.info("Filtering for COVID-19 or Suspected COVID-19 conditions...")
    covid_condition_df = filter_covid_patients(conditions_df)
    logging.info(f"Found {len(covid_condition_df)} COVID-19 condition rows.")

    # Step 3: Build master dataset
    logging.info("Building patient-level dataset with expanded x-variables...")
    master_df = build_patient_level_dataset(
        covid_condition_df=covid_condition_df,
        conditions_df=conditions_df,
        patients_df=patients_df,
        encounters_df=encounters_df,
        observations_df=observations_df,
        procedures_df=procedures_df,
        devices_df=devices_df,
        supplies_df=supplies_df,
        medications_df=medications_df,
        allergies_df=allergies_df,
        careplans_df=careplans_df,
        immunizations_df=immunizations_df,
        target_obs=TARGET_OBSERVATIONS
    )
    logging.info(f"Master dataset shape: {master_df.shape}")

    # Step 4: Label severity
    logging.info("Assigning COVID-19 severity labels...")
    master_df = assign_severity_labels(master_df)

    # Step 5: Run EDA & Save Plots
    logging.info("Running EDA and saving plots...")
    run_eda_and_save_plots(master_df)

    # Step 6: Create enhanced patient journey visualizations
    create_patient_journey_plots(
        master_df=master_df,
        encounters_df=encounters_df,
        conditions_df=conditions_df,
        procedures_df=procedures_df,
        observations_df=observations_df,
        medications_df=medications_df
    )

    # Step 7: Export final dataset
    out_csv_path = RESULTS_DIR / "final_covid19_labeled.csv"
    try:
        master_df.to_csv(out_csv_path, index=False)
        logging.info(f"Final merged and labeled dataset saved to: {out_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save final CSV to {out_csv_path}: {e}")

    logging.info("=== COVID-19 EDA & Preprocessing Complete ===")


if __name__ == "__main__":
    main()

# endregion Main Execution