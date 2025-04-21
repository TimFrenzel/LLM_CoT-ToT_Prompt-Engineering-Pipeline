"""
Author: Tim Frenzel
Version: 1.7
Usage:  python llm_tot_pipeline.py [--mode reasoning_method] [--subset N] [--test_mode] [--simulate_gpt] [--stratify_subset | --balance_subset_mixed] [--no_balance_retrieval]

Description:
------------
Executes and evaluates multiple Large Language Model (LLM) reasoning strategies (CoT, ToT, Self-Consistency, etc.) for predicting COVID-19 severity using preprocessed patient data and few-shot examples.

Key Features:
-----------
1. Multi-Method Reasoning
   - Implements and compares various prompting techniques: Chain-of-Thought (CoT), Tree-of-Thought (ToT), Self-Consistency (SC), Self-Reflection (SR), Fact-Checking (FC), and a Hybrid approach.

2. Few-Shot Learning Integration
   - Leverages semantic retrieval (via `semantic_retrieval.py`) to select contextually relevant patient examples for inclusion in prompts.

3. Configurable Execution & Sampling
   - Supports processing subsets of data, test mode with optional GPT simulation, loading previous results, and different sampling strategies (random, stratified, balanced-mixed).

4. Comprehensive Evaluation
   - Calculates and reports standard classification metrics (accuracy, precision, recall, F1), generates confusion matrices, and analyzes performance across methods and subgroups.

5. Token Usage & Cost Analysis
   - Tracks token consumption for each method and generates visualizations comparing usage and estimated costs.

6. Reasoning Path Visualization
   - Creates graphical representations of the reasoning process, particularly for the Tree-of-Thought method, categorized by clinical factors.

7. Feature Importance Analysis
   - Includes an experimental module to assess the impact of different input features (age, comorbidities, etc.) on LLM predictions.

8. Comparative Analysis Framework
   - Provides functionality to load results from multiple runs/methods and generate comparative performance plots and statistics.

References:
-----------
1. Liu, Jiachang, et al. "What Makes Good In-Context Examples for GPT-3?" Proceedings of Deep
   Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for
   Deep Learning Architectures, 2022, pp. 100–114. (KATE Framework for semantic retrieval)

2. Chen, Banghao, et al. "Unleashing the Potential of Prompt Engineering in Large Language
   Models: A Comprehensive Review." arXiv preprint arXiv:2310.14735, 2024. (Self-consistency voting)

3. Kolbinger, Fiona R., et al. "The future landscape of large language models in medicine" Communications
   Medicine, vol. 3, 2023, p. 141. (Medical fact-checking framework)

4. Liu, Sinuo, et al. "New Trends for Modern Machine Translation with Large Reasoning Models." arXiv
   Preprint, arXiv:2503.10351v2 [cs.CL], 14 Mar. 2025. (Iterative self-reflection mechanism)

5. Wang, Yaoting, et al. "Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey." 2025,
   arXiv:2503.12605v2. (Tree-of-thought reasoning approaches)

6. White, Jules, et al. "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT." arXiv
   preprint arXiv:2302.11382, 2023. (Structured prompt patterns)

NOTE:
 - GPT-4 calls require valid API credentials. This script reads from:
     E:\\001_UT_Austin\\005_AI_in_Healthcare\\005_Homework5.API\\openai_api_key.txt
 - The final results (predictions, confusion matrix, etc.) are stored under
     E:\\001_UT_Austin\\005_AI_in_Healthcare\\005_Homework5\\results\\LLM_inference
 - For "advanced" mode, ensure you have run semantic_retrieval.py beforehand so
   embeddings are available, or let the script automatically compute them.

Example:
  python llm_tot_pipeline.py --mode cot --subset 20                # Chain-of-Thought
  python llm_tot_pipeline.py --mode tot --subset 10                # Tree-of-Thought
  python llm_tot_pipeline.py --mode sc --subset 15                 # Self-Consistency
  python llm_tot_pipeline.py --mode sr --subset 15                 # Self-Reflection
  python llm_tot_pipeline.py --mode fc --subset 15                 # Fact-Checking
  python llm_tot_pipeline.py --mode hybrid --subset 15             # Combined approach
  python llm_tot_pipeline.py --mode compare --methods cot,tot,sc   # Compare multiple methods
  python llm_tot_pipeline.py --test_mode --simulate_gpt            # Quick test without API calls
  python llm_tot_pipeline.py --mode compare --load_previous        # Compare from previous runs

Test mode to verify everything works:
  python llm_tot_pipeline.py --mode compare --methods cot,tot,sc --test_mode --simulate_gpt --subset 10

Run all with actual GPT-3.5 calls (requires API key, lower cost):
  python scripts/llm_tot_pipeline.py --mode compare --methods cot,tot,sc,sr,fc,hybrid --subset 100 --test_mode

"""

import os
import sys
import argparse
import json
import hashlib
import time
import random
import re
import itertools
from datetime import datetime
import pickle
import scipy.stats as stats
from collections import Counter
import networkx as nx
import matplotlib.colors as mcolors
from pathlib import Path
import logging # Added logging
import traceback # Added for detailed error logging

import openai
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# For plots
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
# region Configuration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Derive base directory relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

API_KEY_PATH = BASE_DIR / ".API" / "openai_api_key.txt"  # WARNING: Consider env variables for keys
RESULTS_DIR = BASE_DIR / "results" / "LLM_inference"
COMPARISON_DIR = RESULTS_DIR / "model_comparisons"
REASONING_VIZ_DIR = RESULTS_DIR / "reasoning_visualizations"
TOKEN_USAGE_DIR = RESULTS_DIR / "token_usage"
# Create directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
REASONING_VIZ_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_USAGE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DATA_CSV = BASE_DIR / "results" / "final_covid19_labeled.csv"

# Paths related to semantic retrieval
SEM_RETRIEVAL_SCRIPT = BASE_DIR / "scripts" / "semantic_retrieval.py"
SEM_RETRIEVAL_CACHE_DIR = BASE_DIR / "results" / "semantic_retrieval" # For embeddings, etc.

# Constants (Consider moving to constants.py)
SEVERITY_LABELS = ["Mild", "Moderate", "Severe", "Critical"]

CLINICAL_CATEGORIES = {
    "respiratory": ["oxygen", "saturation", "breathing", "respiratory", "ventilation", "intubation", "SpO2"],
    "cardiovascular": ["heart", "cardiac", "blood pressure", "pulse", "cardiovascular"],
    "comorbidities": ["diabetes", "hypertension", "obesity", "COPD", "asthma", "comorbidity", "comorbidities"],
    "demographics": ["age", "gender", "elderly", "young", "male", "female"],
    "labs": ["d-dimer", "CRP", "ferritin", "troponin", "lab", "marker", "biomarker"]
}

METHOD_NAMES = {
    "cot": "Chain-of-Thought",
    "tot": "Tree-of-Thought",
    "sc": "Self-Consistency",
    "sr": "Self-Reflection",
    "fc": "Fact-Checking",
    "hybrid": "Hybrid Approach",
    "baseline": "Chain-of-Thought (Legacy)",
    "advanced": "Tree-of-Thought (Legacy)"
}

# endregion Configuration

# -------------------------------------------------------------------------
# 1. Utilities (API Key Loading)
# -------------------------------------------------------------------------
# region Utilities

def load_openai_key(key_path: Path) -> str:
    """Loads the OpenAI API key from the specified file path.

    Args:
        key_path (Path): Path object pointing to the API key file.

    Returns:
        str: The loaded API key.

    Raises:
        SystemExit: If the key file is not found.
    """
    if not key_path.is_file():
        # Consider fallback to environment variable OPENAI_API_KEY here
        logging.error(f"OpenAI API key file not found: {key_path}")
        sys.exit(1)
    try:
        with open(key_path, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        if not api_key:
            logging.error(f"API key file is empty: {key_path}")
            sys.exit(1)
        # Basic check if key looks plausible
        if not api_key.startswith("sk-"):
            logging.warning(f"API key read from {key_path} doesn't look like a valid OpenAI key.")
        return api_key
    except Exception as e:
        logging.error(f"Failed to read API key from {key_path}: {e}")
        sys.exit(1)

def init_openai_api() -> OpenAI:
    """Initializes and returns the OpenAI client using the loaded API key.

    Returns:
        OpenAI: An initialized OpenAI client instance.
    """
    openai_api_key = load_openai_key(API_KEY_PATH)
    client = OpenAI(api_key=openai_api_key)
    return client

# endregion Utilities

# -------------------------------------------------------------------------
# 2. Prompt Construction Functions
# -------------------------------------------------------------------------
# region Prompt Construction

def build_chain_of_thought_prompt(row_dict: dict, retrieved_examples: list = None) -> str:
    """
    Basic Chain-of-Thought (CoT) prompt with linear reasoning steps.
    May include a few examples if retrieved_examples is provided.
    """
    # Extract relevant data from row
    age = row_dict.get("AGE", "unknown")
    gender = row_dict.get("GENDER", "unknown")
    htn = "yes" if row_dict.get("HAS_HYPERTENSION", 0) == 1 else "no"
    diab = "yes" if row_dict.get("HAS_DIABETES", 0) == 1 else "no"
    inpatient = row_dict.get("ANY_INPATIENT", 0)
    vent = row_dict.get("USED_MECH_VENT", 0) or row_dict.get("VENT_PROCEDURE", 0)

    # We can create a short summary:
    summary = (
        f"Patient Info: Age={age}, Gender={gender}, "
        f"Hypertension={htn}, Diabetes={diab}, Inpatient={inpatient}, Vent={vent}.\n"
    )

    # If we have retrieved examples, include them in the prompt
    examples_text = ""
    if retrieved_examples:
        examples_text = "Here are a few similar cases with known severities:\n\n"
        for i, (row_d, score, summ) in enumerate(retrieved_examples, 1):
            examples_text += f"Example {i}: {summ}\n\n"
    else:
        # Otherwise, include a single default example
        examples_text = (
            "Example: For a 45-year-old female with no comorbidities and normal vitals, "
            "the final severity was 'Mild'.\n"
        )

    instructions = (
        "Task: Based on the patient info, think step-by-step (chain-of-thought) "
        "and determine whether the final severity is Mild, Moderate, Severe, or Critical.\n"
        "Focus on comorbidities, ventilation, hospital admission, and any risk factors.\n"
        "Follow these structured steps:\n"
        "1. Analyze patient demographics and baseline risk\n"
        "2. Evaluate comorbidities and their impact\n"
        "3. Assess hospital resource utilization\n"
        "4. Consider ventilation needs if any\n"
        "5. Conclude with the final severity assessment\n"
    )

    # Combine
    prompt = (
        "You are a medical assistant LLM specialized in COVID-19 severity assessment.\n\n"
        f"{examples_text}\n"
        "New Case:\n"
        f"{summary}\n"
        f"{instructions}\n"
        "Please provide your step-by-step reasoning followed by the final severity.\n"
    )
    return prompt

def build_tree_of_thought_prompt(row_dict: dict, retrieved_examples: list = None) -> str:
    """
    Tree-of-Thought (ToT) prompt with branching exploration.
    Requires retrieved_examples for few-shot learning.
    """
    # Create patient summary
    patient_summary = (
        f"New patient: Age={row_dict.get('AGE','unknown')}, "
        f"Gender={row_dict.get('GENDER','unknown')}, "
        f"Hypertension={row_dict.get('HAS_HYPERTENSION',0)}, "
        f"Diabetes={row_dict.get('HAS_DIABETES',0)}, "
        f"Inpatient={row_dict.get('ANY_INPATIENT',0)}, "
        f"Vent={row_dict.get('USED_MECH_VENT',0) or row_dict.get('VENT_PROCEDURE',0)}.\n"
    )

    # Initialize prompt
    lines = []
    lines.append("You are a highly sophisticated clinical LLM specialized in COVID-19 severity assessment.")
    
    # Include retrieved examples if available
    if retrieved_examples:
        lines.append("\nBelow are exemplars of patient profiles with final severities.")
        for i, (row_d, score, summ) in enumerate(retrieved_examples, 1):
            lines.append(f"Example {i} (Similarity={score:.2f}):")
            lines.append(summ)
            lines.append("")  # blank line

    # Tree-of-Thought instructions
    lines.append("Now, let's analyze the NEW patient's data using a Tree-of-Thought process.")
    lines.append("\n**NEW Patient**:")
    lines.append(patient_summary)
    lines.append("\nInstructions: Break down different branches based on key clinical factors. "
                 "Consider alternative interpretations of the data, then converge on the best severity label "
                 "(Mild, Moderate, Severe, or Critical).")
    lines.append("\nUse the following structured branches for your analysis:")
    lines.append("1. Branch 1: Respiratory Status Analysis")
    lines.append("   - Consider oxygen requirements, ventilation needs, and respiratory symptoms")
    lines.append("2. Branch 2: Comorbidity Risk Assessment")
    lines.append("   - Analyze how hypertension, diabetes, and other conditions affect prognosis")
    lines.append("3. Branch 3: Age and Demographic Considerations")
    lines.append("   - Evaluate how age and gender impact COVID-19 risk")
    lines.append("4. Branch 4: Hospital Resource Utilization")
    lines.append("   - Assess inpatient status and resource intensity")
    lines.append("5. Convergence: Final severity assessment")
    lines.append("   - Synthesize insights from all branches to determine final severity")
    lines.append("\nFor each branch, explore multiple possibilities before converging on the most likely scenario.")

    final_prompt = "\n".join(lines)
    return final_prompt

def build_self_consistency_prompt(row_dict: dict, retrieved_examples: list = None, attempt_number: int = 1) -> str:
    """
    Self-Consistency prompt for generating multiple reasoning paths.
    Modified CoT prompt with different seed instructions.
    """
    # Create base prompt from CoT
    base_prompt = build_chain_of_thought_prompt(row_dict, retrieved_examples)
    
    # Add self-consistency instructions
    sc_instructions = (
        f"\nThis is reasoning attempt #{attempt_number}. Please follow a unique reasoning path "
        f"that may consider different aspects of the patient case compared to other attempts. "
        f"Focus on providing your best independent assessment for this attempt.\n"
    )
    
    # Modify the seed slightly based on attempt number to get diverse reasoning
    focus_points = [
        "Particularly examine respiratory status and ventilation needs in your analysis.",
        "Give special attention to how comorbidities and age interact as risk factors.",
        "Consider hospital resource utilization as a key severity indicator.",
        "Pay close attention to demographic risk factors in your assessment.",
        "Focus on potential clinical complications given the patient profile."
    ]
    
    # Select a focus point based on attempt number
    if attempt_number <= len(focus_points):
        sc_instructions += focus_points[attempt_number - 1]
    
    return base_prompt + sc_instructions

def build_self_reflection_prompt(initial_prediction: str, patient_data: dict) -> str:
    """
    Creates a prompt for self-reflection on the initial prediction.
    
    Based on: Liu, Sinuo, et al. "New Trends for Modern Machine Translation with Large Reasoning Models."
    arXiv Preprint, arXiv:2503.10351v2 [cs.CL], 14 Mar. 2025.
    """
    # Extract patient details for reflection context
    age = patient_data.get("AGE", "unknown")
    gender = patient_data.get("GENDER", "unknown")
    htn = "yes" if patient_data.get("HAS_HYPERTENSION", 0) == 1 else "no"
    diab = "yes" if patient_data.get("HAS_DIABETES", 0) == 1 else "no"
    inpatient = patient_data.get("ANY_INPATIENT", 0)
    vent = patient_data.get("USED_MECH_VENT", 0) or patient_data.get("VENT_PROCEDURE", 0)
    
    # Create the self-reflection prompt
    reflection_prompt = (
        "You are a thoughtful clinical assistant reflecting on your own reasoning. "
        "Review your initial COVID-19 severity assessment below and critically evaluate your conclusion.\n\n"
        f"Patient: Age={age}, Gender={gender}, Hypertension={htn}, Diabetes={diab}, "
        f"Inpatient={inpatient}, Ventilation={vent}\n\n"
        f"Your initial assessment:\n{initial_prediction}\n\n"
        "Please perform a critical self-reflection by considering:\n"
        "1. What alternative hypotheses might I have overlooked in my reasoning?\n"
        "2. Are there any logical gaps or assumptions in my analysis?\n"
        "3. Did I give appropriate weight to each risk factor based on clinical evidence?\n"
        "4. Would another clinician potentially come to a different conclusion? Why?\n\n"
        "After this reflection, either confirm your initial assessment or provide a revised severity prediction. "
        "Clearly explain any changes in your reasoning and final determination.\n"
    )
    
    return reflection_prompt

def build_fact_checking_prompt(initial_prediction: str, patient_data: dict) -> str:
    """
    Creates a prompt for medical fact-checking of the initial prediction.
    
    Based on: Kolbinger, Fiona R., et al. "The Future Landscape of Large Language Models in."
    Communications Medicine, vol. 3, 2023, p. 141.
    """
    # Extract patient details for fact-checking context
    age = patient_data.get("AGE", "unknown")
    gender = patient_data.get("GENDER", "unknown")
    htn = "yes" if patient_data.get("HAS_HYPERTENSION", 0) == 1 else "no"
    diab = "yes" if patient_data.get("HAS_DIABETES", 0) == 1 else "no"
    inpatient = patient_data.get("ANY_INPATIENT", 0)
    vent = patient_data.get("USED_MECH_VENT", 0) or patient_data.get("VENT_PROCEDURE", 0)
    
    # Create the fact-checking prompt
    fact_check_prompt = (
        "You are a specialized medical fact-checker reviewing COVID-19 severity predictions. "
        "You need to verify if the following prediction aligns with established clinical guidelines.\n\n"
        f"Patient: Age={age}, Gender={gender}, Hypertension={htn}, Diabetes={diab}, "
        f"Inpatient={inpatient}, Ventilation={vent}\n\n"
        f"Initial Prediction:\n{initial_prediction}\n\n"
        "Please check this prediction against established COVID-19 clinical guidelines by answering these questions:\n"
        "1. Does the prediction appropriately consider known risk factors (age, hypertension, diabetes)?\n"
        "2. Is the severity assessment consistent with typical clinical outcomes for similar patients?\n"
        "3. Are there any factual errors or inconsistencies in the reasoning?\n"
        "4. Does the assessment align with WHO or CDC guidelines for COVID-19 severity classification?\n\n"
        "After your verification, provide a final assessment: either confirm the prediction or suggest a correction "
        "with supporting clinical rationale. Be specific about which aspects are correct or need revision.\n"
    )
    
    return fact_check_prompt

def build_hybrid_prompt(row_dict: dict, retrieved_examples: list = None) -> str:
    """
    Hybrid approach combining multiple reasoning methods.
    Integrates ToT branching with self-reflection and fact-checking.
    """
    # Start with the tree-of-thought approach as the foundation
    base_prompt = build_tree_of_thought_prompt(row_dict, retrieved_examples)
    
    # Add hybrid instructions for integrated reasoning
    hybrid_instructions = (
        "\n\nYour analysis should integrate multiple reasoning approaches:\n"
        "1. First, explore multiple branches of reasoning as outlined above\n"
        "2. Then, apply self-consistency by generating multiple potential severity assessments\n"
        "3. Next, perform self-reflection to critically examine your reasoning process\n"
        "4. Finally, verify your conclusion against medical guidelines\n\n"
        "Your response should be structured as follows:\n"
        "- Tree-of-Thought Analysis (with multiple branches)\n"
        "- Alternative Assessments (at least 2 potential severity classifications)\n"
        "- Critical Self-Reflection (questioning assumptions and biases)\n"
        "- Medical Guideline Verification\n"
        "- Final Severity Assessment (with confidence level)\n"
    )
    
    return base_prompt + hybrid_instructions

# endregion Prompt Construction

# -------------------------------------------------------------------------
# 3. GPT-4 Inference with Token Usage Tracking
# -------------------------------------------------------------------------
# region GPT Inference Logic

def get_cached_response(prompt: str, cache_dir: str) -> tuple[str | None, dict | None]:
    """Try to get a cached response for a given prompt."""
    # Create a unique hash for the prompt
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{prompt_hash}.json" # Use pathlib

    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                return cached_data.get("response"), cached_data.get("token_usage")
        except Exception as e:
            logging.warning(f"Failed to load cache file {cache_file}: {e}")
    return None, None

def save_cached_response(prompt: str, response: str, cache_dir: str, token_usage: dict | None = None):
    """Save a response to cache for future use."""
    # Create a unique hash for the prompt
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{prompt_hash}.json" # Use pathlib

    data = {
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }

    if token_usage:
        data["token_usage"] = token_usage

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True) # Ensure cache dir exists
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save cache file {cache_file}: {e}")

def generate_simulated_response(prompt: str, severity_labels: list) -> tuple[str, dict]:
    """Generate a fake GPT response for testing without API calls."""
    # Create a unique hash for this prompt to ensure consistent "random" results
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    random.seed(int(prompt_hash, 16) % 2**32)
    
    # Extract severity hints from the prompt
    prompt_lower = prompt.lower()
    severity_hints = {
        "Critical": prompt_lower.count("ventilat") + prompt_lower.count("death") + prompt_lower.count("icu") * 2,
        "Severe": prompt_lower.count("hospital") + prompt_lower.count("complication") + prompt_lower.count("icu"),
        "Moderate": prompt_lower.count("symptom") + prompt_lower.count("cough") + prompt_lower.count("fever"),
        "Mild": prompt_lower.count("mild") + prompt_lower.count("home") + prompt_lower.count("recovery") 
    }
    
    # Weight the random choice based on hints in the prompt
    weights = [severity_hints.get(label, 1) for label in severity_labels]
    total = sum(weights)
    if total > 0:
        weights = [w/total for w in weights]
        severity = random.choices(severity_labels, weights=weights, k=1)[0]
    else:
        severity = random.choice(severity_labels)
    
    # Generate a plausible reasoning text according to prompt type
    if "tree-of-thought" in prompt_lower or "tree of thought" in prompt_lower:
        # Generate ToT-style reasoning
        reasoning_lines = [
            "# Tree-of-Thought Analysis\n",
            "## Branch 1: Respiratory Status Analysis\n",
            "- Patient's oxygen saturation is " + ("significantly compromised" if "vent=1" in prompt_lower else "within acceptable limits"),
            "- Respiratory rate appears to be " + ("elevated, suggesting respiratory distress" if "vent=1" in prompt_lower else "normal"),
            "- " + ("Need for mechanical ventilation indicates severe respiratory compromise" if "vent=1" in prompt_lower else "No ventilatory support required, suggesting adequate respiratory function"),
            "\n## Branch 2: Comorbidity Risk Assessment\n",
            "- " + ("Hypertension present, which increases risk of severe COVID-19" if "hypertension=yes" in prompt_lower else "No hypertension noted"),
            "- " + ("Diabetes noted, which significantly elevates complications risk" if "diabetes=yes" in prompt_lower else "No diabetes noted"),
            "- Overall comorbidity burden is " + ("high" if "hypertension=yes" in prompt_lower or "diabetes=yes" in prompt_lower else "low"),
            "\n## Branch 3: Age and Demographic Considerations\n",
            "- Patient appears to be " + ("elderly" if "age=7" in prompt_lower or "age=8" in prompt_lower else "middle-aged" if "age=5" in prompt_lower or "age=6" in prompt_lower else "relatively young"),
            "- Gender factors suggest " + ("slightly higher risk" if "male" in prompt_lower else "typical risk profile"),
            "- Age-related risk is " + ("significant" if "age=7" in prompt_lower or "age=8" in prompt_lower else "moderate" if "age=5" in prompt_lower or "age=6" in prompt_lower else "lower"),
            "\n## Branch 4: Hospital Resource Utilization\n",
            "- " + ("Patient required inpatient care, suggesting moderate to severe illness" if "inpatient=1" in prompt_lower else "Patient managed as outpatient"),
            "- Resource intensity appears " + ("high with mechanical ventilation" if "vent=1" in prompt_lower else "moderate with hospitalization" if "inpatient=1" in prompt_lower else "low with outpatient management"),
            "\n## Convergence of Branches\n",
            f"Considering all factors across different reasoning branches, I assess this case as **{severity}**."
        ]
    elif "self-consistency" in prompt_lower or "attempt #" in prompt_lower:
        # Generate self-consistency response with focused content based on attempt number
        if "attempt #1" in prompt_lower:
            reasoning_lines = [
                "# Self-Consistency Reasoning (Attempt #1) - Respiratory Focus\n",
                "Step 1: Analyzing respiratory status",
                "- " + ("Patient requires mechanical ventilation, indicating severe respiratory compromise" if "vent=1" in prompt_lower else "No ventilatory support needed, indicating adequate respiratory function"),
                "- This alone suggests " + ("severe to critical COVID-19" if "vent=1" in prompt_lower else "mild to moderate COVID-19"),
                "\nStep 2: Considering comorbidities",
                "- " + ("Hypertension and/or diabetes present, increasing risk" if "yes" in prompt_lower else "No significant comorbidities noted"),
                "\nStep 3: Evaluating hospital resource needs",
                "- " + ("Inpatient care required" if "inpatient=1" in prompt_lower else "Outpatient management sufficient"),
                "\nStep 4: Age consideration",
                "- Patient is " + ("elderly" if "age=7" in prompt_lower or "age=8" in prompt_lower else "middle-aged" if "age=5" in prompt_lower or "age=6" in prompt_lower else "younger"),
                f"\nBased on my respiratory-focused analysis, I assess this as a {severity} case of COVID-19."
            ]
        elif "attempt #2" in prompt_lower:
            reasoning_lines = [
                "# Self-Consistency Reasoning (Attempt #2) - Comorbidity Focus\n",
                "Step 1: Analyzing comorbidity burden",
                "- " + ("Hypertension present: " + ("Yes" if "hypertension=yes" in prompt_lower else "No")),
                "- " + ("Diabetes present: " + ("Yes" if "diabetes=yes" in prompt_lower else "No")),
                "- Comorbidity risk is " + ("significantly elevated" if "diabetes=yes" in prompt_lower and "hypertension=yes" in prompt_lower else "moderately elevated" if "diabetes=yes" in prompt_lower or "hypertension=yes" in prompt_lower else "low"),
                "\nStep 2: Age-comorbidity interaction",
                "- " + ("Elderly patients with comorbidities face highest risk" if ("age=7" in prompt_lower or "age=8" in prompt_lower) and ("diabetes=yes" in prompt_lower or "hypertension=yes" in prompt_lower) else "No concerning age-comorbidity interaction"),
                "\nStep 3: Clinical interventions needed",
                "- " + ("Ventilation required" if "vent=1" in prompt_lower else "No ventilation needed"),
                "- " + ("Hospitalization required" if "inpatient=1" in prompt_lower else "Outpatient management sufficient"),
                f"\nBased on my comorbidity-focused analysis, I assess this as a {severity} case of COVID-19."
            ]
        else:
            reasoning_lines = [
                f"# Self-Consistency Reasoning (Attempt #3) - Resource Utilization Focus\n",
                "Step 1: Analyzing hospital resource needs",
                "- " + ("Inpatient care required" if "inpatient=1" in prompt_lower else "Outpatient management only"),
                "- " + ("ICU-level care with ventilation needed" if "vent=1" in prompt_lower else "Standard ward care sufficient" if "inpatient=1" in prompt_lower else "Home care sufficient"),
                "\nStep 2: Resource intensity correlation with severity",
                "- " + ("Ventilation strongly correlates with severe/critical COVID-19" if "vent=1" in prompt_lower else "Hospitalization correlates with moderate COVID-19" if "inpatient=1" in prompt_lower else "Outpatient management correlates with mild COVID-19"),
                "\nStep 3: Considering patient factors",
                "- Age and comorbidities support the resource-based assessment",
                f"\nBased on my resource utilization-focused analysis, I assess this as a {severity} case of COVID-19."
            ]
    elif "fact-checker" in prompt_lower:
        # Generate fact-checking response
        reasoning_lines = [
            "# Medical Fact-Checking Assessment\n",
            "## Evaluation of Risk Factor Consideration",
            "- The prediction " + ("appropriately" if random.random() > 0.3 else "inadequately") + " considers the patient's age and comorbidities",
            "- " + ("Hypertension and diabetes are correctly weighted as significant risk factors" if "hypertension=yes" in prompt_lower or "diabetes=yes" in prompt_lower else "The absence of major comorbidities is properly noted"),
            "\n## Consistency with Clinical Outcomes",
            "- The severity assessment is " + ("consistent" if random.random() > 0.3 else "somewhat inconsistent") + " with typical outcomes for similar patients",
            "- " + ("The need for mechanical ventilation strongly indicates severe/critical status" if "vent=1" in prompt_lower else "Outpatient management is typically associated with mild cases"),
            "\n## Factual Accuracy Review",
            "- No significant factual errors were identified in the reasoning",
            "- The causal relationships between risk factors and outcomes are medically sound",
            "\n## Guideline Alignment",
            "- The assessment " + ("aligns with" if random.random() > 0.3 else "partially aligns with") + " WHO/CDC guidelines for COVID-19 severity",
            "- " + ("Ventilation requirement is correctly associated with severe/critical classification" if "vent=1" in prompt_lower else "Hospitalization without ventilation typically indicates moderate severity"),
            "\n## Final Assessment",
            f"I confirm that the prediction of **{severity}** severity is medically appropriate based on established guidelines and the provided patient information."
        ]
    elif "self-reflection" in prompt_lower:
        # Generate self-reflection response
        reasoning_lines = [
            "# Self-Reflection on COVID-19 Severity Assessment\n",
            "## Alternative Hypotheses",
            "- I may have overlooked " + ("the possibility that hypertension alone without other comorbidities carries moderate risk" if "hypertension=yes" in prompt_lower else "that younger patients can sometimes develop severe symptoms despite low risk profiles"),
            "- An alternative interpretation could be that " + ("ventilation was preventative rather than indicative of severity" if "vent=1" in prompt_lower else "outpatient management doesn't always indicate mild disease"),
            "\n## Logical Gaps",
            "- I made an assumption about " + ("the duration of symptoms prior to hospitalization" if "inpatient=1" in prompt_lower else "the adequacy of home monitoring"),
            "- Additional lab values would strengthen my analysis",
            "\n## Risk Factor Weighting",
            "- I may have " + ("overemphasized" if random.random() > 0.5 else "underemphasized") + " the importance of age relative to comorbidities",
            "- Clinical evidence suggests " + ("diabetes carries particularly high risk" if "diabetes=yes" in prompt_lower else "absence of comorbidities significantly reduces risk"),
            "\n## Alternative Clinical Perspectives",
            "- Some clinicians might classify this case as " + (random.choice([l for l in severity_labels if l != severity])) + " based on different guideline interpretations",
            "- There's ongoing debate about the threshold between moderate and severe categories",
            "\n## Revised Assessment",
            f"After careful reflection, I " + ("stand by" if random.random() > 0.3 else "need to revise") + f" my initial assessment. The patient's COVID-19 severity is best classified as **{severity}**."
        ]
    elif "hybrid" in prompt_lower:
        # Generate hybrid approach reasoning
        reasoning_lines = [
            "# Hybrid Reasoning Approach\n",
            "\n## Tree-of-Thought Analysis",
            "\n### Branch 1: Respiratory Status",
            "- Patient's oxygen needs: " + ("Requires ventilation, indicating severe compromise" if "vent=1" in prompt_lower else "No ventilation needed, suggesting adequate function"),
            "- This branch points toward: " + ("Severe/Critical" if "vent=1" in prompt_lower else "Mild/Moderate"),
            
            "\n### Branch 2: Comorbidity Analysis",
            "- Hypertension: " + ("Present, increasing risk" if "hypertension=yes" in prompt_lower else "Absent"),
            "- Diabetes: " + ("Present, significantly elevating risk" if "diabetes=yes" in prompt_lower else "Absent"),
            "- This branch points toward: " + ("Moderate/Severe" if "hypertension=yes" in prompt_lower or "diabetes=yes" in prompt_lower else "Mild"),
            
            "\n### Branch 3: Demographics",
            "- Age risk: " + ("High (elderly)" if "age=7" in prompt_lower or "age=8" in prompt_lower else "Moderate (middle-aged)" if "age=5" in prompt_lower or "age=6" in prompt_lower else "Lower (younger)"),
            "- This branch points toward: " + ("Moderate/Severe" if "age=7" in prompt_lower or "age=8" in prompt_lower else "Mild/Moderate"),
            
            "\n### Branch 4: Resource Needs",
            "- Hospital status: " + ("Inpatient" if "inpatient=1" in prompt_lower else "Outpatient"),
            "- This branch points toward: " + ("Moderate/Severe" if "inpatient=1" in prompt_lower else "Mild"),
            
            "\n## Alternative Assessments",
            "- Assessment 1: " + severity,
            "- Assessment 2: " + (random.choice([l for l in severity_labels if l != severity])),
            "- Majority consensus: " + severity,
            
            "\n## Critical Self-Reflection",
            "- Potential oversight: Interaction between comorbidities may multiply risk",
            "- Possible bias: Over-reliance on ventilation status as severity indicator",
            
            "\n## Medical Guideline Verification",
            "- WHO guidelines: " + ("Ventilation indicates severe/critical disease" if "vent=1" in prompt_lower else "Hospitalization without ventilation typically indicates moderate disease"),
            "- CDC classification: Aligns with assessment",
            
            "\n## Final Severity Assessment",
            f"Based on my comprehensive multi-method analysis, I assess this case as **{severity}** with high confidence."
        ]
    else:
        # Generate CoT-style reasoning
        reasoning_lines = [
            "# Chain-of-Thought Analysis\n",
            "Step 1: Analyzing patient demographics and risk factors",
            "- Age: " + ("Elderly (higher risk)" if "age=7" in prompt_lower or "age=8" in prompt_lower else "Middle-aged (moderate risk)" if "age=5" in prompt_lower or "age=6" in prompt_lower else "Younger (lower risk)"),
            "- Gender: " + ("Male" if "male" in prompt_lower else "Female"),
            
            "\nStep 2: Evaluating comorbidities",
            "- " + ("Patient has significant comorbidities (hypertension and/or diabetes)" if "yes" in prompt_lower else "Patient has minimal comorbidities"),
            "- Comorbidity burden: " + ("High (multiple conditions)" if "hypertension=yes" in prompt_lower and "diabetes=yes" in prompt_lower else "Moderate (single condition)" if "hypertension=yes" in prompt_lower or "diabetes=yes" in prompt_lower else "Low (no major comorbidities)"),
            
            "\nStep 3: Analyzing hospital resource utilization",
            "- " + ("Inpatient care required, suggesting at least moderate severity" if "inpatient=1" in prompt_lower else "Outpatient management sufficient, suggesting milder disease"),
            
            "\nStep 4: Assessing respiratory support needs",
            "- " + ("Mechanical ventilation required, indicating severe respiratory compromise" if "vent=1" in prompt_lower else "No ventilatory support needed, suggesting adequate respiratory function"),
            
            "\nStep 5: Determining final severity classification",
            f"- Based on all factors considered in steps 1-4, I conclude this is a {severity} case of COVID-19."
        ]
    
    # Create simulated token usage data
    simulated_usage = {
        "prompt_tokens": int(len(prompt.split()) * 1.3),  # Ensure integer
        "completion_tokens": int(len("\n".join(reasoning_lines).split()) * 1.3),
        "total_tokens": int((len(prompt.split()) + len("\n".join(reasoning_lines).split())) * 1.3)
    }
    
    return "\n".join(reasoning_lines), simulated_usage

def gpt_inference(client: OpenAI, prompt: str, model="gpt-4", temperature=0.2,
                  test_mode=False, cache_dir=None, simulate=False) -> tuple[str, dict]:
    """
    Sends prompt to GPT-4, returns the response text and token usage.
    # ... (rest of docstring) ...
    """
    # Test mode with simulation - don't make API calls
    if test_mode and simulate:
        logging.info("[TEST] Using simulated GPT response")
        return generate_simulated_response(prompt, SEVERITY_LABELS)

    # Test mode with caching - try to use cached response first
    if test_mode and cache_dir:
        cached_response, cached_usage = get_cached_response(prompt, cache_dir)
        if cached_response:
            logging.info("[TEST] Using cached GPT response")
            return cached_response, cached_usage
    
    # If we need to make an actual API call
    try:
        # Use a faster model in test mode
        if test_mode and model == "gpt-4":
            model = "gpt-3.5-turbo"
            logging.info(f"[TEST] Using {model} instead of gpt-4 for test mode")
        
        # Add a short delay to avoid rate limits in test mode
        if test_mode:
            time.sleep(0.5)
        
        # Make the API call
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        elapsed = time.time() - start_time
        output_text = response.choices[0].message.content
        
        # Extract token usage
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Log API call details in test mode
        if test_mode:
            logging.info(f"[TEST] API call took {elapsed:.2f}s for model {model}, used {token_usage['total_tokens']} tokens")
        
        # Cache the response in test mode
        if test_mode and cache_dir:
            save_cached_response(prompt, output_text, cache_dir, token_usage)
        
        return output_text, token_usage
    except Exception as e:
        logging.error(f"[ERROR] GPT inference failed: {e}")
        if test_mode:
            logging.info("[TEST] Returning simulated fallback response due to API error")
            return generate_simulated_response(prompt, SEVERITY_LABELS)
        return "GPT_ERROR", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def parse_severity_from_response(response_text: str):
    """
    A simple parser to find a final severity label 
    in GPT's response. 
    We look for keywords like 'Mild', 'Moderate', 'Severe', 'Critical'.
    The approach can be as naive or robust as needed.
    """
    # Convert to lower for easy search
    resp_lower = response_text.lower()

    # This approach tries to find the mention of severity in a typical format
    for label in SEVERITY_LABELS:
        if label.lower() in resp_lower:
            return label
    # If none found, return unknown
    return "Unknown"

# -------------------------------------------------------------------------
# 4. Self-Consistency Voting Implementation
# -------------------------------------------------------------------------
def run_self_consistency(client: OpenAI, row_dict: dict, retrieved_examples: list = None, 
                         num_samples=3, test_mode=False, cache_dir=None, simulate=False):
    """
    Implements self-consistency voting by generating multiple reasoning paths
    and aggregating predictions through majority voting.
    
    Based on: Chen, Banghao, et al. "Unleashing the Potential of Prompt Engineering in Large Language
    Models: A Comprehensive Review." arXiv preprint arXiv:2310.14735, 2024.
    """
    # Generate multiple reasoning paths
    responses = []
    predictions = []
    token_usage_list = []
    
    for i in range(num_samples):
        # Generate different reasoning paths using self-consistency prompts
        prompt = build_self_consistency_prompt(row_dict, retrieved_examples, i+1)
        
        # Get the response
        response, token_usage = gpt_inference(
            client, prompt, model="gpt-4", temperature=0.3+i*0.1,  # Slightly increase temperature
            test_mode=test_mode, cache_dir=cache_dir, simulate=simulate
        )
        
        # Parse the severity prediction
        prediction = parse_severity_from_response(response)
        
        responses.append(response)
        predictions.append(prediction)
        token_usage_list.append(token_usage)
    
    # Get the majority vote
    if predictions:
        # Count occurrences of each prediction
        prediction_counts = Counter(predictions)
        
        # Get the most common prediction
        majority_prediction = prediction_counts.most_common(1)[0][0]
        
        # Find the index of a response with the majority prediction
        majority_index = predictions.index(majority_prediction)
        majority_response = responses[majority_index]
        
        # Calculate total token usage across all attempts
        total_token_usage = {
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in token_usage_list if u),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in token_usage_list if u),
            "total_tokens": sum(u.get("total_tokens", 0) for u in token_usage_list if u)
        }
        
        # Return the consensus prediction and reasoning
        consensus_result = {
            "final_prediction": majority_prediction,
            "raw_response": majority_response,
            "all_responses": responses,
            "all_predictions": predictions,
            "prediction_counts": dict(prediction_counts),
            "consistency": max(prediction_counts.values()) / len(predictions) if predictions else 0,
            "token_usage": total_token_usage
        }
        
        return consensus_result
    else:
        return {
            "final_prediction": "Unknown",
            "raw_response": "No valid responses generated",
            "all_responses": [],
            "all_predictions": [],
            "prediction_counts": {},
            "consistency": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

# -------------------------------------------------------------------------
# 5. Reasoning Path Visualization Functions
# -------------------------------------------------------------------------
# region Reasoning Visualization

def parse_tot_branches(response_text: str) -> dict[str, str]:
    """Parse Tree-of-Thought branches from GPT response text.

    Args:
        response_text (str): The raw text output from the LLM.

    Returns:
        dict[str, str]: Dictionary mapping branch titles to their content.
    """
    # Use regex to find branch headers and content
    branch_pattern = r"(?:#{1,2}\s+)?Branch\s+\d+:?\s+([^\n]+)(?:\n|\r\n?)((?:(?!(?:#{1,2}\s+)?Branch\s+\d+:|#{1,2}\s+Convergence).+(?:\n|\r\n?))+)"
    branches = {}

    matches = re.finditer(branch_pattern, response_text, re.IGNORECASE | re.MULTILINE)

    for match in matches:
        branch_title = match.group(1).strip()
        branch_content = match.group(2).strip()
        branches[branch_title] = branch_content

    # Try to find convergence section
    convergence_pattern = r"(?:#{1,2}\s+)?Convergence:?\s+([^\n]+)(?:\n|\r\n?)((?:.+(?:\n|\r\n?))+)"
    convergence_match = re.search(convergence_pattern, response_text, re.IGNORECASE | re.MULTILINE)

    if convergence_match:
        # Ensure convergence content is captured correctly
        convergence_content = convergence_match.group(2).strip()
        branches["Convergence"] = convergence_content

    return branches

def categorize_branch_content(branch_text: str) -> dict[str, int]:
    """Categorize branch content into clinical categories based on keywords.

    Args:
        branch_text (str): The text content of a reasoning branch.

    Returns:
        dict[str, int]: Dictionary mapping clinical category names to keyword counts.
    """
    category_counts = {category: 0 for category in CLINICAL_CATEGORIES}

    for category, keywords in CLINICAL_CATEGORIES.items():
        for keyword in keywords:
            # Use word boundaries for more precise matching (optional)
            # pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            # category_counts[category] += len(re.findall(pattern, branch_text.lower()))
            category_counts[category] += branch_text.lower().count(keyword.lower())

    return category_counts

def create_reasoning_visualization(response_text: str, patient_data: dict, filename: str):
    """Create a visual representation of the reasoning paths in the response."""
    # Parse branches from the response
    branches = parse_tot_branches(response_text)

    if not branches:
        logging.warning("No branches found in response text for visualization")
        return

    # Create a directed graph
    G = nx.DiGraph()
    
    # Patient node
    patient_label = (f"Patient\nAge: {patient_data.get('AGE', 'unknown')}, "
                     f"Gender: {patient_data.get('GENDER', 'unknown')}")
    G.add_node("Patient", label=patient_label, shape="box")
    
    # Add branch nodes
    branch_categories = {}
    for branch_name, branch_content in branches.items():
        G.add_node(branch_name, label=branch_name, shape="ellipse")
        G.add_edge("Patient", branch_name)
        
        # Categorize branch content
        if branch_name != "Convergence":
            branch_categories[branch_name] = categorize_branch_content(branch_content)
    
    # Add convergence node if it exists
    if "Convergence" in branches:
        # Try to extract the severity prediction
        severity = "Unknown"
        for label in SEVERITY_LABELS:
            if label.lower() in branches["Convergence"].lower():
                severity = label
                break
        
        convergence_label = f"Conclusion: {severity}"
        G.add_node("Conclusion", label=convergence_label, shape="diamond")
        
        # Connect all branches to the conclusion
        for branch_name in branches:
            if branch_name != "Convergence":
                G.add_edge(branch_name, "Conclusion")
    
    # Set up the visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Define colors for different node types
    node_colors = []
    for node in G.nodes():
        if node == "Patient":
            node_colors.append("lightblue")
        elif node == "Conclusion":
            node_colors.append("lightgreen")
        else:
            # Create a color based on the dominant category
            if node in branch_categories:
                categories = branch_categories[node]
                dominant_category = max(categories.items(), key=lambda x: x[1])[0]
                
                # Assign colors based on clinical categories
                category_color_map = {
                    "respiratory": "lightcoral",
                    "cardiovascular": "salmon",
                    "comorbidities": "goldenrod",
                    "demographics": "mediumaquamarine",
                    "labs": "mediumpurple"
                }
                
                node_colors.append(category_color_map.get(dominant_category, "lightgray"))
            else:
                node_colors.append("lightgray")
    
    # Define edge colors based on the source node
    edge_colors = []
    for u, v in G.edges():
        if u == "Patient":
            edge_colors.append("gray")
        else:
            if u in branch_categories:
                categories = branch_categories[u]
                dominant_category = max(categories.items(), key=lambda x: x[1])[0]
                
                # Assign colors based on clinical categories
                category_color_map = {
                    "respiratory": "lightcoral",
                    "cardiovascular": "salmon",
                    "comorbidities": "goldenrod",
                    "demographics": "mediumaquamarine",
                    "labs": "mediumpurple"
                }
                
                edge_colors.append(category_color_map.get(dominant_category, "lightgray"))
            else:
                edge_colors.append("lightgray")
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels={node: data["label"] for node, data in G.nodes(data=True)},
            node_color=node_colors, edge_color=edge_colors, width=2.0, node_size=3000,
            font_size=10, font_weight='bold', arrowsize=15)
    
    # Add a title
    severity = "Unknown"
    for label in SEVERITY_LABELS:
        if label.lower() in response_text.lower():
            severity = label
            break
    
    plt.title(f"COVID-19 Severity Reasoning Path: {severity}")
    
    # Add a legend for clinical categories
    category_color_map = {
        "Respiratory": "lightcoral",
        "Cardiovascular": "salmon",
        "Comorbidities": "goldenrod",
        "Demographics": "mediumaquamarine",
        "Laboratory": "mediumpurple",
        "Patient": "lightblue",
        "Conclusion": "lightgreen"
    }
    
    legend_patches = [mpatches.Patch(color=color, label=category) 
                     for category, color in category_color_map.items()]
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=len(category_color_map))
    
    # Save the figure using pathlib
    save_path = REASONING_VIZ_DIR / filename
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Reasoning visualization saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save reasoning visualization to {save_path}: {e}")
    finally:
        plt.close()

def generate_sample_reasoning_visualizations(client: OpenAI, df: pd.DataFrame,
                                             test_mode=False, cache_dir=None, simulate_gpt=False,
                                             no_balance_retrieval=False):
    """Generate sample visualizations of reasoning paths for Tree-of-Thought."""
    if df is None or df.empty:
        logging.warning("DataFrame empty, cannot generate sample visualizations.")
        return

    logging.info("Generating sample reasoning visualizations for ToT...")

    # Sample 3 patients for visualization (or fewer if df is small)
    n_samples = 3
    if len(df) < n_samples:
        logging.warning(f"[WARNING] Only {len(df)} patients available, generating fewer visualizations.")
        n_samples = len(df)
    if n_samples == 0:
        return
    
    try:
        # Try to select diverse patients (stratified sample)
        if 'COVID19_SEVERITY' in df.columns:
            from sklearn.model_selection import train_test_split
            try:
                _, sample_indices = train_test_split(
                    df.index, test_size=n_samples, 
                    stratify=df['COVID19_SEVERITY'],
                    random_state=42
                )
                sample_df = df.loc[sample_indices]
            except ValueError:
                # This might happen if some class has only 1 member
                logging.warning("[WARNING] Stratified sampling for visualization failed, using random sample.")
                sample_df = df.sample(n=n_samples, random_state=42)
        else:
            sample_df = df.sample(n=n_samples, random_state=42)
    except Exception as e:
        logging.warning(f"[WARNING] Error during sampling for visualization: {e}")
        sample_df = df.iloc[:min(n_samples, len(df))]
    
    # Load semantic retrieval data
    scripts_path = str(BASE_DIR / "scripts")
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)
    try:
        import semantic_retrieval
        logging.info("Loading embeddings for visualization examples...")
        # Use pathlib path, convert to string for the function call
        sr_df, sr_embeds = semantic_retrieval.load_and_embed_patients(
            str(FINAL_DATA_CSV),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        if sr_df is None or sr_embeds is None:
            raise ValueError("Failed to load embeddings")
    except Exception as e:
        logging.error(f"Failed to load semantic retrieval data for visualization: {e}")
        return
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        patient_dict = row.to_dict()
        patient_dict['PATIENT'] = row.get('PATIENT', f'Index_{idx}')
        
        # Retrieve examples
        query_text = semantic_retrieval.summarize_patient(row)
        retrieved_examples = semantic_retrieval.retrieve_top_k(
            query_text, sr_embeds, sr_df, k=3,
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            balance_severity=not no_balance_retrieval,
            query_patient_index=idx # Pass original index to prevent self-retrieval
        )
        
        # Generate ToT response
        prompt = build_tree_of_thought_prompt(patient_dict, retrieved_examples)
        tot_response, _ = gpt_inference(
            client, prompt, model="gpt-4", temperature=0.2,
            test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
        )
        
        if tot_response != "GPT_ERROR":
            # Create and save visualization
            severity = row.get("COVID19_SEVERITY", "Unknown")
            safe_id = str(patient_dict.get('PATIENT', f'Index_{idx}')).replace('/', '_').replace('\\', '_') # Added get()
            filename = f"sample_viz_{i+1}_{safe_id}_{severity}.png"
            create_reasoning_visualization(tot_response, patient_dict, filename)
        else:
            logging.warning(f"[WARNING] Skipping visualization for patient {i+1} due to GPT error.")
            
    logging.info(f"[INFO] Sample reasoning visualizations completed ({n_samples} generated).")

# endregion Reasoning Visualization

# -------------------------------------------------------------------------
# 6. Feature Importance Analysis
# -------------------------------------------------------------------------
# region Feature Importance

def analyze_feature_importance(client: OpenAI, baseline_data: dict, df: pd.DataFrame,
                               test_mode=False, cache_dir=None, simulate=False, num_samples=10):
    """Analyze feature importance by perturbing inputs and observing prediction changes."""
    logging.info("Starting feature importance analysis...")
    
    # Select a subset of patients for analysis
    if len(df) > num_samples:
        analysis_df = df.sample(n=num_samples, random_state=42)
    else:
        analysis_df = df
    
    # Features to analyze
    features = ["AGE", "GENDER", "HAS_HYPERTENSION", "HAS_DIABETES", "ANY_INPATIENT", "USED_MECH_VENT"]
    feature_importance = {feature: 0 for feature in features}
    feature_samples = {feature: 0 for feature in features}
    
    # For each patient, vary one feature at a time and measure prediction changes
    for i, row in analysis_df.iterrows():
        row_dict = row.to_dict()
        original_prompt = build_chain_of_thought_prompt(row_dict)
        
        # Get original prediction
        original_response, _ = gpt_inference(
            client, original_prompt, model="gpt-4", temperature=0.2,
            test_mode=test_mode, cache_dir=cache_dir, simulate=simulate
        )
        original_prediction = parse_severity_from_response(original_response)
        
        logging.info(f"[INFO] Analyzing feature importance for patient {i+1}/{len(analysis_df)}...")
        
        # Test each feature
        for feature in features:
            # Skip if feature value is unknown
            if feature not in row_dict or pd.isna(row_dict[feature]):
                continue
            
            # Create a modified row dict with the feature altered
            modified_row = row_dict.copy()
            
            # Modify the feature based on its type
            if feature == "AGE":
                # Increase/decrease age by 20 years
                current_age = modified_row.get("AGE", 50)
                modified_row["AGE"] = max(18, current_age + (20 if current_age < 50 else -20))
            elif feature == "GENDER":
                # Flip gender
                current_gender = modified_row.get("GENDER", "M")
                modified_row["GENDER"] = "F" if current_gender == "M" else "M"
            else:
                # Flip binary features
                current_value = modified_row.get(feature, 0)
                modified_row[feature] = 1 - current_value
            
            # Build modified prompt
            modified_prompt = build_chain_of_thought_prompt(modified_row)
            
            # Get modified prediction
            modified_response, _ = gpt_inference(
                client, modified_prompt, model="gpt-4", temperature=0.2,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate
            )
            modified_prediction = parse_severity_from_response(modified_response)
            
            # Calculate importance (prediction changed = 1, unchanged = 0)
            importance = 0 if original_prediction == modified_prediction else 1
            feature_importance[feature] += importance
            feature_samples[feature] += 1
    
    # Normalize importance scores
    for feature in features:
        if feature_samples[feature] > 0:
            feature_importance[feature] /= feature_samples[feature]
    
    # Create the visualization
    plt.figure(figsize=(10, 6))
    features_sorted = sorted(features, key=lambda x: feature_importance[x], reverse=True)
    importance_values = [feature_importance[feature] for feature in features_sorted]
    
    bars = plt.bar(features_sorted, importance_values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Feature Importance for COVID-19 Severity Prediction')
    plt.xlabel('Feature')
    plt.ylabel('Importance Score (0-1)')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure using pathlib
    out_filepath = RESULTS_DIR / "feature_importance_analysis.png"
    try:
        plt.savefig(out_filepath, dpi=300)
        logging.info(f"Feature importance analysis completed. Results saved to {out_filepath}")
    except Exception as e:
        logging.error(f"Failed to save feature importance plot to {out_filepath}: {e}")
    finally:
        plt.close()
    
    # Return the importance scores
    return feature_importance

# endregion Feature Importance

# -------------------------------------------------------------------------
# 7. Token Usage Analysis
# -------------------------------------------------------------------------
# region Token Analysis

def analyze_token_usage(token_usage_data: dict, method_names: dict, out_dir: Path = TOKEN_USAGE_DIR):
    """Analyze and visualize token usage across different reasoning methods."""
    logging.info("Analyzing token usage across methods...")
    
    # Extract token usage metrics by method
    methods = list(token_usage_data.keys())
    prompt_tokens = [token_usage_data[m].get("prompt_tokens", 0) for m in methods]
    completion_tokens = [token_usage_data[m].get("completion_tokens", 0) for m in methods]
    total_tokens = [token_usage_data[m].get("total_tokens", 0) for m in methods]
    
    # Calculate cost estimates (approximate rates as of 2025)
    # Note: These rates might need updating based on current OpenAI pricing
    prompt_cost = [tokens * 0.00003 for tokens in prompt_tokens]  # $0.03 per 1000 tokens
    completion_cost = [tokens * 0.00006 for tokens in completion_tokens]  # $0.06 per 1000 tokens
    total_cost = [p + c for p, c in zip(prompt_cost, completion_cost)]
    
    # Readable method names for display
    method_labels = [method_names.get(m, m) for m in methods]
    
    # Create visualization for token usage
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Token usage bars
    ax1.bar(x - width/2, prompt_tokens, width/2, label='Prompt Tokens', color='#1f77b4')
    ax1.bar(x - width/2 + width/2, completion_tokens, width/2, label='Completion Tokens', color='#ff7f0e')
    ax1.bar(x + width/2, total_tokens, width, label='Total Tokens', color='#2ca02c', alpha=0.6)
    
    # Add a second y-axis for cost
    ax2 = ax1.twinx()
    ax2.plot(x, total_cost, 'ro-', linewidth=2, label='Est. Cost ($)')
    
    # Labels and formatting
    ax1.set_xlabel('Reasoning Method')
    ax1.set_ylabel('Number of Tokens')
    ax2.set_ylabel('Estimated Cost ($)')
    ax1.set_title('Token Usage and Cost by Reasoning Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=45, ha='right')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save the figure using pathlib
    out_filepath = out_dir / "token_usage_comparison.png"
    try:
        plt.savefig(out_filepath, dpi=300)
        logging.info(f"Token usage comparison plot saved to {out_filepath}")
    except Exception as e:
        logging.error(f"Failed to save token usage plot to {out_filepath}: {e}")
    finally:
        plt.close()
    
    # Create token usage efficiency visualization (accuracy per 1000 tokens)
    if all(m in token_usage_data for m in methods) and all("accuracy" in token_usage_data[m] for m in methods):
        accuracies = [token_usage_data[m].get("accuracy", 0) for m in methods]
        
        # Calculate efficiency (accuracy per 1000 tokens)
        efficiency = [acc * 1000 / tokens if tokens > 0 else 0 for acc, tokens in zip(accuracies, total_tokens)]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(method_labels, efficiency, color='purple')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title('Efficiency: Accuracy per 1000 Tokens')
        plt.xlabel('Reasoning Method')
        plt.ylabel('Accuracy per 1000 Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure using pathlib
        efficiency_filepath = out_dir / "token_efficiency_comparison.png"
        try:
            plt.savefig(efficiency_filepath, dpi=300)
            logging.info(f"Token efficiency analysis saved to {efficiency_filepath}")
        except Exception as e:
            logging.error(f"Failed to save token efficiency plot to {efficiency_filepath}: {e}")
        finally:
            plt.close()
    else:
        logging.info("Skipping token efficiency plot: Accuracy data missing for some methods.")
    
    # Save token usage data as JSON using pathlib
    token_data_filepath = out_dir / "token_usage_data.json"
    token_data_output = {
        "methods": methods,
        "method_labels": method_labels,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost,
        "raw_data": {m: token_usage_data[m] for m in methods}
    }
    
    try:
        with open(token_data_filepath, 'w') as f:
            json.dump(token_data_output, f, indent=2)
        logging.info(f"Token usage data saved to {token_data_filepath}")
    except Exception as e:
        logging.error(f"Failed to save token usage JSON to {token_data_filepath}: {e}")
    
    return token_data_output

# endregion Token Analysis

# -------------------------------------------------------------------------
# 8. Core Pipeline Runner
# -------------------------------------------------------------------------
# region Pipeline Execution

def run_reasoning_pipeline(df: pd.DataFrame, client: OpenAI, method="cot", subset=None,
                           test_mode=False, cache_dir=None, simulate_gpt=False, stratify_subset=False,
                           balance_subset_mixed=False, no_balance_retrieval=False):
    """
    Run the pipeline with a specified reasoning method.
    Returns predictions and token usage data.
    
    Args:
        df (pd.DataFrame): Input patient data
        client (OpenAI): OpenAI client
        method (str): Reasoning method to use ('cot', 'tot', etc.)
        subset (int, optional): Number of patients to process
        test_mode (bool): Whether to run in test mode
        cache_dir (str, optional): Directory for caching responses
        simulate_gpt (bool): Whether to simulate GPT responses
        stratify_subset (bool): Whether to stratify sampling by severity
        balance_subset_mixed (bool): Whether to use capped mixed representation sampling
        no_balance_retrieval (bool): Whether to disable balanced retrieval
    
    Returns:
        tuple: (predictions list, token usage dict)
    """
    logging.info(f"Running pipeline with {METHOD_NAMES.get(method, method)} reasoning...")
    
    # Handle subsetting logic
    processed_df = df.copy()
    total_rows = len(df)
    
    if subset is not None and subset < total_rows:
        if balance_subset_mixed:
            # Capped Mixed Representation Sampling Logic
            logging.info(f"[INFO] Applying Capped Mixed Representation sampling for subset size {subset}...")
            if 'COVID19_SEVERITY' not in df.columns:
                logging.error("[ERROR] Cannot apply mixed balancing: 'COVID19_SEVERITY' column missing.")
                sys.exit(1)
                
            if subset < 4:  # Not enough for one from each class
                logging.warning(f"[WARNING] Subset size {subset} is less than the number of severity classes (4). Cannot guarantee representation. Falling back to random sampling.")
                processed_df = df.sample(n=subset, random_state=42).reset_index(drop=True)
            else:
                try:
                    # Ensure target variable has no NaNs
                    df_clean = df.dropna(subset=['COVID19_SEVERITY'])
                    
                    if len(df_clean) < subset:
                        logging.warning(f"[WARNING] Requested subset {subset} > available clean data {len(df_clean)}. Using {len(df_clean)}.")
                        subset = len(df_clean)
                        
                    if subset < 4:  # Double check after cleaning
                        logging.warning(f"[WARNING] Cleaned subset size {subset} < num classes. Falling back to random sampling.")
                        processed_df = df_clean.sample(n=subset, random_state=42).reset_index(drop=True)
                    else:
                        # --- Start Capped Mixed Sampling --- #
                        final_indices = []
                        remaining_indices = list(df_clean.index)
                        
                        # 1. Mandatory Inclusion (1 from each class)
                        mandatory_indices = []
                        logging.info("[INFO] Selecting mandatory samples (1 per class)...")
                        severity_labels = ["Mild", "Moderate", "Severe", "Critical"]
                        minority_classes = ["Severe", "Critical"]
                        
                        for severity in severity_labels:
                            class_indices = df_clean[df_clean['COVID19_SEVERITY'] == severity].index
                            if len(class_indices) > 0:
                                chosen_idx = random.choice(class_indices)
                                mandatory_indices.append(chosen_idx)
                                if chosen_idx in remaining_indices:
                                    remaining_indices.remove(chosen_idx)  # Remove from pool
                            else:
                                logging.warning(f"[WARNING] No samples found for mandatory inclusion of class '{severity}'.")
                                
                        final_indices.extend(mandatory_indices)
                        logging.info(f"[INFO] Mandatory samples selected: {len(final_indices)}")
                        
                        # 2. Calculate remaining needs and minority cap
                        remaining_slots = subset - len(final_indices)
                        max_minority_total = int(subset * 0.10)  # 10% cap (changed from 30%)
                        
                        # Count how many minority samples are already in mandatory set
                        current_minority_count = sum(1 for idx in final_indices 
                                                   if df_clean.loc[idx, 'COVID19_SEVERITY'] in minority_classes)
                        allowed_additional_minority = max(0, max_minority_total - current_minority_count)
                        
                        logging.info(f"[INFO] Remaining slots: {remaining_slots}. Max minority: {max_minority_total}. "
                                      f"Current minority: {current_minority_count}. Allowed additional minority: "
                                      f"{allowed_additional_minority}.")
                        
                        # 3. Weighted Sampling for Remaining Slots (with cap check)
                        if remaining_slots > 0 and remaining_indices:
                            logging.info(f"[INFO] Performing weighted sampling for remaining {remaining_slots} slots...")
                            
                            minority_weight = 10.0
                            majority_weight = 1.0
                            weights = []
                            candidate_indices = []
                            
                            for idx in remaining_indices:
                                severity = df_clean.loc[idx, 'COVID19_SEVERITY']
                                weight = minority_weight if severity in minority_classes else majority_weight
                                weights.append(weight)
                                candidate_indices.append(idx)
                            
                            if not candidate_indices:
                                logging.warning("[WARNING] No remaining candidates for weighted sampling.")
                            else:
                                # Normalize weights
                                total_weight = sum(weights)
                                probabilities = [w / total_weight for w in weights] if total_weight > 0 else None
                                
                                if probabilities:
                                    # Sample more than needed initially, then filter
                                    num_to_sample = min(len(candidate_indices), remaining_slots * 3)
                                    try:
                                        sampled_indices = np.random.choice(
                                            candidate_indices,
                                            size=num_to_sample,
                                            p=probabilities,
                                            replace=False
                                        )
                                        
                                        # Filter to meet cap and remaining slots
                                        additional_minority_selected = 0
                                        slots_filled = 0
                                        weighted_selection = []
                                        
                                        for idx in sampled_indices:
                                            if slots_filled >= remaining_slots:
                                                break
                                                
                                            is_minority = df_clean.loc[idx, 'COVID19_SEVERITY'] in minority_classes
                                            
                                            if is_minority:
                                                if additional_minority_selected < allowed_additional_minority:
                                                    weighted_selection.append(idx)
                                                    additional_minority_selected += 1
                                                    slots_filled += 1
                                                # else: Skip this minority sample as cap is reached
                                            else:
                                                # Always add majority samples if slots remain
                                                weighted_selection.append(idx)
                                                slots_filled += 1
                                        
                                        logging.info(f"[INFO] Selected {slots_filled} samples via weighted sampling. "
                                              f"({additional_minority_selected} additional minority)")
                                        final_indices.extend(weighted_selection)
                                        
                                        # If we still haven't filled all slots (e.g., minority cap was strict)
                                        # fill the rest randomly from remaining candidates
                                        final_needed = subset - len(final_indices)
                                        if final_needed > 0:
                                            logging.info(f"[INFO] Filling remaining {final_needed} slots randomly...")
                                            current_final_set = set(final_indices)
                                            fill_pool = [idx for idx in remaining_indices if idx not in current_final_set]
                                            
                                            if len(fill_pool) >= final_needed:
                                                final_indices.extend(random.sample(fill_pool, final_needed))
                                            else:  # Not enough unique candidates left
                                                final_indices.extend(fill_pool)
                                                logging.warning(f"[WARNING] Could only add {len(fill_pool)} more unique samples.")
                                    except Exception as e:
                                        logging.error(f"[ERROR] Error during weighted sampling: {e}")
                                        logging.info("[INFO] Falling back to random filling for remaining slots.")
                                        current_final_set = set(final_indices)
                                        fill_pool = [idx for idx in remaining_indices if idx not in current_final_set]
                                        
                                        slots_to_fill = min(remaining_slots, len(fill_pool))
                                        if slots_to_fill > 0:
                                            final_indices.extend(random.sample(fill_pool, slots_to_fill))
                                else:
                                    logging.warning("[WARNING] Could not calculate probabilities for weighted sampling.")
                                    logging.info("[INFO] Filling randomly instead.")
                                    slots_to_fill = min(remaining_slots, len(remaining_indices))
                                    if slots_to_fill > 0:
                                        final_indices.extend(random.sample(remaining_indices, slots_to_fill))
                        
                        # 4. Create and Shuffle Final DataFrame
                        processed_df = df_clean.loc[final_indices].sample(frac=1, random_state=42).reset_index(drop=True)
                        logging.info(f"[INFO] Capped Mixed Representation subset created with {len(processed_df)} rows.")
                        logging.info("Subset Severity Distribution:")
                        logging.info(processed_df['COVID19_SEVERITY'].value_counts(normalize=True))
                
                except Exception as e:
                    logging.error(f"[ERROR] Unexpected error during Capped Mixed Representation sampling: {e}")
                    logging.error(traceback.format_exc())
                    logging.info("[INFO] Falling back to random sampling.")
                    processed_df = df.sample(n=subset, random_state=42).reset_index(drop=True)
        
        elif stratify_subset:
            # Stratified sampling logic
            logging.info(f"[INFO] Applying stratified sampling for subset size {subset} based on COVID19_SEVERITY...")
            if 'COVID19_SEVERITY' not in df.columns:
                logging.error("[ERROR] Cannot stratify: 'COVID19_SEVERITY' column missing.")
                sys.exit(1)
                
            try:
                from sklearn.model_selection import train_test_split
                
                # Ensure target variable has no NaNs
                df_clean = df.dropna(subset=['COVID19_SEVERITY'])
                if len(df_clean) < total_rows:
                    logging.warning(f"[WARNING] Dropped {total_rows - len(df_clean)} rows with missing severity for stratification.")
                
                if len(df_clean) < subset:
                    logging.warning(f"[WARNING] Requested subset {subset} > available clean data {len(df_clean)}. Using {len(df_clean)}.")
                    subset = len(df_clean)
                
                if subset > 0:
                    _, sampled_indices = train_test_split(
                        df_clean.index,
                        test_size=subset,
                        stratify=df_clean['COVID19_SEVERITY'],
                        random_state=42
                    )
                    processed_df = df_clean.loc[sampled_indices].reset_index(drop=True)
                    logging.info(f"[INFO] Stratified subset created with {len(processed_df)} rows.")
                    logging.info("Subset Severity Distribution:")
                    logging.info(processed_df['COVID19_SEVERITY'].value_counts(normalize=True))
                else:
                    logging.warning("[WARNING] Subset size resulted in 0 after cleaning/checking. No data to process.")
                    processed_df = pd.DataFrame(columns=df.columns)
            
            except ValueError as ve:
                logging.error(f"[ERROR] Stratified sampling failed: {ve}")
                logging.info("This might happen if subset size is smaller than the number of classes or a class has only 1 member.")
                logging.info("[INFO] Falling back to random sampling.")
                processed_df = df.sample(n=subset, random_state=42).reset_index(drop=True)
                
            except Exception as e:
                logging.error(f"[ERROR] Unexpected error during stratified sampling: {e}")
                logging.error(traceback.format_exc())
                logging.info("[INFO] Falling back to random sampling.")
                processed_df = df.sample(n=subset, random_state=42).reset_index(drop=True)
        
        else:
            # Random sampling (default)
            logging.info(f"[INFO] Applying random sampling for subset size {subset}...")
            processed_df = df.sample(n=subset, random_state=42).reset_index(drop=True)
            
    elif test_mode and subset is None:
        # Test mode default subset handling
        test_size = min(30, len(df))
        logging.info(f"[TEST] Automatically limiting to {test_size} samples for test mode.")
        processed_df = df.sample(n=test_size, random_state=42).reset_index(drop=True)
    
    # Check for empty DataFrame
    if processed_df.empty:
        logging.warning("[WARNING] Processed DataFrame is empty after subsetting. No rows to process.")
        return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    logging.info(f"[INFO] Starting processing for {len(processed_df)} patients using method '{method}'.")
    
    # Load semantic retrieval if needed for few-shot examples
    sr_df, sr_embeds, semantic_retrieval = None, None, None
    needs_retrieval = method in ["tot", "sc", "sr", "fc", "hybrid", "advanced"]
    
    if needs_retrieval:
        # Adjust sys.path relative to BASE_DIR
        scripts_path = str(BASE_DIR / "scripts")
        if scripts_path not in sys.path:
            sys.path.append(scripts_path)
        try:
            import semantic_retrieval
            logging.info(f"[INFO] Loading embeddings for semantic retrieval...")
            sr_df, sr_embeds = semantic_retrieval.load_and_embed_patients(
                str(FINAL_DATA_CSV),
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            if sr_df is None or sr_embeds is None:
                raise ValueError("Failed to load embeddings")
        except Exception as e:
            logging.error(f"[ERROR] Failed to load semantic retrieval data: {e}")
            logging.error(traceback.format_exc())
            logging.error("[ERROR] Cannot proceed with method that requires retrieval.")
            return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Initialize results and token tracking
    results = []
    total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Start time for progress tracking
    start_time = time.time()
    
    # Process each patient
    for i, row in processed_df.iterrows():
        row_dict = row.to_dict()
        
        # Retrieve examples if needed (for few-shot learning)
        retrieved_examples = None
        if method not in ["cot", "baseline"]:
            query_text = semantic_retrieval.summarize_patient(row)
            retrieved_examples = semantic_retrieval.retrieve_top_k(
                query_text, sr_embeds, sr_df, k=3, 
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Apply the selected reasoning method
        if method in ["cot", "baseline"]:
            # Chain-of-Thought reasoning
            prompt = build_chain_of_thought_prompt(row_dict)
            response, token_usage = gpt_inference(
                client, prompt, model="gpt-4", temperature=0.2,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
            )
            predicted_label = parse_severity_from_response(response)
            method_name = "Chain-of-Thought"
            
        elif method in ["tot", "advanced"]:
            # Tree-of-Thought reasoning
            prompt = build_tree_of_thought_prompt(row_dict, retrieved_examples)
            response, token_usage = gpt_inference(
                client, prompt, model="gpt-4", temperature=0.2,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
            )
            predicted_label = parse_severity_from_response(response)
            method_name = "Tree-of-Thought"
            
        elif method == "sc":
            # Self-Consistency with voting
            consensus_result = run_self_consistency(
                client, row_dict, retrieved_examples, num_samples=3,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
            )
            response = consensus_result["raw_response"]
            predicted_label = consensus_result["final_prediction"]
            token_usage = consensus_result["token_usage"]
            method_name = "Self-Consistency"
            
        elif method == "sr":
            # Self-Reflection approach
            # First generate an initial prediction
            base_prompt = build_chain_of_thought_prompt(row_dict, retrieved_examples)
            initial_response, initial_token_usage = gpt_inference(
                client, base_prompt, model="gpt-4", temperature=0.2,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
            )
            
            # Then apply self-reflection
            reflection_prompt = build_self_reflection_prompt(initial_response, row_dict)
            reflection_response, reflection_token_usage = gpt_inference(
                client, reflection_prompt, model="gpt-4", temperature=0.2,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
            )
            
            # Use reflection response for final prediction
            response = f"Initial Assessment:\n{initial_response}\n\nReflection:\n{reflection_response}"
            predicted_label = parse_severity_from_response(reflection_response)
            
            # Combine token usage
            token_usage = {
                "prompt_tokens": initial_token_usage.get("prompt_tokens", 0) + reflection_token_usage.get("prompt_tokens", 0),
                "completion_tokens": initial_token_usage.get("completion_tokens", 0) + reflection_token_usage.get("completion_tokens", 0),
                "total_tokens": initial_token_usage.get("total_tokens", 0) + reflection_token_usage.get("total_tokens", 0)
            }
            method_name = "Self-Reflection"
            
        elif method == "fc":
            # Fact-Checking approach
            # First generate an initial prediction
            base_prompt = build_chain_of_thought_prompt(row_dict, retrieved_examples)
            initial_response, initial_token_usage = gpt_inference(
                client, base_prompt, model="gpt-4", temperature=0.2,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
            )
            
            # Then apply fact-checking
            fact_check_prompt = build_fact_checking_prompt(initial_response, row_dict)
            fact_check_response, fact_check_token_usage = gpt_inference(
                client, fact_check_prompt, model="gpt-4", temperature=0.2,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
            )
            
            # Use fact-check response for final prediction
            response = f"Initial Assessment:\n{initial_response}\n\nFact Check:\n{fact_check_response}"
            fact_check_label = parse_severity_from_response(fact_check_response)
            predicted_label = fact_check_label if fact_check_label != "Unknown" else parse_severity_from_response(initial_response)
            
            # Combine token usage
            token_usage = {
                "prompt_tokens": initial_token_usage.get("prompt_tokens", 0) + fact_check_token_usage.get("prompt_tokens", 0),
                "completion_tokens": initial_token_usage.get("completion_tokens", 0) + fact_check_token_usage.get("completion_tokens", 0),
                "total_tokens": initial_token_usage.get("total_tokens", 0) + fact_check_token_usage.get("total_tokens", 0)
            }
            method_name = "Fact-Checking"
            
        elif method == "hybrid":
            # Hybrid approach
            prompt = build_hybrid_prompt(row_dict, retrieved_examples)
            response, token_usage = gpt_inference(
                client, prompt, model="gpt-4", temperature=0.2,
                test_mode=test_mode, cache_dir=cache_dir, simulate=simulate_gpt
            )
            predicted_label = parse_severity_from_response(response)
            method_name = "Hybrid"
            
        else:
            raise ValueError(f"Unknown reasoning method: {method}")
        
        # Track token usage
        if token_usage:
            total_token_usage["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            total_token_usage["completion_tokens"] += token_usage.get("completion_tokens", 0)
            total_token_usage["total_tokens"] += token_usage.get("total_tokens", 0)
        
        # Create result entry
        result_entry = {
            "PATIENT": row.get("PATIENT"),
            "ground_truth": row.get("COVID19_SEVERITY"),
            "predicted_label": predicted_label,
            "method": method_name,
            "raw_response": response,
            "token_usage": token_usage,
            # Add these fields to help with analysis
            "age": row.get("AGE", "unknown"),
            "gender": row.get("GENDER", "unknown"),
            "has_hypertension": row.get("HAS_HYPERTENSION", 0),
            "has_diabetes": row.get("HAS_DIABETES", 0),
            "inpatient": row.get("ANY_INPATIENT", 0),
            "ventilated": row.get("USED_MECH_VENT", 0) or row.get("VENT_PROCEDURE", 0)
        }
        
        results.append(result_entry)
        
        # Print progress with estimated time remaining
        if (i+1) % 5 == 0 or test_mode:
            elapsed = time.time() - start_time
            avg_time_per_item = elapsed / (i+1)
            remaining_items = len(processed_df) - (i+1)
            estimated_remaining = avg_time_per_item * remaining_items
            
            logging.info(f"[INFO] Processed {i+1}/{len(processed_df)} rows with {method_name}... " +
                  f"(Avg: {avg_time_per_item:.2f}s/patient, Est. remaining: {estimated_remaining/60:.1f}min)")
    
    # Calculate average token usage
    avg_tokens_per_patient = {
        "prompt_tokens": total_token_usage["prompt_tokens"] / len(processed_df) if len(processed_df) > 0 else 0,
        "completion_tokens": total_token_usage["completion_tokens"] / len(processed_df) if len(processed_df) > 0 else 0,
        "total_tokens": total_token_usage["total_tokens"] / len(processed_df) if len(processed_df) > 0 else 0
    }
    
    logging.info(f"[INFO] {method_name} processing completed. Average {avg_tokens_per_patient['total_tokens']:.1f} tokens per patient.")
    
    return results, total_token_usage

# endregion Pipeline Execution

# -------------------------------------------------------------------------
# 9. Evaluation & Persistence Functions
# -------------------------------------------------------------------------
# region Evaluation & Persistence

def evaluate_predictions(predictions: list, out_filename="evaluation_results.json", test_mode=False):
    """
    Compute confusion matrix, classification report, and store results.
    """
    if not predictions:
        logging.warning("No predictions provided for evaluation.")
        return None, None

    ground_truth = [p.get("ground_truth") for p in predictions]
    pred_labels = [p.get("predicted_label") for p in predictions]

    # Filter out entries where ground truth might be missing or invalid
    valid_indices = [i for i, gt in enumerate(ground_truth) if gt in SEVERITY_LABELS]
    if len(valid_indices) < len(predictions):
        logging.warning(f"Removed {len(predictions) - len(valid_indices)} predictions with invalid ground truth labels.")
    ground_truth_valid = [ground_truth[i] for i in valid_indices]
    pred_labels_valid = [pred_labels[i] for i in valid_indices]

    if not ground_truth_valid:
        logging.warning("No valid ground truth labels found after filtering. Cannot evaluate.")
        return None, None

    # Classification metrics
    cm = confusion_matrix(ground_truth_valid, pred_labels_valid, labels=SEVERITY_LABELS)
    report_str = classification_report(ground_truth_valid, pred_labels_valid, labels=SEVERITY_LABELS, zero_division=0)
    classif_report_dict = classification_report(ground_truth_valid, pred_labels_valid, labels=SEVERITY_LABELS, zero_division=0, output_dict=True)

    # Print to console via logging
    logging.info("Classification Report:")
    logging.info(f"\n{report_str}\n") # Log multiline report

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    out_png = RESULTS_DIR / f"confusion_matrix_{Path(out_filename).stem}.png"
    try:
        plt.savefig(out_png, dpi=150)
        logging.info(f"Confusion matrix plot saved: {out_png}")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix plot to {out_png}: {e}")
    finally:
        plt.close()

    # In test mode, create additional visualizations
    if test_mode:
        # 1. Reasoning length analysis
        plt.figure(figsize=(8, 5))
        reasoning_lengths = [len(p["raw_response"].split()) for p in predictions]
        sns.histplot(reasoning_lengths, kde=True)
        plt.title("Distribution of Reasoning Lengths (word count)")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")
        len_path = RESULTS_DIR / f"reasoning_length_{Path(out_filename).stem}.png"
        try:
            plt.savefig(len_path, dpi=150)
            logging.info(f"Reasoning length plot saved: {len_path}")
        except Exception as e:
            logging.error(f"Failed to save reasoning length plot to {len_path}: {e}")
        finally:
            plt.close()
        
        # 2. Correct vs Incorrect predictions
        plt.figure(figsize=(8, 5))
        correct = [p for p in predictions if p["ground_truth"] == p["predicted_label"]]
        incorrect = [p for p in predictions if p["ground_truth"] != p["predicted_label"]]
        
        # Get severity distribution in each group
        correct_severities = [p["ground_truth"] for p in correct]
        incorrect_severities = [p["ground_truth"] for p in incorrect]
        
        correct_counts = [correct_severities.count(label) for label in SEVERITY_LABELS]
        incorrect_counts = [incorrect_severities.count(label) for label in SEVERITY_LABELS]
        
        # Create a grouped bar chart
        x = np.arange(len(SEVERITY_LABELS))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, correct_counts, width, label='Correct')
        ax.bar(x + width/2, incorrect_counts, width, label='Incorrect')
        
        ax.set_ylabel('Count')
        ax.set_title('Correct vs Incorrect Predictions by Severity')
        ax.set_xticks(x)
        ax.set_xticklabels(SEVERITY_LABELS)
        ax.legend()
        
        cvsi_path = RESULTS_DIR / f"correct_vs_incorrect_{Path(out_filename).stem}.png"
        try:
            plt.savefig(cvsi_path, dpi=150)
            logging.info(f"Correct vs Incorrect plot saved: {cvsi_path}")
        except Exception as e:
            logging.error(f"Failed to save correct vs incorrect plot to {cvsi_path}: {e}")
        finally:
            plt.close()

    # Save JSON with details
    out_data = {
        "classification_report_dict": classif_report_dict,
        "classification_report_str": report_str,
        "confusion_matrix": cm.tolist(),
        "labels": SEVERITY_LABELS,
        "accuracy": classif_report_dict.get("accuracy", 0),
        "macro_avg": classif_report_dict.get("macro avg", {})
    }
    out_json_path = RESULTS_DIR / out_filename
    try:
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2)
        logging.info(f"Evaluation results saved: {out_json_path}")
    except Exception as e:
        logging.error(f"Failed to save evaluation JSON to {out_json_path}: {e}")

    return classif_report_dict, cm

def save_model_results(predictions, method, token_usage=None, run_info=None):
    """Save model predictions and metadata for later comparison."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use pathlib for path construction
    out_path = RESULTS_DIR / f"{method}_predictions_{timestamp}.pkl"

    data = {
        "predictions": predictions,
        "method": method,
        "method_name": METHOD_NAMES.get(method, method),
        "timestamp": timestamp,
        "token_usage": token_usage or {},
        "run_info": run_info or {}
    }
    
    try:
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Model results saved to {out_path}")
    except Exception as e:
        logging.error(f"Failed to save model results pickle to {out_path}: {e}")
    return out_path

def load_model_results(method=None, file_path=None):
    """Load previously saved model results."""
    try:
        if file_path:
            load_path = Path(file_path)
            if not load_path.is_file():
                logging.error(f"Specified result file not found: {load_path}")
                return None
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Loaded results from {load_path}")
            return data
        elif method:
            # Find the most recent file for the specified method
            method_files = sorted([f for f in RESULTS_DIR.iterdir()
                                   if f.name.startswith(f"{method}_predictions_") and f.name.endswith(".pkl")],
                                  reverse=True)
            if not method_files:
                logging.error(f"No saved results found for method '{method}' in {RESULTS_DIR}")
                return None
            load_path = method_files[0]
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Loaded latest results for method '{method}' from {load_path}")
            return data
        else:
            # Load latest results for all available methods
            method_data = {}
            valid_methods = METHOD_NAMES.keys()
            for m in valid_methods:
                method_files = sorted([f for f in RESULTS_DIR.iterdir()
                                       if f.name.startswith(f"{m}_predictions_") and f.name.endswith(".pkl")],
                                      reverse=True)
                if method_files:
                    load_path = method_files[0]
                    try:
                        with open(load_path, 'rb') as f:
                            method_data[m] = pickle.load(f)
                        logging.info(f"Loaded latest results for method '{m}' from {load_path}")
                    except Exception as e_inner:
                        logging.warning(f"Failed to load results file {load_path} for method '{m}': {e_inner}")
            if not method_data:
                logging.error(f"No saved results found in {RESULTS_DIR}")
                return None
            return method_data
    except Exception as e:
        logging.error(f"Error loading model results: {e}")
        return None

# endregion Evaluation & Persistence

# -------------------------------------------------------------------------
# 10. Multi-Method Comparison Functions
# -------------------------------------------------------------------------
# region Multi-Method Comparison

def create_multi_method_comparison(method_results: dict, output_prefix="comparison"):
    """Create comprehensive visualizations comparing multiple reasoning methods."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use pathlib
    comparison_dir_path = COMPARISON_DIR / f"{output_prefix}_{timestamp}"
    comparison_dir_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving comparison results to: {comparison_dir_path}")

    # Extract method names and data
    methods = list(method_results.keys())
    method_names = [method_results[m].get("method_name", METHOD_NAMES.get(m, m)) for m in methods]
    
    if not methods:
        logging.error("No methods found in results to compare.")
        return None
    
    # Collect metrics for each method
    metrics = {}
    token_usage = {}
    all_patients = {}
    
    # Process each method's results
    for m in methods:
        predictions = method_results[m]["predictions"]
        
        # Extract ground truth and predictions
        ground_truth = [p["ground_truth"] for p in predictions if p["ground_truth"] in SEVERITY_LABELS]
        pred_labels = [p["predicted_label"] for p in predictions if p["ground_truth"] in SEVERITY_LABELS]
        
        # Create patient-to-prediction mapping
        for p in predictions:
            if p["PATIENT"] not in all_patients:
                all_patients[p["PATIENT"]] = {}
            all_patients[p["PATIENT"]][m] = {
                "ground_truth": p["ground_truth"],
                "predicted_label": p["predicted_label"],
                "correct": p["ground_truth"] == p["predicted_label"]
            }
        
        # Calculate metrics
        try:
            metrics[m] = {
                "accuracy": accuracy_score(ground_truth, pred_labels),
                "precision_macro": precision_score(ground_truth, pred_labels, average='macro', zero_division=0),
                "recall_macro": recall_score(ground_truth, pred_labels, average='macro', zero_division=0),
                "f1_macro": f1_score(ground_truth, pred_labels, average='macro', zero_division=0),
                "precision_by_class": precision_score(ground_truth, pred_labels, average=None, labels=SEVERITY_LABELS, zero_division=0).tolist(),
                "recall_by_class": recall_score(ground_truth, pred_labels, average=None, labels=SEVERITY_LABELS, zero_division=0).tolist(),
                "f1_by_class": f1_score(ground_truth, pred_labels, average=None, labels=SEVERITY_LABELS, zero_division=0).tolist()
            }
        except Exception as e:
            print(f"[WARNING] Could not calculate metrics for method {m}: {e}")
            metrics[m] = {
                "accuracy": 0,
                "precision_macro": 0,
                "recall_macro": 0,
                "f1_macro": 0,
                "precision_by_class": np.zeros(len(SEVERITY_LABELS)),
                "recall_by_class": np.zeros(len(SEVERITY_LABELS)),
                "f1_by_class": np.zeros(len(SEVERITY_LABELS))
            }
        
        # Get token usage if available
        if "token_usage" in method_results[m]:
            token_usage[m] = method_results[m]["token_usage"]
    
    # Create visualizations
    
    # 1. Overall Performance Comparison
    plt.figure(figsize=(14, 8))
    x = np.arange(len(methods))
    width = 0.2
    
    # Extract metric values for each method
    accuracies = [metrics[m]["accuracy"] for m in methods]
    precisions = [metrics[m]["precision_macro"] for m in methods]
    recalls = [metrics[m]["recall_macro"] for m in methods]
    f1_scores = [metrics[m]["f1_macro"] for m in methods]
    
    # Create grouped bar chart
    plt.bar(x - width*1.5, accuracies, width, label='Accuracy', color='#1f77b4')
    plt.bar(x - width/2, precisions, width, label='Precision', color='#ff7f0e')
    plt.bar(x + width/2, recalls, width, label='Recall', color='#2ca02c')
    plt.bar(x + width*1.5, f1_scores, width, label='F1 Score', color='#d62728')
    
    plt.xlabel('Reasoning Method')
    plt.ylabel('Score')
    plt.title('Performance Comparison Across Reasoning Methods')
    plt.xticks(x, method_names)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(accuracies):
        plt.text(i - width*1.5, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(precisions):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(recalls):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(f1_scores):
        plt.text(i + width*1.5, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    try:
        plt.savefig(comparison_dir_path / "overall_performance_comparison.png", dpi=300)
        logging.info(f"Overall performance plot saved.")
    except Exception as e:
        logging.error(f"Failed to save overall performance plot: {e}")
    finally:
        plt.close()
    
    # 2. Confusion Matrix Grid
    if len(methods) <= 6:  # Only create if we have a reasonable number of methods
        n_cols = min(3, len(methods))
        n_rows = (len(methods) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5))
        axes = np.array(axes).flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for i, m in enumerate(methods):
            if i < len(axes):
                predictions = method_results[m]["predictions"]
                ground_truth = [p["ground_truth"] for p in predictions if p["ground_truth"] in SEVERITY_LABELS]
                pred_labels = [p["predicted_label"] for p in predictions if p["ground_truth"] in SEVERITY_LABELS]
                
                # Calculate confusion matrix
                cm = confusion_matrix(ground_truth, pred_labels, labels=SEVERITY_LABELS)
                
                # Create custom colormap for this method
                colors = plt.cm.tab10(i % 10)
                cmap = LinearSegmentedColormap.from_list(f"method_{i}", ["#ffffff", colors])
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, 
                            xticklabels=SEVERITY_LABELS, yticklabels=SEVERITY_LABELS, 
                            ax=axes[i])
                axes[i].set_title(f"{method_names[i]} Confusion Matrix")
                axes[i].set_xlabel("Predicted")
                axes[i].set_ylabel("Ground Truth")
        
        # Hide unused subplots
        for j in range(len(methods), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        try:
            plt.savefig(comparison_dir_path / "confusion_matrix_grid.png", dpi=300)
            logging.info(f"Confusion matrix grid plot saved.")
        except Exception as e:
            logging.error(f"Failed to save confusion matrix grid plot: {e}")
        finally:
            plt.close()
    else:
        logging.info("Skipping confusion matrix grid plot due to large number of methods.")
    
    # 3. Per-Method F1 Scores by Severity Class
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(SEVERITY_LABELS))
    width = 0.8 / len(methods)
    
    for i, m in enumerate(methods):
        f1_by_class = metrics[m]["f1_by_class"]
        plt.bar(x + (i - len(methods)/2 + 0.5) * width, f1_by_class, width, label=method_names[i])
    
    plt.xlabel('Severity Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Severity Class Across Methods')
    plt.xticks(x, SEVERITY_LABELS)
    plt.legend(loc='best')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    try:
        plt.savefig(comparison_dir_path / "f1_by_severity_comparison.png", dpi=300)
        logging.info(f"F1 by severity plot saved.")
    except Exception as e:
        logging.error(f"Failed to save F1 by severity plot: {e}")
    finally:
        plt.close()
    
    # 4. Method Agreement Analysis
    if len(methods) > 1:
        # Calculate agreement statistics
        common_patients = set.intersection(*[set(p["PATIENT"] for p in method_results[m]["predictions"]) for m in methods])
        
        # Create a dictionary to track agreements
        agreement_counts = {
            "all_agree_correct": 0,
            "all_agree_incorrect": 0,
            "partial_agreement": 0,
            "total_disagreement": 0
        }
        
        # Count the different agreement patterns
        for patient_id in common_patients:
            # Get predictions for this patient across methods
            patient_predictions = {m: all_patients[patient_id][m] for m in methods if patient_id in all_patients and m in all_patients[patient_id]}
            
            if not patient_predictions:
                continue
                
            # Check if all methods predicted the same
            predictions = [patient_predictions[m]["predicted_label"] for m in patient_predictions]
            all_same = len(set(predictions)) == 1
            
            # Check if any of the predictions were correct
            ground_truth = next(iter(patient_predictions.values()))["ground_truth"]
            any_correct = any(patient_predictions[m]["predicted_label"] == ground_truth for m in patient_predictions)
            all_correct = all(patient_predictions[m]["predicted_label"] == ground_truth for m in patient_predictions)
            
            # Categorize
            if all_same and all_correct:
                agreement_counts["all_agree_correct"] += 1
            elif all_same and not any_correct:
                agreement_counts["all_agree_incorrect"] += 1
            elif any_correct:
                agreement_counts["partial_agreement"] += 1
            else:
                agreement_counts["total_disagreement"] += 1
        
        # Create agreement visualization
        plt.figure(figsize=(10, 6))
        
        labels = ["All Agree\n(Correct)", "All Agree\n(Incorrect)", "Some Correct\n(Disagreement)", "All Incorrect\n(Disagreement)"]
        values = [agreement_counts["all_agree_correct"], 
                 agreement_counts["all_agree_incorrect"], 
                 agreement_counts["partial_agreement"], 
                 agreement_counts["total_disagreement"]]
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
        
        bars = plt.bar(labels, values, color=colors)
        
        plt.xlabel('Agreement Pattern')
        plt.ylabel('Number of Patients')
        plt.title('Method Agreement Analysis')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        try:
            plt.savefig(comparison_dir_path / "method_agreement_analysis.png", dpi=300)
            logging.info(f"Method agreement plot saved.")
        except Exception as e:
            logging.error(f"Failed to save method agreement plot: {e}")
        finally:
            plt.close()
    
    # 5. Token Usage Comparison (if available)
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    # Main execution based on mode
    if args.mode == "compare":
        # Parse methods to compare
        methods_to_compare = [m.strip() for m in args.methods.split(',') if m.strip()]
        if not methods_to_compare:
            print("[ERROR] No methods specified for comparison. Use --methods argument.")
            sys.exit(1)
            
        valid_methods = ["cot", "tot", "sc", "sr", "fc", "hybrid", "baseline", "advanced"]
        invalid_methods = [m for m in methods_to_compare if m not in valid_methods]
        if invalid_methods:
            print(f"[ERROR] Invalid method(s) specified: {', '.join(invalid_methods)}")
            print(f"Valid options are: {', '.join(valid_methods)}")
            sys.exit(1)
            
        print(f"[INFO] Running comparison of methods: {', '.join(methods_to_compare)}")
        comparison_dir = run_multi_method_comparison(df, client, methods_to_compare, args)
        
        if comparison_dir:
            print(f"[INFO] Comparison complete. Results saved to {comparison_dir}")
        else:
            print("[ERROR] Comparison failed or produced no results.")
    else:
        # Run single method
        print(f"[INFO] Running single method: {METHOD_NAMES.get(args.mode, args.mode)}")
        
        predictions, token_usage = run_reasoning_pipeline(
            df, client, method=args.mode, 
            subset=args.subset,
            test_mode=args.test_mode, 
            cache_dir=args.cache_dir, 
            simulate_gpt=args.simulate_gpt,
            stratify_subset=args.stratify_subset,
            balance_subset_mixed=args.balance_subset_mixed,
            no_balance_retrieval=args.no_balance_retrieval
        )
        
        if not predictions:
            print(f"[WARNING] No predictions generated for method '{args.mode}'")
        else:
            print(f"[INFO] Evaluating {len(predictions)} predictions for method '{args.mode}'...")
            
            # Generate appropriate filename with sampling method
            sampling_str = ""
            if args.subset:
                if args.balance_subset_mixed: 
                    sampling_str = "_mixed"
                elif args.stratify_subset: 
                    sampling_str = "_stratified"
                else: 
                    sampling_str = "_random"
                    
            subset_str = f"subset_{args.subset}" if args.subset else "all"
            out_filename = f"{result_prefix}evaluation_{args.mode}_{subset_str}{sampling_str}.json"
            
            evaluate_predictions(predictions, out_filename=out_filename, test_mode=args.test_mode)
            
            # Save results
            run_info = vars(args)  # Capture runtime settings
            save_model_results(predictions, args.mode, token_usage, run_info)
            
            # Generate visualizations for ToT
            if args.mode in ["tot", "advanced"]:
                print("[INFO] Generating sample reasoning visualizations...")
                # We need to update this to use args instead of separate parameters
                generate_sample_reasoning_visualizations(client, df, 
                                                        test_mode=args.test_mode, 
                                                        cache_dir=args.cache_dir, 
                                                        simulate_gpt=args.simulate_gpt,
                                                        no_balance_retrieval=args.no_balance_retrieval)
        
        print(f"\n[INFO] {METHOD_NAMES.get(args.mode, args.mode)} reasoning completed.")

if __name__ == "__main__":
    main()