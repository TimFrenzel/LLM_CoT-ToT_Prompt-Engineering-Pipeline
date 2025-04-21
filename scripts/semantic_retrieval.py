"""
Author: Tim Frenzel
Version: 1.4
Usage:  python semantic_retrieval.py [--query_text '...'] [--k N] [--no_balance] [--visualize]

Description:
------------
Generates patient summaries and Sentence-BERT embeddings from preprocessed data, retrieving the most semantically similar patient examples for few-shot learning, with optional severity balancing.

Key Features:
-----------
1. Patient Summarization
   - Creates concise text summaries from structured patient data rows, including demographics, comorbidities, key labs/vitals, and severity.

2. Embedding Generation & Caching
   - Computes Sentence-BERT embeddings for patient summaries or loads them from a cache (`patient_embeddings.npz`) if available.

3. Semantic Retrieval
   - Finds top-K similar patients based on cosine similarity between a query text embedding and the cached patient embeddings.

4. Severity Balancing (Optional)
   - Adjusts the retrieved set of examples to improve representation of minority severity classes (Severe, Critical) if enabled (default).

5. Few-Shot Prompt Building
   - Formats the final retrieved examples (summary, score, severity) into a structured string suitable for inclusion in LLM prompts.

6. Visualization (Optional)
   - Generates a histogram of similarity scores and a PCA scatter plot of embeddings to visualize the retrieval results.

7. Result Export
   - Saves the details of the retrieved examples (rank, score, patient info, summary) to a JSON file for inspection.

Example:
--------
  # Run basic retrieval for a query with default balancing
  python semantic_retrieval.py --query_text "Patient age 70, Female. Diabetes, hospitalized..."

  # Run retrieval without balancing
  python semantic_retrieval.py --query_text "Patient age 70, Female. Diabetes, hospitalized..." --no_balance

"""

import os
import sys
import argparse
import json
from collections import defaultdict
import traceback
from pathlib import Path
import logging

import numpy as np
import pandas as pd

# Suppress specific warnings if needed, e.g., from SentenceTransformer
# import warnings
# warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    # For visualization
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
except ImportError as e:
    # Use logging for errors
    logging.basicConfig(level=logging.ERROR) # Basic config if logging not set yet
    logging.error(f"Missing required libraries ({e}). Please install them:")
    logging.error("pip install sentence-transformers scikit-learn matplotlib pandas numpy")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
# region Configuration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Derive base directory relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

RESULTS_DIR = BASE_DIR / "results" / "semantic_retrieval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DATA_CSV = BASE_DIR / "results" / "final_covid19_labeled.csv"
EMBEDDING_CACHE = RESULTS_DIR / "patient_embeddings.npz"

# Define severity labels for balancing
SEVERITY_LABELS = ["Mild", "Moderate", "Severe", "Critical"]

# endregion Configuration

# ------------------------------------------------------------------------------
# Patient Summarization
# ------------------------------------------------------------------------------
# region Summarization Logic

def summarize_patient(row: pd.Series) -> str:
    """
    Convert a single patient row into a text summary that includes:
      - Age, gender, inpatient status, total encounters
      - Key comorbidities: Hypertension, Diabetes
      - Relevant labs/vitals from the final dataset:
         * LAST_Body_temperature
         * LAST_Oxygen_saturation_in_Arterial_blood
         * LAST_C_reactive_protein_Mass_volume_in_Serum_or_Plasma
      - The final severity label

    Uses a graceful approach for missing data, e.g. "No data" if not available.
    """
    # Basic demographics
    age = row.get("AGE", np.nan)
    if pd.isna(age):
        age_str = "Unknown age"
    else:
        try:
            age_str = f"{int(age)}"
        except (ValueError, TypeError):
            age_str = "Invalid age"

    gender = row.get("GENDER", "Unknown")
    if not isinstance(gender, str) or gender.strip() == "":
        gender = "Unknown gender"

    severity = row.get("COVID19_SEVERITY", "Unknown severity")

    # Inpatient/Encounters
    any_inpatient = row.get("ANY_INPATIENT", 0)
    total_encs = row.get("TOTAL_ENCOUNTERS", 0)

    # Comorbidities
    has_htn = (row.get("HAS_HYPERTENSION", 0) == 1)
    has_diab = (row.get("HAS_DIABETES", 0) == 1)
    comorbs = []
    if has_htn:
        comorbs.append("Hypertension")
    if has_diab:
        comorbs.append("Diabetes")
    if not comorbs:
        comorbs_str = "no major comorbidities"
    else:
        comorbs_str = ", ".join(comorbs)

    # Labs/Vitals (graceful missing)
    def safe_val(col_name, label):
        val = row.get(col_name, np.nan)
        if pd.isna(val) or val == "":
            return f"{label}=NoData"
        else:
            # Attempt to format numeric values nicely, fallback to string
            try:
                return f"{label}={float(val):.1f}"
            except (ValueError, TypeError):
                return f"{label}={val}"

    body_temp_str = safe_val("LAST_Body_temperature", "Temp")
    o2_sat_str = safe_val("LAST_Oxygen_saturation_in_Arterial_blood", "SpO2")
    crp_str = safe_val("LAST_C_reactive_protein_Mass_volume_in_Serum_or_Plasma", "CRP")

    # Additional context
    total_facemask = row.get("TOTAL_FACEMASK_USED", 0)

    summary = (
        f"Patient age {age_str}, gender {gender}, severity={severity}. "
        f"Inpatient={any_inpatient}, totalEnc={total_encs}, comorbidities=({comorbs_str}). "
        f"Vitals/Labs: {body_temp_str}, {o2_sat_str}, {crp_str}. "
        f"FaceMaskUsed={total_facemask}."
    )
    return summary

# endregion Summarization Logic

# ------------------------------------------------------------------------------
# Embedding Generation
# ------------------------------------------------------------------------------
# region Embedding Computation & Caching

def _compute_and_store_embeddings(summaries: pd.Series, model_name: str) -> np.ndarray:
    """Computes Sentence-BERT embeddings for summaries and caches them.

    Internal helper function called by load_and_embed_patients if cache is missed.

    Args:
        summaries (pd.Series): Series containing textual patient summaries.
        model_name (str): Name of the Sentence Transformer model to use.

    Returns:
        np.ndarray: Computed embeddings, or an empty array on failure.
    """
    logging.info(f"Loading Sentence-BERT model: {model_name}")
    try:
        embed_model = SentenceTransformer(model_name)
    except Exception as e:
        logging.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
        logging.error(traceback.format_exc())
        return np.array([])

    logging.info("Generating embeddings...")
    summary_list = [str(s) if pd.notna(s) else "" for s in summaries]
    try:
        embed_arr = embed_model.encode(summary_list, show_progress_bar=True)
        logging.info(f"Done. Embeddings shape: {embed_arr.shape}")
    except Exception as e:
        logging.error(f"Failed during embedding generation: {e}")
        logging.error(traceback.format_exc())
        return np.array([])

    # Save the embeddings
    try:
        np.savez_compressed(EMBEDDING_CACHE, embeddings=embed_arr)
        logging.info(f"Embeddings saved to {EMBEDDING_CACHE}")
    except Exception as e:
        logging.error(f"Failed to save embeddings to {EMBEDDING_CACHE}: {e}")
        logging.error(traceback.format_exc())

    return embed_arr

def load_and_embed_patients(csv_path: str, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Loads the final labeled CSV, converts each row to a textual summary,
    and obtains (or loads) a Sentence-BERT embedding for each summary.

    Returns:
      df          : original DataFrame with a 'SUMMARY' column, or None on failure
      embeddings  : numpy array of shape (N, D), or None on failure
    """
    if not Path(csv_path).is_file(): # Use pathlib
        logging.error(f"CSV not found: {csv_path}")
        return None, None

    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        logging.error(f"Failed to load CSV {csv_path}: {e}")
        return None, None

    # Fill NA severity temporarily for summary generation if needed
    if "COVID19_SEVERITY" in df.columns:
        df["COVID19_SEVERITY"] = df["COVID19_SEVERITY"].fillna("Unknown")
    else:
        logging.warning("'COVID19_SEVERITY' column not found. Cannot balance examples by severity.")
        df["COVID19_SEVERITY"] = "Unknown" # Add dummy column if missing

    # Build patient summaries with the updated function
    try:
        df["SUMMARY"] = df.apply(summarize_patient, axis=1)
    except Exception as e:
        logging.error(f"Failed to generate summaries: {e}")
        logging.error(traceback.format_exc())
        return df, None # Return df but no embeddings

    # If a cached embedding file exists, load it
    embeddings = None
    if EMBEDDING_CACHE.is_file():
        logging.info(f"Found cached embeddings at {EMBEDDING_CACHE}, loading...")
        try:
            # Ensure file is not empty
            if EMBEDDING_CACHE.stat().st_size > 0:
                cache_data = np.load(EMBEDDING_CACHE, allow_pickle=True)
                embeddings = cache_data["embeddings"]
                # Optional sanity check
                if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
                    logging.warning("Cached embeddings are not a valid 2D numpy array. Recomputing...")
                    embeddings = None # Force recompute
                elif embeddings.shape[0] != len(df):
                    logging.warning(f"Cached embeddings size ({embeddings.shape[0]}) mismatch with DataFrame size ({len(df)})! Recomputing...")
                    embeddings = None # Force recompute
                else:
                    logging.info(f"Loaded embeddings of shape {embeddings.shape}.")
            else:
                logging.warning(f"Cached embedding file {EMBEDDING_CACHE} is empty. Recomputing...")
                embeddings = None
        except Exception as e:
            logging.warning(f"Failed to load cached embeddings: {e}. Recomputing...")
            embeddings = None # Force recompute

    if embeddings is None:
        logging.info("Computing embeddings from scratch...")
        embeddings = _compute_and_store_embeddings(df["SUMMARY"], model_name)
        if embeddings.size == 0: # Check if computation failed
            logging.error("Embedding computation failed.")
            return df, None

    if embeddings.shape[0] != len(df):
        logging.error(f"Final embedding count ({embeddings.shape[0]}) does not match DataFrame rows ({len(df)}). Aborting.")
        return df, None

    return df, embeddings

# endregion Embedding Computation & Caching

# ------------------------------------------------------------------------------
# Semantic Retrieval Logic
# ------------------------------------------------------------------------------
# region Retrieval Functions

def retrieve_top_k(query_text: str, embeddings: np.ndarray, df: pd.DataFrame,
                   k: int = 3, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                   balance_severity: bool = True, initial_pool_factor: int = 10,
                   query_patient_index: int = None) -> list:
    """
    Given a query text, finds the top candidates based on cosine similarity
    and then selects a final set of 'k' examples.

    If balance_severity is True, it attempts to ensure representation from
    minority severity classes ('Severe', 'Critical') in the final 'k' examples.

    Args:
        query_text (str): The text describing the patient case to find examples for.
        embeddings (np.ndarray): Precomputed embeddings for all patients in df.
        df (pd.DataFrame): DataFrame containing patient data, including 'SUMMARY' and 'COVID19_SEVERITY'.
        k (int): The final number of examples to retrieve.
        model_name (str): The Sentence Transformer model name.
        balance_severity (bool): Whether to apply severity balancing logic.
        initial_pool_factor (int): Multiplier for 'k' to determine the initial pool size for candidate selection.
        query_patient_index (int, optional): The index of the query patient in the original df, to avoid self-retrieval.

    Returns:
        list: A list of tuples, where each tuple is (row_dict, similarity_score, summary_text).
              The list contains the final 'k' selected examples, sorted by similarity.
    """
    if embeddings is None or embeddings.size == 0:
        logging.error("Embeddings are missing or empty. Cannot retrieve.")
        return []
    if df is None or df.empty:
        logging.error("DataFrame is missing or empty. Cannot retrieve.")
        return []
    if embeddings.shape[0] != len(df):
        logging.error(f"Embeddings shape ({embeddings.shape}) mismatch with DataFrame length ({len(df)}). Cannot retrieve.")
        return []
    if k <= 0:
        return []

    if balance_severity and "COVID19_SEVERITY" not in df.columns:
        logging.warning("'COVID19_SEVERITY' column missing. Cannot perform severity balancing.")
        balance_severity = False

    try:
        embed_model = SentenceTransformer(model_name)
        query_vec = embed_model.encode([query_text])  # shape (1, D)
        sims = cosine_similarity(query_vec, embeddings)[0]  # shape (N,)
    except Exception as e:
        logging.error(f"Failed during embedding or similarity calculation: {e}")
        logging.error(traceback.format_exc())
        return []

    # 1. Get initial pool of candidates based purely on similarity
    # Ensure pool size is reasonable and within bounds
    n_pool = max(k * initial_pool_factor, k + 20) # Ensure a decent pool size
    n_pool = min(n_pool, len(df)) # Cap at total number of patients
    # Get indices sorted by similarity (highest first)
    sorted_indices = np.argsort(-sims)

    initial_candidates = []
    added_indices_set = set()
    if query_patient_index is not None and 0 <= query_patient_index < len(df):
        added_indices_set.add(query_patient_index)

    # Iterate through sorted indices to build initial pool, skipping query patient
    for idx in sorted_indices:
        if int(idx) not in added_indices_set: # Ensure index is treated as int for set lookup
            score = float(sims[idx]) # Ensure score is float
            # Basic check if index is valid for df
            if 0 <= idx < len(df):
                row_data = df.iloc[idx].copy() # Use copy to avoid modifying original df
                row_data['_index'] = int(idx) # Store original index for reference
                initial_candidates.append((row_data, score))
                added_indices_set.add(int(idx))
                if len(initial_candidates) >= n_pool:
                    break # Stop once pool is full
            else:
                logging.warning(f"Skipping invalid index {idx} during initial pool creation.")

    if not initial_candidates:
        logging.warning("No candidates found after filtering query patient.")
        return []

    # Determine final selection logic
    final_selection = []
    if not balance_severity:
        # If not balancing, just return the top k by similarity from initial pool
        final_selection = initial_candidates[:k]
    else:
        # 2. Balance the selection by severity
        candidates_by_severity = defaultdict(list)
        for row_data, score in initial_candidates:
            severity = row_data.get("COVID19_SEVERITY", "Unknown")
            candidates_by_severity[severity].append((row_data, score))

        # Sort each severity group by similarity score (descending)
        for severity in candidates_by_severity:
            candidates_by_severity[severity].sort(key=lambda x: x[1], reverse=True)

        selected_candidates_data = []
        selected_indices_set = set() # Keep track of added indices to avoid duplicates

        # Define priority order
        minority_classes = ["Critical", "Severe"]
        majority_classes = ["Moderate", "Mild", "Unknown"] # Treat Unknown as majority for selection

        # Try to add at least one from each minority class if available
        added_count = 0
        for severity in minority_classes:
            if added_count < k and candidates_by_severity[severity]:
                candidate_row, score = candidates_by_severity[severity].pop(0) # Take and remove best
                candidate_idx = candidate_row['_index']
                if candidate_idx not in selected_indices_set:
                    selected_candidates_data.append((candidate_row, score))
                    selected_indices_set.add(candidate_idx)
                    added_count += 1

        # Fill remaining slots - build a combined pool of remaining candidates
        pool = []
        # Add remaining minority candidates first
        for severity in minority_classes:
            pool.extend(candidates_by_severity[severity])
        # Then add majority candidates
        for severity in majority_classes:
            pool.extend(candidates_by_severity[severity])

        # Sort the combined pool primarily by score
        pool.sort(key=lambda x: x[1], reverse=True)

        # Add from the sorted pool until k examples are selected
        for row_data, score in pool:
            if added_count >= k:
                break
            candidate_idx = row_data['_index']
            # Double check index isn't already added
            if candidate_idx not in selected_indices_set:
                selected_candidates_data.append((row_data, score))
                selected_indices_set.add(candidate_idx)
                added_count += 1

        final_selection = selected_candidates_data[:k] # Ensure exactly k items

    # 3. Format results (row_dict, score, summary)
    results = []
    for row_data, score in final_selection:
        # Use original index stored earlier
        original_idx = row_data['_index']
        # Get the full row dictionary from the original DataFrame for safety
        full_row_dict = df.iloc[original_idx].to_dict()
        # Ensure summary is present, regenerate if needed (shouldn't happen often)
        row_summary = full_row_dict.get("SUMMARY", summarize_patient(df.iloc[original_idx]))
        # Add score and original index back if needed by consumer
        # We already have the full dict, just ensure score is float
        results.append((full_row_dict, float(score), row_summary))

    # Sort final results by similarity score for consistency before returning
    results.sort(key=lambda x: x[1], reverse=True)

    return results

def re_rank_candidates(query_text: str, df: pd.DataFrame,
                       top_candidates: list, top_n_for_re_rank=10,
                       final_k=3) -> list:
    """
    Placeholder for domain-specific or GPT-based re-ranking.
    NOTE: With the introduction of severity balancing in retrieve_top_k (v1.4),
    this function's role is diminished. Currently, it simply truncates the list.

    Args:
        query_text (str): The query text.
        df (pd.DataFrame): The main DataFrame.
        top_candidates (list): List of (row_data, score, summary) tuples.
        top_n_for_re_rank (int): Max candidates considered (mostly ignored now).
        final_k (int): The final number of candidates to return.

    Returns:
        list: The truncated list of candidates.
    """
    # Simple truncation based on the list received from retrieve_top_k
    # print("[INFO] re_rank_candidates called (performs simple truncation).")
    return top_candidates[:final_k]

# endregion Retrieval Functions

# ------------------------------------------------------------------------------
# Prompt Building
# ------------------------------------------------------------------------------
# region Prompt Generation

def build_gpt_prompt(query_text: str, retrieved_examples: list) -> str:
    """
    Creates a basic prompt structure including the query and retrieved examples.
    Handles potential None or non-string summaries.
    """
    prompt_lines = []
    prompt_lines.append("Query:")
    prompt_lines.append(str(query_text) if query_text is not None else "[No Query Text Provided]")
    prompt_lines.append("")
    prompt_lines.append("Similar Examples:")

    if not retrieved_examples:
        prompt_lines.append("None found.")
    else:
        for i, (row_data, score, summary) in enumerate(retrieved_examples, 1):
            # Ensure summary is a string before formatting
            summary_str = str(summary) if pd.notna(summary) else "[Summary Unavailable]"
            # Basic escaping of potential issues - more robust might be needed
            summary_str = summary_str.replace('\n', ' ').replace('\r', '') # Remove newlines

            prompt_lines.append(f"Example {i} (Similarity={score:.3f}):")
            prompt_lines.append(f"  Summary: {summary_str}")
            # Optionally add more details from row_data if needed
            prompt_lines.append(f"  Actual Severity: {row_data.get('COVID19_SEVERITY', 'N/A')}")
            prompt_lines.append("")

    return "\n".join(prompt_lines)

# endregion Prompt Generation

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------
# region Visualization Functions

def visualize_results(query_text: str, df: pd.DataFrame, embeddings: np.ndarray,
                      retrieved: list):
    """
    Generate visualizations:
    1. Histogram of cosine similarities.
    2. PCA scatter plot of embeddings highlighting query and retrieved points.
    """
    if not retrieved:
        logging.warning("No retrieved examples to visualize.")
        return

    if embeddings is None or len(embeddings) == 0:
        logging.warning("No embeddings available for visualization.")
        return

    model_name = 'sentence-transformers/all-MiniLM-L6-v2' # Use consistent model
    try:
        embed_model = SentenceTransformer(model_name)
        query_vec = embed_model.encode([query_text])
        # Ensure embeddings are 2D array for cosine similarity
        if embeddings.ndim == 1:
            logging.warning("Embeddings array is 1D, attempting reshape for visualization.")
            if len(df) == embeddings.shape[0]: # Basic check if it might be flattened per patient
                embeddings = embeddings.reshape(len(df), -1)
            else:
                raise ValueError("Cannot reshape 1D embeddings of unknown structure.")
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
        sims = cosine_similarity(query_vec, embeddings)[0]
    except Exception as e:
        logging.error(f"[ERROR] Failed encoding query or calculating similarities for viz: {e}")
        logging.error(traceback.format_exc())
        return

    # Use the original index stored in the row_dict (first element of tuple)
    retrieved_indices = [r[0].get('original_index', r[0].get('_index', -1)) for r in retrieved]
    # Filter out invalid indices (-1 or out of bounds)
    valid_retrieved_indices = [idx for idx in retrieved_indices if isinstance(idx, int) and 0 <= idx < len(df)]

    if not valid_retrieved_indices:
        logging.warning("No valid indices found for retrieved examples in visualization.")
        # Proceed without highlighting retrieved points if indices are bad
    # Get scores corresponding to valid indices
    retrieved_scores = [r[1] for r, idx in zip(retrieved, retrieved_indices) if isinstance(idx, int) and 0 <= idx < len(df)]

    # 1. Similarity Histogram
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(sims, bins=50, alpha=0.7, label='All Patient Similarities')
        # Use valid scores for plotting
        if retrieved_scores:
            plt.scatter(retrieved_scores, [5]*len(retrieved_scores), color='red', s=50, zorder=5, label=f'Top {len(retrieved_scores)} Retrieved')
        plt.title('Cosine Similarity Distribution (Query vs. All Patients)')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        hist_path = RESULTS_DIR / "similarity_histogram.png"
        plt.savefig(hist_path, dpi=150)
        plt.close()
        logging.info(f"[INFO] Similarity histogram saved to {hist_path}")
    except Exception as e:
        logging.error(f"[ERROR] Failed to generate/save similarity histogram: {e}")
        logging.error(traceback.format_exc())
        plt.close() # Ensure plot is closed even if error occurs

    # 2. PCA Visualization
    if len(embeddings) < 2:
        logging.warning("Not enough data points (>1) for PCA visualization.")
        return

    try:
        pca = PCA(n_components=2)
        # Fit PCA on all embeddings + query embedding
        all_embeddings_for_pca = np.vstack([embeddings, query_vec])
        # Ensure correct shape before transform
        if all_embeddings_for_pca.ndim != 2:
            raise ValueError(f"Data for PCA must be 2D, got shape {all_embeddings_for_pca.shape}")
        embeds_2d = pca.fit_transform(all_embeddings_for_pca)

        query_2d = embeds_2d[-1] # Last point is the query
        patients_2d = embeds_2d[:-1] # All other points are patients

        plt.figure(figsize=(10, 8))
        plt.scatter(patients_2d[:, 0], patients_2d[:, 1], alpha=0.3, s=10, label='All Patients')

        # Highlight retrieved points using valid indices if available
        if valid_retrieved_indices:
            retrieved_2d = patients_2d[valid_retrieved_indices]
            plt.scatter(retrieved_2d[:, 0], retrieved_2d[:, 1], color='red', s=50, label=f'Top {len(valid_retrieved_indices)} Retrieved')

        # Highlight query point
        plt.scatter(query_2d[0], query_2d[1], color='black', marker='*', s=150, label='Query Patient')

        plt.title('PCA Visualization of Patient Embeddings')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        pca_path = RESULTS_DIR / "pca_visualization.png"
        plt.savefig(pca_path, dpi=150)
        plt.close()
        logging.info(f"[INFO] PCA visualization saved to {pca_path}")
    except Exception as e:
        logging.error(f"[ERROR] Failed to generate/save PCA visualization: {e}")
        logging.error(traceback.format_exc())
        plt.close()

# endregion Visualization Functions

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
# region Main Execution

def main():
    parser = argparse.ArgumentParser(description="Semantic Retrieval for Few-Shot Learning (v1.4)")
    parser.add_argument('--query_text', type=str, default="Patient age 65, Male. Has Hypertension and Diabetes. Admitted to inpatient care.",
                        help="Text summary of the patient case to find examples for.")
    parser.add_argument('--k', type=int, default=3, help="Number of examples to retrieve.")
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help="Sentence Transformer model name.")
    parser.add_argument('--visualize', action='store_true', help="Generate visualizations.")
    parser.add_argument('--no_balance', action='store_true', help="Disable severity balancing in retrieval.")

    args = parser.parse_args()

    logging.info("--- Running Semantic Retrieval Script (v1.4) ---")

    # Load data and embeddings
    df, embeddings = load_and_embed_patients(FINAL_DATA_CSV, args.model_name)

    if df is None or embeddings is None:
        logging.error("Failed to load data or embeddings. Exiting.")
        sys.exit(1)

    # Retrieve top-k examples (with balancing enabled by default)
    logging.info(f"[INFO] Retrieving top {args.k} examples for query: '{args.query_text[:100]}...' (Balancing: {not args.no_balance})")
    retrieved_examples = retrieve_top_k(
        args.query_text,
        embeddings,
        df,
        k=args.k,
        model_name=args.model_name,
        balance_severity=(not args.no_balance), # Balance unless --no_balance is set
        query_patient_index=None # Not applicable when running script directly
    )

    # Re-ranking step is now mostly bypassed as balancing is in retrieve_top_k
    final_examples = re_rank_candidates(args.query_text, df, retrieved_examples, final_k=args.k)

    logging.info(f"\n[INFO] Retrieved {len(final_examples)} examples:")
    # Use the data structure returned by retrieve_top_k: (row_dict, score, summary)
    for i, (row_data, score, summary) in enumerate(final_examples, 1):
        original_index = row_data.get('original_index', row_data.get('_index', 'N/A')) # Get stored index
        logging.info(f"  {i}. Score={score:.4f}, Severity={row_data.get('COVID19_SEVERITY', 'N/A')}, Index={original_index}")
        # Ensure summary is printable
        summary_str = str(summary) if pd.notna(summary) else "[Summary Unavailable]"
        logging.info(f"     Summary: {summary_str[:150]}{'...' if len(summary_str) > 150 else ''}")

    # Build a sample prompt
    gpt_prompt = build_gpt_prompt(args.query_text, final_examples)
    logging.info("\n[INFO] Sample Generated Prompt:")
    logging.info("-" * 30)
    logging.info(f"\n{gpt_prompt}\n") # Log multiline prompt
    logging.info("-" * 30)

    # Visualize if requested
    if args.visualize:
        logging.info("\n[INFO] Generating visualizations...")
        visualize_results(args.query_text, df, embeddings, final_examples)

    # Save results to JSON (optional, useful for debugging)
    try:
        results_json = []
        for i, (row_data, score, summary) in enumerate(final_examples, 1):
            # Ensure data is serializable - convert numpy types, filter complex objects
            serializable_row = {}
            for k, v in row_data.items():
                if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                    serializable_row[k] = int(v)
                elif isinstance(v, (np.float64, np.float32, np.float16)):
                    serializable_row[k] = float(v)
                elif isinstance(v, (np.bool_)):
                    serializable_row[k] = bool(v)
                elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                    serializable_row[k] = v
                # else: skip other types like numpy arrays etc.

            results_json.append({
                "rank": i,
                "score": float(score),
                "patient_info": serializable_row,
                "summary": str(summary) if pd.notna(summary) else ""
            })
        out_json_path = RESULTS_DIR / "retrieval_results_last_run.json"
        with open(out_json_path, 'w', encoding='utf-8') as fout:
            json.dump(results_json, fout, indent=2, ensure_ascii=False)
        logging.info(f"[INFO] Retrieval results for this run saved to: {out_json_path}")
    except Exception as e:
        logging.warning(f"[WARNING] Failed to save retrieval results to JSON: {e}")
        logging.warning(traceback.format_exc())

    logging.info("\n[INFO] Semantic retrieval script finished.")

if __name__ == "__main__":
    main()

# endregion Main Execution