# LLM_CoT-ToT_Prompt-Engineering-Pipeline

## Abstract
Large Language Models (LLMs) offer new possibilities in clinical settings, though effectively tailoring their reasoning steps remains a challenge. This repository presents a reproducible framework for prompt engineering using **Chain-of-Thought (CoT)** and **Tree-of-Thought (ToT)** strategies, applied to **synthetic COVID-19** patient data from Synthea. Through integration of **semantic retrieval** of relevant patient examples with multi-step LLM prompts, the goal is to predict COVID-19 **severity levels**—Mild, Moderate, Severe, or Critical—based on patient attributes including vital signs, laboratory results, and comorbidities. This approach addresses a key clinical question of risk stratification and complements existing AI pipelines with structured prompting techniques. Structured reasoning leads to improvements in classification performance and interpretability, showing that methodical prompt design can produce more consistent model outputs.

---

## Project Objectives
1. **Demonstrate Advanced Prompt Engineering**  
   Combine Chain-of-Thought and Tree-of-Thought approaches to guide LLM reasoning step by step.

2. **Explore Multi-Step Reasoning & Semantic Retrieval**  
   Employ RoBERTa-based retrieval to find contextually similar patients as few-shot examples, enhancing LLM output reliability.

3. **Quantify Performance on COVID-19 Severity Prediction**  
   Evaluate how well different prompt methods classify synthetic patient severity levels (Mild to Critical).

4. **Facilitate Interpretability & Clinical Relevance**  
   Present structured LLM outputs that clinicians can examine to understand model decision pathways and logic.

---

## System Requirements & Dataset Considerations

### Synthea COVID-19 Dataset
- **Data Source**: [Synthea](https://github.com/synthetichealth/synthea) with its COVID-19 module producing realistic (but synthetic) patient records (demographics, conditions, labs, medications, encounters, etc.).
- **Clinical Fields**: Encounters, Observations, Conditions, relevant procedure codes (e.g., ventilation).
- **Severity Labels**: Derived from outcomes such as ICU admission and mortality.  
- **Data Volume**: Typically hundreds to thousands of synthetic patient records, exportable as CSV or FHIR resources.

### Computational Environment
- **GPU (NVIDIA RTX 4070)** recommended for embedding and large-scale LLM calls.
- **Python 3.x** environment with packages: `sentence-transformers`, `openai`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, etc.
- **Memory**: ~8–16 GB system RAM for medium-size synthetic datasets; more for extensive runs or large LLM requests.

### Data Usage
- **Synthetic Data**: No real patient PHI; Synthea data is freely distributable but remains realistic in structure and format.
- **Scalability**: Larger Synthea exports (10k+ patients) require optimized data merges and GPU resource management.

---

## Project Workflow Overview
The pipeline follows these core steps:

1. **Data Ingestion & Preprocessing**  
   - Load Synthea COVID-19 CSV files (patients, conditions, observations, etc.).  
   - Derive severity labels (Mild, Moderate, Severe, Critical).

2. **Semantic Retrieval**  
   - Encode patient profiles using Sentence-BERT (RoBERTa variant) to find top-k similar patient examples.
   - Store embeddings and retrieve them dynamically for in-context LLM prompts.

3. **Prompt Engineering with LLMs**  
   - **Chain-of-Thought (CoT)**: Linear step-by-step reasoning instructions.  
   - **Tree-of-Thought (ToT)**: Branching instructions guiding multiple reasoning paths before converging on a final severity prediction.

4. **Model Inference & Evaluation**  
   - Compare baseline prompting vs. advanced CoT/ToT strategies.  
   - Compute accuracy, F1, confusion matrix on the severity classification task.

5. **Interpretability & Analysis**  
   - Review LLM rationales (chain-of-thought transcripts).  
   - Visualize how different prompt methods or example retrieval sets affect severity predictions.

A simplified illustration of the end-to-end approach:

![CoT and ToT Prompt Engineering Workflow](workflow.png)

---

## Methodological Framework

1. **Data Pipeline**  
   - **Merging & Labeling**: A single per-patient dataset with final severity labels.  
   - **RoBERTa Embeddings**: Vector representations of patient summaries for semantic retrieval.

2. **Few-Shot Retrieval**  
   - Weighted approach to ensure minority classes (Severe, Critical) appear among retrieved exemplars.  
   - KNN-like similarity in embedding space to pick examples most relevant to the query patient.

3. **Prompt Construction**  
   - **Chain-of-Thought**: Nudges the LLM to reason systematically (step 1 → step 2 → conclusion).  
   - **Tree-of-Thought**: Explores multiple branches (e.g., respiratory status, comorbidities) before final severity selection.

4. **Evaluation Metrics**  
   - Classification metrics (accuracy, precision, recall, F1).  
   - Subgroup analysis (older patients, missing labs, comorbidities).  
   - Token usage tracking for cost/time trade-offs in different prompting methods.

---

## Exploratory Data Analysis & Interpretability

### Patient Severity Distributions
- EDA reveals **majority** of patients recover with Mild or Moderate severity.  
- Synthetic ICU or ventilation flags drive the Severe/Critical classes.

### Example Patient Journeys
Illustrate day-by-day changes in labs and encounters.  
![Patient Journey Example](results/EDA/sample_patient_journey.png)

### Model Explanation (Chain-of-Thought Rationale)
- Stepwise reasoning text from GPT clarifies how factors like age, oxygen saturation, and comorbidities influence severity.

---

## Baseline vs. Advanced Prompting Approaches

1. **Baseline**  
   - Minimal instructions to LLM, e.g., “Predict severity from these features.”  
   - Tends to produce less structured or consistent outputs.

2. **Chain-of-Thought (CoT)**  
   - Directs the LLM to articulate intermediate reasoning steps.  
   - Observed improvement in classification consistency and less “hallucination” about severity.

3. **Tree-of-Thought (ToT)**  
   - Explicit branching: separate reasoning about labs, comorbidities, demographic risk, then converge.  
   - Gains in interpretability, though potentially higher token usage.

### Comparative Results
- **Accuracy**: ToT slightly outperforms CoT, especially in borderline moderate–severe cases.  
- **F1-Score**: Gains seen particularly in the minority classes (Severe, Critical) when using advanced prompting with few-shot retrieval.  
- **Token Overheads**: ToT prompts can be ~20–40% more tokens than baseline.

---

## Key Domain Insights
1. **Contextual Examples Improve Generalization**  
   Few-shot retrieval of patients with similar profiles reduces misclassification for severe outliers.

2. **Structured Reasoning Helps**  
   Breaking the reasoning into discrete steps or branches clarifies risk factors (ventilation, age, comorbidities).

3. **Severity Overlaps**  
   Patients near the boundary of moderate vs. severe benefit the most from advanced ToT reasoning.

---

## Broader Evaluation & Takeaways
- **Robust Prompt Design**: Providing a consistent, structured methodology significantly enhances LLM reliability.  
- **Cost-Benefit**: Higher token usage from advanced prompts can be justified if the outcome is more accurate severity labeling.  
- **Clinical Applicability**: Synthetic data is no substitute for real-world validation, but it lays the foundation for safe experimentation.  
- **Future Prospects**: Combining chain-of-thought with retrieval from real EHR data may yield potent triage tools if regulated properly.

---

## Limitations & Future Directions
- **Data Realism**: While Synthea data is realistic, it lacks real patient variability or complexities like missing or contradictory records.  
- **LLM Biases**: Even with structured prompts, LLMs can produce inaccurate or inconsistent rationales.  
- **Scalability**: Extremely large patient sets require efficient embedding retrieval pipelines.  
- **Expansion**: Incorporate advanced self-consistency or fact-checking prompts for further improvement.

---

## Ethical Considerations and Data Usage
- **Synthetic Data**: Minimizes privacy concerns but does not replicate real patient heterogeneity precisely.  
- **Bias & Safety**: Predictions must be validated before any clinical usage.  
- **OpenAI & GPT-4**: Complies with usage policies; no protected health information is processed here.  
- **Transparent Methodology**: All code is open-sourced for reproducibility and auditing.

---

## License
**Apache License 2.0** or **MIT License** recommended for broad academic and industry adoption.  
This repository adopts the [MIT License](LICENSE) to foster open collaboration while acknowledging no warranties for clinical outcomes.
