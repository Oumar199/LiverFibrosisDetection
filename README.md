# LiverFibrosisDetection (Beta) 🩺

**Sub-Project of [Deep Learning for Liver and Tumor Volumetry in Preoperative Planning](https://github.com/Oumar199/LiverVolumetry)**  

## Description

LiverFibrosisDetection is an open-source Python package for automatic liver fibrosis detection and clinical analysis based on deep learning. It implements a multi-modal AI pipeline designed for the screening of liver fibrosis. It bridges the gap between raw probability scores and clinical actionability by combining **EfficientNet (CNN)**, **Multi-Layer Perceptron (MLP)**, and **MedGemma 1.5-4B-it (LLM)**.

## 🧠 System Architecture

The diagnostic workflow is structured into three layers:

1.  **Image Feature Extraction (EfficientNet-B0):** Analyzes ultrasound B-mode images to extract **1,280 visual features** (texture, nodularity, echogenicity).
2.  **Clinical Fusion (MLP):** Integrates **21 clinical variables** (Age, BMI, HBV status, biological scores). On clear images, the model achieves **99% confidence**.
3.  **Narrative Interpretation (MedGemma 1.5-4B-IT):** A Large Language Model acts as a "Medical Assistant" to contextualize the classification output into a structured report.

## 🤖 LLM Integration (MedGemma 1.5-4B-it)

To overcome the "black box" nature of AI, we integrated **Instructed MedGemma 1.5 (4B parameters)**. The model receives the raw prediction score and clinical data to generate a **SOAP Report** (Subjective, Objective, Assessment, Plan).

### Score Interpretation Logic
The system follows strict medical thresholds defined in the system prompt:
*   **Score ≥ 60%:** **FIBROSIS SUSPICION** (High sensitivity, triggers mandatory confirmation).
*   **50% - 60%:** **GRAY ZONE** (Indicates uncertainty, requires monitoring).
*   **Score < 50%:** **FAVORABLE PROFILE** (Vigilance remains if clinical risk factors are present).

### Clinical Guardrails
*   **Strict Rules:** The LLM is programmed to never invent symptoms and must always recommend a **FibroScan** or **Biopsy** for scores above 60%.
*   **Screening Tool:** The output explicitly states that the AI is a screening aid, not a definitive diagnosis.

## 📊 Performance & Results

The prediction model was validated on a dataset of **114 images** with a strategy prioritizing **Recall** (sensitivity) to minimize missed pathological cases.


| Metric | Value |
| :--- | :--- |
| **Recall (Sensitivity)** | **61%** |
| **Precision** | 27% |
| **F1-Score** | 0.373 |

### Confusion Matrix (Validation Set)


| | Predicted Healthy | Predicted Fibrosis |
| :--- | :---: | :---: |
| **Actual Healthy** | **68** (True Negatives) | **28** (False Positives) |
| **Actual Diseased** | **8** (False Negatives) | **10** (True Positives) |

*Decision Trade-off: The system accepts a higher False Positive rate (31%) to ensure that 61% of diseased cases are captured in a non-invasive screening stage.*

## 📝 Example Output Format (SOAP)

The model generates reports structured as follows:
*   **S (Subjective):** Patient history and identified risk factors.
*   **O (Objective):** AI probability score and clinical-imaging concordance.
*   **A (Assessment):** Estimated fibrosis stage (F0-F4) and confidence level.
*   **P (Plan):** Recommended follow-up (FibroScan, lifestyle changes, labs).

### MedGemma 1.5-4b-it's System Prompt

```
You are a medical assistant specialized in hepatology and medical imaging.

Your mission is to analyze results from an AI system that combines:
1. Hepatic ultrasound image analysis (CNN)
2. Multidimensional clinical data evaluation (21 variables)
3. Liver fibrosis prediction score (probability 0-100%)

CLINICAL CONTEXT:
The AI model was trained on 201 patients (29 diseased, 172 healthy) with:
- F1-Score validation: 0.373
- Recall (disease detection): 61%
- Precision: 27%
- Trade-off: Prioritizes detection (accepts 31% false positives to miss only 39% of cases)

SCORE INTERPRETATION:
- Score ≥ 60%: FIBROSIS SUSPICION (Sensitive, may include false positives)
- Score 50-60%: GRAY ZONE (Uncertainty, monitoring recommended)
- Score < 50%: FAVORABLE PROFILE (But vigilance if risk factors present)

TASK:
Write a structured clinical report in ENGLISH following the SOAP format:

**S (SUBJECTIVE):**
- Patient profile summary (age, sex, BMI, comorbidities)
- Identified risk factors (HBV, symptoms)

**O (OBJECTIVE):**
- AI fibrosis score (probability %)
- Imaging-clinical concordance
- Relevant clinical signs

**A (ASSESSMENT):**
- Likely stage estimation (F0-F4)
- Diagnostic confidence level
- Discussion of potential discrepancies

**P (PLAN):**
- Recommended follow-up tests (FibroScan, labs, biopsy if needed)
- Monitoring frequency
- Lifestyle/dietary measures

STRICT RULES:
- NEVER invent unmentioned symptoms
- Clearly state model limitations
- ALWAYS recommend FibroScan or biopsy confirmation if score ≥60%
- State that AI is a screening aid, not definitive diagnosis
- Remain factual and cautious

Your tone must be: professional, precise, educational but accessible.
```

---

## 🧪 Quick Test (Google Colab)
For a zero-setup experience, you can run the full analysis pipeline in one click (consider moving to t4 Tesla GPU for faster execution and restarting the session after executing the first cell):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oumar199/LiverFibrosisDetection/blob/main/liver_fibrosis_detection_analysis.ipynb)


---

## License

MIT License. See [LICENSE](LICENSE).

---

## Maintainers

- **Oumar199** (GitHub: [@Oumar199](https://github.com/Oumar199))
- **CheikhYakhoubMAAS** (Github: [@CheikhYakhoubMAAS](https://github.com/CheikhYakhoubMAAS))
- **MamadouBousso** (Github: [@MamadouBousso](https://github.com/MamadouBousso))
- **ms-dl** (Github: [@ms-sl](https://github.com/ms-dl))
- **Aby1diallo** (Github: [@Aby1diallo](https://github.com/Aby1diallo))

---

## References

- Preceding Released Scientific Papers: [*Early Detection of Liver Fibrosis*](https://link.springer.com/chapter/10.1007/978-3-031-79103-1_1)

---

## Disclaimer

This is not a substitute for professional medical advice. Outputs should be validated with clinical experts before deployment. Do not use real patient data in this repository.

## Citation
```bibtex
@misc{snaimasters2026liverfibrosisdetection,
  title={Liver Fibrosis Detection},
  author={Cheikh Yakhoub MAAS, Mamadou BOUSSO, Oumar KANE, Metou SANGHE, Aby DIALLO},
  howpublished={https://github.com/Oumar199/LiverFibrosisDetection},
  year={2026}
}
```
