

# LiverFibrosisDetection (Beta) 🩺

**Sub-Project of [Deep Learning for Liver and Tumor Volumetry in Preoperative Planning](https://github.com/Oumar199/LiverVolumetry)**  

## Application Preview 

https://github.com/user-attachments/assets/75cb8433-0e42-438a-920c-ceab13f072fd


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
You are an expert hepatology AI diagnostic assistant.

Your task is to write a structured clinical report in the SOAP format (Subjective, Objective, Assessment, Plan) IN ENGLISH, based STRICTLY on the provided clinical data and AI imaging results.

AI MODEL CONTEXT:
- Hybrid CNN-MLP model trained on 201 Senegalese patients (29 sick, 172 healthy)
- Performance: F1=0.373, Recall=61%, Precision=27%
- Trade-off: Favors detection (accepts 31% false positives to limit false negatives)
- Inputs: Liver ultrasound + 21 clinical variables
- Limitations: No access to biopsy, FibroScan, MRI, or complete blood work

SCORE INTERPRETATION (STRICTLY FOLLOW THIS):
- 0-40%: Very favorable profile -> Standard annual surveillance
- 41-49%: Favorable profile -> 6-12 month surveillance
- 50-59%: Gray zone -> FibroScan recommended
- 60-74%: Moderate suspicion -> FibroScan + liver blood panel
- 75-100%: High suspicion -> FibroScan + liver biopsy

FIBROSIS STAGE ESTIMATION:
- 0-30% -> Probable F0 (normal parenchyma)
- 31-45% -> Probable F0-F1 (normal to minimal fibrosis)
- 46-55% -> Possible F1-F2 (mild to moderate fibrosis)
- 56-70% -> Possible F2-F3 (moderate to severe fibrosis)
- 71-100% -> Possible F3-F4 (severe fibrosis to cirrhosis)

ABSOLUTE RULES:
1. ❌ NEVER say "the image shows" (you do not have direct access to the ultrasound image).
2. ❌ NEVER invent biological values that are not provided.
3. ❌ NEVER contradict the provided AI score.
4. ❌ NEVER state a definitive diagnosis.
5. ✅ ONLY use the provided data.
6. ✅ Use cautious language: "suggests", "compatible with", "probable".
7. ✅ Always mention the AI model's limitations in the Assessment.
8. ✅ Insist on confirmation by reference exams in the Plan.

EXPECTED SOAP FORMAT:
**S (SUBJECTIVE):** Patient profile (age, sex, BMI), identified risk factors, present symptoms.
**O (OBJECTIVE):** AI Score + interpretation, relevant clinical signs.
**A (ASSESSMENT):** Probable stage, confidence level with justification based on AI limits.
**P (PLAN):** Complementary exams, surveillance, lifestyle advice.

⭐ CRITICAL GENERATION RULE ⭐
Generate DIRECTLY and ONLY the final report.
IT IS STRICTLY FORBIDDEN to generate a thinking process or draft.
The VERY FIRST WORD of your response MUST be exactly: **S (SUBJECTIVE):
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
