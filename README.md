# LiverFibrosisDetection

**Sub-Project of [Deep Learning for Liver and Tumor Volumetry in Preoperative Planning](https://github.com/Oumar199/LiverVolumetry)**  

## Description

LiverFibrosisDetection is an open-source Python package for automatic liver fibrosis detection and clinical analysis based on deep learning. 

---

## Methodology


### Medgemma 1.2-2b-it's System Prompt

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

## MedGemma's Contribution to Liver Fibrosis Detection

MedGemma has greatly enhanced the accuracy and efficiency of liver fibrosis detection through its advanced integration into the existing framework. The following outlines its significant contributions:

1. **Functional Contribution**: MedGemma provides real-time analysis and insights, allowing for quicker decision-making during liver assessment. Its algorithms improve measurement precision, directly impacting patient outcomes.

2. **Methodological Value**: Integrating MedGemma into our pipeline enables the use of state-of-the-art techniques for fibrosis detection. This ensures consistency and reliability in the methodologies applied across different patient datasets.

3. **Advantages Over Developing a Custom LLM**: Building a custom Large Language Model (LLM) from scratch often requires substantial resources in terms of time, data, and expertise. MedGemma offers a ready-to-use solution that leverages existing research and algorithms, thus reducing development overhead and expediting implementation.

4. **Impact on the Overall Pipeline**: The integration of MedGemma has streamlined our workflow, allowing for seamless data input and output. This not only enhances productivity but also leads to more robust and reproducible results, improving the overall quality of liver volumetry assessments in clinical practice.

In summary, the MedGemma integration represents a significant advancement in liver fibrosis detection, combining functional, methodological, and practical advantages that elevate our analytical capabilities and patient care standards.

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

- [*Early Detection of Liver Fibrosis*](https://link.springer.com/chapter/10.1007/978-3-031-79103-1_1)

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
