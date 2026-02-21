""" 🏥 SYSTÈME DE DIAGNOSTIC HÉPATIQUE V4.1 (SANS FIBROSCAN)
============================================================================
Architecture : Hybrid CNN-MLP (V4.1) + MedGemma 2B
Améliorations:
- Architecture compatible V4.1 (Sampler seul)
- Prompt médical optimisé (structure SOAP)
- Heatmap avec GradCAM amélioré
- Rapport clinique structuré
"""

from liver_fibrosis_detection import warnings, torch, plt, np
from liver_fibrosis_detection.utils.visualization import visualize_prediction
from liver_fibrosis_detection.inference.models import HybridModelV4_1, GradCAMV4_1
from liver_fibrosis_detection.utils.data_preparation import COLUMN_MAP, preprocess_image, load_scaler_params, preprocess_clinical_v4_1

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 Initialisation FibroDetect V4.1 sur {device}...")

# ============================================================================
# 1. EXPERT MEDICAL CONFIGURATION
# ============================================================================

# ⭐ OPTIMIZED SYSTEM PROMPT (Medical SOAP Structure)
SYSTEM_PROMPT = """You are a medical assistant specialized in hepatology and medical imaging.

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

Your tone must be: professional, precise, educational but accessible."""

# ============================================================================
# 6. REPORT GENERATION WITH MEDGEMMA
# ============================================================================

def format_patient_for_llm(patient_data, probability, prediction):
    """
    Formats data for LLM prompt
    """
    # Demographic data
    sexe_str = "Male" if patient_data['sexe'] == 1.0 else "Female"
    age = int(patient_data['age'])
    imc = patient_data['imc']

    # Positive clinical signs
    signes = []
    symptomes_keys = ['asthenie', 'amaigrissement', 'oedeme', 'palleur', 'ictere',
                      'hepatomegalie', 'hepatalgie', 'ascite', 'circulation_veineuse',
                      'hippocratisme', 'exantheme']

    for key in symptomes_keys:
        if patient_data.get(key, 0.0) == 1.0:
            signes.append(COLUMN_MAP.get(key, key))

    # Risk factors
    risques = []
    if patient_data.get('presence_vhb', 0.0) == 1.0:
        risques.append("Chronic Hepatitis B")
    if imc > 30:
        risques.append(f"Obesity (BMI={imc:.1f})")
    elif imc > 25:
        risques.append(f"Overweight (BMI={imc:.1f})")

    # Build text
    patient_summary = f"""PATIENT PROFILE:
- Sex: {sexe_str}
- Age: {age} years
- BMI: {imc:.1f} kg/m²
- Blood Pressure: {patient_data['tension_systolique']:.0f}/{patient_data['tension_diastolique']:.0f} mmHg
- Heart Rate: {patient_data['freq_cardiaque']:.0f} bpm
- Temperature: {patient_data['temperature']:.1f}°C

CLINICAL SIGNS PRESENT:
{chr(10).join(['- ' + s for s in signes]) if signes else '- No evident clinical signs'}

RISK FACTORS:
{chr(10).join(['- ' + r for r in risques]) if risques else '- No major risk factors identified'}

AI RESULTS:
- Fibrosis Score: {probability*100:.1f}%
- Interpretation: {prediction}
- Confidence: {'High' if probability > 0.7 or probability < 0.3 else 'Moderate'}"""

    return patient_summary

def generate_medical_report(llm, patient_data, probability, prediction):
    """
    Generates the structured medical report
    """
    print("\n" + "="*70)
    print("📝 MEDICAL REPORT GENERATION (MedGemma 2B)")
    print("="*70 + "\n")

    patient_text = format_patient_for_llm(patient_data, probability, prediction)

    # Complete prompt
    full_prompt = f"""{SYSTEM_PROMPT}

{patient_text}

Generate the structured clinical report now following the SOAP format:"""

    try:
        # Generation with optimized parameters
        response = llm(
            full_prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.2,  # Low for consistency
            top_p=0.9,
            repetition_penalty=1.1
        )

        # Extract generated text
        generated_text = response[0]['generated_text']

        # Clean prompt from response
        if full_prompt in generated_text:
            report = generated_text.replace(full_prompt, "").strip()
        else:
            report = generated_text

        # Formatted display
        print("─" * 70)
        print(report)
        print("─" * 70)

        # Disclaimer
        print("\n⚠️ WARNING:")
        print("This report is generated by AI and serves as a screening aid.")
        print("It does NOT replace a medical diagnosis by a healthcare professional.")
        print("Confirmation by FibroScan and/or liver biopsy is recommended.")

    except Exception as e:
        print(f"❌ Report generation error: {e}")
        print("\n📋 BASIC REPORT:")
        print(f"  Patient: {int(patient_data['age'])} years, BMI {patient_data['imc']:.1f}")
        print(f"  Prediction: {prediction} ({probability*100:.1f}%)")
        print(f"  Recommendation: {'Specialist consultation + FibroScan' if probability > 0.6 else 'Regular monitoring'}")

# ============================================================================
# 7. COMPLETE PIPELINE
# ============================================================================

def run_full_analysis(model_path, scaler_path, img_path, patient_data, llm=None):
    """
    Complete pipeline: Loading → Inference → Visualization → Report
    """
    print("\n" + "="*70)
    print("🔬 ANALYSIS IN PROGRESS...")
    print("="*70)

    # ───────────────────────────────────────────────────────────────────
    # 1. Model loading
    # ───────────────────────────────────────────────────────────────────
    print("\n1️⃣ Loading V4.1 model...")

    model = HybridModelV4_1(num_clinical=21).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # ───────────────────────────────────────────────────────────────────
    # 2. Preprocessing
    # ───────────────────────────────────────────────────────────────────
    print("\n2️⃣ Data preprocessing...")

    # Image
    img_tensor, img_pil = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)

    # Clinical data
    scaler_params = load_scaler_params(scaler_path)
    clin_tensor = preprocess_clinical_v4_1(patient_data, scaler_params)
    clin_tensor = clin_tensor.to(device)

    print("   ✅ Data ready")

    # ───────────────────────────────────────────────────────────────────
    # 3. Inference + GradCAM
    # ───────────────────────────────────────────────────────────────────
    print("\n3️⃣ AI inference...")

    with torch.set_grad_enabled(True):  # Required for GradCAM
        gradcam = GradCAMV4_1(model)
        heatmap, probability = gradcam.generate(img_tensor, clin_tensor, target_class=1)

    # Final prediction
    prediction = "FIBROSIS SUSPICION" if probability >= 0.5 else "FAVORABLE PROFILE"

    print(f"   ✅ Fibrosis score: {probability*100:.1f}%")
    print(f"   ✅ Prediction: {prediction}")

    # ───────────────────────────────────────────────────────────────────
    # 4. Visualization
    # ───────────────────────────────────────────────────────────────────
    print("\n4️⃣ Generating visualizations...")

    if heatmap is not None and prediction == "FIBROSIS SUSPICION":
        visualize_prediction(np.array(img_pil), heatmap, probability, prediction)
    else:
        print("   ⚠️ Heatmap not available")
        plt.figure(figsize=(6, 6))
        plt.imshow(np.array(img_pil))
        plt.title(f"{prediction} ({probability*100:.1f}%)", fontweight='bold')
        plt.axis('off')
        plt.show()

    # ───────────────────────────────────────────────────────────────────
    # 5. Medical report
    # ───────────────────────────────────────────────────────────────────
    if llm is not None:
        print("\n5️⃣ Generating medical report...")
        generate_medical_report(llm, patient_data, probability, prediction)
    else:
        print("\n📋 SUMMARY (without LLM):")
        print(f"   Score: {probability*100:.1f}%")
        print(f"   Interpretation: {prediction}")
