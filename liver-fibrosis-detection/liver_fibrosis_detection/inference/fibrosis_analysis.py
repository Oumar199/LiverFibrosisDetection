""" 🏥 SYSTÈME DE DIAGNOSTIC HÉPATIQUE V4.1 (SANS FIBROSCAN)
============================================================================
Architecture : Hybrid CNN-MLP (V4.1) + MedGemma 1.5 4B IT
Améliorations:
- Architecture compatible V4.1 (Sampler seul)
- Prompt médical optimisé (structure SOAP)
- Heatmap avec GradCAM amélioré
- Rapport clinique structuré
"""

from liver_fibrosis_detection import warnings, torch, plt, np, re
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
SYSTEM_PROMPT = """You are an expert hepatology AI diagnostic assistant based on MedGemma 1.5.

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
The VERY FIRST WORD of your response MUST be exactly: **S (SUBJECTIVE):**"""

# SYSTEM_PROMPT = """You are a medical assistant specialized in hepatology and medical imaging.

# Your mission is to analyze results from an AI system that combines:
# 1. Hepatic ultrasound image analysis (CNN)
# 2. Multidimensional clinical data evaluation (21 variables)
# 3. Liver fibrosis prediction score (probability 0-100%)

# CLINICAL CONTEXT:
# The AI model was trained on 201 patients (29 diseased, 172 healthy) with:
# - F1-Score validation: 0.373
# - Recall (disease detection): 61%
# - Precision: 27%
# - Trade-off: Prioritizes detection (accepts 31% false positives to miss only 39% of cases)

# SCORE INTERPRETATION:
# - Score ≥ 60%: FIBROSIS SUSPICION (Sensitive, may include false positives)
# - Score 50-60%: GRAY ZONE (Uncertainty, monitoring recommended)
# - Score < 50%: FAVORABLE PROFILE (But vigilance if risk factors present)

# TASK:
# Write a structured clinical report in ENGLISH following the SOAP format:

# **S (SUBJECTIVE):**
# - Patient profile summary (age, sex, BMI, comorbidities)
# - Identified risk factors (HBV, symptoms)

# **O (OBJECTIVE):**
# - AI fibrosis score (probability %)
# - Imaging-clinical concordance
# - Relevant clinical signs

# **A (ASSESSMENT):**
# - Likely stage estimation (F0-F4)
# - Diagnostic confidence level
# - Discussion of potential discrepancies

# **P (PLAN):**
# - Recommended follow-up tests (FibroScan, labs, biopsy if needed)
# - Monitoring frequency
# - Lifestyle/dietary measures

# STRICT RULES:
# - NEVER invent unmentioned symptoms
# - Clearly state model limitations
# - ALWAYS recommend FibroScan or biopsy confirmation if score ≥60%
# - State that AI is a screening aid, not definitive diagnosis
# - Remain factual and cautious

# Your tone must be: professional, precise, educational but accessible."""

# ============================================================================
# 6. REPORT GENERATION WITH MEDGEMMA
# ============================================================================

def format_patient_for_llm(patient_data, probability):
    """
    Formats data for LLM prompt
    """
    sexe_str = "Male" if patient_data['sexe'] == 1.0 else "Female"
    age = int(patient_data['age'])
    imc = patient_data['imc']

    if imc < 18.5: imc_cat = "Underweight"
    elif imc < 25: imc_cat = "Normal weight"
    elif imc < 30: imc_cat = "Overweight"
    elif imc < 35: imc_cat = "Obesity grade 1"
    elif imc < 40: imc_cat = "Obesity grade 2"
    else: imc_cat = "Obesity grade 3"

    if probability < 0.40: score_interp, stade_probable = "Very favorable profile", "F0 (normal parenchyma)"
    elif probability < 0.50: score_interp, stade_probable = "Favorable profile", "F0-F1 (normal to minimal fibrosis)"
    elif probability < 0.60: score_interp, stade_probable = "Gray zone", "F1-F2 possible"
    elif probability < 0.75: score_interp, stade_probable = "Moderate suspicion", "F2-F3 possible"
    else: score_interp, stade_probable = "High suspicion", "F3-F4 possible"

    signes_positifs = []
    symptomes = {
        'asthenie': 'Asthenia', 'amaigrissement': 'Weight loss', 'oedeme': 'Edema',
        'palleur': 'Pallor', 'ictere': 'Jaundice', 'hepatomegalie': 'Hepatomegaly',
        'hepatalgie': 'Hepatalgia', 'ascite': 'Ascites',
        'circulation_veineuse': 'Collateral circulation', 'hippocratisme': 'Digital clubbing',
        'etat_conscience': 'Altered consciousness', 'exantheme': 'Exanthema'
    }

    for key, label in symptomes.items():
        if patient_data.get(key, 0.0) == 1.0: signes_positifs.append(label)

    facteurs = []
    if patient_data.get('presence_vhb', 0.0) == 1.0: facteurs.append("Chronic Hepatitis B (HBV+)")
    if imc >= 30: facteurs.append(f"Obesity ({imc_cat}, BMI={imc:.1f})")
    elif imc >= 25: facteurs.append(f"Overweight (BMI={imc:.1f})")
    if age > 50: facteurs.append(f"Age >50 years")

    context = f"""═══════════════════════════════════════════════════════════════════════
PATIENT DATA
═══════════════════════════════════════════════════════════════════════

PROFILE:
- Sex: {sexe_str}
- Age: {age} years
- BMI: {imc:.1f} kg/m² ({imc_cat})

VITALS:
- Blood Pressure: {patient_data['tension_systolique']:.0f}/{patient_data['tension_diastolique']:.0f} mmHg
- Heart Rate: {patient_data['freq_cardiaque']:.0f} bpm
- Temperature: {patient_data['temperature']:.1f}°C

CLINICAL SIGNS:
{chr(10).join(['- ' + s for s in signes_positifs]) if signes_positifs else '- No overt clinical signs'}

RISK FACTORS:
{chr(10).join(['- ' + f for f in facteurs]) if facteurs else '- No identified risk factors'}

═══════════════════════════════════════════════════════════════════════
AI MODEL RESULT
═══════════════════════════════════════════════════════════════════════

AI SCORE: {probability*100:.1f}%
INTERPRETATION: {score_interp}
PROBABLE STAGE: {stade_probable}

MODEL PERFORMANCE:
- Recall: 61% (detects 6/10 sick)
- Precision: 27% (3/10 FP accepted)

═══════════════════════════════════════════════════════════════════════"""

    return context

def generate_medical_report(llm, patient_data, probability, prediction):
    """
    Generates the structured medical report
    """
    print("\n" + "="*70)
    print("📝 MEDICAL REPORT GENERATION (MedGemma 1.5-4b-it)")
    print("="*70 + "\n")

    context = format_patient_for_llm(patient_data, probability)

    # Complete prompt
    full_prompt = f"""<start_of_turn>user
{SYSTEM_PROMPT}

Here is the patient data to analyze:
{context}

Generate the clinical SOAP report in English now:<end_of_turn>
<start_of_turn>model"""

    try:
        # Generation with optimized parameters
        response = llm(
            full_prompt,
            max_new_tokens=1500,
            max_length=None,
            do_sample=True,
            temperature=0.1,
            top_p=0.85,
            repetition_penalty=1.15,
            return_full_text=False
        )

        if isinstance(response, list):
            raw_report = response[0]['generated_text'].strip()
        else:
            raw_report = response['generated_text'].strip()

        last_s_index = raw_report.rfind("**S (SUBJECTIVE)")

        if last_s_index != -1:
            report_clean = raw_report[last_s_index:].strip()
        elif "**S**" in raw_report:
            last_s_index = raw_report.rfind("**S**")
            report_clean = raw_report[last_s_index:].strip()
        else:
            report_clean = raw_report.strip()

        report_clean = re.sub(r'<unused94>thought.*?(?:</unused94>|$)', '', report_clean, flags=re.DOTALL).strip()

        # Formatted display
        print("─" * 70)
        print(report_clean)

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
