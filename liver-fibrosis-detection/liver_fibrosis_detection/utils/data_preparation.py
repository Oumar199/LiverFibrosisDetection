"""Module of preparation of data

- Loading scaler config
- Preprocessing clinical data and image
- Input of patient information
"""

from liver_fibrosis_detection import json, torch, Image, np, transforms


# Mapping colonnes pour affichage
COLUMN_MAP = {
    "age": "Âge",
    "sexe": "Sexe",
    "imc": "IMC",
    "freq_cardiaque": "Fréquence cardiaque",
    "temperature": "Température",
    "tension_systolique": "Tension systolique",
    "tension_diastolique": "Tension diastolique",
    "asthenie": "Asthénie",
    "amaigrissement": "Amaigrissement",
    "oedeme": "Œdème",
    "palleur": "Pâleur",
    "ictere": "Ictère",
    "hepatomegalie": "Hépatomégalie",
    "hepatalgie": "Hépatalgies",
    "ascite": "Ascite",
    "circulation_veineuse": "Circulation collatérale",
    "hippocratisme": "Hippocratisme digital",
    "etat_conscience": "État de conscience",
    "stade_encephalopathie": "Encéphalopathie",
    "exantheme": "Exanthème",
    "presence_vhb": "Hépatite B"
}

# ============================================================================
# 4. PREPROCESSING AND DATA INPUT
# ============================================================================

def load_scaler_params(json_path):
    """Load V4.1's scaler parameters"""
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        return params
    except:
        print("⚠️ scaler_params_final.json is unavailable, usage of default parameters")
        return None


def preprocess_clinical_v4_1(patient_data, scaler_params):
    """
    Preprocessing clinical data (identical to the training)

    Args:
        patient_data: dict including the patient's 21 features
        scaler_params: dict including 'features', 'center', 'scale'

    Returns:
        torch.Tensor: tensor of dimension [1, 21]
    """
    if scaler_params is None:
        # Fallback if not available scaler
        features = list(patient_data.values())
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    feature_names = scaler_params['features']
    centers = scaler_params['center']
    scales = scaler_params['scale']

    # Create the vector 
    input_vector = []
    for i, fname in enumerate(feature_names):
        raw_value = patient_data.get(fname, 0.0)

        # Apply a RobustScaler normalizer
        normalized = (raw_value - centers[i]) / scales[i]

        # Clipping [-5, 5]
        normalized = np.clip(normalized, -5.0, 5.0)

        input_vector.append(normalized)

    return torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)


def preprocess_image(img_path):
    """Preprocessing image (ImageNet default normalization)"""
    img = Image.open(img_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0), img


def collect_patient_data(necessary_only: bool = True):
    """Simplified patient data entry interface

    Args:
        necessary_only (bool, optional): only necessary attributes will be asked. Defaults to True.

    Returns:
        dict: patient information
    """
    print("\n" + "="*60)
    print("📋 PATIENT DATA ENTRY")
    print("="*60)

    data = {}

    # Essential data
    data['age'] = float(input("  Age (years): ") or "50")

    sexe = input("  Sex (M/F): ").strip().upper()
    data['sexe'] = 1.0 if sexe in ['H', 'M', 'HOMME'] else 0.0

    data['imc'] = float(input("  BMI (kg/m²): ") or "25")

    # Vital signs (defaults = normal values)
    print("\n--- Vital Signs (Enter = normal value) ---")
    data['freq_cardiaque'] = float(input("  Heart rate (bpm): ") or "75")
    data['temperature'] = float(input("  Temperature (°C): ") or "37.0")

    tension = input("  Blood pressure (ex: 120/80): ").strip()
    if '/' in tension:
        sys, dia = tension.split('/')
        data['tension_systolique'] = float(sys)
        data['tension_diastolique'] = float(dia)
    else:
        data['tension_systolique'] = 120.0
        data['tension_diastolique'] = 80.0

    # Symptoms (yes/no)
    print("\n--- Symptoms (y/N) ---")
    symptomes = ['asthenie', 'amaigrissement', 'oedeme', 'palleur', 'ictere',
                 'hepatomegalie', 'hepatalgie', 'ascite', 'exantheme']

    for symp in symptomes:
        label = COLUMN_MAP.get(symp, symp)
        rep = input(f"  {label}: ").strip().lower() if not necessary_only else "N"
        data[symp] = 1.0 if rep in ['o', 'oui', 'y', 'yes', '1'] else 0.0

    if necessary_only:
        print("  ")
        # Print English names for positive symptoms only
        symptom_names = {
            'asthenie': 'Asthenia',
            'amaigrissement': 'Weight loss', 
            'oedeme': 'Edema',
            'palleur': 'Pallor',
            'ictere': 'Jaundice',
            'hepatomegalie': 'Hepatomegaly',
            'hepatalgie': 'Hepatic pain',
            'ascite': 'Ascites',
            'exantheme': 'Rash'
        }
        
        for symp in symptomes:
            if data[symp] == 1.0:
                print(f"  {symptom_names[symp]}: {data[symp]}")


    # Rare clinical signs
    print("\n--- Advanced Clinical Signs (y/N) ---")
    col_ven = input("  Collateral circulation: ").lower() if not necessary_only else "N"
    data['circulation_veineuse'] = 1.0 if col_ven in ['o', 'oui', 'y'] else 0.0
    hip = input("  Hippocratic fingers: ").lower() if not necessary_only else "N"
    data['hippocratisme'] = 1.0 if hip in ['o', 'oui', 'y'] else 0.0
    et_consc = input("  Consciousness disorder: ").lower() if not necessary_only else "N"
    data['etat_conscience'] = 1.0 if et_consc in ['o', 'oui', 'y'] else 0.0
    data['stade_encephalopathie'] = 1.0 if data['etat_conscience'] > 0 else 0.0
    
    if necessary_only:
        
        print(f"  Collateral circulation: {col_ven}")
        print(f"  Hippocratic fingers: {hip}")
        print(f"  Consciousness disorder: {et_consc}")
        print(f"  Collateral circulation: {col_ven}")
        print(f"  Encephalopatic Stage: {data['stade_encephalopathie']}")

    # Risk factors
    print("\n--- Risk Factors ---")
    pvhb = input("  Hepatitis B (HBV): ").lower() if not necessary_only else "N"
    data['presence_vhb'] = 1.0 if pvhb in ['o', 'oui', 'y'] else 0.0

    return data



