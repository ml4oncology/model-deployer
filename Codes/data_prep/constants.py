demo_cols = [
    'height',
    'weight',
    'body_surface_area',
]
symp_cols = [
    'esas_pain',
    'esas_tiredness',
    'esas_nausea',
    'esas_depression',
    'esas_anxiety',
    'esas_drowsiness',
    'esas_appetite',
    'esas_well_being',
    'esas_shortness_of_breath',
    'patient_ecog',
]
lab_cols = [
    'alanine_aminotransferase',
    'albumin',
    'alkaline_phosphatase',
    'aspartate_aminotransferase',
    'basophil',
    'bicarbonate',
    'chloride',
    'creatinine',
    'eosinophil',
    'glucose',
    'hematocrit',
    'hemoglobin',
    'lactate_dehydrogenase',
    'lymphocyte',
    'magnesium',
    'mean_corpuscular_hemoglobin',
    'mean_corpuscular_hemoglobin_concentration',
    'mean_corpuscular_volume',
    'mean_platelet_volume',
    'monocyte',
    'neutrophil',
    'phosphate',
    'platelet',
    'potassium',
    'red_blood_cell',
    'red_cell_distribution_width',
    'sodium',
    'total_bilirubin',
    'white_blood_cell',
]
symp_change_cols = [f'{col}_change' for col in symp_cols]
lab_change_cols = [f'{col}_change' for col in lab_cols]
DROP_CLINIC_COLUMNS = [
    'MRN','Lab Type', 'Collected Date', 'Result Date', 'Finalized Date', 
    'Last Update', 'Accession', 'Order ID','Specimen Source', 'Specimen Type',
    'Test Type', 'Lab Status', 'Agency','Organism', 'Comment', 'Narrative'
]
