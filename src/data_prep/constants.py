DROP_CLINIC_COLUMNS = [
    'MRN','Lab Type', 'Collected Date', 'Result Date', 'Finalized Date', 
    'Last Update', 'Accession', 'Order ID','Specimen Source', 'Specimen Type',
    'Test Type', 'Lab Status', 'Agency','Organism', 'Comment', 'Narrative'
]


# TODO: automatically store this info in ml-common.prep
FILL_VALS = {
    'treatment': {'days_since_last_treatment': 4746, 'days_since_prev_ED_visit': 1822}, 
    'clinic': {'days_since_last_treatment': 28, 'days_since_prev_ED_visit': 1821}
}