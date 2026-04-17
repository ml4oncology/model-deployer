import math

def create_patient_overview(mrn, next_sched_trt, cancer, age, gender, risk_score, risk_level):
    """Create a patient info card"""

    # Determine risk badge color and icon
    # if risk_level.lower() == "high risk":
    #     badge_color = "#F14949"
    #     risk_icon = "⚠"
    # else:
    #     badge_color = "#00cc44"
    #     risk_icon = "✓"

    html_content = f"""
    <div class="overview-container">
        <div class="overview-header">
            AIM2REDUCE ED Risk Prediction
        </div>
        <div class="patient-info-grid">
            <div class="patient-info-section">
                <div class="patient-info-label">MRN</div>
                <div class="patient-info-value">{mrn}</div>
            </div>
            <div class="patient-info-section">
                <div class="patient-info-label">Next Scheduled Treatment</div>
                <div class="patient-info-value">{next_sched_trt}</div>
            </div>
            <div class="patient-info-section">
                <div class="patient-info-label">Primary Site</div>
                <div class="patient-info-value">{cancer}</div>
            </div>
            <div class="patient-info-section">
                <div class="patient-info-label">Age / Gender</div>
                <div class="patient-info-value">{age} / {gender}</div>
            </div>
            <div class="patient-info-section">
                <div class="patient-info-label">30d Probability of ED Visit</div>
                <div class="risk-score-container">
                    <span class="risk-score">{risk_score:.2f}</span>
                    <!-- <span class="risk-badge" style="background-color: [badge_color];">
                        <span class="risk-icon">[risk_icon]</span>
                        [risk_level]
                    </span> -->
                </div>
            </div>
        </div>
    </div>
    """

    return html_content


def create_model_overview():
    """Create a model info card"""
    html_content = """
    <div class="overview-container">
        <div class="overview-header">
            Model Overview
        </div>
        <div class="model-info-grid">
            <div class="model-info-section">
                <div class="model-info-label">Model Name/Version</div>
                <div class="model-info-value">AIM2REDUCE v1.0.0</div>
            </div>
            <div class="model-info-section">
                <div class="model-info-label">Date of Release</div>
                <div class="model-info-value">Oct 15, 2024</div>
            </div>
            <div class="model-info-section">
                <div class="model-info-label">Model Type</div>
                <div class="model-info-value">XGBoost</div>
            </div>
            <div class="model-info-section">
                <div class="model-info-label">Calibration</div>
                <div class="model-info-value">Isotonic Regression</div>
            </div>
            <div class="model-info-section">
                <div class="model-info-label">Dataset</div>
                <div class="model-info-value">
                    Train: 10,152 samples (1,927 GI pts, 2012-2018)<br>
                    Test: 1,082 samples (389 GI pts, 2018-2020)<br>
                    213 features
                </div>
            </div>
            <div class="model-info-section">
                <div class="model-info-label">Training Procedure</div>
                <div class="model-info-value">3-fold stratified cross-validation</div>
            </div>
            <div class="model-info-section">
                <div class="model-info-label">Test AUROC</div>
                <div class="model-info-value">0.73</div>
            </div>
        </div>
    </div>
    """
    # <div class="model-card-toggle">
        #     <input type="checkbox" id="show-hyper" style="display:none;">
        #     <label for="show-hyper">Show Hyperparameters</label>
        #     <div class="model-card-hyper">
        #         <ul>
        #             <li>n_estimators: 64</li>
        #             <li>max_depth: 6</li>
        #             <li>learning_rate: 0.3</li>
        #             <li>min_split_loss: 0.5</li>
        #             <li>min_child_weight: 21</li>
        #             <li>reg_lambda: 1.0</li>
        #             <li>reg_alpha: 30</li>
        #         </ul>
        #     </div>
        # </div>
    return html_content


def create_percentile_overview(p_all, p_same, cancer):
    def get_suffix(n):
        if 10 <= n <= 20:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    all_suffix = get_suffix(p_all)

    same_section = ""
    if not (isinstance(p_same, float) and math.isnan(p_same)):
        same_suffix = get_suffix(p_same)
        same_section = f"""
        <div class="percentile-info-section">
            <div class="patient-info-label">Risk Percentile - Same Diagnosis</div>
            <div class="percentile-info-value">{p_same}{same_suffix}</div>
            <div class="patient-info-label">Higher risk than {p_same}% of patients with {cancer.lower()} cancer</div>
        </div>"""

    return f"""
    <div class="percentile-info-grid">
        <div class="percentile-info-section">
            <div class="patient-info-label">Risk Percentile - All Patients</div>
            <div class="percentile-info-value">{p_all}{all_suffix}</div>
            <div class="patient-info-label">Higher risk than {p_all}% of all patients</div>
        </div>
        {same_section}
    </div>
    """