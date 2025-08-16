import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from streamlit_lottie import st_lottie
import requests
import matplotlib
matplotlib.use('Agg')
from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper to store and apply optimal threshold"""
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        
    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def __getattr__(self, attr):
        return getattr(self.model, attr)

# Configure page
st.set_page_config(
    page_title="Loan Approval Predictor Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .approved { background-color: #e6f7e6; color: #2d572c; border-left: 5px solid #4CAF50; padding: 20px; border-radius: 5px; margin: 10px 0; }
    .rejected { background-color: #ffebee; color: #b71c1c; border-left: 5px solid #F44336; padding: 20px; border-radius: 5px; margin: 10px 0; }
    .grade-A { color: #006400; font-weight: bold; }
    .grade-B { color: #228B22; font-weight: bold; }
    .grade-C { color: #FFD700; font-weight: bold; }
    .grade-D { color: #FFA500; font-weight: bold; }
    .grade-E { color: #FF8C00; font-weight: bold; }
    .grade-F { color: #FF4500; font-weight: bold; }
    .grade-G { color: #FF0000; font-weight: bold; }
    .feature-importance { margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
    .stButton>button { width: 100%; }
    .history-table { margin-top: 20px; }
    .tips-section { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-top: 20px; }
    .what-if-box { border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-top: 15px; background-color: #f9f9f9; }
    .health-meter { padding: 15px; border-radius: 10px; margin-bottom: 15px; background: linear-gradient(90deg, #ff4d4d 0%, #ffcc00 50%, #4CAF50 100%); color: white; text-align: center; }
    .health-score { font-size: 24px; font-weight: bold; margin: 10px 0; }
    .health-status { font-size: 18px; margin-bottom: 10px; }
    .eligibility-box { margin: 10px 0; padding: 15px; border-radius: 10px; background-color: #f8f9fa; }
    .criteria-met { color: #4CAF50; }
    .criteria-not-met { color: #F44336; }
    .quick-check { border-left: 5px solid #4a6fa5; padding: 15px; margin-bottom: 15px; }
    .input-error { color: #ff0000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Loan grade information
GRADE_INFO = {
    "A": {"risk": "Lowest", "rate": "5-8%", "tip": "Excellent credit (700+), high income, long credit history"},
    "B": {"risk": "Low", "rate": "8-10%", "tip": "Good credit (650-700), stable income"},
    "C": {"risk": "Medium", "rate": "10-12%", "tip": "Fair credit (600-650), moderate income"},
    "D": {"risk": "High", "rate": "12-15%", "tip": "Below-average credit (550-600)"},
    "E": {"risk": "Very High", "rate": "15-20%", "tip": "Poor credit (500-550), limited history"},
    "F": {"risk": "Severe", "rate": "20-25%", "tip": "Very poor credit (450-500)"},
    "G": {"risk": "Extreme", "rate": "25-30%", "tip": "Worst credit (<450), recent defaults"}
}

# Constants for validation
MAX_INCOME = 10_000_000  # $10 million
MAX_LOAN = 5_000_000     # $5 million
MAX_AGE = 100
MAX_EMPLOYMENT_YEARS = 50
MAX_CREDIT_HISTORY = 50

def validate_inputs(input_data):
    """Validate all input fields for reasonable ranges"""
    errors = []
    
    if not (18 <= input_data['person_age'] <= MAX_AGE):
        errors.append(f"Age must be between 18 and {MAX_AGE}")
    
    if not (10_000 <= input_data['person_income'] <= MAX_INCOME):
        errors.append(f"Income must be between $10,000 and ${MAX_INCOME:,}")
    
    if not (0 <= input_data['person_emp_length'] <= MAX_EMPLOYMENT_YEARS):
        errors.append(f"Employment years must be between 0 and {MAX_EMPLOYMENT_YEARS}")
    
    if not (500 <= input_data['loan_amnt'] <= MAX_LOAN):
        errors.append(f"Loan amount must be between $500 and ${MAX_LOAN:,}")
    
    if not (0 <= input_data['loan_int_rate'] <= 30):
        errors.append("Interest rate must be between 0% and 30%")
    
    if not (0 <= input_data['loan_percent_income'] <= 1):
        errors.append("Loan-to-income ratio must be between 0 and 1")
    
    if not (0 <= input_data['cb_person_cred_hist_length'] <= MAX_CREDIT_HISTORY):
        errors.append(f"Credit history must be between 0 and {MAX_CREDIT_HISTORY} years")
    
    # Validate loan-to-income ratio matches actual calculation
    calculated_ratio = input_data['loan_amnt'] / (input_data['person_income'] + 1)
    if abs(calculated_ratio - input_data['loan_percent_income']) > 0.01:
        errors.append("Loan-to-income ratio doesn't match actual calculation")
    
    return errors

def calculate_financial_health(applicant_data):
    """Calculate financial health score (0-100) based on applicant information"""
    score = 0
    
    # Income factors (max 30 points)
    income_ratio = applicant_data['person_income'] / (applicant_data['loan_amnt'] + 1)
    if income_ratio > 0.5:
        score += 30
    elif income_ratio > 0.3:
        score += 20
    elif income_ratio > 0.2:
        score += 10
        
    # Credit history factors (max 25 points)
    if applicant_data['cb_person_default_on_file'] == 'N':
        score += 15
    if applicant_data['cb_person_cred_hist_length'] > 7:
        score += 10
    elif applicant_data['cb_person_cred_hist_length'] > 3:
        score += 5
        
    # Employment stability (max 20 points)
    if applicant_data['person_emp_length'] > 5:
        score += 20
    elif applicant_data['person_emp_length'] > 2:
        score += 10
        
    # Age factor (max 10 points)
    if 25 <= applicant_data['person_age'] <= 60:
        score += 10
        
    # Home ownership (max 10 points)
    if applicant_data['person_home_ownership'] == 'OWN':
        score += 10
    elif applicant_data['person_home_ownership'] == 'MORTGAGE':
        score += 5
        
    # Loan purpose (max 5 points)
    if applicant_data['loan_intent'] in ['EDUCATION', 'HOMEIMPROVEMENT']:
        score += 5
        
    return min(score, 100)

def get_health_status(score):
    """Get health status based on score"""
    if score >= 80: return "Excellent", "#4CAF50"
    elif score >= 60: return "Good", "#8BC34A"
    elif score >= 40: return "Fair", "#FFC107"
    elif score >= 20: return "Poor", "#FF9800"
    else: return "Critical", "#F44336"

def calculate_loan_grade(applicant_data):
    """Calculate loan grade based on applicant information"""
    score = 0
    
    # Credit score factors
    if applicant_data['cb_person_default_on_file'] == 'N':
        score += 30
    else:
        score -= 20
        
    # Income-to-loan ratio
    income_ratio = applicant_data['person_income'] / (applicant_data['loan_amnt'] + 1)
    if income_ratio > 0.5:
        score += 25
    elif income_ratio > 0.3:
        score += 15
    elif income_ratio > 0.2:
        score += 5
        
    # Employment history
    if applicant_data['person_emp_length'] > 5:
        score += 15
    elif applicant_data['person_emp_length'] > 2:
        score += 5
        
    # Credit history length
    if applicant_data['cb_person_cred_hist_length'] > 7:
        score += 20
    elif applicant_data['cb_person_cred_hist_length'] > 3:
        score += 10
        
    # Age factor
    if 25 <= applicant_data['person_age'] <= 60:
        score += 10
        
    # Home ownership bonus
    if applicant_data['person_home_ownership'] == 'OWN':
        score += 10
    elif applicant_data['person_home_ownership'] == 'MORTGAGE':
        score += 5
        
    # Loan intent adjustment
    if applicant_data['loan_intent'] in ['EDUCATION', 'HOMEIMPROVEMENT']:
        score += 5
    elif applicant_data['loan_intent'] == 'DEBTCONSOLIDATION':
        score -= 5
        
    # Determine grade based on total score
    if score >= 90: return "A"
    elif score >= 80: return "B"
    elif score >= 65: return "C"
    elif score >= 50: return "D"
    elif score >= 35: return "E"
    elif score >= 20: return "F"
    else: return "G"

def load_lottie_url(url: str):
    """Load Lottie animation from URL with proper error handling"""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

APPROVED_ANIMATION = "https://assets1.lottiefiles.com/packages/lf20_7KXJtD.json"
REJECTED_ANIMATION = "https://assets1.lottiefiles.com/packages/lf20_kcsr6f2n.json"

@st.cache_resource
def load_model():
    """Load trained model package with error handling"""
    try:
        import sys
        import types
        module = types.ModuleType('threshold_module')
        module.ThresholdClassifier = ThresholdClassifier
        sys.modules['threshold_module'] = module
        
        model_package = joblib.load('loan_model.pkl')
        
        required_keys = ['pipeline', 'metrics', 'best_model', 'feature_names']
        for key in required_keys:
            if key not in model_package:
                st.error(f"Model package is missing required key: {key}")
                st.stop()
                
        best_model_metrics = model_package['metrics'].get(model_package['best_model'], {})
        if 'optimal_threshold' not in best_model_metrics:
            st.error("Model is missing optimal threshold information")
            st.stop()
            
        return model_package
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def predict_loan_approval(input_data, model_package):
    """Make prediction with the trained model"""
    try:
        # First validate inputs
        validation_errors = validate_inputs(input_data)
        if validation_errors:
            return {
                'approved': False,
                'error': True,
                'error_messages': validation_errors,
                'grade': 'G'
            }
        
        pipeline = model_package['pipeline']
        threshold = model_package['metrics'][model_package['best_model']]['optimal_threshold']
        
        required_features = model_package['feature_names']
        missing_features = [f for f in required_features if f not in input_data]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        input_df = pd.DataFrame([input_data])[required_features]
        proba = pipeline.predict_proba(input_df)[0][1]
        
        return {
            'approved': proba < threshold,
            'probability': proba,
            'threshold': threshold,
            'input_features': input_df.iloc[0].to_dict(),
            'grade': input_data.get('loan_grade', 'C'),
            'error': False
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def display_grade_info(grade):
    """Display information about the selected loan grade"""
    info = GRADE_INFO.get(grade, {})
    with st.container():
        st.markdown(f"""
        **Risk Level:** <span class="grade-{grade}">{info.get('risk', 'Unknown')}</span>  
        **Typical Interest Rate:** {info.get('rate', 'N/A')}  
        **Approval Tips:** {info.get('tip', '')}
        """, unsafe_allow_html=True)
        
        st.progress((7 - ["A","B","C","D","E","F","G"].index(grade))/7)
        
        st.markdown("""
        **To improve your loan grade:**
        - Increase your credit score
        - Reduce existing debt
        - Maintain longer credit history
        - Show stable employment
        """)

def what_if_calculator():
    """Interactive calculator to show how changes affect approval"""
    with st.container():
        st.write("Test how different scenarios would affect your approval chances:")
        
        col1, col2 = st.columns(2)
        with col1:
            new_income = st.number_input("Alternative Income ($)", 
                                      min_value=10_000, max_value=MAX_INCOME, value=50_000, step=1000)
            new_loan = st.number_input("Alternative Loan Amount ($)", 
                                    min_value=500, max_value=MAX_LOAN, value=10_000, step=500)
            new_credit = st.number_input("Alternative Credit History (years)", 
                                      min_value=0, max_value=MAX_CREDIT_HISTORY, value=3)
        
        with col2:
            new_default = st.selectbox("Alternative Default Status", ["N", "Y"])
            new_home = st.selectbox("Alternative Housing Status", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            new_intent = st.selectbox("Alternative Loan Purpose", [
                "PERSONAL", "EDUCATION", "MEDICAL", 
                "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
            ])
            new_age = st.number_input("Alternative Age", min_value=18, max_value=MAX_AGE, value=35)
        
        if st.button("Calculate This Scenario"):
            with st.spinner('Calculating...'):
                safe_income = max(new_income, 1)
                hypothetical_data = {
                    'person_age': new_age,
                    'person_income': safe_income,
                    'person_emp_length': 5,
                    'loan_amnt': new_loan,
                    'loan_int_rate': 10.0,
                    'loan_percent_income': new_loan / safe_income,
                    'cb_person_cred_hist_length': new_credit,
                    'cb_person_default_on_file': new_default,
                    'person_home_ownership': new_home,
                    'loan_intent': new_intent,
                    'debt_to_income': new_loan / safe_income,
                    'income_to_loan': safe_income / (new_loan + 1),
                    'emp_to_age': 5 / (new_age + 1)
                }
                
                # Validate inputs first
                validation_errors = validate_inputs(hypothetical_data)
                if validation_errors:
                    st.error("Invalid scenario inputs:")
                    for error in validation_errors:
                        st.error(f"- {error}")
                    return
                
                hypothetical_data['loan_grade'] = calculate_loan_grade(hypothetical_data)
                result = predict_loan_approval(hypothetical_data, model_package)
                
                if result:
                    st.markdown(f"""
                    <div class="what-if-box">
                        <h4>Scenario Results</h4>
                        <p><strong>Approval Status:</strong> {'Approved' if result['approved'] else 'Rejected'}</p>
                        <p><strong>Approval Probability:</strong> {(1 - result['probability']) * 100:.1f}%</p>
                        <p><strong>Risk Score:</strong> {result['probability']:.1%}</p>
                        <p><strong>Loan Grade:</strong> <span class="grade-{result['grade']}">{result['grade']}</span></p>
                        <p><strong>Key Factors:</strong></p>
                        <ul>
                            <li>Income: ${safe_income:,.0f}</li>
                            <li>Loan Amount: ${new_loan:,.0f}</li>
                            <li>Loan-to-Income Ratio: {(new_loan/safe_income)*100:.1f}%</li>
                            <li>Credit History: {new_credit} years</li>
                            <li>Default Status: {'Yes' if new_default == 'Y' else 'No'}</li>
                            <li>Housing: {new_home}</li>
                            <li>Purpose: {new_intent}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

def display_application_form():
    """Display the loan application form with enhancements"""
    with st.form("loan_form"):
        st.header("📝 Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=MAX_AGE, value=35)
            person_income = st.number_input("Annual Income ($)", min_value=10_000, max_value=MAX_INCOME, value=75_000, step=1000)
            person_emp_length = st.number_input("Employment History (years)", min_value=0, max_value=MAX_EMPLOYMENT_YEARS, value=5)
            loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=MAX_LOAN, value=15_000, step=1000)
            
        with col2:
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=MAX_CREDIT_HISTORY, value=5)
            loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=8.5, step=0.1)
            
            # Calculate loan-to-income ratio automatically
            if 'person_income' in st.session_state and 'loan_amnt' in st.session_state:
                calculated_ratio = st.session_state.loan_amnt / (st.session_state.person_income + 1)
                loan_percent_income = st.number_input("Loan-to-Income Ratio", 
                                                    min_value=0.0, max_value=1.0, 
                                                    value=min(calculated_ratio, 1.0), 
                                                    step=0.01,
                                                    disabled=True)
            else:
                loan_percent_income = st.number_input("Loan-to-Income Ratio", 
                                                    min_value=0.0, max_value=1.0, 
                                                    value=0.2, step=0.01)
            
            cb_person_default_on_file = st.selectbox("Previous Defaults", ["N", "Y"])
        
        st.header("ℹ️ Additional Details")
        col3, col4 = st.columns(2)
        
        with col3:
            person_home_ownership = st.selectbox("Housing Status", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            loan_intent = st.selectbox("Loan Purpose", [
                "PERSONAL", "EDUCATION", "MEDICAL", 
                "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
            ])

            # Loan Eligibility Quick Check
            with st.container():
                st.write("Get instant feedback on your loan eligibility:")
                
                eligibility_met = 0
                total_criteria = 5
                
                crit1 = person_income >= 30_000
                crit2 = loan_amnt / (person_income + 1) <= 0.5
                crit3 = cb_person_cred_hist_length >= 2
                crit4 = cb_person_default_on_file == "N"
                crit5 = person_emp_length >= 1
                
                if crit1: eligibility_met += 1
                if crit2: eligibility_met += 1
                if crit3: eligibility_met += 1
                if crit4: eligibility_met += 1
                if crit5: eligibility_met += 1
                
                eligibility_score = (eligibility_met / total_criteria) * 100
                status_color = "#4CAF50" if eligibility_score >= 80 else "#FFC107" if eligibility_score >= 50 else "#F44336"
                status_text = "Strong" if eligibility_score >= 80 else "Moderate" if eligibility_score >= 50 else "Weak"
                
                st.markdown(f"""
                <div class="quick-check">
                    <h4>Eligibility Score: <span style="color: {status_color}">{eligibility_score:.0f}%</span></h4>
                    <p>Status: <strong>{status_text}</strong></p>
                    <progress value="{eligibility_score}" max="100" style="width: 100%; height: 10px;"></progress>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Criteria Details"):
                    st.markdown(f"""
                    - <span class="{'criteria-met' if crit1 else 'criteria-not-met'}">Income ≥ $30,000</span>
                    - <span class="{'criteria-met' if crit2 else 'criteria-not-met'}">Loan ≤ 50% of income</span>
                    - <span class="{'criteria-met' if crit3 else 'criteria-not-met'}">Credit history ≥ 2 years</span>
                    - <span class="{'criteria-met' if crit4 else 'criteria-not-met'}">No previous defaults</span>
                    - <span class="{'criteria-met' if crit5 else 'criteria-not-met'}">Employment ≥ 1 year</span>
                    """, unsafe_allow_html=True)

        with col4:
            health_data = {
                'person_age': person_age,
                'person_income': person_income,
                'person_emp_length': person_emp_length,
                'loan_amnt': loan_amnt,
                'cb_person_cred_hist_length': cb_person_cred_hist_length,
                'cb_person_default_on_file': cb_person_default_on_file,
                'person_home_ownership': person_home_ownership,
                'loan_intent': loan_intent
            }
            
            health_score = calculate_financial_health(health_data)
            health_status, health_color = get_health_status(health_score)
            
            st.markdown(f"""
            <div class="health-meter" style="border-left: 5px solid {health_color}">
                <h3>Financial Health Meter</h3>
                <div class="health-score">{health_score}/100</div>
                <div class="health-status">Status: {health_status}</div>
                <progress value="{health_score}" max="100" style="width: 100%; height: 10px;"></progress>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("💡 How to improve your financial health"):
                tips = []
                
                if health_score < 80:
                    if loan_amnt / (person_income + 1) > 0.3:
                        tips.append("Consider borrowing less relative to your income")
                    if cb_person_cred_hist_length < 3:
                        tips.append("Build longer credit history by maintaining accounts")
                    if cb_person_default_on_file == "Y":
                        tips.append("Avoid any new defaults on payments")
                    if person_emp_length < 2:
                        tips.append("Longer employment history improves stability")
                    if not tips:
                        tips.append("Your financial health is already good!")
                
                for tip in tips:
                    st.write(f"- {tip}")
            
            with st.expander("💰 Payment Preview"):
                loan_term = st.selectbox("Loan Term (years)", [1, 2, 3, 5, 7, 10], index=2)
                if loan_amnt > 0 and loan_int_rate > 0:
                    monthly_rate = loan_int_rate / 100 / 12
                    num_payments = loan_term * 12
                    monthly_payment = (loan_amnt * monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
                    st.metric("Estimated Monthly Payment", f"${monthly_payment:,.2f}")
                    st.caption(f"Total interest: ${(monthly_payment * num_payments - loan_amnt):,.2f}")
                else:
                    st.warning("Enter loan amount and interest rate to see payment estimate")
        
        submitted = st.form_submit_button("✅ Check Eligibility")
    
    if submitted:
        # Store current inputs in session state
        st.session_state.person_income = person_income
        st.session_state.loan_amnt = loan_amnt
        
        # Calculate derived features with safe division
        safe_income = max(person_income, 1)
        debt_to_income = loan_amnt / safe_income
        income_to_loan = safe_income / (loan_amnt + 1)
        emp_to_age = person_emp_length / (person_age + 1)
        
        applicant_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_emp_length': person_emp_length,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_amnt / safe_income,  # Use calculated ratio
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'cb_person_default_on_file': cb_person_default_on_file,
            'person_home_ownership': person_home_ownership,
            'loan_intent': loan_intent,
            'debt_to_income': debt_to_income,
            'income_to_loan': income_to_loan,
            'emp_to_age': emp_to_age,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'financial_health_score': health_score,
            'eligibility_score': eligibility_score
        }
        
        applicant_data['loan_grade'] = calculate_loan_grade(applicant_data)
        return applicant_data
    return None

def display_decision(result):
    """Display the loan decision with enhanced feedback"""
    if not result:
        return
        
    if result.get('error', False):
        st.error("**Invalid Inputs Detected:**")
        for error in result.get('error_messages', []):
            st.error(f"- {error}")
        st.markdown(
            f'''<div class="rejected">
                <h2>❌ APPLICATION ERROR</h2>
                <p>Please correct the input errors and try again.</p>
            </div>''',
            unsafe_allow_html=True
        )
        return
        
    grade_class = f"grade-{result['grade']}"
    grade_info = GRADE_INFO.get(result['grade'], {})
    
    if result['approved']:
        approved_anim = load_lottie_url(APPROVED_ANIMATION)
        if approved_anim:
            st_lottie(approved_anim, height=200, key="approved")
        else:
            st.balloons()
        
        st.markdown(
            f'''<div class="approved">
                <h2>🎉 LOAN APPROVED!</h2>
                <p>Congratulations! Your loan application has been approved.</p>
                <p><strong>Risk Score:</strong> {result['probability']:.1%}</p>
                <p><strong>Loan Grade:</strong> <span class="{grade_class}">{result['grade']}</span> ({grade_info.get('risk', '')} Risk)</p>
            </div>''',
            unsafe_allow_html=True
        )
        
        st.success(f"""
        **Approval Details:**
        - Your {result['grade']} grade indicates {grade_info.get('risk', '')} risk
        - Estimated interest rate: {grade_info.get('rate', 'N/A')}
        - Strong financial profile with good repayment capacity
        """)
        
        with st.expander("🔄 Next Steps"):
            st.write("""
            1. **Documentation:** You'll receive approval documents within 2 business days
            2. **Funding:** Funds will be disbursed within 5 business days
            3. **Repayment:** Access your schedule in our mobile app
            """)
            
    else:
        rejected_anim = load_lottie_url(REJECTED_ANIMATION)
        if rejected_anim:
            st_lottie(rejected_anim, height=200, key="rejected")
        else:
            st.snow()
        
        st.markdown(
            f'''<div class="rejected">
                <h2>❌ LOAN REJECTED</h2>
                <p>We regret to inform you that your application wasn't approved.</p>
                <p><strong>Risk Score:</strong> {result['probability']:.1%}</p>
                <p><strong>Loan Grade:</strong> <span class="{grade_class}">{result['grade']}</span> ({grade_info.get('risk', '')} Risk)</p>
            </div>''',
            unsafe_allow_html=True
        )
        
        st.error(f"""
        **Grade-Specific Reasons:**
        - {grade_info.get('tip', 'General risk factors present')}
        
        **Key Improvement Areas:**
        1. **Credit Score:** {'' if result['grade'] in ['A','B'] else 'Needs improvement'}
        2. **Income vs Loan:** ${result['input_features']['person_income']:,.0f} income vs ${result['input_features']['loan_amnt']:,.0f} loan
        3. **Credit History:** {result['input_features']['cb_person_cred_hist_length']} years
        """)

        display_grade_info(result['grade'])

def display_application_history():
    """Display previous application history"""
    if 'applications' not in st.session_state or len(st.session_state.applications) == 0:
        return
        
    st.header("📋 Application History")
    
    history_df = pd.DataFrame(st.session_state.applications)
    display_cols = ['timestamp', 'loan_amnt', 'person_income', 'loan_grade', 'financial_health_score', 'eligibility_score', 'decision']
    
    history_df['loan_amnt'] = history_df['loan_amnt'].apply(lambda x: f"${x:,.0f}")
    history_df['person_income'] = history_df['person_income'].apply(lambda x: f"${x:,.0f}")
    
    def color_grade(grade):
        return f"<span class='grade-{grade}'>{grade}</span>"
    history_df['loan_grade'] = history_df['loan_grade'].apply(color_grade)
    
    def color_health(score):
        status, color = get_health_status(score)
        return f"<span style='color: {color}'>{score} ({status})</span>"
    history_df['financial_health_score'] = history_df['financial_health_score'].apply(color_health)
    
    def color_eligibility(score):
        color = "#4CAF50" if score >= 80 else "#FFC107" if score >= 50 else "#F44336"
        return f"<span style='color: {color}'>{score:.0f}%</span>"
    history_df['eligibility_score'] = history_df['eligibility_score'].apply(color_eligibility)
    
    st.write(history_df[display_cols].to_html(escape=False, index=False), unsafe_allow_html=True)
    
    if st.button("Clear History"):
        st.session_state.applications = []
        st.rerun()

def main():
    """Main application function"""
    global model_package
    model_package = load_model()
    
    st.title("🏦 Smart Loan Approval System Pro")
    st.markdown("""
    Get instant loan decisions with detailed feedback and improvement tips. 
    Complete the form below to check your eligibility.
    """)
    
    with st.sidebar:
        st.header("💡 Loan Grade Guide")
        for grade, info in GRADE_INFO.items():
            st.markdown(f"""
            <span class="grade-{grade}">Grade {grade}</span>: {info['risk']} Risk ({info['rate']})
            """, unsafe_allow_html=True)
        
        st.header("📞 Support")
        st.write("""
        Questions? Contact us:
        - Email: support@loanapp.com
        - Phone: 1800-91011-5077
        """)
        
        what_if_calculator()
    
    applicant_data = display_application_form()
    
    if applicant_data:
        with st.spinner('Analyzing your application...'):
            result = predict_loan_approval(applicant_data, model_package)
            
            if result:
                grade_class = f"grade-{result['grade']}"
                health_status, health_color = get_health_status(applicant_data['financial_health_score'])
                
                st.markdown(f"""
                <div style="margin: 20px 0; padding: 15px; border-left: 4px solid #4a6fa5; background-color: #f8f9fa;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <h3>Your Calculated Loan Grade: <span class="{grade_class}">{result['grade']}</span></h3>
                            <p>{GRADE_INFO.get(result['grade'], {}).get('tip', '')}</p>
                        </div>
                        <div style="text-align: right;">
                            <h3>Financial Health: <span style="color: {health_color}">{health_status}</span></h3>
                            <p>Score: {applicant_data['financial_health_score']}/100</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                display_decision(result)
                
                if not result.get('error', False):
                    if 'applications' not in st.session_state:
                        st.session_state.applications = []
                    
                    app_record = {
                        **applicant_data,
                        'risk_score': result['probability'],
                        'decision': 'Approved' if result['approved'] else 'Rejected'
                    }
                    st.session_state.applications.append(app_record)
    
    display_application_history()

if __name__ == "__main__":
    main()