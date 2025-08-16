import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import joblib
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.base import BaseEstimator, ClassifierMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('loan_model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Constants
NUMERIC_FEATURES = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'debt_to_income',
    'income_to_loan', 'emp_to_age'
]

CATEGORICAL_FEATURES = [
    'person_home_ownership', 'loan_intent',
    'loan_grade', 'cb_person_default_on_file'
]

def load_and_preprocess_data() -> pd.DataFrame:
    """Load and preprocess the loan dataset"""
    try:
        df = pd.read_csv('credit_risk_dataset.csv')
        logger.info("Data loaded successfully. Shape: %s", df.shape)
        
        # Data cleaning
        df = df[df['person_age'] <= 100]
        df = df[(df['person_income'] > 0) & (df['loan_amnt'] > 0)]
        
        # Create new features
        df = df.assign(
            debt_to_income=df['loan_amnt'] / (df['person_income'] + 1),
            income_to_loan=df['person_income'] / (df['loan_amnt'] + 1),
            emp_to_age=df['person_emp_length'] / (df['person_age'] + 1)
        )
        
        # Handle missing values
        df = df.assign(
            person_emp_length=df['person_emp_length'].fillna(df['person_emp_length'].median()),
            loan_int_rate=df['loan_int_rate'].fillna(df['loan_int_rate'].median())
        )
        
        # Convert target and reset index
        df = df.assign(loan_status=df['loan_status'].astype(int))
        df = df.reset_index(drop=True)
        
        logger.info("Data after preprocessing. Shape: %s", df.shape)
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                  loan_amounts: pd.Series, model_name: str) -> Dict[str, Any]:
    """Generate comprehensive evaluation metrics"""
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        optimal_threshold = find_optimal_threshold(y_test, y_proba, loan_amounts)
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'optimal_threshold': optimal_threshold,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'business_metrics': calculate_business_metrics(y_test, y_pred, loan_amounts),
            'feature_importances': get_feature_importances(model, X_test.columns) if hasattr(model, 'feature_importances_') else None
        }
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
        
        return metrics
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

def find_optimal_threshold(y_true: pd.Series, y_proba: np.ndarray, 
                          loan_amounts: pd.Series) -> float:
    """Find threshold that maximizes expected profit"""
    thresholds = np.linspace(0.1, 0.9, 50)
    profits = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        profit = calculate_profit(y_true, y_pred, loan_amounts)
        profits.append(profit)
        
    optimal_idx = np.argmax(profits)
    return thresholds[optimal_idx]

def calculate_profit(y_true: pd.Series, y_pred: pd.Series, 
                    loan_amounts: pd.Series) -> float:
    """Calculate expected profit from loan decisions"""
    if len(y_true) == 0 or len(loan_amounts) == 0:
        return 0.0
        
    approved_mask = (y_pred == 0)  # 0 is approved
    good_loans = (y_true == 0) & approved_mask
    bad_loans = (y_true == 1) & approved_mask
    
    profit = loan_amounts[good_loans].sum() * 0.1 if good_loans.any() else 0
    loss = loan_amounts[bad_loans].sum() if bad_loans.any() else 0
    
    return profit - loss

def calculate_business_metrics(y_true: pd.Series, y_pred: pd.Series, 
                             loan_amounts: pd.Series) -> Dict[str, float]:
    """Calculate business-oriented metrics"""
    approved_mask = (y_pred == 0)
    profit = calculate_profit(y_true, y_pred, loan_amounts)
    
    return {
        'expected_profit': profit,
        'approval_rate': np.mean(approved_mask),
        'default_rate': np.mean(y_true[approved_mask] == 1) if approved_mask.any() else 0,
        'rejection_rate': np.mean(y_pred == 1)
    }

def get_feature_importances(model, feature_names):
    """Extract feature importances from model"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        return dict(zip(feature_names, model.coef_[0]))
    return None

def train_enhanced_model() -> Dict[str, Any]:
    """Main training function with 5-model ensemble"""
    try:
        logger.info("Starting enhanced model training pipeline...")
        df = load_and_preprocess_data()
        
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']
        loan_amounts = pd.Series(df['loan_amnt'].values, index=df.index)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), NUMERIC_FEATURES),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), CATEGORICAL_FEATURES)
            ],
            verbose_feature_names_out=False,
            remainder='drop'
        )
        
        models = {
            'RandomForest': make_imb_pipeline(
                preprocessor,
                ADASYN(random_state=42),
                RandomForestClassifier(class_weight={0:1, 1:3}, random_state=42, n_jobs=-1)
            ),
            'XGBoost': make_imb_pipeline(
                preprocessor,
                ADASYN(random_state=42),
                XGBClassifier(scale_pos_weight=3, eval_metric='aucpr', random_state=42, n_jobs=-1)
            ),
            'LightGBM': make_imb_pipeline(
                preprocessor,
                ADASYN(random_state=42),
                LGBMClassifier(class_weight={0:1, 1:3}, random_state=42, n_jobs=-1, verbose=-1)
            ),
            'GradientBoosting': make_imb_pipeline(
                preprocessor,
                ADASYN(random_state=42),
                GradientBoostingClassifier(random_state=42)
            ),
            'LogisticRegression': make_imb_pipeline(
                preprocessor,
                ADASYN(random_state=42),
                LogisticRegression(class_weight={0:1, 1:3}, random_state=42, max_iter=1000, n_jobs=-1)
            )
        }
        
        model_metrics = {}
        successful_models = {}
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test, loan_amounts.loc[y_test.index], name)
                model_metrics[name] = metrics
                successful_models[name] = model
                logger.info(f"Successfully trained {name}")
            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
        
        if len(successful_models) >= 2:
            try:
                stacked_model = StackingClassifier(
                    estimators=[(name, model) for name, model in successful_models.items()],
                    final_estimator=LogisticRegression(),
                    cv=5,
                    n_jobs=-1
                )
                logger.info("Training StackedEnsemble...")
                stacked_model.fit(X_train, y_train)
                metrics = evaluate_model(stacked_model, X_test, y_test, loan_amounts.loc[y_test.index], 'StackedEnsemble')
                model_metrics['StackedEnsemble'] = metrics
                successful_models['StackedEnsemble'] = stacked_model
                logger.info("Successfully trained StackedEnsemble")
            except Exception as e:
                logger.error(f"Failed to train StackedEnsemble: {str(e)}")
            
            try:
                voting_model = VotingClassifier(
                    estimators=[(name, model) for name, model in successful_models.items() if name != 'StackedEnsemble'],
                    voting='soft',
                    n_jobs=-1
                )
                logger.info("Training VotingClassifier...")
                voting_model.fit(X_train, y_train)
                metrics = evaluate_model(voting_model, X_test, y_test, loan_amounts.loc[y_test.index], 'VotingClassifier')
                model_metrics['VotingClassifier'] = metrics
                successful_models['VotingClassifier'] = voting_model
                logger.info("Successfully trained VotingClassifier")
            except Exception as e:
                logger.error(f"Failed to train VotingClassifier: {str(e)}")
        
        if not model_metrics:
            raise ValueError("No models were successfully trained")
        
        best_model_name = max(model_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
        best_model = ThresholdClassifier(
            successful_models[best_model_name],
            model_metrics[best_model_name]['optimal_threshold']
        )
        
        try:
            if hasattr(best_model.model, 'named_steps'):
                X_test_transformed = best_model.model.named_steps['columntransformer'].transform(X_test)
                explainer = shap.Explainer(best_model.model.named_steps[list(best_model.model.named_steps.keys())[-1]], X_test_transformed)
                shap_values = explainer(X_test_transformed)
                shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", show=False)
                plt.savefig('feature_importance.png')
                plt.close()
            else:
                explainer = None
        except Exception as e:
            logger.error(f"Could not create SHAP explanation: {str(e)}")
            explainer = None
        
        model_package = {
            'pipeline': best_model,
            'all_models': {k: ThresholdClassifier(v, model_metrics[k]['optimal_threshold']) 
                          for k, v in successful_models.items()},
            'metrics': model_metrics,
            'best_model': best_model_name,
            'feature_names': NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            'class_distribution': dict(y.value_counts()),
            'timestamp': pd.Timestamp.now(),
            'shap_explainer': explainer
        }
        
        joblib.dump(model_package, 'loan_model.pkl')
        logger.info(f"Model training completed. Best model: {best_model_name}")
        
        return model_package
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    train_enhanced_model()