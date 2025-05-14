"""
Utilities for loading and processing heart disease data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_heart_disease_data():
    """
    Load the Framingham Heart Study dataset
    """
    try:
        # Load the dataset
        df = pd.read_csv('attached_assets/framingham_heart_study.csv')
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_data(df):
    """
    Clean the heart disease dataset by handling missing values and converting data types
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying the original dataframe
    df_cleaned = df.copy()
    
    # Convert target column to int
    df_cleaned['has_heart_disease'] = df_cleaned['has_heart_disease'].astype(int)
    
    # Handle missing values
    # For numeric columns: replace with median
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # For categorical columns: replace with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    # Convert education to category if it exists
    if 'education' in df_cleaned.columns:
        df_cleaned['education'] = df_cleaned['education'].fillna(0).astype(float)
    
    return df_cleaned

def preprocess_data(df, target_column='has_heart_disease'):
    """
    Preprocess the data for machine learning models:
    - Split into features and target
    - Split into train and test sets
    - Scale features
    """
    if df is None:
        return None, None, None, None, None, None
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Get feature names for later use
    feature_names = X.columns.tolist()
    
    # Split into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def get_dataset_description():
    """
    Return description of the Framingham Heart Study dataset
    """
    description = {
        "en": """
        ## Framingham Heart Study Dataset
        
        The Framingham Heart Study is a long-term, ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. 
        The study began in 1948 with 5,209 adult subjects from Framingham, and is now on its third generation of participants.
        
        ### Dataset Features:
        
        * **male**: Gender (1 = male, 0 = female)
        * **age**: Age at examination (years)
        * **education**: Education level (1-4)
        * **currentSmoker**: Current smoking status (1 = yes, 0 = no)
        * **cigsPerDay**: Number of cigarettes smoked per day
        * **BPMeds**: Whether the patient was on blood pressure medication (1 = yes, 0 = no)
        * **prevalentStroke**: Whether the patient had previously had a stroke (1 = yes, 0 = no)
        * **prevalentHyp**: Whether the patient was hypertensive (1 = yes, 0 = no)
        * **diabetes**: Whether the patient had diabetes (1 = yes, 0 = no)
        * **totChol**: Total cholesterol level (mg/dL)
        * **sysBP**: Systolic blood pressure (mmHg)
        * **diaBP**: Diastolic blood pressure (mmHg)
        * **BMI**: Body Mass Index (weight in kg / (height in meters)^2)
        * **heartRate**: Heart rate (beats per minute)
        * **glucose**: Glucose level (mg/dL)
        * **has_heart_disease**: 10-year risk of coronary heart disease (1 = yes, 0 = no)
        """,
        "ar": """
        ## بيانات دراسة فرامينغهام لأمراض القلب
        
        دراسة فرامينغهام لأمراض القلب هي دراسة طويلة الأمد ومستمرة للأمراض القلبية الوعائية على سكان مدينة فرامينغهام، ماساتشوستس.
        بدأت الدراسة في عام 1948 مع 5,209 من البالغين من فرامينغهام، وهي الآن في جيلها الثالث من المشاركين.
        
        ### خصائص مجموعة البيانات:
        
        * **male**: الجنس (1 = ذكر، 0 = أنثى)
        * **age**: العمر عند الفحص (بالسنوات)
        * **education**: مستوى التعليم (1-4)
        * **currentSmoker**: حالة التدخين الحالية (1 = نعم، 0 = لا)
        * **cigsPerDay**: عدد السجائر المدخنة يومياً
        * **BPMeds**: ما إذا كان المريض يتناول أدوية ضغط الدم (1 = نعم، 0 = لا)
        * **prevalentStroke**: ما إذا كان المريض قد أصيب بالسكتة الدماغية سابقاً (1 = نعم، 0 = لا)
        * **prevalentHyp**: ما إذا كان المريض يعاني من ارتفاع ضغط الدم (1 = نعم، 0 = لا)
        * **diabetes**: ما إذا كان المريض مصاباً بالسكري (1 = نعم، 0 = لا)
        * **totChol**: مستوى الكوليسترول الكلي (ملغ/ديسيلتر)
        * **sysBP**: ضغط الدم الانقباضي (ملم زئبق)
        * **diaBP**: ضغط الدم الانبساطي (ملم زئبق)
        * **BMI**: مؤشر كتلة الجسم (الوزن بالكيلوغرام / (الطول بالمتر)^2)
        * **heartRate**: معدل ضربات القلب (نبضة في الدقيقة)
        * **glucose**: مستوى الجلوكوز (ملغ/ديسيلتر)
        * **has_heart_disease**: خطر الإصابة بأمراض القلب التاجية خلال 10 سنوات (1 = نعم، 0 = لا)
        """
    }
    return description

def get_column_mapping():
    """
    Return mapping of column names to display names
    """
    column_mapping = {
        "en": {
            "male": "Gender (1=Male, 0=Female)",
            "age": "Age (years)",
            "education": "Education Level (1-4)",
            "currentSmoker": "Current Smoker (1=Yes, 0=No)",
            "cigsPerDay": "Cigarettes per Day",
            "BPMeds": "Blood Pressure Medication (1=Yes, 0=No)",
            "prevalentStroke": "Previous Stroke (1=Yes, 0=No)",
            "prevalentHyp": "Hypertensive (1=Yes, 0=No)",
            "diabetes": "Diabetes (1=Yes, 0=No)",
            "totChol": "Total Cholesterol (mg/dL)",
            "sysBP": "Systolic BP (mmHg)",
            "diaBP": "Diastolic BP (mmHg)",
            "BMI": "BMI (kg/m²)",
            "heartRate": "Heart Rate (bpm)",
            "glucose": "Glucose (mg/dL)",
            "has_heart_disease": "Heart Disease Risk (1=Yes, 0=No)"
        },
        "ar": {
            "male": "الجنس (1=ذكر, 0=أنثى)",
            "age": "العمر (سنوات)",
            "education": "مستوى التعليم (1-4)",
            "currentSmoker": "مدخن حالياً (1=نعم, 0=لا)",
            "cigsPerDay": "عدد السجائر في اليوم",
            "BPMeds": "دواء ضغط الدم (1=نعم, 0=لا)",
            "prevalentStroke": "سكتة دماغية سابقة (1=نعم, 0=لا)",
            "prevalentHyp": "ارتفاع ضغط الدم (1=نعم, 0=لا)",
            "diabetes": "مرض السكري (1=نعم, 0=لا)",
            "totChol": "الكوليسترول الكلي (ملغ/ديسيلتر)",
            "sysBP": "ضغط الدم الانقباضي (ملم زئبق)",
            "diaBP": "ضغط الدم الانبساطي (ملم زئبق)",
            "BMI": "مؤشر كتلة الجسم (كغم/م²)",
            "heartRate": "معدل ضربات القلب (نبضة/دقيقة)",
            "glucose": "الجلوكوز (ملغ/ديسيلتر)",
            "has_heart_disease": "خطر أمراض القلب (1=نعم, 0=لا)"
        }
    }
    return column_mapping

def get_input_ranges():
    """
    Return valid input ranges for each feature
    """
    input_ranges = {
        "male": (0, 1, 1),  # (min, max, step)
        "age": (20, 90, 1),
        "education": (1, 4, 1),
        "currentSmoker": (0, 1, 1),
        "cigsPerDay": (0, 70, 1),
        "BPMeds": (0, 1, 1),
        "prevalentStroke": (0, 1, 1),
        "prevalentHyp": (0, 1, 1),
        "diabetes": (0, 1, 1),
        "totChol": (100, 500, 1),
        "sysBP": (80, 250, 1),
        "diaBP": (40, 150, 1),
        "BMI": (15.0, 50.0, 0.1),  # Make sure all values are float
        "heartRate": (40, 150, 1),
        "glucose": (40, 400, 1)
    }
    return input_ranges