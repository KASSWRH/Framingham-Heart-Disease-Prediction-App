"""
This module contains translations for the Heart Disease Prediction App
"""

translations = {
    "en": {
        # General
        "app_title": "Heart Disease Prediction App",
        "app_subtitle": "Predict the risk of heart disease using machine learning",
        "language": "Language",
        "loading_data": "Loading data...",
        "disclaimer": "Disclaimer: This app is for educational purposes only and not intended to provide medical advice. Always consult with a healthcare professional.",
        
        # Navigation
        "nav_home": "Home",
        "nav_data_exploration": "Data Exploration",
        "nav_model_training": "Model Training",
        "nav_feature_importance": "Feature Importance",
        "nav_prediction": "Make Prediction",
        
        # Home page
        "welcome_title": "Welcome to the Heart Disease Prediction App",
        "welcome_description": "This application uses machine learning to predict the risk of heart disease based on various health parameters from the Framingham Heart Study dataset.",
        "about_section": "About This App",
        "about_content": """
        This app demonstrates how machine learning can be used in healthcare for risk prediction.
        
        **Features:**
        * Explore the dataset with interactive visualizations
        * Train and evaluate multiple machine learning models
        * Analyze feature importance to understand key risk factors
        * Make predictions with your own health parameters
        
        **How it works:**
        The app uses supervised learning algorithms trained on the Framingham Heart Study data to identify patterns associated with heart disease risk.
        """,
        "how_to_use": "How to Use This App",
        "how_to_use_content": """
        1. **Explore the data** to understand the distribution of health parameters and their relationship with heart disease
        2. **Train models** to see how different machine learning algorithms perform on this dataset
        3. **Analyze feature importance** to understand which health factors contribute most to heart disease risk
        4. **Make a prediction** by entering your health parameters
        """,
        
        # Data exploration page
        "data_exploration_title": "Data Exploration",
        "dataset_info": "Dataset Information",
        "dataset_stats": "Dataset Statistics",
        "dataset_description": "Dataset Description",
        "data_visualizations": "Data Visualizations",
        "correlation_matrix": "Correlation Matrix",
        "feature_distributions": "Feature Distributions",
        "feature_boxplots": "Feature Boxplots",
        "feature_relationships": "Feature Relationships",
        
        # Model training page
        "model_training_title": "Model Training and Evaluation",
        "training_description": "Train multiple machine learning models and evaluate their performance",
        "train_models_button": "Train Models",
        "training_results": "Training Results",
        "model_metrics": "Model Metrics",
        "confusion_matrices": "Confusion Matrices",
        "roc_curves": "ROC Curves",
        "pr_curves": "Precision-Recall Curves",
        
        # Feature importance page
        "feature_importance_title": "Feature Importance Analysis",
        "feature_importance_description": "Analyze which features contribute most to heart disease prediction",
        "feature_importance_results": "Feature Importance Results",
        
        # Prediction page
        "prediction_title": "Heart Disease Prediction",
        "prediction_description": "Enter your health parameters to predict heart disease risk",
        "patient_information": "Patient Information",
        "medical_history": "Medical History",
        "vital_signs": "Vital Signs",
        "lab_results": "Laboratory Results",
        "predict_button": "Predict Heart Disease Risk",
        "prediction_results": "Prediction Results",
        "prediction_explanation": "What does this mean?",
        "high_risk_explanation": "A high risk (>50%) suggests a significant chance of developing heart disease within 10 years based on the provided parameters.",
        "low_risk_explanation": "A low risk (<50%) suggests a lower chance of developing heart disease within 10 years based on the provided parameters.",
        "risk_factors_identified": "Key risk factors identified:",
        "disclaimer_prediction": "Remember: This is not a medical diagnosis. Consult with a healthcare professional for proper evaluation and advice.",
    },
    
    "ar": {
        # General
        "app_title": "تطبيق التنبؤ بأمراض القلب",
        "app_subtitle": "التنبؤ بمخاطر الإصابة بأمراض القلب باستخدام تعلم الآلة",
        "language": "اللغة",
        "loading_data": "جاري تحميل البيانات...",
        "disclaimer": "تنبيه: هذا التطبيق مخصص للأغراض التعليمية فقط وليس المقصود منه تقديم المشورة الطبية. استشر دائمًا أخصائي الرعاية الصحية.",
        
        # Navigation
        "nav_home": "الرئيسية",
        "nav_data_exploration": "استكشاف البيانات",
        "nav_model_training": "تدريب النماذج",
        "nav_feature_importance": "أهمية المتغيرات",
        "nav_prediction": "إجراء التنبؤ",
        
        # Home page
        "welcome_title": "مرحبًا بك في تطبيق التنبؤ بأمراض القلب",
        "welcome_description": "يستخدم هذا التطبيق تعلم الآلة للتنبؤ بمخاطر الإصابة بأمراض القلب بناءً على مختلف المعلمات الصحية من مجموعة بيانات دراسة فرامينغهام للقلب.",
        "about_section": "نبذة عن هذا التطبيق",
        "about_content": """
        يوضح هذا التطبيق كيف يمكن استخدام تعلم الآلة في الرعاية الصحية للتنبؤ بالمخاطر.
        
        **الميزات:**
        * استكشاف مجموعة البيانات باستخدام رسومات تفاعلية
        * تدريب وتقييم نماذج متعددة للتعلم الآلي
        * تحليل أهمية المتغيرات لفهم عوامل الخطر الرئيسية
        * إجراء تنبؤات باستخدام معلماتك الصحية الخاصة
        
        **كيف يعمل:**
        يستخدم التطبيق خوارزميات التعلم الخاضع للإشراف المدربة على بيانات دراسة فرامينغهام للقلب لتحديد الأنماط المرتبطة بمخاطر الإصابة بأمراض القلب.
        """,
        "how_to_use": "كيفية استخدام هذا التطبيق",
        "how_to_use_content": """
        1. **استكشف البيانات** لفهم توزيع المعلمات الصحية وعلاقتها بأمراض القلب
        2. **درب النماذج** لمعرفة كيف تعمل خوارزميات التعلم الآلي المختلفة على مجموعة البيانات هذه
        3. **حلل أهمية المتغيرات** لفهم أي العوامل الصحية تساهم أكثر في مخاطر الإصابة بأمراض القلب
        4. **قم بإجراء تنبؤ** عن طريق إدخال معلماتك الصحية
        """,
        
        # Data exploration page
        "data_exploration_title": "استكشاف البيانات",
        "dataset_info": "معلومات مجموعة البيانات",
        "dataset_stats": "إحصائيات مجموعة البيانات",
        "dataset_description": "وصف مجموعة البيانات",
        "data_visualizations": "عرض مرئي للبيانات",
        "correlation_matrix": "مصفوفة الارتباط",
        "feature_distributions": "توزيعات المتغيرات",
        "feature_boxplots": "مخططات صندوقية للمتغيرات",
        "feature_relationships": "العلاقات بين المتغيرات",
        
        # Model training page
        "model_training_title": "تدريب وتقييم النماذج",
        "training_description": "قم بتدريب نماذج متعددة للتعلم الآلي وقيّم أدائها",
        "train_models_button": "تدريب النماذج",
        "training_results": "نتائج التدريب",
        "model_metrics": "مقاييس النماذج",
        "confusion_matrices": "مصفوفات الالتباس",
        "roc_curves": "منحنيات ROC",
        "pr_curves": "منحنيات الدقة-الاسترجاع",
        
        # Feature importance page
        "feature_importance_title": "تحليل أهمية المتغيرات",
        "feature_importance_description": "تحليل المتغيرات التي تساهم أكثر في التنبؤ بأمراض القلب",
        "feature_importance_results": "نتائج أهمية المتغيرات",
        
        # Prediction page
        "prediction_title": "التنبؤ بأمراض القلب",
        "prediction_description": "أدخل معلماتك الصحية للتنبؤ بمخاطر الإصابة بأمراض القلب",
        "patient_information": "معلومات المريض",
        "medical_history": "التاريخ الطبي",
        "vital_signs": "العلامات الحيوية",
        "lab_results": "نتائج المختبر",
        "predict_button": "التنبؤ بمخاطر الإصابة بأمراض القلب",
        "prediction_results": "نتائج التنبؤ",
        "prediction_explanation": "ماذا يعني هذا؟",
        "high_risk_explanation": "تشير المخاطر العالية (>50%) إلى فرصة كبيرة للإصابة بأمراض القلب خلال 10 سنوات بناءً على المعلمات المقدمة.",
        "low_risk_explanation": "تشير المخاطر المنخفضة (<50%) إلى فرصة أقل للإصابة بأمراض القلب خلال 10 سنوات بناءً على المعلمات المقدمة.",
        "risk_factors_identified": "عوامل الخطر الرئيسية المحددة:",
        "disclaimer_prediction": "تذكر: هذا ليس تشخيصًا طبيًا. استشر أخصائي الرعاية الصحية للتقييم والنصيحة المناسبين.",
    }
}

def get_translation(key, lang):
    """
    Get translation for a key in the specified language
    If key is not found, return the key itself
    """
    if lang in translations and key in translations[lang]:
        return translations[lang][key]
    return key