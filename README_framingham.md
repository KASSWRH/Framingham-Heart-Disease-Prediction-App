# تطبيق التنبؤ بأمراض القلب | Heart Disease Prediction App

تطبيق ويب يستخدم تقنيات التعلم الآلي للتنبؤ باحتمالية إصابة الشخص بأمراض القلب بناءً على بيانات دراسة فرامينغهام للقلب.

## ميزات التطبيق

- عرض وتحليل بيانات دراسة فرامينغهام للقلب
- تدريب وتقييم 6 نماذج مختلفة للتعلم الآلي
- تحليل أهمية المتغيرات المستخدمة في التنبؤ
- إمكانية إدخال البيانات الصحية والحصول على تنبؤ فوري
- دعم اللغتين العربية والإنجليزية

## المتطلبات

```
streamlit==1.26.0
pandas==2.0.3
numpy==1.24.4
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1
scikit-learn==1.3.0
```

## التثبيت والتشغيل

1. قم بتثبيت المتطلبات:
```
pip install -r requirements_framingham.txt
```

2. قم بتشغيل التطبيق:
```
streamlit run app.py
```

## البيانات المستخدمة

يستخدم التطبيق مجموعة بيانات دراسة فرامينغهام للقلب، وهي دراسة طويلة المدى لأمراض القلب والأوعية الدموية بدأت عام 1948.

### متغيرات البيانات الرئيسية:

- **الجنس**: ذكر (1) أو أنثى (0)
- **العمر**: عمر المريض بالسنوات
- **التدخين**: مدخن حالي (1) أو غير مدخن (0)
- **ضغط الدم**: ضغط الدم الانقباضي والانبساطي
- **الكوليسترول الكلي**: مستوى الكوليسترول (ملغ/ديسيلتر)
- **مرض السكري**: مصاب بالسكري (1) أو غير مصاب (0)
- **مؤشر كتلة الجسم**: الوزن بالكيلوغرام مقسوماً على مربع الطول بالمتر

## تفاصيل النماذج المستخدمة

- الانحدار اللوجستي (Logistic Regression)
- الغابات العشوائية (Random Forest)
- تعزيز التدرج (Gradient Boosting)
- آلة المتجه الداعم (SVM)
- أقرب الجيران (KNN)
- آداببوست (AdaBoost)

## إخلاء المسؤولية

هذا التطبيق مخصص لأغراض تعليمية وبحثية فقط. لا ينبغي استخدام التنبؤات كبديل عن المشورة الطبية المهنية. يرجى استشارة مقدم الرعاية الصحية للتشخيص والعلاج المناسبين.

---

# Heart Disease Prediction App

A web application that uses machine learning techniques to predict the likelihood of a person having heart disease based on the Framingham Heart Study data.

## Features

- Display and analyze Framingham Heart Study data
- Train and evaluate 6 different machine learning models
- Analyze the importance of variables used in prediction
- Input health data and get immediate prediction
- Support for both Arabic and English languages

## Requirements

```
streamlit==1.26.0
pandas==2.0.3
numpy==1.24.4
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1
scikit-learn==1.3.0
```

## Installation and Running

1. Install the requirements:
```
pip install -r requirements_framingham.txt
```

2. Run the application:
```
streamlit run app.py
```

## Data Used

The application uses the Framingham Heart Study dataset, which is a long-term cardiovascular study that started in 1948.

### Key Data Variables:

- **Gender**: Male (1) or Female (0)
- **Age**: Age in years
- **Smoking**: Current smoker (1) or Non-smoker (0)
- **Blood Pressure**: Systolic and Diastolic blood pressure
- **Total Cholesterol**: Cholesterol level (mg/dL)
- **Diabetes**: Has diabetes (1) or No diabetes (0)
- **BMI**: Body Mass Index - weight in kg divided by the square of height in meters

## Details of Models Used

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- AdaBoost

## Disclaimer

This application is intended for educational and research purposes only. The predictions should not be used as a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.