import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Admission_Predict.csv")
df1 = df.drop(['Serial No.'],axis=1)
x = df.iloc[:,[1,2,3,4,5,6,7]].values
y = df.iloc[:,8].values
x_d = pd.DataFrame(x,columns = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research'])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
model = LogisticRegression()
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train1 = [1 if x > 0.75 else 0 for x in y_train]
y_test1 = [1 if x > 0.75 else 0 for x in y_test]
model.fit(x_train,y_train1)
st.title("Graduation Admission Predictor")
st.markdown("
Graduate admissions hinge significantly on GRE and TOEFL scores, serving as key metrics for evaluating candidates' academic aptitude and English proficiency, respectively. The GRE assesses fundamental skills essential for advanced studies, offering a standardized benchmark for comparison. Meanwhile, TOEFL measures English language proficiency crucial for effective communication in academic settings. However, while scores provide valuable insights, they aren't sole determinants. Holistic evaluations, considering academic transcripts, recommendations, and personal statements, ensure a comprehensive assessment. Challenges exist, including concerns over test bias and limited scope. Yet, by supplementing scores with qualitative measures, institutions strive for inclusivity, identifying diverse talents poised to contribute meaningfully to academia.")
st.markdown()
st.markdown("Provide the necessary details below to predict if you can get the admission.")
GREScore = st.text_input("Enter your GRE Score:")
TOEFLScore = st.text_input("Enter your TOEFL Score:")
UniversityRating = st.text_input("Enter your University rating:")
Sop = st.text_input("Enter SOP:")
Lor	= st.text_input("Enter LOR:")
Cgpa = st.text_input("Enter your cgpa:")
Research = st.text_input("Enter 1 if you have any research experience:")
st.text('Result will be displayed here!')
if st.button("Submit"):
    result = model.predict(np.array([[int(GREScore),int(TOEFLScore),int(UniversityRating),float(Sop),float(Lor),float(Cgpa),int(Research)]]))
    if result == 1:
        st.text("Congratulations!! You are eligible to admit.")
    else:
        st.text("Sorry, you are not eligible to admit. Better luck next time!")
