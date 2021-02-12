from model import *
import streamlit as st
import time

def main():
    st.title('Drug Prescription Recommender')

    activities = ['Recommender', 'About']

    choice = st.sidebar.radio("Select Action", activities)
    
    if choice == 'Recommender':
        st.subheader('Details')
        age = st.slider('Age', min_value = 0.1, max_value = 100.0, value = 40.0, step = 0.1)
        format_temp = '%f °F'
        temperature = st.slider('Temperature', min_value = 95.0, max_value = 106.0, value = 98.6, step = 0.1, format = format_temp)
        
        specialities = ['Cardiologist', 'Dermatologist', 'Gastroenterologist', 'General Physician', 'Gynaecologist', 'Neurologist ', 'Orthopedic', 'Pediatrician']
        speciality = st.selectbox('Speciality of the Doctor concerned', specialities, index = 3)

        all_findings = ['abdominal pain', 'acid reflux', 'acute abdomen', 'allergic disorder of skin', 'amenorrhea', 'asthenia', 'backache', 'bodyache', 'chest pain', 'common cold', 'constipation', 'cough', 'diarrhea', 'epigastric pain', 'fever', 'foot pain', 'furuncle', 'gastritis', 'generalised weakness', 'generalized aches and pains', 'headache', 'hypertensive disorder', 'infestation caused by sarcoptes scabiei var hominis', 'injury of foot', 'itching', 'joint injury', 'joint pain', 'knee pain', 'loose stool', 'neck pain', 'otalgia', 'pain in eye', 'pain in throat', 'pregnant', 'pruritus of vulva', 'respiratory tract infection', 'shoulder pain', 'systemic arterial', 'tinea corporis', 'tinea cruris', 'toothache', 'ulcer of mouth', 'upper respiratory infection', 'urticaria', 'vertigo', 'vomiting'] 
        all_f = [f.title() for f in all_findings]
        findings = st.multiselect('Symptoms', all_f, default = ['Cough', 'Bodyache'])
        findings = [f.lower() for f in findings]

        model = Prescription('model/model.json', 'model/weights.h5', 'model/scaler.pickle', 'model/enc_input.pickle', 'model/enc_output.pickle')
        ans = model.predict_prescription(age, temperature, speciality, findings)
        ans = [a.title() for a in ans]

        df = pd.DataFrame(ans, columns = ['Salt'])
        st.write('\n\n')
        st.subheader("Recommended Prescription")
        st.dataframe(df)
        
        
    else:
        st.write('The application is a drug-prescription recommendation system. On feeding information about both the patient and the doctor this recommender yields a suitable medical prescription for the patient.')
        st.write('\n')
        st.write("~ Eklavya Jain")

if __name__ == '__main__':
    main()

