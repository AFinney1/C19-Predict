FROM python:3.8.5

WORKDIR /C19-Predict

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY C19-Predict/ .

CMD ["python", "streamlit run Case_Dash_st.py"]