import streamlit as st


from multipage import MultiPage
import pull_eval_cases_page
import forecast_modeling_page


app = MultiPage()

st.title("Covid-19 Dashboard")

app.add_page("Confirmed Cases per Region", pull_eval_cases_page.app)
app.add_page("Covid Case Forecast Model", forecast_modeling_page.app)

app.run()