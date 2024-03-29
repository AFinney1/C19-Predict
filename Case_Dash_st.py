from re import T
import streamlit as st
from Case_Analysis import *
import tkinter
import matplotlib
#from st import pydeck_chart
import pydeck as pdk

matplotlib.use("Tkagg")

#Introduction and Data Exploration'''
st.title(""" COVID-19 Case Projection App""")
st.subheader("All U.S. Cases")
case_df = get_cases()
case_df.rename(columns = {'Lat':'lat', 'Long_':'lon'}, inplace = True)
st.map(case_df)
st.write(case_df)

#'''Preprocessing'''
preprocessed_data = preprocessing(case_df)
initial_start_date = preprocessed_data[-5]
val_start_date = preprocessed_data[-4]
#st.subheader(str(val_start_date))
test_start_date = preprocessed_data[-3]
county_name = preprocessed_data[-2]
lastdate = preprocessed_data[-1]
training_df, val_df, test_df, training_mean, training_std = train_test_val_split(preprocessed_data=preprocessed_data)
training_df = normalize(training_df, training_mean, training_std)
val_df = normalize(val_df, training_mean, training_std)
test_df = normalize(test_df, training_mean, training_std)

#'''Modeling'''
model = build_time_series_model(test_df, training_df, val_df)
saved_model = model_save_function(model)
model_test= test_predictions(model, test_df, training_mean, training_std)
#predicted_cases = plot_case_predictions(model_test, county_name, saved_model)

#'''Data Visualization'''
plot_cases(cases = pd.DataFrame(denormalize(training_df, training_mean= training_mean, training_std=training_std)), county_name = county_name, startdate=initial_start_date, lastdate = val_start_date, history = True)
plot_cases(model_test, county_name, startdate=test_start_date, lastdate = lastdate, history = False)
selected_region = preprocessed_data[2]
#selected_region = selected_region.columns
print(selected_region, type(selected_region))
selected_region.rename(columns = {'Lat':'lat', 'Long_':'lon'}, inplace = True)
case_df.rename(columns = {'Lat':'lat', 'Long_':'lon'}, inplace = True)
st.subheader("Projected Case Map (in development)")
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v10',
    initial_view_state=pdk.ViewState(
        latitude=37.76,
        longitude=-90.4,
        zoom=1,
        pitch=50,
     ),
     layers=[
         pdk.Layer(
            'ScatterplotLayer',
            data=selected_region,
            get_position='[lon, lat]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
         ),
         pdk.Layer(
             'HexagonLayer',
             data=selected_region,
             get_position='[lon, lat]',
             get_color='[200, 30, 0, 160]',
             get_radius=200,
             pickable=True,
             extruded=True
         ),
     ],
 ))