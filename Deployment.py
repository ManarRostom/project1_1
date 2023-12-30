
import streamlit as st
import numpy as np
import joblib 
import pandas as pd
import sklearn
import category_encoders

inputs_dict = joblib.load('Data/inputs_dict.pkl')
Model = joblib.load('Data/Model.pkl')
mlb_dict = joblib.load('Data/mlb_dict.pkl')

def encode_cats(df_cats):
    rest_type_df = pd.DataFrame(mlb_dict['mlb_rest_type'].transform(df_cats['rest_type']), columns=mlb_dict['mlb_rest_type'].classes_)
    rest_type_df = rest_type_df[inputs_dict['Rest_Type_Cols']]
    rest_type_df.columns = ['Rest_Type_' + col for col in rest_type_df.columns]
    
    cuisines_df = pd.DataFrame(mlb_dict['mlb_cuisines'].transform(df_cats['cuisines']), columns=mlb_dict['mlb_cuisines'].classes_)
    cuisines_df = cuisines_df[inputs_dict['Cuisines_Cols']]
    cuisines_df.columns = ['Cuisines_' + col for col in  cuisines_df.columns]

        
    listed_type_df = pd.DataFrame(mlb_dict['mlb_type'].transform(df_cats['listed_type']), columns=mlb_dict['mlb_type'].classes_)
    listed_type_df = listed_type_df[inputs_dict['Listed_in_Type']]
    listed_type_df.columns = ['Listed_in_Type_' + col for col in listed_type_df.columns]
    
    df = pd.concat([rest_type_df, cuisines_df, listed_type_df], axis=1)
    return df
    
def Prediction(online_order, book_table, location, approx_cost, rest_type, cuisines, listed_type):
    ## Create df_test
    df_test = pd.DataFrame(columns=inputs_dict['columns_names'])
    df_test.at[0,'Original_Columns_online_order'] = online_order
    df_test.at[0,'Original_Columns_book_table'] = book_table
    df_test.at[0,'Original_Columns_location'] = location
    df_test.at[0,'Original_Columns_approx_cost(for two people)'] = approx_cost
    
    ## Create df_cats
    df_cats = pd.DataFrame(columns=['rest_type','cuisines','listed_type'])
    df_cats.at[0,'rest_type'] = rest_type
    df_cats.at[0,'cuisines'] = cuisines
    df_cats.at[0,'listed_type'] = listed_type
    ## Call encode_cats Function
    encoded_df = encode_cats(df_cats)
    df_test2 = pd.concat([df_test,encoded_df], axis=1)
    return Model.predict(df_test2)[0]


def Main():
    
    st.markdown('<p style="font-size:50px;text-align:center;"><strong>Predict New Restaurant Success in Bangalore</strong></p>',unsafe_allow_html=True)
    col1_1 , col1_2 = st.columns([2,2]) 
    col2_1 , col2_2 = st.columns([2,2])
    col3_1 , col3_2 = st.columns([2,2])
    col4_1 , col4_2 = st.columns([2,2])
    
    with col1_1:
        online_order = st.selectbox('Is your restaurant offers Online orders : ', ['Yes','No'])
    with col1_2:
        book_table = st.selectbox('Is your restaurant offers Booking tables : ', ['Yes','No'])
        
    with col2_1:    
        location = st.selectbox('Select Location of your restaurant : ', inputs_dict['Location'])
    with col2_2:
        approx_cost = st.slider(label='What is the approximate cost for two people : ', value=1000, step=100, min_value=10, max_value=10000)
        
    with col3_1:    
        rest_type = st.multiselect('Select type of you restaurant : ', inputs_dict['Rest_Type_Cols'])
    with col3_2:
        cuisines = st.multiselect('Select the cuisines available at your restaurant  : ', inputs_dict['Cuisines_Cols'])
        
    with col4_1:
        listed_type = st.multiselect('Select category type of your restaurant',inputs_dict['Listed_in_Type'])
    with col4_2:
        st.text('')
        st.text('')
        if st.button("Predict"):
            res = Prediction(online_order, book_table, location, approx_cost, rest_type, cuisines, listed_type)
            if res == 1:
                model_prediction = "Will Successed"
            else:
                model_prediction = "Will not Successed"

            st.write(f"We predict that your restaurant {model_prediction}")
Main()        
