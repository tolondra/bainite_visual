import streamlit as st
import pandas as pd

# import numpy as np
from helper_Functions import *


# Cache to import the datasets just once
@st.cache_data
def importdata():
    """
    Import and preprocess the datasets.

    Returns:
        tuple: A tuple containing the main dataframe, experimental dataframe, and columns to filter.
    """
    # Read the main dataset
    df = pd.read_csv(
        "DATA_ALL.csv",
        float_precision="round_trip",
    )
    # Sample a fraction of the dataset for faster processing
    df = df.sample(frac=1, replace=True, random_state=1)
    # Round the values to 2 decimal places
    df = df.round(2)

    # Rename columns for consistency
    df["YS"] = df["Ys"]
    df["Sigma YS"] = df["Yserr"]
    df["Sigma UEL"] = df["UELerr"]
    df = df.drop(columns=["Yserr", "UELerr", "Ys"])

    # Read the experimental dataset
    experimental_df = pd.read_csv(
        "YsUELExdata.csv", sep=";"
    )
    # Rename columns for consistency
    experimental_df["YS"] = experimental_df["Ys"]
    experimental_df["Sigma YS"] = experimental_df["Ys_std"]
    experimental_df["Sigma UEL"] = experimental_df["UEL_std"]
    experimental_df["T"] = experimental_df["Th"]
    experimental_df = experimental_df.drop(
        columns=["Ys_std", "UEL_std", "Ys", "Ys_std", "Th"]
    )

    # Sort the dataframes by YS column
    df = df.sort_values(by="YS")
    experimental_df = experimental_df.sort_values(by="YS")

    # Columns to filter with sliders
    columns_to_filter = [
        "C",
        "Si",
        "Mn",
        "Cr",
        "Mo",
        "V",
        "Al",
        "T",
        "Sigma UEL",
        "Sigma YS",
    ]

    # Convert columns to numeric, handling non-numeric values gracefully
    for column in columns_to_filter:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Drop rows with NaN values in the filtered columns
    df = df.dropna(subset=columns_to_filter)

    return (df, experimental_df, columns_to_filter)


def create_slider(df, df_exp, columns_to_filter, keyval):
    """
    Create sliders in the sidebar for filtering the dataframes.

    Args:
        df (DataFrame): The main dataframe.
        df_exp (DataFrame): The experimental dataframe.
        columns_to_filter (list): List of columns to filter with sliders.
        keyval (str): The key value for the form.

    Returns:
        tuple: A tuple containing sliders dictionary, filtered experimental dataframe, and filtered main dataframe.
    """
    # Create sliders in the sidebar within a form
    with st.sidebar.form(key=keyval):
        cols = st.columns(2, gap="medium")
        sliders = {}
        for column in columns_to_filter:

            # Determine which column to write to (left or right)
            if columns_to_filter.index(column) < 5:
                col2write = 0
            else:
                col2write = 1

            # Get the min and max values for the slider
            min_value, max_value = round(df[column].min(), 3), round(
                df[column].max(), 3
            )
            # Create the slider for the column
            sliders[column] = cols[col2write].slider(
                column + " range",
                min_value,
                max_value,
                (min_value, max_value),
            )

        # Filter the experimental and simulation datasets based on slider values
        filtered_df_exp = filter_data(df_exp, sliders, change=None)
        filtered_df_sim = filter_data(df, sliders, change=None)

        # Update the dataset when the form is submitted
        st.form_submit_button(label="Update Dataset")

    return sliders, filtered_df_exp, filtered_df_sim
