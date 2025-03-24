import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from functions import *
from helper_Functions import *


if __name__ == "__main__":
    # """
    # Main function to run the Streamlit app. It sets up the sidebar, imports data, creates sliders,
    # filters the data, and provides options for visualizing and downloading the data.
    # """
    # Set the title for the sidebar
    st.sidebar.title("Bainite Visualizer")
    st.sidebar.write("")

    # Import data and get the necessary columns for filtering
    df_sim, df_exp, columns_to_filter = importdata()

    # Create a button in the sidebar to reset the sliders
    resetSlider = st.sidebar.button(
        label="Reset sliders",
    )  # , use_container_width=True)

    if resetSlider:
        # If reset button is clicked, create sliders with reset key and empty the filtered dataframes
        sliders, filtered_df_exp, filtered_df_sim = create_slider(
            df_sim, df_exp, columns_to_filter, keyval="reset"
        )
        filtered_df_exp = pd.DataFrame()
        filtered_df_sim = pd.DataFrame()

    else:
        # Otherwise, initialize the sliders normally
        sliders, filtered_df_exp, filtered_df_sim = create_slider(
            df_sim, df_exp, columns_to_filter, keyval="initialize"
        )

    if not filtered_df_sim.empty:
        # If there are data points in the filtered simulation dataframe

        st.write("Data points in the DataFrame: ", len(filtered_df_sim))

        # Radio button for selecting actions
        action_sel = st.radio(
            "",
            [
                "Plot Dataset",
                "Plot Pareto Front",
                r"Plot Pareto Front +/- $\sigma$ Plot",
                "Download  PDF",
            ],
            horizontal=True,
        )

        st.markdown(
            """<hr style="height:5px;border:none;color:#808080;background-color:#808080;" /> """,
            unsafe_allow_html=True,
        )

        if action_sel == "Plot Dataset":
            # Create checkbox for showing experimental data
            show_experimental_checkbox = st.checkbox(
                "Show Experimental Data", key="scatter"
            )
            # Plot scatter graph
            plot_scatter(
                filtered_df_sim,
                filtered_df_exp,
                show_experimental_checkbox,
                change=None,
            )

        # Options for dropdown menu colorcode
        options = (
            columns_to_filter
            + ["Diff. in chem. comp. to next Exp. point"]
            + ["Rel Error UEL/YS"]
        )
        default_value = "C"

        if action_sel == "Plot Pareto Front":
            # Create checkbox for showing experimental data
            show_experimental_checkbox = st.checkbox(
                "Show Experimental Data", key="paretograph"
            )
            # Create a dropdown for color-coding options
            dropdown = st.selectbox(
                label="Color-Coding",  # Description
                options=options,  # Options for the dropdown
                index=0,  # Default index (adjust based on default_value)
                help="Select an option for color-coding",  # Optional help tooltip
                key="pareto",
            )
            # Plot pareto front and content variations
            plot_pareto_and_content_variations(
                filtered_df_sim,
                filtered_df_exp,
                show_experimental_checkbox,
                dropdown,
            )

        if action_sel == r"Plot Pareto Front +/- $\sigma$ Plot":
            # Create checkbox for showing experimental data
            show_experimental_checkbox = st.checkbox(
                "Show Experimental Data", key="paretosigama"
            )
            # Create a dropdown for color-coding options
            dropdown = st.selectbox(
                label="Color-Coding",  # Description
                options=options,  # Options for the dropdown
                index=0,  # Default index (adjust based on default_value)
                help="Select an option for color-coding",  # Optional help tooltip
                key="pareto_variation",
            )
            # Plot pareto front and variations
            plot_pareto_and_variations(
                filtered_df_sim,
                filtered_df_exp,
                show_experimental_checkbox,
                dropdown,
            )

        # Stored figures for printing PDF function
        stored_figures = []
        if action_sel == "Download  PDF":
            # Create checkbox for showing experimental data
            show_experimental_checkbox = st.checkbox(
                "Show Experimental Data", key="downloadexp"
            )
            # Create a dropdown for color-coding options
            dropdown = st.selectbox(
                label="Color-Coding",  # Description
                options=options,  # Options for the dropdown
                index=0,  # Default index (adjust based on default_value)
                help="Select an option for color-coding",  # Optional help tooltip
                key="download",
            )
            stored_figures.clear()

            # Generate and store all plots when the download button is clicked
            # Call all the plotting functions to generate the plots
            stored_figures.append(
                plot_scatter(
                    filtered_df_sim,
                    filtered_df_exp,
                    show_experimental_checkbox,
                    show=False,
                    change=None,
                )
            )

            results = plot_pareto_and_content_variations(
                filtered_df_sim,
                filtered_df_exp,
                show_experimental_checkbox,
                dropdown,
                show=False,
            )
            stored_figures.extend(results[:2])

            stored_figures.append(
                plot_pareto_and_variations(
                    filtered_df_sim,
                    filtered_df_exp,
                    show_experimental_checkbox,
                    dropdown,
                    show=False,
                )
            )
            # Save all plots into a PDF file
            with PdfPages("all_plots.pdf") as pdf:
                for fig in stored_figures:
                    pdf.savefig(fig)  # Save the figure to the PDF
                    plt.close(fig)  # Close the figure to avoid conflicts
            st.write("All plots have been saved into 'all_plots.pdf'.")
