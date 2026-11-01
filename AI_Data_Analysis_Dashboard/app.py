import streamlit as st
import pandas as pd
import plotly.express as px

from data_cleaning import clean_data
from data_analysis import generate_insights, detect_anomalies


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Data Analysis Dashboard")

st.markdown(
    "Analyze datasets, visualize trends, "
    "discover patterns and generate intelligent insights."
)


# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls", "json"]
)

missing_method = st.sidebar.selectbox(
    "Missing Value Handling",
    [
        "Keep unchanged",
        "Mean / Mode",
        "Drop Rows",
        "Fill with 0 / Unknown"
    ]
)

remove_duplicates = st.sidebar.checkbox(
    "Remove Duplicate Records",
    value=True
)


# =====================================
# LOAD DATASET
# =====================================

if uploaded_file is None:

    st.info(
        "Please upload a CSV, Excel or JSON dataset "
        "using the sidebar."
    )

    st.subheader("Dashboard Features")

    st.write("✅ Dataset Upload")
    st.write("✅ Data Cleaning")
    st.write("✅ Dataset Overview")
    st.write("✅ Interactive Visualization")
    st.write("✅ AI Insights")
    st.write("✅ Anomaly Detection")
    st.write("✅ Search and Filter")
    st.write("✅ Report Generation")

    st.stop()


try:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    elif file_name.endswith((".xlsx", ".xls")):

        df = pd.read_excel(uploaded_file)

    elif file_name.endswith(".json"):

        df = pd.read_json(uploaded_file)

    else:

        st.error("Unsupported file format.")
        st.stop()

except Exception as error:

    st.error(f"Dataset loading error: {error}")
    st.stop()


if df.empty:

    st.error("Uploaded dataset is empty.")
    st.stop()


# =====================================
# DATA PREPROCESSING
# =====================================

cleaned_df, cleaning_messages = clean_data(
    df,
    missing_method,
    remove_duplicates
)

if cleaned_df.empty:

    st.error("No records remain after preprocessing.")
    st.stop()


st.success("Dataset uploaded successfully!")


# =====================================
# DASHBOARD TABS
# =====================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Overview",
        "📈 Visualization",
        "🤖 AI Insights",
        "📄 Reports"
    ]
)


# =====================================
# TAB 1: DATASET OVERVIEW
# =====================================

with tab1:

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rows",
        len(cleaned_df)
    )

    col2.metric(
        "Total Columns",
        len(cleaned_df.columns)
    )

    col3.metric(
        "Missing Values",
        int(cleaned_df.isnull().sum().sum())
    )

    col4.metric(
        "Duplicate Rows",
        int(cleaned_df.duplicated().sum())
    )

    # Cleaning summary
    st.subheader("Data Cleaning Summary")

    for message in cleaning_messages:

        st.write("✅", message)

    # Column information
    st.subheader("Column Information")

    column_info = pd.DataFrame({

        "Column Name": cleaned_df.columns,

        "Data Type": [
            str(cleaned_df[col].dtype)
            for col in cleaned_df.columns
        ],

        "Missing Values": [
            cleaned_df[col].isnull().sum()
            for col in cleaned_df.columns
        ],

        "Unique Values": [
            cleaned_df[col].nunique()
            for col in cleaned_df.columns
        ]

    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    # Statistical summary
    st.subheader("Statistical Summary")

    st.dataframe(
        cleaned_df.describe(include="all").T,
        use_container_width=True
    )

    # Dataset preview
    st.subheader("Dataset Preview")

    st.dataframe(
        cleaned_df.head(100),
        use_container_width=True
    )

    # =================================
    # SEARCH AND FILTER
    # =================================

    st.divider()

    st.subheader("🔍 Search and Filter Records")

    search_text = st.text_input(
        "Search across all columns"
    )

    filtered_data = cleaned_df.copy()

    if search_text:

        search_mask = filtered_data.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
        ).any(axis=1)

        filtered_data = filtered_data[
            search_mask
        ]

    filter_column = st.selectbox(
        "Select Column to Filter",
        ["None"] + cleaned_df.columns.tolist()
    )

    if filter_column != "None":

        values = (
            filtered_data[filter_column]
            .dropna()
            .unique()
            .tolist()
        )

        selected_values = st.multiselect(
            "Select Values",
            values
        )

        if selected_values:

            filtered_data = filtered_data[
                filtered_data[filter_column].isin(
                    selected_values
                )
            ]

    st.write(
        f"Filtered Records: {len(filtered_data)}"
    )

    st.dataframe(
        filtered_data,
        use_container_width=True
    )

    st.download_button(
        "Download Filtered Data",
        filtered_data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_data.csv",
        mime="text/csv"
    )


# =====================================
# TAB 2: DATA VISUALIZATION
# =====================================

with tab2:

    st.header("Interactive Data Visualization")

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Bar Chart",
            "Line Chart",
            "Pie Chart",
            "Histogram",
            "Scatter Plot",
            "Correlation Heatmap"
        ]
    )

    numeric_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns.tolist()

    all_columns = cleaned_df.columns.tolist()

    # Histogram
    if chart_type == "Histogram":

        if numeric_columns:

            column = st.selectbox(
                "Select Numeric Column",
                numeric_columns
            )

            fig = px.histogram(
                cleaned_df,
                x=column,
                title=f"Distribution of {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning("No numeric columns found.")

    # Scatter plot
    elif chart_type == "Scatter Plot":

        if len(numeric_columns) >= 2:

            x = st.selectbox(
                "Select X Axis",
                numeric_columns
            )

            y_options = [
                col for col in numeric_columns
                if col != x
            ]

            y = st.selectbox(
                "Select Y Axis",
                y_options
            )

            fig = px.scatter(
                cleaned_df,
                x=x,
                y=y,
                title=f"{x} vs {y}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numeric columns are required."
            )

    # Correlation heatmap
    elif chart_type == "Correlation Heatmap":

        if len(numeric_columns) >= 2:

            correlation = cleaned_df[
                numeric_columns
            ].corr()

            fig = px.imshow(
                correlation,
                text_auto=True,
                title="Correlation Heatmap",
                aspect="auto"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numeric columns are required."
            )

    # Bar chart, line chart and pie chart
    else:

        x = st.selectbox(
            "Select Category Column",
            all_columns
        )

        if chart_type == "Bar Chart":

            if numeric_columns:

                y = st.selectbox(
                    "Select Value Column",
                    numeric_columns
                )

                chart_data = (
                    cleaned_df.groupby(
                        x,
                        dropna=False
                    )[y]
                    .mean()
                    .reset_index()
                )

                fig = px.bar(
                    chart_data,
                    x=x,
                    y=y,
                    title=f"Average {y} by {x}"
                )

            else:

                chart_data = (
                    cleaned_df[x]
                    .fillna("Missing")
                    .value_counts()
                    .rename_axis(x)
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    chart_data,
                    x=x,
                    y="Count",
                    title=f"Count by {x}"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        elif chart_type == "Line Chart":

            if numeric_columns:

                y = st.selectbox(
                    "Select Numeric Value",
                    numeric_columns
                )

                chart_data = (
                    cleaned_df.groupby(
                        x,
                        dropna=False
                    )[y]
                    .mean()
                    .reset_index()
                )

                fig = px.line(
                    chart_data,
                    x=x,
                    y=y,
                    markers=True,
                    title=f"{y} Trend"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Line chart requires numeric data."
                )

        elif chart_type == "Pie Chart":

            if numeric_columns:

                y = st.selectbox(
                    "Select Value Column",
                    numeric_columns
                )

                chart_data = (
                    cleaned_df.groupby(
                        x,
                        dropna=False
                    )[y]
                    .sum()
                    .reset_index()
                )

                fig = px.pie(
                    chart_data,
                    names=x,
                    values=y,
                    title=f"{y} Distribution"
                )

            else:

                chart_data = (
                    cleaned_df[x]
                    .fillna("Missing")
                    .value_counts()
                    .rename_axis(x)
                    .reset_index(name="Count")
                )

                fig = px.pie(
                    chart_data,
                    names=x,
                    values="Count",
                    title=f"{x} Distribution"
                )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =====================================
# TAB 3: AI INSIGHTS
# =====================================

with tab3:

    st.header("🤖 AI-Powered Insights")

    insights = generate_insights(
        cleaned_df
    )

    st.subheader("Automatic Data Analysis")

    for insight in insights:

        st.info("💡 " + insight)

    st.divider()

    st.subheader("Anomaly Detection")

    anomalies = detect_anomalies(
        cleaned_df
    )

    if anomalies.empty:

        st.warning(
            "Not enough suitable numeric data "
            "to detect anomalies."
        )

    else:

        st.write(
            f"Potential unusual records: {len(anomalies)}"
        )

        st.dataframe(
            anomalies,
            use_container_width=True
        )

        st.caption(
            "Anomaly flags identify unusual records "
            "for further investigation."
        )


# =====================================
# TAB 4: REPORT GENERATION
# =====================================

with tab4:

    st.header("📄 Analysis Report")

    insights = generate_insights(
        cleaned_df
    )

    st.subheader("Dataset Information")

    st.write(
        f"Total Rows: {len(cleaned_df)}"
    )

    st.write(
        f"Total Columns: {len(cleaned_df.columns)}"
    )

    st.write(
        f"Missing Values: "
        f"{cleaned_df.isnull().sum().sum()}"
    )

    st.subheader("Cleaning Operations")

    for message in cleaning_messages:

        st.write("•", message)

    st.subheader("Important Findings")

    for insight in insights:

        st.write("💡", insight)

    # Create report
    report = [
        "# AI-Powered Data Analysis Report",
        "",
        f"Total Rows: {len(cleaned_df)}",
        f"Total Columns: {len(cleaned_df.columns)}",
        f"Missing Values: {cleaned_df.isnull().sum().sum()}",
        "",
        "## Data Cleaning"
    ]

    report.extend(cleaning_messages)

    report.extend([
        "",
        "## AI Insights"
    ])

    report.extend(insights)

    report_text = "\n".join(report)

    st.download_button(
        "⬇️ Download Analysis Report",
        data=report_text,
        file_name="analysis_report.txt",
        mime="text/plain"
    )

    st.download_button(
        "⬇️ Download Cleaned Dataset",
        data=cleaned_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    st.success(
        "Your analysis report is ready!"
    )