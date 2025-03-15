import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

def generate_all_plots(df):
    """
    Function to generate and display various plots for data analysis in Streamlit.
    """
    
    # 🎯 1️⃣ Data Distribution (Histogram for each column)
    def plot_histograms():
        st.subheader("📊 Data Distribution After Cleaning")
        fig, ax = plt.subplots(figsize=(14, 10))
        df.hist(bins=20, color="skyblue", edgecolor="black", ax=ax)
        st.pyplot(fig)  # Pass the figure to Streamlit

    # 🎯 2️⃣ Correlation Heatmap
    def plot_correlation_heatmap():
        st.subheader("🔍 Correlation Heatmap After Cleaning")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
        st.pyplot(fig)  # Pass the figure to Streamlit

    # 🎯 3️⃣ Boxplot for Outlier Detection
    def plot_boxplots():
        st.subheader("🚀 Boxplot for Outlier Detection")
        fig, ax = plt.subplots(figsize=(14, 6))
        df.boxplot(rot=45, patch_artist=True, boxprops=dict(facecolor="lightblue"), ax=ax)
        st.pyplot(fig)  # Pass the figure to Streamlit

    # 🎯 4️⃣ Pairplot for Feature Relationships
    def plot_pairplot():
        st.subheader("🔄 Pairplot for Feature Relationships")
        fig, ax = plt.subplots(figsize=(10, 8))
        sample_df = df.sample(min(500, len(df)))  # Limit for better performance
        sns.pairplot(sample_df, diag_kind="kde", plot_kws={"alpha": 0.6})
        st.pyplot(fig)  # Pass the figure to Streamlit

    # 🎯 5️⃣ Time-Series Trends (If Date Exists)
    def plot_time_series():
        st.subheader("📅 Monthly Trends")
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            for date_col in date_cols:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                fig, ax = plt.subplots(figsize=(12, 6))
                df.resample("M").mean().plot(ax=ax, title="📅 Monthly Trends")
                st.pyplot(fig)  # Pass the figure to Streamlit
        else:
            st.write("📅 No date column found for time-series trends")

    # 🎯 6️⃣ KDE Plot (Data Density)
    def plot_kde():
        st.subheader("📊 Data Density (KDE Plot)")
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in df.select_dtypes(include=np.number).columns[:5]:  # Limit to first 5 cols
            sns.kdeplot(df[col], fill=True, label=col, alpha=0.5, ax=ax)
        ax.set_title("📊 Data Density (KDE Plot)")
        ax.legend()
        st.pyplot(fig)  # Pass the figure to Streamlit

    # 🎯 7️⃣ Category Count Plots
    def plot_category_counts():
        st.subheader("📊 Distribution of Categorical Data")
        cat_cols = df.select_dtypes(include="object").columns
        if len(cat_cols) > 0:
            for col in cat_cols[:3]:  # Limit to first 3 categorical columns
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(y=df[col], palette="coolwarm", ax=ax)
                ax.set_title(f"📊 Distribution of {col}")
                st.pyplot(fig)  # Pass the figure to Streamlit
        else:
            st.write("📊 No categorical columns found")

    # 🎯 8️⃣ Scatter Plot (Top 2 Correlated Features)
    def plot_top_correlation():
        st.subheader("📉 Relationship between Top 2 Correlated Features")
        corr_matrix = df.corr()
        top_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
        top_pair = top_corr[top_corr < 1].index[0]  # Pick most correlated pair
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=df[top_pair[0]], y=df[top_pair[1]], alpha=0.7, edgecolor="black", ax=ax)
        ax.set_title(f"📉 Relationship between {top_pair[0]} & {top_pair[1]}")
        ax.set_xlabel(top_pair[0])
        ax.set_ylabel(top_pair[1])
        st.pyplot(fig)  # Pass the figure to Streamlit

    # 🎯 9️⃣ Violin Plot (Data Spread)
    def plot_violin():
        st.subheader("🎻 Violin Plot (Data Spread Analysis)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df.select_dtypes(include=np.number), palette="coolwarm", ax=ax)
        ax.set_title("🎻 Violin Plot (Data Spread Analysis)")
        st.pyplot(fig)  # Pass the figure to Streamlit

    # Generate and display all plots
    plot_histograms()
    plot_correlation_heatmap()
    plot_boxplots()
    plot_pairplot()
    plot_time_series()
    plot_kde()
    plot_category_counts()
    plot_top_correlation()
    plot_violin()

# Main execution: Run the Streamlit App and display plots
# def main():
#     # Load data
#     df = pd.read_csv("cleaned_data.csv")  # Replace with your actual dataset
#     st.title("📊 Data Visualizations")
#     generate_all_plots(df)

# if __name__ == "__main__":
#     main()
