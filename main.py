import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import time

# Page configuration
st.set_page_config(
    page_title="Kruskal-Wallis H Test Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .results-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 20px 0;
    }
    .results-header {
        color: #0D47A1;
        font-size: 1.3rem;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Kruskal-Wallis H Test Calculator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">
    A powerful non-parametric statistical tool for comparing three or more independent groups
    </p>
    """, unsafe_allow_html=True)
    
    # Create sidebar for instructions and information
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/mwaskom/seaborn/master/doc/_static/logo-mark-lightbg.svg", width=100)
        st.markdown("### About This Tool")
        st.info("""
        The Kruskal-Wallis H test is used when:
        - You have 3+ independent groups
        - Data doesn't meet normality assumptions
        - You want to compare medians between groups
        """)
        
        st.markdown("### Quick Guide")
        st.markdown("""
        1. Enter the number of groups
        2. Input data for each group (comma-separated)
        3. Set significance level
        4. Click "Run Analysis"
        """)
        
        st.markdown("### Example Data")
        if st.button("Load Example Data"):
            return_example_data()
            
        st.markdown("### 📚 References")
        st.markdown("""
        - [Kruskal-Wallis Test (Wikipedia)](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance)
        - [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)
        """)

        st.markdown("### 👥 Team Members")
        team_members = ["Mustafa A", "Karthik Saran", "Kishaan","Naveen Bharathi","Jayanth Kumar","Mithunavanan","Bensingh","Diwakaran"]
        for member in team_members:
            st.markdown(f"- {member}")

    # Main content area - create two columns
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown('<h2 class="subheader">Input Data</h2>', unsafe_allow_html=True)
        
        # Input section
        num_groups = st.number_input("Number of Groups", min_value=3, max_value=10, value=3, step=1)
        
        group_data = []
        group_names = []
        
        # Create tabs for data input
        tabs = st.tabs([f"Group {i+1}" for i in range(num_groups)])
        
        for i, tab in enumerate(tabs):
            with tab:
                group_names.append(st.text_input("Group Name (optional)", value=f"Group {i+1}", key=f"name_{i}"))
                group_input = st.text_area(
                    "Enter values (comma-separated)",
                    height=100,
                    key=f"group_{i}",
                    help="Enter numeric values separated by commas"
                )
                
                if group_input:
                    try:
                        group = [float(x.strip()) for x in group_input.replace('\n', ',').split(',') if x.strip()]
                        group_data.append(group)
                        st.write(f"Count: {len(group)} values")
                        st.write(f"Range: {min(group):.2f} to {max(group):.2f}")
                    except ValueError:
                        st.error("Invalid input. Please enter numbers only.")
                        group_data.append([])
                else:
                    group_data.append([])
        
        # Advanced options
        with st.expander("Advanced Options"):
            alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
            show_details = st.checkbox("Show detailed calculations", value=False)
            
        # Run test button
        if st.button("Run Analysis", type="primary", use_container_width=True):
            # Validate input data
            if len([g for g in group_data if len(g) > 0]) < 2:
                st.error("Please enter data for at least 2 groups.")
            elif not all(len(g) > 0 for g in group_data):
                st.error("All groups must have at least one value.")
            else:
                # Show progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Process data and get results
                results = calculate_kruskal_wallis(group_data, group_names, alpha, show_details)
                
                # Display results in the second column
                with col2:
                    display_results(results, group_data, group_names, alpha, show_details)

    # Footer
    st.markdown('<div class="footer">© 2025 Statistical Analysis Tools</div>', unsafe_allow_html=True)


def calculate_kruskal_wallis(group_data, group_names, alpha, show_details):
    # Calculate test statistics
    valid_groups = [g for g in group_data if len(g) > 0]
    valid_names = [group_names[i] for i, g in enumerate(group_data) if len(g) > 0]
    
    h_stat, p_val = kruskal(*valid_groups)
    df = len(valid_groups) - 1
    
    # Calculate ranks for detailed output
    all_data = []
    group_indices = []
    
    for i, group in enumerate(valid_groups):
        all_data.extend(group)
        group_indices.extend([i] * len(group))
    
    ranks = pd.DataFrame({
        'Value': all_data,
        'Group': [valid_names[i] for i in group_indices]
    })
    
    # Calculate mean ranks
    ranks['Rank'] = ranks['Value'].rank()
    mean_ranks = ranks.groupby('Group')['Rank'].mean().reset_index()
    
    # Determine result
    significant = p_val < alpha
    
    return {
        'h_stat': h_stat,
        'p_val': p_val,
        'df': df,
        'significant': significant,
        'mean_ranks': mean_ranks,
        'all_data': ranks
    }


def display_results(results, group_data, group_names, alpha, show_details):
    st.markdown('<h2 class="subheader">Analysis Results</h2>', unsafe_allow_html=True)
    
    # Results container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="results-header">📊 Kruskal-Wallis H Test Results</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("H Statistic", f"{results['h_stat']:.4f}")
    with col2:
        st.metric("p-value", f"{results['p_val']:.4f}")
    with col3:
        st.metric("Degrees of Freedom", f"{results['df']}")
    
    # Significance result
    if results['significant']:
        st.success(f"🔍 **Significant difference detected** between groups (p < {alpha})")
        st.markdown("""
        The null hypothesis (all groups have the same distribution) can be rejected.
        This suggests that at least one group is statistically different from the others.
        """)
    else:
        st.info(f"🔍 **No significant difference** between groups (p ≥ {alpha})")
        st.markdown("""
        Failed to reject the null hypothesis. There's insufficient evidence 
        to conclude that the groups differ significantly.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data visualization
    st.markdown('<h2 class="subheader">Data Visualization</h2>', unsafe_allow_html=True)
    
    # Create a DataFrame for visualization
    df_viz = pd.DataFrame()
    for i, (group, name) in enumerate(zip(group_data, group_names)):
        if len(group) > 0:
            df_temp = pd.DataFrame({
                'Value': group,
                'Group': [name] * len(group)
            })
            df_viz = pd.concat([df_viz, df_temp])
    
    # Create multiple visualization tabs
    viz_tabs = st.tabs(["Box Plot", "Violin Plot", "Mean Ranks", "Data Table"])
    
    with viz_tabs[0]:
        # Box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Group', y='Value', data=df_viz, palette='Blues', ax=ax)
        ax.set_title('Distribution of Values by Group')
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')
        st.pyplot(fig)
    
    with viz_tabs[1]:
        # Violin plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='Group', y='Value', data=df_viz, palette='Blues', inner='quartile', ax=ax)
        ax.set_title('Distribution Density by Group')
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')
        st.pyplot(fig)
    
    with viz_tabs[2]:
        # Mean ranks visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Group', y='Rank', data=results['mean_ranks'], palette='Blues', ax=ax)
        ax.set_title('Mean Ranks by Group')
        ax.set_xlabel('Group')
        ax.set_ylabel('Mean Rank')
        st.pyplot(fig)
        
        # Show mean ranks table
        st.subheader("Mean Ranks by Group")
        st.table(results['mean_ranks'].set_index('Group'))
    
    with viz_tabs[3]:
        # Data table
        st.subheader("Raw Data and Ranks")
        st.dataframe(results['all_data'])
    
    # Detailed calculation section (optional)
    if show_details:
        st.markdown('<h2 class="subheader">Detailed Calculations</h2>', unsafe_allow_html=True)
        
        with st.expander("View Calculation Details"):
            # Formula
            st.latex(r'''
            H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
            ''')
            
            # Explanation of the formula components
            st.markdown("""
            Where:
            - N is the total number of observations
            - k is the number of groups
            - Ri is the sum of ranks for group i
            - ni is the number of observations in group i
            """)
            
            # Display rank details
            st.subheader("Rank Details")
            st.dataframe(results['all_data'])


def return_example_data():
    # Predefined example data sets
    example_data = [
        [8.5, 9.2, 9.6, 8.8, 9.1, 9.3],
        [7.5, 7.8, 8.2, 7.1, 7.3, 7.6],
        [6.5, 6.8, 6.2, 6.9, 6.4, 6.3]
    ]
    
    example_names = ["Treatment A", "Treatment B", "Control Group"]
    
    # Set the data in the session state
    for i, (data, name) in enumerate(zip(example_data, example_names)):
        st.session_state[f"name_{i}"] = name
        st.session_state[f"group_{i}"] = ", ".join(map(str, data))
    
    st.success("Example data loaded! Please go to the input tabs to see the data.")

if __name__ == "__main__":
    main()