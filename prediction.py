import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Union

# --- Page Configuration ---
st.set_page_config(
    page_title="College Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Configuration ---
# NOTE: Ensure this CSV file is in the same directory as your Streamlit script.
FILE_PATH = "cet_cutoffs_with_categories.csv" 

# --- Caching the Data Loading ---
@st.cache_data # This decorator caches the data, so it's loaded only once.
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Loads the cutoff data from a CSV file and performs basic cleanup.
    This function is cached to improve performance.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Normalize column names to a predictable format
        df.rename(columns=lambda c: c.strip().lower().replace(' ', '_'), inplace=True)

        # Accept either 'closing_percentile' or 'cutoff_percentile' in source; normalize to 'closing_percentile'
        if 'cutoff_percentile' in df.columns and 'closing_percentile' not in df.columns:
            df.rename(columns={'cutoff_percentile': 'closing_percentile'}, inplace=True)

        required_columns = ['college_name', 'branch_name', 'category', 'closing_percentile']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Error: CSV is missing required columns. Found: {df.columns.tolist()}. Required: {required_columns}")
            return pd.DataFrame()

        df['closing_percentile'] = pd.to_numeric(df['closing_percentile'], errors='coerce')
        df.dropna(subset=['closing_percentile'], inplace=True)
        
        # Group by the main fields and find the minimum cutoff percentile for each group
        df_cutoffs = df.groupby(
            ['college_name', 'branch_name', 'category']
        )['closing_percentile'].min().reset_index()
        df_cutoffs.rename(columns={'closing_percentile': 'cutoff_percentile'}, inplace=True)
        
        return df_cutoffs
    except FileNotFoundError:
        st.error(f"FATAL ERROR: The data file was NOT found at '{file_path}'. Please ensure the file is in the same directory as the script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during file loading: {e}")
        return pd.DataFrame()

# --- Prediction Logic ---

def predict_admission(
    df: pd.DataFrame, percentile: float, college_name: str, branch_name: str, category: str
) -> Dict[str, Union[str, float, None]]:
    """Predicts admission probability for a specific college choice."""
    if df.empty:
        return {"status": "Error", "message": "Cutoff data is not available.", "cutoff": None}

    match = df[
        (df['college_name'] == college_name) &
        (df['branch_name'] == branch_name) &
        (df['category'] == category)
    ]

    if match.empty:
        return {"status": "Unknown", "message": f"No cutoff data found for the selected combination.", "cutoff": None}

    cutoff_value = match['cutoff_percentile'].iloc[0]

    # Determine admission probability based on the percentile difference
    if percentile >= cutoff_value + 0.1:
        status = 'High Probability'
    elif percentile >= cutoff_value - 0.5:
        status = 'Borderline'
    else:
        status = 'Low Probability'
    
    return {"status": status, "cutoff": cutoff_value}

def find_admissible_options(
    df: pd.DataFrame, student_percentile: float, student_category: str, margin_of_safety: float = 0.5
) -> pd.DataFrame:
    """Finds all colleges where the student has a chance of admission."""
    if df.empty: return pd.DataFrame()

    category_matches = df[df['category'] == student_category].copy()
    if category_matches.empty: return pd.DataFrame()

    # Find colleges where the cutoff is within the student's percentile + margin
    admissible_options = category_matches[
        category_matches['cutoff_percentile'] <= (student_percentile + margin_of_safety)
    ].copy()

    # Calculate the difference for sorting and display
    admissible_options['score_diff'] = student_percentile - admissible_options['cutoff_percentile']
    admissible_options.sort_values(by='score_diff', ascending=True, inplace=True)
    
    # Return a clean DataFrame for display
    return admissible_options[['college_name', 'branch_name', 'cutoff_percentile', 'score_diff']]


# --- Streamlit UI ---

st.title("🎓 Maharashtra College Admission Predictor")
st.markdown("This tool helps you predict college admissions based on historical cutoff data.")

# Load data and handle potential errors
cutoff_data = load_and_prepare_data(FILE_PATH)

if cutoff_data.empty:
    st.warning("Prediction engine cannot start because the data failed to load. Please check the file path and format.")
    st.stop() # Halts the app if data isn't loaded

st.success(f"Successfully loaded {len(cutoff_data)} unique college/branch/category combinations.")

# --- Sidebar for Mode Selection and Inputs ---
st.sidebar.header("⚙️ Controls")
app_mode = st.sidebar.radio(
    "Choose your tool:",
    ('Reverse Search (Find Possible Colleges)', 'Direct Prediction (Check a Specific College)')
)

# --- UI for Reverse Search ---
if app_mode == 'Reverse Search (Find Possible Colleges)':
    st.header("Find All Admissible Colleges")
    st.markdown("Enter your details to see a list of colleges you have a chance of getting into.")

    unique_categories = sorted(cutoff_data['category'].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        student_percentile = st.number_input("Enter your Percentile Score", min_value=0.0, max_value=100.0, value=95.0, step=0.01)
    with col2:
        student_category = st.selectbox("Select your Category", options=unique_categories, index=unique_categories.index('GOPENS') if 'GOPENS' in unique_categories else 0)

    margin = st.slider(
        "Margin of Safety (How far below the cutoff should we look?)",
        min_value=0.0, max_value=5.0, value=1.0, step=0.1,
        help="A margin of 1.0 means we will show colleges where the cutoff is up to 1 percentile point *higher* than your score."
    )

    if st.button("🔍 Find Colleges", type="primary"):
        with st.spinner("Searching for options..."):
            results_df = find_admissible_options(cutoff_data, student_percentile, student_category, margin)
        
        if results_df.empty:
            st.warning("No colleges found matching your criteria. Try increasing the margin of safety or checking a different category.")
        else:
            st.success(f"Found {len(results_df)} potential options for you!")
            
            # Renaming for better readability in the app
            display_df = results_df.rename(columns={
                'college_name': 'College Name',
                'branch_name': 'Branch',
                'cutoff_percentile': 'Last Year Cutoff %',
                'score_diff': 'Your Score vs Cutoff'
            })
            
            # Formatting the difference column to show + or -
            display_df['Your Score vs Cutoff'] = display_df['Your Score vs Cutoff'].apply(lambda x: f"{x:+.2f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# --- UI for Direct Prediction ---
elif app_mode == 'Direct Prediction (Check a Specific College)':
    st.header("Check Admission for a Specific College")
    st.markdown("Select a specific college and branch to see your admission probability.")

    # Get unique lists for dropdown menus
    unique_colleges = sorted(cutoff_data['college_name'].dropna().unique())
    unique_categories = sorted(cutoff_data['category'].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        student_percentile = st.number_input("Enter your Percentile Score", min_value=0.0, max_value=100.0, value=90.0, step=0.01)
        selected_college = st.selectbox("Select College", options=unique_colleges)

    with col2:
        student_category = st.selectbox("Select your Category", options=unique_categories, index=unique_categories.index('GOPENS') if 'GOPENS' in unique_categories else 0)
        
        # Dynamically update branch selection based on the chosen college
        available_branches = sorted(cutoff_data[cutoff_data['college_name'] == selected_college]['branch_name'].dropna().unique())
        selected_branch = st.selectbox("Select Branch", options=available_branches)

    if st.button("🔮 Predict My Chances", type="primary"):
        result = predict_admission(cutoff_data, student_percentile, selected_college, selected_branch, student_category)
        
        st.subheader("Prediction Result")
        
        if result['status'] == 'Error' or result['status'] == 'Unknown':
            st.error(f"Could not make a prediction. Reason: {result.get('message', 'No data available.')}")
        else:
            status = result['status']
            cutoff = result['cutoff']
            
            # Display colored status messages
            if status == 'High Probability':
                st.success(f"✅ **{status}** of Admission!")
            elif status == 'Borderline':
                st.warning(f"⚠️ **{status}** Chance.")
            else: # Low Probability
                st.error(f"❌ **{status}** of Admission.")
            
            # Use st.metric to display the cutoff comparison clearly
            st.metric(label=f"Last Year's Cutoff for {student_category}", value=f"{cutoff:.2f} %ile", delta=f"{(student_percentile - cutoff):.2f} %ile vs Your Score")
            st.markdown(f"Your score of **{student_percentile:.2f}%** is compared to the cutoff of **{cutoff:.2f}%**.")