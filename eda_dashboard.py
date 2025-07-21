import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from groq import Groq
import warnings
import plotly.express as px 
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Enhanced EDA + AI Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)


# === TITLE ===
st.markdown('📊 Enhanced EDA + Agentic AI Dashboard</h1>', unsafe_allow_html=True)

# === SIDEBAR FOR MODE SELECTION ===
st.sidebar.markdown('🎛️ Dashboard Mode</h2>', unsafe_allow_html=True)
mode = st.sidebar.selectbox("Select Mode", ["Manual EDA", "AI-Based EDA"])

# === FILE UPLOAD ===

st.markdown("### 📁 Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


if uploaded_file:
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
    
    df = st.session_state.df
    st.success("File uploaded successfully!")
    
    # === DATASET OVERVIEW ===
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📋 Dataset Overview</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", df.shape[0])
    with col2:
        st.metric("📈 Columns", df.shape[1])
    with col3:
        st.metric("❌ Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("🔢 Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    
    st.markdown("### 👀 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # === AI BASED EDA ===
    if mode == "AI-Based EDA":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">🤖 AI-Based Insights and Visualization</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Dataset Summary")
            st.write(df.describe())
            
            st.markdown("#### Data Types")
            st.write(df.dtypes)

        with col2:
            st.markdown("#### Missing Values")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                st.bar_chart(missing_data)
            else:
                st.success("No missing values found!")

        st.markdown("#### Top Correlations")
        try:
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                top_corr = corr.abs().unstack().sort_values(key=lambda x: x, ascending=False).drop_duplicates()
                st.dataframe(top_corr[top_corr < 1].head(10))

                st.markdown("#### Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")

        # === GROQ AI INSIGHTS ===
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        if st.button("🧠 Generate AI Insights", key="ai_insights"):
            with st.spinner("🔍 Analyzing your data with AI..."):
                try:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    
                    # Create a comprehensive dataset summary
                    summary_info = {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": df.dtypes.to_dict(),
                        "missing_values": df.isnull().sum().to_dict(),
                        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "No numeric columns"
                    }
                    
                    prompt = f"""
                    Analyze this dataset and provide insights:
                    
                    Dataset Info: {summary_info}
                    
                    Please provide:
                    1. Key insights about the data
                    2. Recommended visualizations
                    3. Data quality observations
                    4. Suggested preprocessing steps
                    5. Potential analysis directions
                    
                    Keep response concise and actionable.
                    """

                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are an expert data analyst. Provide clear, actionable insights."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama3-70b-8192"
                    )

                    st.markdown("#### 🔍 AI-Generated Insights")
                    st.info(chat_completion.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Error generating AI insights: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # === MANUAL EDA ===
    else:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">🛠️ Manual EDA & Analysis</h2>', unsafe_allow_html=True)

        # === TASK SELECTION ===
        st.markdown("### 🎯 Select Analysis Task")
        # ...existing code...
        task = st.radio(
            "### 🎯 Select Analysis Task",
            ["🧹 Data Cleaning", "🔄 Data Transformation", "📊 Visualization", "🎯 Classification"],
            horizontal=True
        )

        data_cleaning = task == "🧹 Data Cleaning"
        data_transform = task == "🔄 Data Transformation"
        visualization = task == "📊 Visualization"
        classification = task == "🎯 Classification"
        
        
        st.markdown('</div>', unsafe_allow_html=True)

        # === DATA CLEANING ===
        if data_cleaning:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">🧹 Data Cleaning Options</h3>', unsafe_allow_html=True)
            
            # Missing Values Section
            st.markdown("#### Missing Values Analysis")
            missing_summary = df.isnull().sum()
            missing_summary = missing_summary[missing_summary > 0]
            
            if len(missing_summary) > 0:
                st.write("Columns with missing values:")
                st.dataframe(missing_summary.to_frame("Missing Count"))
                
                # Missing value handling options
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Handle Missing Values")
                    missing_action = st.selectbox(
                        "Choose action for missing values",
                        ["Select Action", "Drop rows with any missing values", "Drop rows with all missing values", 
                         "Fill with mean (numeric)", "Fill with median (numeric)", "Fill with mode", "Fill with custom value"]
                    )
                    
                    if missing_action == "Drop rows with any missing values":
                        if st.button("Apply: Drop rows with any missing values"):
                            st.session_state.df = df.dropna()
                            st.success("Rows with missing values dropped!")
                            st.rerun()
                    
                    elif missing_action == "Drop rows with all missing values":
                        if st.button("Apply: Drop rows with all missing values"):
                            st.session_state.df = df.dropna(how='all')
                            st.success("Rows with all missing values dropped!")
                            st.rerun()
                    
                    elif missing_action == "Fill with mean (numeric)":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        selected_cols = st.multiselect("Select numeric columns to fill with mean", numeric_cols)
                        if st.button("Apply: Fill with mean") and selected_cols:
                            for col in selected_cols:
                                st.session_state.df[col].fillna(df[col].mean(), inplace=True)
                            st.success("Missing values filled with mean!")
                            st.rerun()
                    
                    elif missing_action == "Fill with median (numeric)":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        selected_cols = st.multiselect("Select numeric columns to fill with median", numeric_cols)
                        if st.button("Apply: Fill with median") and selected_cols:
                            for col in selected_cols:
                                st.session_state.df[col].fillna(df[col].median(), inplace=True)
                            st.success("Missing values filled with median!")
                            st.rerun()
                    
                    elif missing_action == "Fill with mode":
                        selected_cols = st.multiselect("Select columns to fill with mode", df.columns)
                        if st.button("Apply: Fill with mode") and selected_cols:
                            for col in selected_cols:
                                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                                st.session_state.df[col].fillna(mode_value, inplace=True)
                            st.success("Missing values filled with mode!")
                            st.rerun()
                    
                    elif missing_action == "Fill with custom value":
                        selected_col = st.selectbox("Select column", df.columns)
                        custom_value = st.text_input("Enter custom value")
                        if st.button("Apply: Fill with custom value") and custom_value:
                            st.session_state.df[selected_col].fillna(custom_value, inplace=True)
                            st.success(f"Missing values in {selected_col} filled with {custom_value}!")
                            st.rerun()
                
                with col2:
                    st.markdown("##### Column-specific Actions")
                    col_to_clean = st.selectbox("Select column for specific cleaning", df.columns)
                    
                    if st.button(f"Drop column '{col_to_clean}'"):
                        st.session_state.df = df.drop(columns=[col_to_clean])
                        st.success(f"Column '{col_to_clean}' dropped!")
                        st.rerun()
                    
                    if st.button(f"Drop rows with missing values in '{col_to_clean}'"):
                        st.session_state.df = df.dropna(subset=[col_to_clean])
                        st.success(f"Rows with missing values in '{col_to_clean}' dropped!")
                        st.rerun()
            else:
                st.success("No missing values found!")
            
            # Duplicate handling
            st.markdown("#### Duplicate Values")
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
            
            if duplicates > 0:
                if st.button("Remove duplicate rows"):
                    st.session_state.df = df.drop_duplicates()
                    st.success("Duplicate rows removed!")
                    st.rerun()
            
            # Outlier detection
            st.markdown("#### Outlier Detection")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)
                
                # Calculate IQR
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                st.metric("Outliers Found", len(outliers))
                
                if len(outliers) > 0:
                    st.write("Outlier bounds:", f"{lower_bound:.2f} to {upper_bound:.2f}")
                    if st.button("Remove outliers"):
                        st.session_state.df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                        st.success("Outliers removed!")
                        st.rerun()

        # === DATA TRANSFORMATION ===
        elif data_transform:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">🔄 Data Transformation Options</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Encoding Options")
                
                # Label Encoding
                st.markdown("##### Label Encoding")
                cat_cols = df.select_dtypes(include='object').columns.tolist()
                if len(cat_cols) > 0:
                    label_encode_cols = st.multiselect("Select columns for Label Encoding", cat_cols)
                    if st.button("Apply Label Encoding") and label_encode_cols:
                        le = LabelEncoder()
                        for col in label_encode_cols:
                            st.session_state.df[col] = le.fit_transform(df[col].astype(str))
                        st.success("Label Encoding Applied!")
                        st.rerun()
                
                # One-Hot Encoding
                st.markdown("##### One-Hot Encoding")
                if len(cat_cols) > 0:
                    onehot_cols = st.multiselect("Select columns for One-Hot Encoding", cat_cols)
                    if st.button("Apply One-Hot Encoding") and onehot_cols:
                        df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
                        st.session_state.df = df_encoded
                        st.success("One-Hot Encoding Applied!")
                        st.rerun()
                
                # Feature Creation
                st.markdown("#### Feature Engineering")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1_feat = st.selectbox("Select first column", numeric_cols, key="feat1")
                    col2_feat = st.selectbox("Select second column", numeric_cols, key="feat2")
                    operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide"])
                    new_col_name = st.text_input("New column name", f"{col1_feat}_{operation.lower()}_{col2_feat}")
                    
                    if st.button("Create Feature") and new_col_name:
                        if operation == "Add":
                            st.session_state.df[new_col_name] = df[col1_feat] + df[col2_feat]
                        elif operation == "Subtract":
                            st.session_state.df[new_col_name] = df[col1_feat] - df[col2_feat]
                        elif operation == "Multiply":
                            st.session_state.df[new_col_name] = df[col1_feat] * df[col2_feat]
                        elif operation == "Divide":
                            st.session_state.df[new_col_name] = df[col1_feat] / df[col2_feat]
                        st.success(f"Feature '{new_col_name}' created!")
                        st.rerun()
            
            with col2:
                st.markdown("#### Scaling Options")
                
                # Standard Scaling
                st.markdown("##### Standard Scaling")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    std_scale_cols = st.multiselect("Select columns for Standard Scaling", numeric_cols)
                    if st.button("Apply Standard Scaling") and std_scale_cols:
                        scaler = StandardScaler()
                        for col in std_scale_cols:
                            st.session_state.df[col] = scaler.fit_transform(df[[col]])
                        st.success("Standard Scaling Applied!")
                        st.rerun()
                
                # Min-Max Scaling
                st.markdown("##### Min-Max Scaling")
                if len(numeric_cols) > 0:
                    minmax_scale_cols = st.multiselect("Select columns for Min-Max Scaling", numeric_cols)
                    if st.button("Apply Min-Max Scaling") and minmax_scale_cols:
                        scaler = MinMaxScaler()
                        for col in minmax_scale_cols:
                            st.session_state.df[col] = scaler.fit_transform(df[[col]])
                        st.success("Min-Max Scaling Applied!")
                        st.rerun()
                
                # Binning
                st.markdown("#### Binning")
                if len(numeric_cols) > 0:
                    bin_col = st.selectbox("Select column for binning", numeric_cols)
                    n_bins = st.slider("Number of bins", 2, 10, 5)
                    bin_labels = st.text_input("Bin labels (comma-separated)", "Low,Medium,High")
                    
                    if st.button("Apply Binning"):
                        labels = [label.strip() for label in bin_labels.split(',')]
                        if len(labels) == n_bins:
                            st.session_state.df[f"{bin_col}_binned"] = pd.cut(df[bin_col], bins=n_bins, labels=labels)
                            st.success(f"Binning applied to {bin_col}!")
                            st.rerun()
                        else:
                            st.error(f"Number of labels ({len(labels)}) must match number of bins ({n_bins})")
            st.markdown('</div>', unsafe_allow_html=True)

        # === VISUALIZATION ===
        elif visualization:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">📊 Custom Visualization</h3>', unsafe_allow_html=True)
            
            plot_col1, plot_col2 = st.columns(2)
            
            with plot_col1:
                plot_type = st.selectbox("Choose plot type", 
                    ["Histogram", "Scatterplot", "Boxplot", "Heatmap", "Countplot", "Pairplot", "Violinplot"])
                
                # Dynamic column selection based on plot type
                if plot_type in ["Histogram", "Boxplot", "Violinplot"]:
                    x_axis = st.selectbox("Select column", df.columns)
                    y_axis = None
                elif plot_type == "Countplot":
                    x_axis = st.selectbox("Select categorical column", df.select_dtypes(include='object').columns)
                    y_axis = None
                elif plot_type == "Heatmap":
                    x_axis = None
                    y_axis = None
                elif plot_type == "Pairplot":
                    x_axis = None
                    y_axis = None
                else:
                    x_axis = st.selectbox("X-axis", df.columns)
                    y_axis = st.selectbox("Y-axis", df.columns)
                
                hue = st.selectbox("Hue (optional)", [None] + df.columns.tolist())
                
                # Customization options
                fig_width = st.slider("Figure width", 5, 15, 10)
                fig_height = st.slider("Figure height", 3, 10, 6)
                
           
                
                # ...inside your visualization section...
            if st.button("Generate Plot"):
                try:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                    if plot_type == "Histogram":
                        sns.histplot(data=df, x=x_axis, hue=hue if hue else None, kde=True, ax=ax)
                        plt.title(f"Histogram of {x_axis}")
                        st.pyplot(fig)
                    elif plot_type == "Scatterplot":
                        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue if hue else None, ax=ax)
                        plt.title(f"Scatterplot of {x_axis} vs {y_axis}")
                        st.pyplot(fig)
                    elif plot_type == "Boxplot":
                        sns.boxplot(data=df, x=x_axis, hue=hue if hue else None, ax=ax)
                        plt.title(f"Boxplot of {x_axis}")
                        st.pyplot(fig)
                    elif plot_type == "Violinplot":
                        sns.violinplot(data=df, x=x_axis, hue=hue if hue else None, ax=ax)
                        plt.title(f"Violinplot of {x_axis}")
                        st.pyplot(fig)
                    elif plot_type == "Countplot":
                        sns.countplot(data=df, x=x_axis, hue=hue if hue else None, ax=ax)
                        plt.title(f"Countplot of {x_axis}")
                        st.pyplot(fig)
                    elif plot_type == "Heatmap":
                        numeric_df = df.select_dtypes(include=[np.number])
                        if len(numeric_df.columns) > 1:
                            corr = numeric_df.corr()
                            sns.heatmap(corr, annot=True, ax=ax)
                            plt.title("Correlation Heatmap")
                            st.pyplot(fig)
                        else:
                            st.error("Need at least 2 numeric columns for heatmap")
                    elif plot_type == "Pairplot":
                        numeric_df = df.select_dtypes(include=[np.number])
                        if len(numeric_df.columns) > 1:
                            if hue and hue in df.columns:
                                g = sns.pairplot(df, vars=numeric_df.columns, hue=hue)
                            else:
                                g = sns.pairplot(df, vars=numeric_df.columns)
                            st.pyplot(g.figure)
                        else:
                            st.error("Need at least 2 numeric columns for pairplot")
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
            st.markdown('</div>',unsafe_allow_html=True )        

        # === CLASSIFICATION ===
        elif classification:
            st.markdown("### 🎯 Machine Learning Classification")
            
            ml_col1, ml_col2 = st.columns(2)
            
            with ml_col1:
                target = st.selectbox("Select target variable", df.columns)
                features = st.multiselect("Select features", df.columns.difference([target]))
                
                test_size = st.slider("Test size", 0.1, 0.5, 0.2)
                random_state = st.number_input("Random state", 0, 100, 42)
                
            with ml_col2:
                model_type = st.selectbox("Select model", 
                    ["Random Forest", "Extra Trees", "Gradient Boosting"])
                
                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of trees", 10, 200, 100)
                    max_depth = st.slider("Max depth", 1, 20, 5)
                
            if st.button("Train Model") and features:
                try:
                    X = df[features]
                    y = df[target]
                    
                    # Check for missing values
                    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                        st.warning("Please handle missing values before training.")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                        
                        if model_type == "Random Forest":
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                random_state=random_state
                            )
                        
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        
                        st.markdown("#### Model Performance")
                        st.text(classification_report(y_test, preds))
                        
                        # Feature importance
                        if hasattr(model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'feature': features,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            st.markdown("#### Feature Importance")
                            st.bar_chart(importance_df.set_index('feature'))
                        
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")


    
 # === PRESET DATASET VISUALIZATIONS ===
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">✨ Quick Dataset Visualizations</h3>', unsafe_allow_html=True)
    viz_tabs = st.tabs(["Column Types", "Missing Values", "Target Distribution", "Correlation"])

    # 1. Column Types Pie Chart
    with viz_tabs[0]:
        st.markdown("#### Column Data Types")
        col_types = df.dtypes.value_counts()
        fig_types = px.pie(
            names=col_types.index.astype(str),
            values=col_types.values,
            title="Column Data Types",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_types, use_container_width=True)

    # 2. Missing Values Bar Chart
    with viz_tabs[1]:
        st.markdown("#### Missing Values per Column")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            fig_missing = px.bar(
                x=missing.index,
                y=missing.values,
                labels={'x': 'Column', 'y': 'Missing Values'},
                title="Missing Values per Column",
                color=missing.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values detected in any column.")

    # 3. Target Distribution Bar Chart (if suitable)
    with viz_tabs[2]:
        st.markdown("#### Target Feature Distribution")
        possible_targets = [col for col in df.columns if df[col].nunique() < df.shape[0] // 2]
        if possible_targets:
            target_col = st.selectbox("Select a target/label column", possible_targets, key="quick_target")
            value_counts = df[target_col].value_counts()
            fig_target = px.bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                labels={'x': target_col, 'y': 'Count'},
                title=f"Distribution of {target_col}",
                color=value_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            st.info("No suitable target column found for distribution analysis.")

    # 4. Correlation Heatmap
    with viz_tabs[3]:
        st.markdown("#### Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")
            st.markdown('</div>', unsafe_allow_html=True)

        # === RESET BUTTON ===
        st.markdown("---")
        if st.button("🔄 Reset to Original Dataset"):
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("Dataset reset to original!")
            st.rerun()
else:
    st.info("Please upload a CSV file to get started!")