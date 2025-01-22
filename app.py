import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from itertools import combinations
from sklearn.metrics import (
    mean_squared_error, 
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf

# session state
def init_session_state():
    session_state_keys = {
        'df': None,
        'train_df': None,
        'test_df': None,
        'selected_features': [],
        'target_column': None,
        'model': None,
        'class_weights': None
    }
    
    for key, default in session_state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session_state()

def main():
    st.set_page_config(page_title="ML Pipeline", layout="wide")
    st.title("📊 End-to-End ML Pipeline")

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", [
            "Data Upload",
            "Data Exploration",
            "Data Cleaning",
            "Feature Engineering",
            "Model Training",
            "Prediction",
            "Export"
        ])

    if page == "Data Upload":
        render_data_upload()
    elif page == "Data Exploration":
        render_data_exploration()
    elif page == "Data Cleaning":
        render_data_cleaning()
    elif page == "Feature Engineering":
        render_feature_engineering()
    elif page == "Model Training":
        render_model_training()
    elif page == "Prediction":
        render_prediction()
    elif page == "Export":
        render_export()

def render_data_upload():
    st.header("📤 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")

            st.subheader("Dataset Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(st.session_state.df))
            col2.metric("Columns", len(st.session_state.df.columns))
            col3.metric("Missing Values", st.session_state.df.isna().sum().sum())
            
            #data preview
            with st.expander("Preview First 10 Rows"):
                st.dataframe(st.session_state.df.head(10))
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def render_data_exploration():
    st.header("🔍 Data Exploration")
    
    if st.session_state.df is None:
        st.warning("Please upload data first!")
        return

    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.df.head(10), height=300)
    
    st.subheader("Column Data Types")
    dtype_info = pd.DataFrame({
        'Column Name': st.session_state.df.columns,
        'Data Type': st.session_state.df.dtypes.astype(str),
        'Missing Values': st.session_state.df.isna().sum().values
    })
    st.table(dtype_info.style.format({'Missing Values': '{:,.0f}'}).set_properties(**{
        'text-align': 'left',
        'white-space': 'pre-wrap'
    }))

    # Visualization 
    st.subheader("Data Visualization")
    cols = st.session_state.df.columns.tolist()
    col1, col2 = st.columns(2)
    plot_type = col1.selectbox("Select Visualization Type", [
        "Histogram", 
        "Scatter Plot", 
        "Box Plot", 
        "Correlation Matrix"
    ])
    
    if plot_type in ["Scatter Plot"]:
        x_col = col2.selectbox("X Axis", cols)
        y_col = col2.selectbox("Y Axis", cols)
    
    if st.button("Generate Visualization"):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == "Histogram":
                selected_col = col2.selectbox("Select Column", cols)
                sns.histplot(st.session_state.df[selected_col], kde=True, ax=ax)
                plt.title(f"Distribution of {selected_col}")
                
            elif plot_type == "Scatter Plot":
                sns.scatterplot(
                    data=st.session_state.df,
                    x=x_col,
                    y=y_col,
                    ax=ax
                )
                plt.title(f"{x_col} vs {y_col}")
                
            elif plot_type == "Box Plot":
                selected_col = col2.selectbox("Select Column", cols)
                sns.boxplot(x=st.session_state.df[selected_col], ax=ax)
                plt.title(f"Box Plot of {selected_col}")
                
            elif plot_type == "Correlation Matrix":
                numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                corr_matrix = st.session_state.df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                plt.title("Correlation Matrix")
                
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

def render_data_cleaning():
    st.header("🧼 Data Cleaning")
    
    if st.session_state.df is None:
        st.warning("Please upload data first!")
        return
    
    st.subheader("Missing Values Handling")
    missing_cols = st.session_state.df.columns[st.session_state.df.isnull().any()].tolist()
    
    if missing_cols:
        col1, col2 = st.columns(2)
        selected_col = col1.selectbox("Select Column", missing_cols)
        strategy = col2.selectbox("Handling Method", ['drop', 'mean', 'median', 'mode', 'custom'])
        
        custom_value = None
        if strategy == 'custom':
            custom_value = st.text_input("Enter Custom Value")
            
        if st.button("Handle Missing Values"):
            try:
                if strategy == 'drop':
                    st.session_state.df = st.session_state.df.dropna(subset=[selected_col])
                elif strategy == 'custom':
                    st.session_state.df[selected_col] = st.session_state.df[selected_col].fillna(custom_value)
                else:
                    if strategy == 'mean':
                        fill_val = st.session_state.df[selected_col].mean()
                    elif strategy == 'median':
                        fill_val = st.session_state.df[selected_col].median()
                    elif strategy == 'mode':
                        fill_val = st.session_state.df[selected_col].mode()[0]
                        
                    st.session_state.df[selected_col] = st.session_state.df[selected_col].fillna(fill_val)
                    
                st.success(f"Handled missing values in {selected_col}!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.success("🎉 No missing values found!")

def render_feature_engineering():
    st.header("⚙️ Feature Engineering")
    
    if st.session_state.df is None:
        st.warning("Please upload data first!")
        return
    
    operation = st.selectbox("Select Operation", [
        'Transformations', 
        'One-Hot Encoding', 
        'Convert to Numeric'
    ])

    if operation == 'Transformations':
        available_cols = st.session_state.df.columns.tolist()
        selected_cols = st.multiselect("Select Columns", available_cols)
        
        feat_type = st.selectbox("Transformation Type", [
            'polynomial', 'interaction', 'log', 'sqrt'
        ])
        
        params = {}
        if feat_type == 'polynomial':
            params['degree'] = st.number_input("Polynomial Degree", 2, 5, 2)
        
        if st.button("Apply Transformation"):
            if not selected_cols:
                st.warning("Please select at least one column!")
                return
                
            try:
                if feat_type == 'polynomial':
                    for col in selected_cols:
                        poly = PolynomialFeatures(params['degree'], include_bias=False)
                        transformed = poly.fit_transform(st.session_state.df[[col]])
                        feature_names = [f"{col}_poly_{i+1}" for i in range(transformed.shape[1])]
                        st.session_state.df[feature_names] = transformed
                
                elif feat_type == 'interaction':
                    for a, b in combinations(selected_cols, 2):
                        st.session_state.df[f"{a}_x_{b}"] = (
                            st.session_state.df[a] * st.session_state.df[b]
                        )
                
                elif feat_type == 'log':
                    for col in selected_cols:
                        st.session_state.df[f"{col}_log"] = np.log1p(st.session_state.df[col])
                
                elif feat_type == 'sqrt':
                    for col in selected_cols:
                        st.session_state.df[f"{col}_sqrt"] = np.sqrt(st.session_state.df[col])
                
                st.success("Transformations applied successfully!")
                st.rerun()
                    
            except Exception as e:
                st.error(f"Transformation failed: {str(e)}")

    elif operation == 'One-Hot Encoding':
        st.subheader("One-Hot Encoding")
        categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            selected_cols = st.multiselect("Select Categorical Columns", categorical_cols)
            max_unique = st.number_input("Max Unique Categories", 2, 100, 10,
                                       help="Columns with more unique values than this will be excluded")
            
            if st.button("Apply One-Hot Encoding"):
                try:
                    for col in selected_cols:
                        if st.session_state.df[col].nunique() <= max_unique:
                            dummies = pd.get_dummies(st.session_state.df[col], prefix=col)
                            st.session_state.df = pd.concat([st.session_state.df, dummies], axis=1)
                            st.session_state.df.drop(col, axis=1, inplace=True)
                        else:
                            st.warning(f"Skipped {col} - too many unique values (> {max_unique})")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("No categorical columns found for one-hot encoding")

    elif operation == 'Convert to Numeric':
        st.subheader("Convert to Numeric")
        convert_col = st.selectbox("Select Column", st.session_state.df.columns)
        handling_strategy = st.selectbox("Handle Non-Numeric Values", [
            'coerce', 
            'fill with median', 
            'fill with mean'
        ])
        
        if st.button("Convert to Numeric"):
            try:
                st.session_state.df[convert_col] = pd.to_numeric(
                    st.session_state.df[convert_col], 
                    errors='coerce'
                )
                
                if handling_strategy != 'coerce':
                    if handling_strategy == 'fill with median':
                        fill_val = st.session_state.df[convert_col].median()
                    else:
                        fill_val = st.session_state.df[convert_col].mean()
                        
                    st.session_state.df[convert_col].fillna(fill_val, inplace=True)
                    
                st.rerun()
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")


def render_model_training():
    st.header("🤖 Model Training")
    
    if st.session_state.df is None:
        st.warning("Please upload data first!")
        return
    
    available_features = st.session_state.df.columns.tolist()
    st.session_state.selected_features = st.multiselect("Select Features", available_features, default=available_features)
    st.session_state.target_column = st.selectbox("Target Column", available_features)
    problem_type = st.selectbox("Problem Type", ["regression", "classification"])
    
    model_options = {
        "regression": ["Linear Regression", "Random Forest", "SVR", "Neural Network"],
        "classification": ["Logistic Regression", "Random Forest", "SVC", "Neural Network"]
    }
    model_type = st.selectbox("Model Type", model_options[problem_type])
    
    advanced_col1, advanced_col2 = st.columns(2)
    with advanced_col1:
        class_weight = None
        if problem_type == "classification":
            class_weight = st.selectbox("Class Weight Handling", ['None', 'balanced', 'custom'])
            if class_weight == 'custom':
                classes = st.session_state.df[st.session_state.target_column].unique()
                class_weights = {cls: st.number_input(f"Weight for {cls}", 0.1, 2.0, 1.0) for cls in classes}
                st.session_state.class_weights = class_weights

        early_stopping = False
        if model_type == "Neural Network":
            early_stopping = st.checkbox("Enable Early Stopping")
            if early_stopping:
                patience = st.number_input("Patience Epochs", 1, 10, 3)
                monitor = st.selectbox("Monitor Metric", ['val_loss', 'val_accuracy'])

    with advanced_col2:
        custom_loss = None
        if model_type == "Neural Network":
            loss_options = {
                "regression": ['mean_squared_error', 'mean_absolute_error'],
                "classification": ['binary_crossentropy', 'categorical_crossentropy']
            }
            custom_loss = st.selectbox("Loss Function", loss_options[problem_type])
        
        cm_normalize = False
        cm_percentage = False
        if problem_type == "classification":
            cm_normalize = st.checkbox("Normalize Confusion Matrix")
            cm_percentage = st.checkbox("Show Percentages in CM")

    nn_params = {}
    if model_type == "Neural Network":
        st.subheader("Neural Network Configuration")
        nn_params['num_hidden_layers'] = st.number_input("Number of Hidden Layers", 1, 5, 1)
        nn_params['neurons_per_layer'] = st.number_input("Neurons per Hidden Layer", 1, 256, 64)
        nn_params['activation'] = st.selectbox("Hidden Activation", ['relu', 'sigmoid', 'tanh', 'elu'])
        nn_params['output_activation'] = st.selectbox("Output Activation", ['linear', 'relu'] if problem_type == "regression" else ['sigmoid', 'softmax'])
        nn_params['optimizer'] = st.selectbox("Optimizer", ['adam', 'sgd', 'rmsprop'])
        nn_params['learning_rate'] = st.number_input("Learning Rate", 0.0001, 1.0, 0.001, format="%.4f")
        nn_params['epochs'] = st.slider("Epochs", 1, 100, 10)
        nn_params['batch_size'] = st.selectbox("Batch Size", [16, 32, 64, 128])
    
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    
    if st.button("Train Model"):
        if not st.session_state.selected_features:
            st.error("Please select at least one feature!")
            return
            
        try:
            X = st.session_state.df[st.session_state.selected_features]
            y = st.session_state.df[st.session_state.target_column]
            X = pd.get_dummies(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            if model_type == "Neural Network":
                if problem_type == "regression":
                    output_neurons, loss = 1, custom_loss or 'mean_squared_error'
                    y_train_vals = y_train.values.reshape(-1, 1)
                    y_test_vals = y_test.values.reshape(-1, 1)
                else:
                    num_classes = len(np.unique(y_train))
                    output_neurons = 1 if num_classes == 2 else num_classes
                    loss = custom_loss or ('binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy')
                    y_train_vals = y_train.values if num_classes == 2 else tf.keras.utils.to_categorical(y_train, num_classes)
                    y_test_vals = y_test.values if num_classes == 2 else tf.keras.utils.to_categorical(y_test, num_classes)

                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Dense(nn_params['neurons_per_layer'], activation=nn_params['activation'], input_shape=(X_train.shape[1],)))
                for _ in range(nn_params['num_hidden_layers'] - 1):
                    model.add(tf.keras.layers.Dense(nn_params['neurons_per_layer'], activation=nn_params['activation']))
                model.add(tf.keras.layers.Dense(output_neurons, activation=nn_params['output_activation']))
                
                optimizer = tf.keras.optimizers.get(nn_params['optimizer'])
                optimizer.learning_rate = nn_params['learning_rate']
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'] if problem_type == "classification" else ['mae'])

                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_container = st.container()
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values.astype(np.float32), y_train_vals)).batch(nn_params['batch_size'])
                
                history = {'loss': [], 'val_loss': []}
                if problem_type == "classification":
                    history.update({'accuracy': [], 'val_accuracy': []})
                
                best_val, wait = np.inf, 0
                for epoch in range(nn_params['epochs']):
                    epoch_loss, epoch_acc = [], []
                    
                    for batch, (x_batch, y_batch) in enumerate(train_dataset):
                        batch_metrics = model.train_on_batch(x_batch, y_batch)
                        progress = ((epoch * len(train_dataset) + batch + 1) / (nn_params['epochs'] * len(train_dataset)))
                        progress_bar.progress(min(progress, 1.0))
                        epoch_loss.append(batch_metrics[0])
                        if problem_type == "classification": 
                            epoch_acc.append(batch_metrics[1])
                        status_text.markdown(f"**Epoch {epoch+1}/{nn_params['epochs']} | Batch {batch+1}/{len(train_dataset)}**\nLoss: {batch_metrics[0]:.4f}")

                    val_metrics = model.evaluate(X_test.values.astype(np.float32), y_test_vals, verbose=0)
                    history['loss'].append(np.mean(epoch_loss))
                    history['val_loss'].append(val_metrics[0])
                    
                    if problem_type == "classification":
                        history['accuracy'].append(np.mean(epoch_acc))
                        history['val_accuracy'].append(val_metrics[1])
                        metrics_text = f"Train Loss: {history['loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}\nTrain Acc: {history['accuracy'][-1]:.4f} | Val Acc: {history['val_accuracy'][-1]:.4f}"
                    else:
                        metrics_text = f"Train Loss: {history['loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}"
                    
                    with metrics_container:
                        st.markdown(f"**Epoch {epoch+1} Summary**\n{metrics_text}")

                    if early_stopping:
                        current_val = val_metrics[0] if monitor == 'val_loss' else val_metrics[1]
                        if (monitor == 'val_loss' and current_val < best_val) or (monitor == 'val_accuracy' and current_val > best_val):
                            best_val, wait = current_val, 0
                        else:
                            wait += 1
                            if wait >= patience:
                                status_text.markdown("**Early stopping triggered!**")
                                break

                fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                ax[0].plot(history['loss'], label='Train Loss')
                ax[0].plot(history['val_loss'], label='Val Loss')
                ax[0].set_title('Loss History')
                ax[0].legend()
                
                if problem_type == "classification":
                    ax[1].plot(history['accuracy'], label='Train Acc')
                    ax[1].plot(history['val_accuracy'], label='Val Acc')
                    ax[1].set_title('Accuracy History')
                    ax[1].legend()
                
                st.pyplot(fig)
                st.session_state.model = model

            else:
                model_map = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor() if problem_type == "regression" else RandomForestClassifier(),
                    "SVR": SVR(),
                    "Logistic Regression": LogisticRegression(),
                    "SVC": SVC()
                }
                model = model_map[model_type]
                if problem_type == "classification" and class_weight == 'balanced':
                    model.class_weight = 'balanced'
                elif problem_type == "classification" and class_weight == 'custom':
                    model.class_weight = st.session_state.class_weights
                model.fit(X_train, y_train)
                st.session_state.model = model
                y_pred = model.predict(X_test)

            st.subheader("Evaluation Results")
            if problem_type == "regression":
                y_pred = st.session_state.model.predict(X_test).flatten() if model_type == "Neural Network" else y_pred
                mse = mean_squared_error(y_test, y_pred)
                col1, col2 = st.columns(2)
                col1.metric("MSE", f"{mse:.2f}")
                col2.metric("RMSE", f"{np.sqrt(mse):.2f}")
            else:
                if model_type == "Neural Network":
                    y_pred_probs = st.session_state.model.predict(X_test)
                    y_pred = (y_pred_probs > 0.5).astype(int).flatten() if num_classes == 2 else y_pred_probs.argmax(axis=1)
                    y_true = y_test_vals.flatten() if num_classes == 2 else y_test_vals.argmax(axis=1)
                else:
                    y_pred = model.predict(X_test)
                    y_true = y_test
                
                st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                st.code(classification_report(y_true, y_pred))
                
                cm = confusion_matrix(y_true, y_pred)
                if cm_normalize:
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='.2%' if cm_percentage else 'd', cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                st.pyplot(plt.gcf())
            
            st.success("Model trained successfully!")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
def render_prediction():
    st.header("🔮 Prediction")
    
    if not st.session_state.model:
        st.warning("Please train a model first!")
        return
    
    #  input form
    input_data = {}
    cols = st.session_state.selected_features
    
    st.subheader("Input Features")
    for i, col in enumerate(cols):
        if i % 3 == 0:
            columns = st.columns(3)
        input_data[col] = columns[i%3].text_input(col, value="0")
    
    if st.button("Predict"):
        try:
            #  input DataFrame
            input_df = pd.DataFrame([input_data])
            
            for col in cols:
                input_df[col] = input_df[col].astype(st.session_state.df[col].dtype)
            
            input_df = pd.get_dummies(input_df)
            train_cols = pd.get_dummies(st.session_state.df[cols]).columns
            input_df = input_df.reindex(columns=train_cols, fill_value=0)
            
            #  prediction
            if isinstance(st.session_state.model, tf.keras.Model):
                prediction = st.session_state.model.predict(input_df)
                if st.session_state.model.output_shape[1] == 1: 
                    prediction = prediction.flatten()[0]
                else:
                    prediction = prediction.argmax()
            else:
                prediction = st.session_state.model.predict(input_df)[0]
            
            st.success(f"Prediction: {prediction}")
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

def render_export():
    st.header("📤 Export Results")
    
    #  export
    st.subheader("Export Data")
    export_format = st.selectbox("Select Format", ["CSV", "Excel", "JSON"])
    
    if st.button("Export Dataset"):
        try:
            if export_format == "CSV":
                csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="dataset.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                excel_buffer = BytesIO()
                st.session_state.df.to_excel(excel_buffer, index=False)
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name="dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = st.session_state.df.to_json(orient='records')
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="dataset.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    # Model export
    st.subheader("Export Model")
    if st.button("Export Trained Model"):
        try:
            if hasattr(st.session_state, 'model') and st.session_state.model is not None:
                if isinstance(st.session_state.model, tf.keras.Model):
                    import tempfile
                    import os

                    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                        st.session_state.model.save(tmp.name)
                        tmp.seek(0)
                        model_bytes = tmp.read()
                    os.unlink(tmp.name)

                    st.download_button(
                        label="Download Model",
                        data=model_bytes,
                        file_name="model.h5",
                        mime="application/octet-stream"
                    )
                else:
                    model_buffer = BytesIO()
                    joblib.dump(st.session_state.model, model_buffer)
                    st.download_button(
                        label="Download Model",
                        data=model_buffer.getvalue(),
                        file_name="model.joblib",
                        mime="application/octet-stream"
                    )
            else:
                st.warning("No trained model to export!")
        except Exception as e:
            st.error(f"Model export failed: {str(e)}")

if __name__ == "__main__":
    main()
