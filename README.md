# Employee Attrition Predictor

This project provides a robust, production-ready pipeline for predicting employee attrition. Using machine learning to identify high-risk employees, HR departments can proactively address turnover by understanding key drivers like workload, compensation, and career growth.


## ✨ Features

  * **Advanced Preprocessing:** Uses `RobustScaler` and `OneHotEncoder` within an `sklearn` pipeline to ensure data consistency.
  * **Dimensionality Reduction:** Implements `PCA` to optimize model training time while maintaining 95% data variance.
  * **Class Imbalance Handling:** Uses `SMOTE` (Synthetic Minority Over-sampling Technique) to ensure the model effectively learns from the minority attrition class.
  * **Interactive Dashboard:** Includes a world-class Streamlit UI with real-time risk assessment and strategy recommendations.
  * **Interpretability:** Uses `SHAP` values to provide transparency into how the model calculates risk for individual employees.

## 📁 Project Structure

```text
├── data/                    # Dataset folder
├── models/                  # Exported .pkl model files
├── notebooks/               # Jupyter notebooks for EDA and training
├── app.py                   # Streamlit application
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## 🚀 Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/cognixai/employee-attrition-prediction.git
    cd employee-attrition-prediction
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### Launch the Web App

Run the following command to start the interactive dashboard:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to start predicting attrition risk.

### Model Training

To retrain the model with updated data, run the provided Jupyter notebook in the `notebooks/` directory. Ensure you export the final pipeline as `attrition_model_pipeline.pkl`.

## 📊 Model Performance

The pipeline is evaluated using a 5-fold cross-validation strategy.

  * **Primary Metric:** F1-Score (chosen for balanced precision and recall on the imbalanced target class).
  * **Secondary Metrics:** ROC-AUC and Recall.
  * **Current Best Model:** XGBoost Classifier.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
