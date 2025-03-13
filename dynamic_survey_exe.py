import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output
from flask import Flask
import webbrowser
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize Flask Server
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Simulated dataset
data = {
    "Research_Type": ["Exploratory", "Descriptive", "Experimental", "Exploratory", "Descriptive", "Experimental"],
    "Survey_Goal": ["Qualitative", "Quantitative", "Mixed", "Quantitative", "Qualitative", "Mixed"],
    "Target_Audience": ["General Public", "Businesses", "Experts", "Policymakers", "General Public", "Businesses"],
    "Sample_Size": [500, 1000, 200, 5000, 150, 800],
    "Budget": ["Low", "Medium", "High", "High", "Low", "Medium"],
    "Time_Constraint": ["Short-term", "Longitudinal", "Short-term", "Longitudinal", "Short-term", "Longitudinal"],
    "Data_Sensitivity": ["Public", "Confidential", "Public", "Confidential", "Open-Source", "Confidential"],
    "Best_Survey_Method": ["Online Survey", "Face-to-Face", "A/B Testing", "Phone Interview", "Paper Survey", "Panel Survey"]
}

df = pd.DataFrame(data)
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=["Best_Survey_Method"])
y = df["Best_Survey_Method"]
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Dash Layout
app.layout = html.Div([
    html.H1("SMS - Survey Method Selector", style={'textAlign': 'center'}),

    html.Label("Select Research Type"),
    dcc.Dropdown(id="research_type", options=[{"label": i, "value": i} for i in label_encoders["Research_Type"].classes_], value="Exploratory"),

    html.Label("Select Survey Goal"),
    dcc.Dropdown(id="survey_goal", options=[{"label": i, "value": i} for i in label_encoders["Survey_Goal"].classes_], value="Qualitative"),

    html.Label("Select Target Audience"),
    dcc.Dropdown(id="target_audience", options=[{"label": i, "value": i} for i in label_encoders["Target_Audience"].classes_], value="General Public"),

    html.Label("Enter Sample Size"),
    dcc.Input(id="sample_size", type="number", value=500),

    html.Label("Select Budget"),
    dcc.Dropdown(id="budget", options=[{"label": i, "value": i} for i in label_encoders["Budget"].classes_], value="Low"),

    html.Label("Select Time Constraint"),
    dcc.Dropdown(id="time_constraint", options=[{"label": i, "value": i} for i in label_encoders["Time_Constraint"].classes_], value="Short-term"),

    html.Label("Select Data Sensitivity"),
    dcc.Dropdown(id="data_sensitivity", options=[{"label": i, "value": i} for i in label_encoders["Data_Sensitivity"].classes_], value="Public"),

    html.Button("Get Best Survey Method", id="submit_button", n_clicks=0, style={'margin-top': '10px'}),

    html.Div(id="output_div", style={'margin-top': '20px', 'font-size': '20px', 'textAlign': 'center', 'font-weight': 'bold'}),

    dcc.Graph(id="confidence_graph")
])

# Callback for prediction and visualization
@app.callback(
    [Output("output_div", "children"), Output("confidence_graph", "figure")],
    [Input("submit_button", "n_clicks")],
    [Input("research_type", "value"), Input("survey_goal", "value"), Input("target_audience", "value"),
     Input("sample_size", "value"), Input("budget", "value"), Input("time_constraint", "value"), Input("data_sensitivity", "value")]
)
def predict_survey_method(n_clicks, research_type, survey_goal, target_audience, sample_size, budget, time_constraint, data_sensitivity):
    input_data = pd.DataFrame([[
        label_encoders["Research_Type"].transform([research_type])[0],
        label_encoders["Survey_Goal"].transform([survey_goal])[0],
        label_encoders["Target_Audience"].transform([target_audience])[0],
        sample_size,
        label_encoders["Budget"].transform([budget])[0],
        label_encoders["Time_Constraint"].transform([time_constraint])[0],
        label_encoders["Data_Sensitivity"].transform([data_sensitivity])[0]
    ]], columns=X.columns)

    prediction_encoded = rf_model.predict(input_data)[0]
    prediction = label_encoders["Best_Survey_Method"].inverse_transform([prediction_encoded])[0]
    probabilities = rf_model.predict_proba(input_data)[0]

    # Creating confidence score visualization
    methods = label_encoders["Best_Survey_Method"].inverse_transform(np.arange(len(probabilities)))
    fig = px.bar(x=methods, y=probabilities, labels={'x': 'Survey Methods', 'y': 'Confidence Score'},
                 title="Prediction Confidence Scores", color_discrete_sequence=["skyblue"])

    return f"Recommended Survey Method: {prediction}", fig

# Function to open the browser automatically
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

# Run Dash App
if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()  # Open browser after delay
    app.run_server(debug=False)
