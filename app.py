# Libraries

# For data analysis
import pandas as pd

# For model creation and performance evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# For visualizations and interactive dashboard creation
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np

# Load dataset
data = pd.read_csv("winequality-red.csv", sep=";")

# Pre process the data

print(data.isnull().sum())  # check for missing values
data.dropna(inplace=True)  # drop rows with missing values
data.drop_duplicates(keep="first")  # Drop duplicate rows

# Label quality into Good (1) and Bad (0)
data["quality"] = data["quality"].apply(lambda x: 1 if x >= 6.0 else 0)

X = data.drop("quality", axis=1)  # Drop the target variable
y = data["quality"]  # Set the target variable as the label

# Split the data into training and testing sets (20% testing and 80% training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Create an object of the logistic regression model
logreg_model = LogisticRegression()

# Fit the model to the training data
logreg_model.fit(X_train, y_train)


# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the dashboard
app.layout = html.Div(
    children=[
        html.H1(
            "CO544-2023 Lab 3: Wine Quality Prediction", style={"textAlign": "center"}
        ),
        dcc.Tabs(
            id="tabs",
            children=[
                # Tab for exploratory data analysis
                dcc.Tab(
                    label="Exploratory Data Analysis",
                    children=[
                        html.Div(
                            [
                                html.H3("Exploratory Data Analysis"),
                                html.Label("Feature 1 (X-axis)"),
                                dcc.Dropdown(
                                    id="x_feature",
                                    options=[
                                        {"label": col, "value": col}
                                        for col in data.columns
                                    ],
                                    value=data.columns[0],
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "marginRight": "2rem",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Feature 2 (Y-axis)"),
                                dcc.Dropdown(
                                    id="y_feature",
                                    options=[
                                        {"label": col, "value": col}
                                        for col in data.columns
                                    ],
                                    value=data.columns[1],
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "marginRight": "2rem",
                            },
                        ),
                        dcc.Graph(
                            id="correlation_plot",
                            style={"marginTop": "3rem",
                                   "marginBottom": "2rem"},
                        ),
                    ],
                ),
                # Tab for wine quality prediction
                dcc.Tab(
                    label="Wine Quality Prediction",
                    children=[
                        html.H3("Wine Quality Prediction"),
                        html.Table(
                            [
                                html.Tr(
                                    [
                                        html.Td(html.Label("Fixed Acidity")),
                                        html.Td(
                                            dcc.Input(
                                                id="fixed_acidity",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label(
                                            "Volatile Acidity")),
                                        html.Td(
                                            dcc.Input(
                                                id="volatile_acidity",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label("Citric Acid")),
                                        html.Td(
                                            dcc.Input(
                                                id="citric_acid",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label("Residual Sugar")),
                                        html.Td(
                                            dcc.Input(
                                                id="residual_sugar",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label("Chlorides")),
                                        html.Td(
                                            dcc.Input(
                                                id="chlorides",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label(
                                            "Free Sulfur Dioxide")),
                                        html.Td(
                                            dcc.Input(
                                                id="free_sulfur_dioxide",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label(
                                            "Total Sulfur Dioxide")),
                                        html.Td(
                                            dcc.Input(
                                                id="total_sulfur_dioxide",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label("Density")),
                                        html.Td(
                                            dcc.Input(
                                                id="density",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label("pH")),
                                        html.Td(
                                            dcc.Input(
                                                id="ph", type="number", required=True
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label("Sulphates")),
                                        html.Td(
                                            dcc.Input(
                                                id="sulphates",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Label("Alcohol")),
                                        html.Td(
                                            dcc.Input(
                                                id="alcohol",
                                                type="number",
                                                required=True,
                                            )
                                        ),
                                    ]
                                ),
                            ],
                            style={"marginBottom": "1rem"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Clear",
                                    id="clear-button",
                                    n_clicks=0,
                                    style={"marginRight": "1rem"},
                                ),
                                html.Button(
                                    "AutoFill",
                                    id="autofill-button",
                                    n_clicks=0,
                                    style={"marginRight": "1rem"},
                                ),
                                html.Button(
                                    "Predict", id="predict-button", n_clicks=0),
                            ]
                        ),
                        html.Div(
                            [
                                html.H4("Predicted Quality"),
                                html.Div(id="prediction-output"),
                            ]
                        ),
                    ],
                ),
            ],
            content_style={
                "marginLeft": "3rem",
                "marginRight": "3rem",
                "marginTop": "2rem",
            },
        ),
    ],
    style={
        "fontFamily": "Google Sans, sans-serif",
        "marginLeft": "3rem",
        "marginRight": "3rem",
    },
)

# Define the callback to update the correlation plot


@app.callback(
    dash.dependencies.Output("correlation_plot", "figure"),
    [
        dash.dependencies.Input("x_feature", "value"),
        dash.dependencies.Input("y_feature", "value"),
    ],
)
def update_correlation_plot(x_feature, y_feature):
    fig = px.scatter(data, x=x_feature, y=y_feature, color="quality")
    fig.update_layout(title=f"Correlation between {x_feature} and {y_feature}")
    return fig


# Define the callback function to predict wine quality


@app.callback(
    Output("prediction-output", "children"),
    [Input("predict-button", "n_clicks")],
    [
        State("fixed_acidity", "value"),
        State("volatile_acidity", "value"),
        State("citric_acid", "value"),
        State("residual_sugar", "value"),
        State("chlorides", "value"),
        State("free_sulfur_dioxide", "value"),
        State("total_sulfur_dioxide", "value"),
        State("density", "value"),
        State("ph", "value"),
        State("sulphates", "value"),
        State("alcohol", "value"),
    ],
)
def predict_quality(
    n_clicks,
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    ph,
    sulphates,
    alcohol,
):
    # Create input features array for prediction
    input_features = np.array(
        [
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            ph,
            sulphates,
            alcohol,
        ]
    ).reshape(1, -1)

    # Predict the wine quality (0 = bad, 1 = good)
    prediction = logreg_model.predict(input_features)[0]

    # Return the prediction
    if prediction == 1:
        return 'This wine is predicted to be good quality.'
    else:
        return 'This wine is predicted to be bad quality.'

# Define the callback function to autofill the input field and clear the input field


@app.callback(
    Output("fixed_acidity", "value"),
    Output("volatile_acidity", "value"),
    Output("citric_acid", "value"),
    Output("residual_sugar", "value"),
    Output("chlorides", "value"),
    Output("free_sulfur_dioxide", "value"),
    Output("total_sulfur_dioxide", "value"),
    Output("density", "value"),
    Output("ph", "value"),
    Output("sulphates", "value"),
    Output("alcohol", "value"),
    [Input("clear-button", "n_clicks"), Input("autofill-button", "n_clicks")],
)
def autofill(clear_click, autofill_click):
    # get the id of the button that triggered the callback
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "clear-button":
        return None, None, None, None, None, None, None, None, None, None, None
    else:
        # get set of values from a row in the dataset
        input_features = data.sample(n=1).values

        # return the values
        return (
            input_features[0][0],
            input_features[0][1],
            input_features[0][2],
            input_features[0][3],
            input_features[0][4],
            input_features[0][5],
            input_features[0][6],
            input_features[0][7],
            input_features[0][8],
            input_features[0][9],
            input_features[0][10],
        )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
