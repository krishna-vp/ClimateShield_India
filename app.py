from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# --------------------------------------------------
# Load trained model and supporting files
# --------------------------------------------------

model = joblib.load("ClimateShield_RF_Model.pkl")
scaler = joblib.load("ClimateShield_Scaler.pkl")
feature_columns = joblib.load("ClimateShield_Feature_Columns.pkl")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("Climate_data.csv")

# Remove unnecessary unnamed/serial column
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# --------------------------------------------------
# Get dropdown values
# --------------------------------------------------

states = sorted(
    df["state_ut"].dropna().unique()
)

diseases = sorted(
    df["Disease"].dropna().unique()
)

weeks = sorted(
    df["week_of_outbreak"].dropna().unique()
)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():

    # IMPORTANT:
    # prediction is None when page is opened.
    # Therefore no risk result will be displayed.

    return render_template(
        "index.html",
        states=states,
        diseases=diseases,
        weeks=weeks,
        prediction=None
    )


# --------------------------------------------------
# Get Districts based on selected State
# --------------------------------------------------

@app.route("/districts/<state>")
def get_districts(state):

    district_list = sorted(
        df[df["state_ut"] == state]["district"]
        .dropna()
        .unique()
    )

    return {
        "districts": district_list
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------------
        # Get values from HTML form
        # ------------------------------------------

        state = request.form["state"]
        district = request.form["district"]
        disease = request.form["disease"]
        week = request.form["week"]

        day = int(request.form["day"])
        month = int(request.form["month"])
        year = int(request.form["year"])

        temp = float(request.form["temp"])
        rain = float(request.form["rain"])
        lai = float(request.form["lai"])

        # ------------------------------------------
        # Get Latitude and Longitude
        # ------------------------------------------

        location_data = df[
            (df["state_ut"] == state) &
            (df["district"] == district)
        ]

        if location_data.empty:

            return render_template(
                "index.html",
                states=states,
                diseases=diseases,
                weeks=weeks,
                prediction=None,
                error="Selected district information was not found."
            )

        lat = location_data.iloc[0]["Latitude"]
        lon = location_data.iloc[0]["Longitude"]

        # ------------------------------------------
        # Create input dataframe
        # ------------------------------------------

        input_df = pd.DataFrame({

            "week_of_outbreak": [week],

            "state_ut": [state],

            "district": [district],

            "Disease": [disease],

            "day": [day],

            "mon": [month],

            "year": [year],

            "Latitude": [lat],

            "Longitude": [lon],

            "preci": [rain],

            "LAI": [lai],

            "Temp": [temp]

        })

        # ------------------------------------------
        # One-hot encoding
        # ------------------------------------------

        input_encoded = pd.get_dummies(
            input_df,
            columns=[
                "week_of_outbreak",
                "state_ut",
                "district",
                "Disease"
            ],
            drop_first=True
        )

        # ------------------------------------------
        # Match training feature columns
        # ------------------------------------------

        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ------------------------------------------
        # Numerical features
        # ------------------------------------------

        numeric_features = [
            "day",
            "mon",
            "year",
            "Latitude",
            "Longitude",
            "preci",
            "LAI",
            "Temp"
        ]

        # ------------------------------------------
        # Apply same scaler used during training
        # ------------------------------------------

        input_encoded[numeric_features] = scaler.transform(
            input_encoded[numeric_features]
        )

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction_value = model.predict(
            input_encoded
        )[0]

        # Convert numerical prediction to risk level

        risk_mapping = {
            0: "Low Risk",
            1: "Medium Risk",
            2: "High Risk"
        }

        prediction = risk_mapping.get(
            prediction_value,
            "Unknown"
        )

        # ------------------------------------------
        # Return result page
        # ------------------------------------------

        district_list = sorted(
            df[df["state_ut"] == state]["district"]
            .dropna()
            .unique()
        )

        return render_template(
            "index.html",

            states=states,

            districts=district_list,

            diseases=diseases,

            weeks=weeks,

            prediction=prediction,

            selected_state=state,

            selected_district=district,

            selected_disease=disease,

            selected_week=week,

            selected_day=day,

            selected_month=month,

            selected_year=year,

            selected_temp=temp,

            selected_rain=rain,

            selected_lai=lai

        )

    except Exception as e:

        return render_template(
            "index.html",

            states=states,

            diseases=diseases,

            weeks=weeks,

            prediction=None,

            error=str(e)

        )


# --------------------------------------------------
# Run Flask Application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )