import joblib
import pandas as pd


MODEL_PATH = "final_model.joblib"
SCALER_PATH = "scaler_for_api.joblib"


SAMPLES = [
    {
        "name": "sample_1_non_fraud",
        "transaction": {
            "Time": 160760.0,
            "V1": -0.674466064578314,
            "V2": 1.40810501967799,
            "V3": -1.11062205357093,
            "V4": -1.32836577843066,
            "V5": 1.38899603254837,
            "V6": -1.30843906707795,
            "V7": 1.88587890268717,
            "V8": -0.614232966299775,
            "V9": 0.311652212453101,
            "V10": 0.65075700363522,
            "V11": -0.857784661547805,
            "V12": -0.229961445775592,
            "V13": -0.19981700479103,
            "V14": 0.266371326329879,
            "V15": -0.0465441684754424,
            "V16": -0.741398089749789,
            "V17": -0.605616644106022,
            "V18": -0.39256818789208,
            "V19": -0.162648311024695,
            "V20": 0.394321820843914,
            "V21": 0.0800842396026648,
            "V22": 0.810033595602455,
            "V23": -0.224327230436412,
            "V24": 0.707899237446867,
            "V25": -0.13583702273753,
            "V26": 0.0451021964988772,
            "V27": 0.533837219064273,
            "V28": 0.291319252625364,
            "Amount": 23.0,
        },
    },
    {
        "name": "sample_2_non_fraud",
        "transaction": {
            "Time": 12345.0,
            "V1": -1.359807,
            "V2": -0.072781,
            "V3": 2.536347,
            "V4": 1.378155,
            "V5": -0.338320,
            "V6": 0.462388,
            "V7": 0.239599,
            "V8": 0.098698,
            "V9": 0.363787,
            "V10": 0.090794,
            "V11": -0.551600,
            "V12": -0.617801,
            "V13": -0.991390,
            "V14": -0.311169,
            "V15": 1.468177,
            "V16": -0.470401,
            "V17": 0.207971,
            "V18": 0.025791,
            "V19": 0.403993,
            "V20": 0.251412,
            "V21": -0.018307,
            "V22": 0.277838,
            "V23": -0.110474,
            "V24": 0.066928,
            "V25": 0.128539,
            "V26": -0.189115,
            "V27": 0.133558,
            "V28": -0.021053,
            "Amount": 149.62,
        },
    },
    {
        "name": "sample_3_fraud",
        "transaction": {
            "Time": 472.0,
            "V1": -10.0,
            "V2": 10.0,
            "V3": -8.0,
            "V4": 8.0,
            "V5": 6.0,
            "V6": -0.466906204,
            "V7": -0.27962483,
            "V8": -1.002212354,
            "V9": 0.697576059,
            "V10": -6.0,
            "V11": 0.36350478,
            "V12": -5.0,
            "V13": -0.009949673,
            "V14": -5.0,
            "V15": 5.0,
            "V16": -5.0,
            "V17": 0.167444318,
            "V18": 0.162747754,
            "V19": -0.198271501,
            "V20": 0.264561858,
            "V21": 0.661969248,
            "V22": 0.052824982,
            "V23": -0.207963384,
            "V24": -0.071624686,
            "V25": 0.252934812,
            "V26": 0.399710356,
            "V27": 0.141045763,
            "V28": 0.265415307,
            "Amount": 0.01,
        },
    },
]


def expected_columns_for_model(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def preprocess_transaction(transaction, expected_columns, scaler):
    frame = pd.DataFrame([transaction])

    for col in expected_columns:
        if col not in frame.columns:
            frame[col] = 0.0

    frame = frame[expected_columns]
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame[["Time", "Amount"]] = scaler.transform(frame[["Time", "Amount"]])
    return frame


def main():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_columns = expected_columns_for_model(model)

    print("Loaded artifacts successfully")
    print(f"Model: {type(model).__name__}")
    print(f"Features expected: {len(expected_columns)}")
    print("-" * 70)

    for sample in SAMPLES:
        sample_name = sample["name"]
        transaction = sample["transaction"]

        processed = preprocess_transaction(transaction, expected_columns, scaler)
        prediction = int(model.predict(processed)[0])
        probability = float(model.predict_proba(processed)[:, 1][0])

        label = "Fraud" if prediction == 1 else "Non-Fraud"
        print(
            f"{sample_name}: prediction={prediction} ({label}), fraud_probability={probability:.6f}"
        )


if __name__ == "__main__":
    main()