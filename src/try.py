print("Starting...")

try:
    from xgboost import XGBClassifier
    print("XGBoost imported successfully")
except Exception as e:
    print("Import failed:", e)

print("Finished")