from predict import predict_parkinson

result = predict_parkinson(
    "Data/raw/drawings/spiral/testing/healthy/V01HE01.png",
    "Data/raw/drawings/wave/testing/healthy/V01HO01.png"
)

print(result)