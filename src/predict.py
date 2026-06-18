def predict_yield(temp, humid, co2):

    # Temporary formula
    prediction = (
        0.2 * temp
        + 0.05 * humid
        + 0.003 * co2
    )

    return prediction