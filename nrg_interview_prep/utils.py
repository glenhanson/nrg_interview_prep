def marginal_effects(weight, age, male, model_params):
    """Return partial derivatives wrt weight and age."""
    d_height_d_weight = (
        model_params['weight']
        + 2 * model_params['I(weight ** 2)'] * weight
        + model_params.get('weight:male', 0) * male
    )
    d_height_d_age = (
        model_params['age']
        + 2 * model_params['I(age ** 2)'] * age
        + model_params.get('age:male', 0) * male
    )
    return d_height_d_weight, d_height_d_age
