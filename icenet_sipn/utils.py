
def drop_variables(data, variable_names: list):
    """Drop variables from an xarray dataset if they exist

    Args:
        data: Xarray dataset
        variable_names: List of variable names to drop
    """
    for var in variable_names:
        if var in data:
            data = data.drop_vars(var)
    return data