import yaml

def load_config(config_path='config.yaml'):

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters
    numerical_cols = config['numerical_cols']
    categorical_cols = config['categorical_cols']
    power_transform_cols = config['power_transform_cols']
    ordinal_columns = config['ordinal_columns']
    nominal_columns = config['nominal_columns']
    
    # Convert ordinal_categories from dict to list format
    # This matches the original format: [['High School', "Master's", "Bachelor's", 'Doctorate']]
    ordinal_categories = [config['ordinal_categories'][col] for col in ordinal_columns]
    
    return {
        'columns': config['columns'],
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'power_transform_cols': power_transform_cols,
        'ordinal_columns': ordinal_columns,
        'ordinal_categories': ordinal_categories,
        'nominal_columns': nominal_columns
    }


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_config('schema.yaml')
    
    # Access parameters exactly as in your original code
    all_columns = config['columns']
    numerical_cols = config['numerical_cols']
    categorical_cols = config['categorical_cols']
    power_transform_cols = config['power_transform_cols']
    ordinal_columns = config['ordinal_columns']
    ordinal_categories = config['ordinal_categories']
    nominal_columns = config['nominal_columns']
    
    # Print to verify
    print(len(all_columns))
    print("Numerical columns:", numerical_cols)
    print("\nCategorical columns:", categorical_cols)
    print("\nPower transform columns:", power_transform_cols)
    print("\nOrdinal columns:", ordinal_columns)
    print("\nOrdinal categories:", ordinal_categories)
    print("\nNominal columns:", nominal_columns)
    
    # You can also access the column definitions
    #print("\nColumn definitions:", config['columns'])