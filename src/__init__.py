def get_model(model_type_str: str): 
    from src.qubo.model import QuboModel 
    models = [QuboModel]
    for model in models:
        if model.__name__ == model_type_str:
            return model
    raise ValueError(f'Model {model_type_str} not found. Available models: {[model.__name__ for model in models]}')

def get_init_model(hparams: dict, dm):
    return get_model(hparams["model"]["type"])(hparams, dm)