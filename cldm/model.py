from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)

def create_model(config, **kwargs):
    model = instantiate_from_config(config.model).cpu()
    return model
