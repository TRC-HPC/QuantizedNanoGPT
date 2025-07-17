# This file will contain methods that determine which layers will have what type of quantization.
# In order to generalize and clarify for any model we want to have this method to be based on some json file that will specifically state for each layer:
# 1. weight quantization
# 2. input quantization
# 3. output quantization
# Thus, during the wrapping of the model we wrapper will also ask an instructor how exactly to define each layer in terms of its quantization (also of gradients)
# The options that are for the entire model are: lp for reference and the L4 option which requires changing the optimizer (which we already did)

# The goal here is to generalize the L1, L2, L3, L4 notations to allow maximum flaxibility and make it easier to define what we are quantizing. This format can then be used for any model we want. 


# Function that receives a model and constructs a json template that the user can fill out to define the specific quantizations throughout the model - this is to be used wihtout a specific instance

# The init will receive a json file as instructions. The init will also flatten the original instruction json to simplify the use (the user should see the heirarchical structure and but the system would like to have it generalized)

# A function that receives a layer and provide the instructions for it.


import json
import torch.nn as nn
DEFAULT_Q_VALUE = 'fp32'

class ModelQuantizationConfig:

    def __init__(self, json_path):
        # Load the JSON structure
        with open(json_path, 'r') as f:
            self.model_structure = json.load(f)
        
        # Flatten the model structure to make layer lookup easier
        self.flattened = self._flatten_structure(self.model_structure)

    @staticmethod
    def generate_template(model: nn.Module, output_path: str):
        """
        Static method that takes a model and creates a JSON file
        representing the hierarchical structure with quantization defaults.
        """
        def build_structure(module):
            children = dict(module.named_children())
            if not children:
                return {
                    "weight_q": DEFAULT_Q_VALUE,
                    "input_q": DEFAULT_Q_VALUE,
                    "output_q": DEFAULT_Q_VALUE,
                    "gradient_q": DEFAULT_Q_VALUE
                }
            return {
                name: build_structure(child)
                for name, child in children.items()
            }

        structure = build_structure(model)
        with open(output_path, 'w') as f:
            json.dump(structure, f, indent=2)

    def _flatten_structure(self, structure, prefix=""):
        flat = {}
        for key, value in structure.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and all(k in value for k in ["weight_q", "input_q", "output_q", "gradient_q"]):
                flat[path] = value
            else:
                flat.update(self._flatten_structure(value, prefix=path))
        return flat

    def get_quantization_config(self, layer_name: str):
        """
        Returns the quantization settings for a given layer name.
        """
        result =  self.flattened.get(layer_name, None)
        if result:
            return result['weight_q'], result['input_q'], result['output_q'], result['gradient_q']
        else:
            return DEFAULT_Q_VALUE, DEFAULT_Q_VALUE, DEFAULT_Q_VALUE, DEFAULT_Q_VALUE

