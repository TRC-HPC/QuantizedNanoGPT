Classic usage:
```
quantization_config_path = MxPTraining(os.path.join("MixedPrecision", "QuantizationConfigs", "GPTBase.json"))
if not os.path.isfile(quantization_config_path):
    JsonReader.ModelQuantizationConfig.generate_template(model, quantization_config_path)
instructions = JsonReader.ModelQuantizationConfig(quantization_config_path)
model = Wrappers.wrap_linear_layers(model, MxPWrappers.StatWrapper, MxPWrappers.DropoutWrapper, instructions)
```
See example in train.py