Classic integration of quantization into training:
```
quantization_config_path = MxPTraining(os.path.join("MixedPrecision", "QuantizationConfigs", "GPTBase.json"))
if not os.path.isfile(quantization_config_path):
    JsonReader.ModelQuantizationConfig.generate_template(model, quantization_config_path)
instructions = JsonReader.ModelQuantizationConfig(quantization_config_path)
model = Wrappers.wrap_linear_layers(model, MxPWrappers.StatWrapper, MxPWrappers.DropoutWrapper, instructions)
```
See example in train.py.

Schedulers handle changing the quantizers at runtime.  Schedulers.no_change() is a no-op - a placeholder that doesn't actually change anything.
Example (pointless) usage of schedulers:
```
scheduler = schedulers.no_change()
if iter_num % update_interval == 0 and master_process:   
    Wrappers.update_precision(model, scheduler.update_module, scheduler.sorting_attr, scheduler.filter_func, iter_num, update=True)
```
with iter_num designating the number of training iterations, update interval designating every how often to change the training precision, update_precision from 
