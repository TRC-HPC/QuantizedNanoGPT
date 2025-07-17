Classic usage:
```
if not os.path.isfile(model_structure_file):
    instruction_reader.ModelQuantizationConfig.generate_template(model, model_structure_file)
instructions = instruction_reader.ModelQuantizationConfig(model_structure_file)
model = wrap_linear_layers(model, StatWrapper, DropoutWrapper, optimizer, est_interval, instructions, mode=scheduler.return_mode(), layers_of_interest=scheduler.layers_of_interest(), now=now, output_dir=intermidiate_file_path)
```
* model inheriting from nn.Module
* StatWrapper and DropoutWrapper classes imported from stat_wrapper_utils.py
* optimizer inheriting from torch.optim.Optimizer; e.g. torch.optim.AdamW(model.parameters()
* est_interval e.g. 250 (how often to collect statistics)
* model_structure_file = "baseline_fp16.json"

See example in train.py