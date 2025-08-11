import torch.nn as nn
import numpy as np
from .Quantizers import IntActQuantizer, IntWeightQuantizer, FloatQuantizer

DEFAULT_Q_VALUE = 'fp32'

# TODO: not sure this function should be here. This is a simple mapping function that maps the string input to the specific quantization needed 
def map_quantizer(input_str, weight=False):
    if input_str.startswith('fp'):
        precision = int(input_str[len('fp'):])
        return FloatQuantizer(precision)
    
    if input_str.endswith('bit'):
        precision = int(input_str[:-len('bit')])
        if weight:
            quantizer = IntWeightQuantizer()
            quantizer.configure(bits=precision)  # TODO: many parameters of quantizer...
            return quantizer
        else:
            quantizer = IntActQuantizer()
            quantizer.configure(bits=precision)
            return quantizer
    
    raise ValueError("Only float (given as 'fp...') and int (given as '...bit') quantizations supported")
    
    
class LinearWrapper(nn.Module):
    def __init__(self, child, est_interval, weight_q='fp16', input_q='fp16', output_q='fp16', gradient_q='fp16', bias=True, beta=0.99, **kwargs):
        super().__init__()
        self.linear = child 
        
        if 'parent_name' in kwargs:
            self.full_name = kwargs['parent_name']   
        else:
            self.full_name = ''
        
        # Quantizers
        self.weight_q = weight_q
        self.input_q = input_q
        self.output_q = output_q
        self.gradient_q = gradient_q
        
        self.quantizer_w = map_quantizer(self.weight_q, True)
        self.quantizer_input = map_quantizer(self.input_q, False)
        self.quantizer_output = map_quantizer(self.output_q, False)

    def forward(self, input):
        quantized_weight = self.quantizer_w(self.linear.weight)
        self.quantizer_input.find_params(input)       
        x_dtype = input.dtype
        quantized_input = self.quantizer_input(input).to(x_dtype)
        self.quantizer_input.free()
                       
        output = nn.functional.linear(input=quantized_input, weight=quantized_weight, bias=self.linear.bias)
        
        self.quantizer_output.find_params(output)       
        x_dtype = output.dtype
        quantized_output = self.quantizer_output(output).to(x_dtype)

        # Uncomment below code to watch the gradients computed (hooks are added in the same order as their backpropagation order)
        # try:
        #     def stop_grad(grad, name):
        #         import pdb; pdb.set_trace()
        #     from functools import partial
        #     quantized_output.register_hook(partial(stop_grad, name=self.full_name + ".quantized_output"))
        #     output.register_hook(partial(stop_grad, name=self.full_name + ".output"))
        #     quantized_input.register_hook(partial(stop_grad, name=self.full_name + ".quantized_input"))
        #     quantized_weight.register_hook(partial(stop_grad, name=self.full_name + ".quantized_weight"))
        #     self.linear.weight.register_hook(partial(stop_grad, name=self.full_name + ".self.linear.weight"))
        #     input.register_hook(partial(stop_grad, name=self.full_name + ".input"))
        # except RuntimeError:  # don't register hooks when gradients are disabled, such as during evaluation
        #     pass
        
        self.quantizer_output.free()
             
        return quantized_output

def update_precision(module, func, sort_func, filter_func, iter_num, parent_name=None, update=False): #, newVal=None):
    if update:
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if isinstance(child, LinearWrapper):
                child.weight_q, child.input_q, child.output_q, child.gradient_q = func(child, iter_num, full_name)
                child.quantizer_w = map_quantizer(child.weight_q, True)
                child.quantizer_input = map_quantizer(child.input_q, False)
                child.quantizer_output = map_quantizer(child.output_q, False)
                
            else:
                update_precision(child, func, sort_func, filter_func, iter_num, parent_name=full_name, update=update)

# TODO: this should be extended also to LayerNorms (what about embeddings??). It should receive instead of a run_flag an instructor that will tell us how to define each layer
def wrap_linear_layers(module, wrapper_cls, instructions=None, parent_name = None, num_wrapped=0, **kwargs):
    """
    Recursively replace all nn.Linear layers in `module` with instances of `wrapper_cls`.

    Args:
        module (nn.Module): The model or submodule to modify.
        wrapper_cls (type): A class like TrackedLinear that wraps nn.Linear.
        est_interval: the interval used for creating an estimator
        **kwargs: Extra arguments to pass to the wrapper_cls (like beta).
    """    
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        # TODO: This does not wrap the LayerNorm which is not nn.LayerNorm but locally defined!

        if isinstance(child, nn.Linear):
            if instructions is not None:
                bias = child.bias is not None
                # We want to get from a predefined instructor how to initialize the given layer:
                weight_q, input_q, output_q, gradient_q = instructions.get_quantization_config(full_name)       
                wrapped = wrapper_cls(child, weight_q, input_q, output_q, gradient_q, bias=bias, parent_name=full_name, **kwargs)  # Replace the linear layer in the parent module
            else:
                wrapped = wrapper_cls(child, **kwargs)
            setattr(module, name, wrapped)
            num_wrapped += 1
        else:
            # Recursively wrap submodules
            if num_wrapped < float("inf"):
                wrap_linear_layers(child, wrapper_cls, instructions, parent_name=full_name, num_wrapped=num_wrapped, **kwargs)
    return module
