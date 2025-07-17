import torch.nn as nn
import numpy as np
from .quantizing import quantize_block, ActQuantizer, WeightQuantizer, BasicQuantizer

DEFAULT_Q_VALUE = 'fp32'

# TODO: not sure this function should be here. This is a simple mapping function that maps the string input to the specific quantization needed 
def map_quantizer(input_str, weight=False):
    
    if input_str == 'fp16':
        return BasicQuantizer(16)
    
    if input_str == 'fp32':
        return BasicQuantizer(32)
    
    if input_str == '8bit':
        if weight:
            quantizer = WeightQuantizer()
            quantizer.configure(bits=8)  # TODO: many parameters of quantizer...
            return quantizer
        else:
            quantizer = ActQuantizer()
            quantizer.configure(bits=8)
            return quantizer
        
    if input_str == '4bit':
        if weight:
            quantizer = WeightQuantizer()
            quantizer.configure(bits=4)  # TODO: many parameters of quantizer...
            return quantizer
        else:
            quantizer = ActQuantizer()
            quantizer.configure(bits=4)
            return quantizer
    
class DropoutWrapper(nn.Module):
    def __init__(self, child, est_interval, weight_q='fp16', input_q='fp16', output_q='fp16', gradient_q='fp16', bias=True, beta=0.99, **kwargs):
        super().__init__()
        self.linear = child

        # Quantizers
        self.weight_q = weight_q
        self.input_q = input_q
        self.output_q = output_q
        self.gradient_q = gradient_q
        
        # TODO: We would like the constructor to receive the quantizer so we can change this easily...
        self.quantizer_w = map_quantizer(self.weight_q, True)
        self.quantizer_input = map_quantizer(self.input_q, False)
        self.quantizer_output = map_quantizer(self.output_q, False)
        
    def forward(self, input):
        
        # We quantize: weight, input and output as defined 
        self.linear = quantize_block(self.linear, self.quantizer_w)
        self.quantizer_input.find_params(input)       
        x_dtype = input.dtype
        input = self.quantizer_input(input).to(x_dtype)
        self.quantizer_input.free()
                       
        output = self.linear(input)
        
        self.quantizer_output.find_params(output)       
        x_dtype = output.dtype
        input = self.quantizer_output(output).to(x_dtype)
        self.quantizer_output.free()
             
        return output
    
class StatWrapper(nn.Module):
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
        # If necessary we quantize: weight, input and output as defined 
        self.linear = quantize_block(self.linear, self.quantizer_w)
        self.quantizer_input.find_params(input)       
        x_dtype = input.dtype
        input = self.quantizer_input(input).to(x_dtype)
        self.quantizer_input.free()
                       
        output = self.linear(input)
        
        self.quantizer_output.find_params(output)       
        x_dtype = output.dtype
        input = self.quantizer_output(output).to(x_dtype)
        self.quantizer_output.free()
             
        return output

def update_precision(module, func, sort_func, filter_func, iter_num, parent_name=None, update=False): #, newVal=None):
    if update:
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if (isinstance(child, DropoutWrapper) or isinstance(child, StatWrapper)):
                child.weight_q, child.input_q, child.output_q, child.gradient_q = func(child, iter_num, full_name)
                child.quantizer_w = map_quantizer(child.weight_q, True)
                child.quantizer_input = map_quantizer(child.input_q, False)
                child.quantizer_output = map_quantizer(child.output_q, False)
                
            else:
                update_precision(child, func, sort_func, filter_func, iter_num, parent_name=full_name, update=update)

# TODO: this should be extended also to LayerNorms (what about embeddings??). It should receive instead of a run_flag an instructor that will tell us how to define each layer
def wrap_linear_layers(module, wrapper_cls, wrapper_cls_quantize, instructions, parent_name = None, **kwargs):
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
        if isinstance(child, nn.Linear) or isinstance(child, nn.LayerNorm):
            bias = child.bias is not None
            # We want to get from a predefined instructor how to initialize the given layer:
            weight_q, input_q, output_q, gradient_q = instructions.get_quantization_config(full_name)       
            wrapped = wrapper_cls(child, weight_q, input_q, output_q, gradient_q, bias=bias, parent_name=full_name, **kwargs)  # Replace the linear layer in the parent module
            setattr(module, name, wrapped)
        elif isinstance(child, nn.Dropout):
            weight_q, input_q, output_q, gradient_q = instructions.get_quantization_config(full_name)
            wrapped = wrapper_cls_quantize(child, weight_q, input_q, output_q, gradient_q, bias=None, **kwargs)
            setattr(module, name, wrapped)  # Replace the linear layer in the parent module
        else:
            # Recursively wrap submodules
            wrap_linear_layers(child, wrapper_cls, wrapper_cls_quantize, instructions, parent_name=full_name, **kwargs)
    return module
