# This file will contain basic schedulers that will determine 
# how to change the run_flag parameter. 
# A schedular can decide on how to change teh run_flag value in different ways. This makes the inputs very different. 
# Is there a way we can generalize? 
import numpy as np

# This is a simple scheduler that will change from the init run_flag to an updated run_flag when we reach iter_max. 
# The main goal of this is to perform basic experiments and learn how to construct a general scheduler
class scheduler_basic:
    
    def __init__(self, iter_max=0, precision_seq =["fp32", 'fp16', '8bit', '4bit'], fp8_flag=False, lp_flag=False, optimizer_states_flag = False, **kwargs):
        
        # TODO: We are still using the fp8_plag as a string to indicate not only fp8 but also the L3 and L4 states. TO FIX!
        self.fp8_flag = fp8_flag
        self.lp_flag = lp_flag
        self.optimizer_states_flag = optimizer_states_flag # TODO: should this be per-layer??? 
        #self.updated_fp8_flag = updated_fp8_flag
        self.iter_max = iter_max
        # We assume this is ordered from least to most 
        self.precision_seq = precision_seq
        
        if 'layers' in kwargs:
            self.layers = kwargs['layers']
        
        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
            
        if 'max_reductions' in kwargs:
            self.max_reductions = kwargs['max_reductions']
            self.number_reductions = 0
            
        if 'mode' in kwargs:
            self.mode = kwargs['mode']
        
        
    def return_fp8_flag(self):
        
        return self.fp8_flag
    
    def return_lp_flag(self):
        return self.lp_flag
    
    def return_optimizer_states_flag(self):
        return self.optimizer_states_flag
    
    def return_current_fp8_flag(self):
        return self.fp8_flag
    
    def layers_of_interest(self):
        if hasattr(self, 'layers'):
            return self.layers
        else:
            return None 
        
    def return_mode(self):
        if hasattr(self, 'mode'):
            return self.mode
        else:
            return 'nothing' 
    
    # TODO: The default value of sorting_attr would be to return the same value for all elements ??? Check this!
    def sorting_attr(self, module):
        return 1
    
    # TODO: The default value of filter_func would be to return True for all elements ??? Check this!
    def filter_func(self, module):
        return True
    
    # This would be in order to update the general run_flag of the model based on variables that are outside the scope of the model
    #def update_fp8_flag(self, iter_num):
    #    
    #    if iter_num >= self.iter_max:
    #        self.current_fp8_flag = self.updated_fp8_flag
    
    # This would be used to update the model based on its internal variables:
    #def update_model(self, *args, **kwargs):
    # Eventually this will be a no implementaiton and specific implementations will be in the inherited classes... 
    # raise NotImplementedError("Subclasses must implement this method")
    # But for noe we will implement something here. 
    # TODO: We might want this to have kwargs as we will determine whether to update in specific layers according to the status ..... not sure 
    def update_module(self, module, iter_num, full_name):    
        pass
        
        
    def reduce(self, precision):
        
        if precision in self.precision_seq:
            index = self.precision_seq.index(precision)
            if index >= 0 and index < len(self.precision_seq)-1:
                index +=1
        else:
            index = 0
        return self.precision_seq[index]
        
        
    def increase(self, percision):
        
        if percision in self.precision_seq:
            index = self.precision_seq.index(percision)
            if index < len(self.precision_seq) and index > 0:
                index -=1
        else:
            index = 0
        return self.precision_seq[index]

            
    def return_scheduler_name(self):
        pass
            
class no_change(scheduler_basic):
    
    def update_module(self, module, iter_num, full_name):
        
        return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'no change'

# These are very basic examples of schedulers: 
#############################################            
class single_reduction(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max: 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'single reduction at iter ' + str(self.iter_max)
    
    
class double_reduction(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max: 
            return self.reduce(self.reduce(module.weight_q)), self.reduce(self.reduce(module.input_q)), self.reduce(self.reduce(module.output_q)), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'double reduction at iter ' + str(self.iter_max)
    

class single_increase(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max: 
            return self.increase(module.weight_q), self.increase(module.input_q), self.increase(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'single increase at iter ' + str(self.iter_max)
    
class two_reductions(scheduler_basic):
    
    def __init__(self, iter_max=0, precision_seq =["fp32", 'fp16', '8bit', '4bit'], fp8_flag=False, lp_flag=False, optimizer_states_flag = False, additional_iter=0): 
        super().__init__(iter_max=iter_max, precision_seq =precision_seq, fp8_flag=fp8_flag, lp_flag=lp_flag, optimizer_states_flag = optimizer_states_flag)
        self.additional_iter = additional_iter
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max or iter_num == self.additional_iter: 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:            
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'two reductions at iter ' + str(self.additional_iter) + ' and at iter ' + str(self.iter_max)
######################


# This is a temporary class that can reduce percision of layers that have their std tracked if their std in the last two time steps had small changes
# Note that currently we may have more than a single reduction in a specific layer... 
# Now I added the constraint of reducing a layer only once, by adding the reduced attribute.
# We also limit the total number of reductions (here to 3)
# This means that the order in which we perform these reductions matters. 
class automatic_reduction(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        number_of_values = 3
        
        if hasattr(module, "mode") and module.mode == "layer_deep_dive" and len(module.std_values) >= number_of_values+1 and (hasattr(module, 'reduced') == False):
            differences = np.diff(module.std_values)
            mask = differences <= self.epsilon
            if all(mask[-number_of_values:]):
                if hasattr(self, 'iter_reduced_min') and hasattr(self, 'iter_reduced_max'):
                    self.iter_reduced_min = min(self.iter_reduced_min, iter_num)
                    self.iter_reduced_max = max(self.iter_reduced_max, iter_num)
                else:
                    self.iter_reduced_min = iter_num
                    self.iter_reduced_max = iter_num
                module.reduced = True
                return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
            else:
                return module.weight_q, module.input_q, module.output_q, module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        if hasattr(self, 'iter_reduced_min') and hasattr(self, 'iter_reduced_max'):
            return 'automatic reduction: std-diff < ' + str(self.epsilon) + 'between: ' + str(self.iter_reduced_min) +',' + str(self.iter_reduced_max) 
        else:
            return 'automatic reduction: std-diff < ' + str(self.epsilon) + 'non identified'

# This version adds the set time and reduces once only those that comply with the epsilon requirement:
class automatic_reduction_set_time(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        number_of_values = 3
        
        if iter_num == self.iter_max and hasattr(module, "mode") and module.mode == "layer_deep_dive" and len(module.std_values) >= number_of_values+1 and (hasattr(module, 'reduced') == False):
            differences = np.diff(module.std_values)
            mask = differences <= self.epsilon
            if all(mask[-number_of_values:]):
                if hasattr(self, 'iter_reduced_min') and hasattr(self, 'iter_reduced_max'):
                    self.iter_reduced_min = min(self.iter_reduced_min, iter_num)
                    self.iter_reduced_max = max(self.iter_reduced_max, iter_num)
                else:
                    self.iter_reduced_min = iter_num
                    self.iter_reduced_max = iter_num
                module.reduced = True
                return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
            else:
                return module.weight_q, module.input_q, module.output_q, module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        return 'automatic reduction at iter :' + str(self.iter_max) +' std-diff < ' + str(self.epsilon)
    
# This version is the first to incorporate the aorting_attr and the filter_func. These allow the update_percision which has a higher-level view of the model
# To sort the layers in a specific order according to some attribute (aorting_attr) and also to filter out some layers according to some attr. 
class automatic_limited_reductions_set_time_smaller(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        number_of_values = 3
        
        if iter_num == self.iter_max and len(module.std_values) >= number_of_values+1 and self.number_reductions < self.max_reductions:
            differences = np.diff(module.std_values)
            mask = differences <= self.epsilon
            if all(mask[-number_of_values:]):
                if hasattr(self, 'iter_reduced_min') and hasattr(self, 'iter_reduced_max'):
                    self.iter_reduced_min = min(self.iter_reduced_min, iter_num)
                    self.iter_reduced_max = max(self.iter_reduced_max, iter_num)
                else:
                    self.iter_reduced_min = iter_num
                    self.iter_reduced_max = iter_num
                module.reduced = True
                self.number_reductions +=1 
                return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
            else:
                return module.weight_q, module.input_q, module.output_q, module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
    
    def sorting_attr(self, module):
        return module.std_values[-1]
    
    def filter_func(self, module):
        
        if hasattr(module, "mode") and module.mode == "layer_deep_dive" and hasattr(module, "std_values") and len(module.std_values)>0 and (hasattr(module, 'reduced') == False) and 'lm_head' not in module.full_name:
            return True
        else:
            return False
        
    def return_scheduler_name(self):
        return 'automatic limited reduction at iter :' + str(self.iter_max) +' std-diff < ' + str(self.epsilon)
        
        
# This version uses the set time and reduces once only those that comply with the epsilon requirement:
class automatic_limited_reductions_set_time_smaller_reverse(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        number_of_values = 3
        
        if iter_num == self.iter_max and len(module.std_values) >= number_of_values+1 and self.number_reductions < self.max_reductions:
            differences = np.diff(module.std_values)
            mask = differences <= self.epsilon
            if all(mask[-number_of_values:]):
                if hasattr(self, 'iter_reduced_min') and hasattr(self, 'iter_reduced_max'):
                    self.iter_reduced_min = min(self.iter_reduced_min, iter_num)
                    self.iter_reduced_max = max(self.iter_reduced_max, iter_num)
                else:
                    self.iter_reduced_min = iter_num
                    self.iter_reduced_max = iter_num
                module.reduced = True
                self.number_reductions +=1 
                return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
            else:
                return module.weight_q, module.input_q, module.output_q, module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
    
    def sorting_attr(self, module):
        return (-1)*module.std_values[-1]
    
    def filter_func(self, module):
        
        if hasattr(module, "mode") and module.mode == "layer_deep_dive" and hasattr(module, "std_values") and len(module.std_values)>0 and (hasattr(module, 'reduced') == False) and 'lm_head' not in module.full_name:
            return True
        else:
            return False
        
    def return_scheduler_name(self):
        return 'automatic reverse limited reduction at iter :' + str(self.iter_max) +' std-diff < ' + str(self.epsilon)
        

# This version uses the set time and reduces once only those that comply with the epsilon requirement:
class automatic_limited_reductions_set_time_larger(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        number_of_values = 3
        
        if iter_num == self.iter_max and hasattr(module, "mode") and module.mode == "layer_deep_dive" and self.number_reductions < self.max_reductions and len(module.std_values) >= number_of_values+1 and (hasattr(module, 'reduced') == False):
            differences = np.diff(module.std_values)
            mask = differences > self.epsilon
            if all(mask[-number_of_values:]):
                if hasattr(self, 'iter_reduced_min') and hasattr(self, 'iter_reduced_max'):
                    self.iter_reduced_min = min(self.iter_reduced_min, iter_num)
                    self.iter_reduced_max = max(self.iter_reduced_max, iter_num)
                else:
                    self.iter_reduced_min = iter_num
                    self.iter_reduced_max = iter_num
                module.reduced = True
                self.number_reductions +=1 
                return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
            else:
                return module.weight_q, module.input_q, module.output_q, module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        return 'automatic limited reduction at iter :' + str(self.iter_max) +' std-diff > ' + str(self.epsilon)
        

        

# TODO: all these can be replaced by providing the scheduler with a layers object.
 
# A layer that is being reduced stores specific information for this purpose and can be tagged as such through the wrapper. 
# Thus, only in initialization we need to check this. After that we can simply examine the tag....
# This can be removed!!!!    
# We are trying to unify all the below classes to a single class that will receive the list of layers that we want to quantize (reduce) and the time
class reduction_in_layers(scheduler_basic):
     
    def update_module(self, module, iter_num, full_name):
        
        matches = [item in full_name for item in self.layers]
        
        if self.layers and iter_num == self.iter_max and any(matches): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction: ' + str(self.layers) + 'iter ' + str(self.iter_max)
    


# This is a preliminary version of the idea of reducing only in specific layers.
class reduction_in_blocks(scheduler_basic): # reduction_in_layers(iter_max, layers=['transformer.h'])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and 'transformer.h' in full_name: 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block layers at iter ' + str(self.iter_max)


class reduction_in_blocks_3to5(scheduler_basic):  # reduction_in_layers(iter_max, layers=['transformer.h.5', 'transformer.h.4', 'transformer.h.3' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.5' in full_name or 'transformer.h.4' in full_name or 'transformer.h.3' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block layers 3 to 5 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to2(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0', 'transformer.h.1', 'transformer.h.2' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0' in full_name or 'transformer.h.1' in full_name or 'transformer.h.2' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block layers 0 to 2 at iter ' + str(self.iter_max)
       
class reduction_in_blocks_0to1(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0', 'transformer.h.1' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0' in full_name or 'transformer.h.1' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block layers 0 to 1 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0'])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block layers 0 only at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to2_MLP(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0.mlp', 'transformer.h.1.mlp', 'transformer.h.2.mlp' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0.mlp' in full_name or 'transformer.h.1.mlp' in full_name or 'transformer.h.2.mlp' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-MLP layers 0 to 2 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to2_MLP_c_fc(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0.mlp.c_fc', 'transformer.h.1.mlp.c_fc', 'transformer.h.2.mlp.c_fc' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0.mlp.c_fc' in full_name or 'transformer.h.1.mlp.c_fc' in full_name or 'transformer.h.2.mlp.c_fc' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-MLP_c_fc layers 0 to 2 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to2_MLP_proj(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0.mlp.c_proj', 'transformer.h.1.mlp.c_proj', 'transformer.h.2.mlp.c_proj' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0.mlp.c_proj' in full_name or 'transformer.h.1.mlp.c_proj' in full_name or 'transformer.h.2.mlp.c_proj' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-MLP-proj layers 0 to 2 at iter ' + str(self.iter_max)

class reduction_in_blocks_0to2_ATTN(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0.attn', 'transformer.h.1.attn', 'transformer.h.2.attn' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0.attn' in full_name or 'transformer.h.1.attn' in full_name or 'transformer.h.2.attn' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-ATTN layers 0 to 2 at iter ' + str(self.iter_max)

class reduction_in_blocks_0to2_ATTN_C(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0.attn.c_attn', 'transformer.h.1.attn.c_attn', 'transformer.h.2.attn.c_attn' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0.attn.c_attn' in full_name or 'transformer.h.1.attn.c_attn' in full_name or 'transformer.h.2.attn.c_attn' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-ATTN_C layers 0 to 2 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to2_ATTN_proj(scheduler_basic): #reduction_in_layers(iter_max, layers=['transformer.h.0.attn.c_proj', 'transformer.h.1.attn.c_proj', 'transformer.h.2.attn.c_proj' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h.0.attn.c_proj' in full_name or 'transformer.h.1.attn.c_proj' in full_name or 'transformer.h.2.attn.c_proj' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-ATTN_proj layers 0 to 2 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to5_MLP(scheduler_basic): #reduction_in_layers(iter_max, layers=['h.0.mlp', 'h.1.mlp', 'h.2.mlp', 'h.3.mlp', 'h.4.mlp', 'h.5.mlp' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h' in full_name and 'mlp' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-MLP layers 0 to 5 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to5_MLP_c_fc(scheduler_basic): #reduction_in_layers(iter_max, layers=['h.0.mlp.c_fc', 'h.1.mlp.c_fc', 'h.2.mlp.c_fc', 'h.3.mlp.c_fc', 'h.4.mlp.c_fc', 'h.5.mlp.c_fc' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h' in full_name and 'mlp.c_fc' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-MLP_c_fc layers 0 to 5 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to5_MLP_proj(scheduler_basic): #reduction_in_layers(iter_max, layers=['h.0.mlp.c_proj', 'h.1.mlp.c_proj', 'h.2.mlp.c_proj', 'h.3.mlp.c_proj', 'h.4.mlp.c_proj', 'h.5.mlp.c_proj' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h' in full_name and 'mlp.c_proj' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-MLP-proj layers 0 to 5 at iter ' + str(self.iter_max)

class reduction_in_blocks_0to5_ATTN(scheduler_basic): #reduction_in_layers(iter_max, layers=['h.0.attn', 'h.1.attn', 'h.2.attn', 'h.3.attn', 'h.4.attn', 'h.5.attn' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h' in full_name and 'attn' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-ATTN layers 0 to 5 at iter ' + str(self.iter_max)

class reduction_in_blocks_0to5_ATTN_C(scheduler_basic): #reduction_in_layers(iter_max, layers=['h.0.attn.c_attn', 'h.1.attn.c_attn', 'h.2.attn.c_attn', 'h.3.attn.c_attn', 'h.4.attn.c_attn', 'h.5.attn.c_attn' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h' in full_name and 'attn.c_attn' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-ATTN_C layers 0 to 5 at iter ' + str(self.iter_max)
    
class reduction_in_blocks_0to5_ATTN_proj(scheduler_basic): #reduction_in_layers(iter_max, layers=['h.0.attn.c_proj', 'h.1.attn.c_proj', 'h.2.attn.c_proj', 'h.3.attn.c_proj', 'h.4.attn.c_proj', 'h.5.attn.c_proj' ])
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.iter_max and ('transformer.h' in full_name and 'attn.c_proj' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'reduction in block-ATTN_proj layers 0 to 5 at iter ' + str(self.iter_max)


#### These include a reduction and an increase. This is very specific not sure worth going into... 
# These include two time points. Reduction and increase. 
class reduction_increase_in_blocks_0to2(scheduler_basic):
    
    def __init__(self, iter_max=0, precision_seq =["fp32", 'fp16', '8bit', '4bit'], fp8_flag=False, lp_flag=False, optimizer_states_flag = False, additional_iter=0): 
        super().__init__(iter_max=iter_max, precision_seq =precision_seq, fp8_flag=fp8_flag, lp_flag=lp_flag, optimizer_states_flag = optimizer_states_flag)
        self.additional_iter = additional_iter
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.additional_iter and ('transformer.h.0' in full_name or 'transformer.h.1' in full_name or 'transformer.h.2' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        elif iter_num == self.iter_max and ('transformer.h.0' in full_name or 'transformer.h.1' in full_name or 'transformer.h.2' in full_name): 
            return self.increase(module.weight_q), self.increase(module.input_q), self.increase(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'in blocks layers 0to2: reductions at iter ' + str(self.additional_iter) + ' and increase at iter ' + str(self.iter_max)

class reduction_increase_in_blocks_0to1(scheduler_basic):
    
    def __init__(self, iter_max=0, precision_seq =["fp32", 'fp16', '8bit', '4bit'], fp8_flag=False, lp_flag=False, optimizer_states_flag = False, additional_iter=0): 
        super().__init__(iter_max=iter_max, precision_seq =precision_seq, fp8_flag=fp8_flag, lp_flag=lp_flag, optimizer_states_flag = optimizer_states_flag)
        self.additional_iter = additional_iter
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.additional_iter and ('transformer.h.0' in full_name or 'transformer.h.1' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        elif iter_num == self.iter_max and ('transformer.h.0' in full_name or 'transformer.h.1' in full_name): 
            return self.increase(module.weight_q), self.increase(module.input_q), self.increase(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'in blocks layers 0to1: reductions at iter ' + str(self.additional_iter) + ' and increase at iter ' + str(self.iter_max)

class reduction_increase_in_blocks_0(scheduler_basic):
    
    def __init__(self, iter_max=0, precision_seq =["fp32", 'fp16', '8bit', '4bit'], fp8_flag=False, lp_flag=False, optimizer_states_flag = False, additional_iter=0): 
        super().__init__(iter_max=iter_max, precision_seq =precision_seq, fp8_flag=fp8_flag, lp_flag=lp_flag, optimizer_states_flag = optimizer_states_flag)
        self.additional_iter = additional_iter
     
    def update_module(self, module, iter_num, full_name):
        
        if iter_num == self.additional_iter and ('transformer.h.0' in full_name): 
            return self.reduce(module.weight_q), self.reduce(module.input_q), self.reduce(module.output_q), module.gradient_q
        elif iter_num == self.iter_max and ('transformer.h.0' in full_name): 
            return self.increase(module.weight_q), self.increase(module.input_q), self.increase(module.output_q), module.gradient_q
        else:
            return module.weight_q, module.input_q, module.output_q, module.gradient_q
        
    def return_scheduler_name(self):
        #return 'simple_'+ str(self.run_flag_init) + '_to_' + str(self.updated_run_flag)
        return 'in blocks layers 0: reductions at iter ' + str(self.additional_iter) + ' and increase at iter ' + str(self.iter_max)