import torch
import torch.nn as nn
import random

import os
from scipy.stats import norm, laplace, halfnorm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from quantizing import quantize_block, ActQuantizer, WeightQuantizer, BasicQuantizer
import instruction_reader

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
    def __init__(self, child, optimizer, est_interval, weight_q='fp16', input_q='fp16', output_q='fp16', gradient_q='fp16', bias=True, beta=0.99, **kwargs):
        super().__init__()
        self.linear = child 
        self.beta1 = beta
        self.beta2 = beta
        self.optimizer = optimizer
        self.est_interval = est_interval
        # Integrating the quantization into the wrapper
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
    def __init__(self, child, optimizer, est_interval, weight_q='fp16', input_q='fp16', output_q='fp16', gradient_q='fp16', bias=True, beta=0.99, **kwargs):
        super().__init__()
        self.linear = child 
        if 'mode' in kwargs:
            self.mode=kwargs['mode']
        else:
            self.mode='optimizer_est'
        
        if 'parent_name' in kwargs:
            self.full_name = kwargs['parent_name']   
        else:
            self.full_name = ''
        
        # TODO: are these relevant only for the "layer_deep_dive"???     
        if self.mode == 'layer_deep_dive':
            if 'output_dir' in kwargs:     
                self.directory = f"{kwargs['output_dir']}"
            else:
                self.directory = f"stat_output/{kwargs['now']}/"
   
        self.beta1 = beta
        self.beta2 = beta
        self.optimizer = optimizer
        self.est_interval = est_interval
        # Integrating the quantization into the wrapper
        self.weight_q = weight_q
        self.input_q = input_q
        self.output_q = output_q
        self.gradient_q = gradient_q
        
        self.quantizer_w = map_quantizer(self.weight_q, True)
        self.quantizer_input = map_quantizer(self.input_q, False)
        self.quantizer_output = map_quantizer(self.output_q, False)
        
        # For now... 
        self.track_bias = False 
        # We can determine whether to loos at bias according to:
        #if bias and self.linear.bias is not None:
        #    all_params = torch.cat([all_params, self.linear.bias.view(-1)])
        #    self._has_bias = True
        #else:
        #    self._has_bias = False
        
        # TODO: All this is used in the optimizer_est case only, right?
        # TODO: should we create another case for optimizer_std??? 
        # TODO: Iin optimizer_std we do not want to randomly pick a single element in a layer. 
        # I want to extract the best mean estimator using two counters for each weight (element we will define as ints).
        # based on these we would like to add an std estimation for each layer. 
        # I will duplicate the hook for this new case and and ass vectors of ints like for the moments1 that we currently have. 
        # We can add this to the deep_dive option for now, so we will already have the ability to perform over subsets of layers and also have the true std calculation
        # We only need to add the hook and extend the mean claulcation to all weights within the layer, and from it perform the std calculation. 
        # We then want to compare it with the true std (we can add that to the printout we currently have of the stds)
        # We want to preserve support of the following:
        # 1. Flexibility of sharing information among the elements of the layer (like the delat element)
        # 2. flexibility of how many weights are taken into account
        # 3. switching to variance estimation (but leave the optimizer_est out to continue with that later on)
        # Anything else? 
        # Out best estimator for the mean was est2
        # We will build the support to create a delat for each weight but also a single element for all weights, they will both hold a moving avergae either for each weight on its own or for all weights together
    
        if self.track_bias:
            self.tracked_tensor = self.linear.bias        # Full tensor
        else:
            self.tracked_tensor = self.linear.weight      # Full tensor
    
        self.alpha = self.get_lr_for_param(self.optimizer, self.tracked_tensor)
        if self.alpha == None:
            self.alpha = 0.001
        
        self.register_buffer("step", torch.tensor(0))
        
        if self.mode == 'optimizer_est':
            
            self.tracked_index = random.randint(0, self.tracked_tensor.numel() - 1)
            
            # Initialize EMA variables and step counter
            self.register_buffer("moment1", torch.tensor(0.0))
            self.register_buffer("moment2", torch.tensor(0.0))
            #self.register_buffer("step2", torch.tensor(0))
            
            self.T_window = 100 # 1000
            self.register_buffer("windowStep", torch.tensor(0))
        
            # Full history buffers (Python lists) - for comparison. These can provide optimal estimators.
            self.moment1_history = []
            self.moment2_history = []
            self.value_history = []
            
            ## Gradient EMA tracking for Adam-like behavior
            #self.register_buffer("grad_moment1", torch.tensor(0.0))
            #self.register_buffer("grad_moment2", torch.tensor(0.0))
            #self.grad_moment1_history = []
            #self.grad_moment2_history = []
            
            self.variance_est_history = []
            #self.adam_ratio_history = []
            #self.adam_diff_history = []
            self.adam_delta_history = []
            self.adam_mean_est1_history = []
            self.adam_mean_est2_history = []
            self.adam_mean_est3_history = []
            self.adam_var_est1_history = []
            self.adam_var_est2_history = []
            self.adam_var_est3_history = []
            self.adam_var_est4_history = []
            self.adam_var_est5_history = []
            
            # Assisting values
            self.positive = 0    
            # The init_counter is helpful with initializing the counter when we shift the window  
            self.init_counter = 0
            
            # Additional counters to enhance our variance estimator:
            self.pos_high = 0
            self.pos_low = 0
            self.neg_high = 0
            self.neg_low = 0
            
            self.init_pos_high = 0
            self.init_pos_low = 0
            self.init_neg_high = 0
            self.init_neg_low = 0
            
            self.a1 = 1.2
            self.a2 = 0.8
            # Since we are holding the delta history, for now we can use an average over it (over positive or negative) and take half of it to be a threshold to distinguish between low and high. 
         
            self.register_gradient_hook()


        if self.mode == 'layer_deep_dive':
            
            # Define the dictionary of std values across iterations:
            self.std_values = []
            self.est_std_values = []
            
            self.T_window = 100 # 1000
            self.register_buffer("windowStep", torch.tensor(0))
            self.ma_d = 1 / self.T_window
                               
            flat_size = self.linear.weight.numel()
            # Register buffer to keep track of moving-average estimator of mean. 
            # This is important since we do not create these graphs at every time-step, we create them once every est_interval
            self.register_buffer("moments1", torch.zeros(flat_size))
            
            # For the estimator of std based on the optimizer:
            self.register_buffer("delta_single_ma", torch.tensor(0.0))
            self.register_buffer("deltas_ma", torch.zeros(flat_size))
            # int8, since the T_window is currently set to 100 so we are well within the range of int8: 
            self.register_buffer("pos_counter", torch.zeros(flat_size, dtype=torch.int8))
            self.register_buffer("init_counter", torch.zeros(flat_size, dtype=torch.int8))
            
            
            self.adam_mean_est2_history = []
            self.adam_mean_est4_history = []
            
            self.register_gradient_hook_std()
            
        if self.mode == "nothing":
            print('For this layer we do nothing')            
                

    def get_lr_for_param(self, optimizer, param):
        device_param = param.device
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.shape == param.shape and torch.equal(p, param.to(p.device)): 
                    param.to(device_param)
                    return group['lr']
        return None

    def register_gradient_hook(self):
        
        self.tracked_tensor.retain_grad()

        def grad_hook(_):
            
            with torch.no_grad():
                state = self.optimizer.state.get(self.tracked_tensor)
                if state is None:
                    return

                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None or exp_avg_sq is None:
                    return

                m1 = exp_avg.view(-1)[self.tracked_index]
                m2 = exp_avg_sq.view(-1)[self.tracked_index]
                
                # Updating a general positive delta counter:
                if m1 > 0.0:
                    self.positive += 1
                    
                adam_delta = m1 / torch.sqrt((m2 + 1e-8))
                
                # Calculating a threshold to be used for the more delicate counters (a threshold can be calculated per-layer to reduce memory)
                mean_interval = int(self.windowStep.item()/self.est_interval * 1)
                adam_delta_window = self.adam_delta_history[-mean_interval:]
                            
                positive_adam_delta_window = [x for x in adam_delta_window if x > 0]
                negative_adam_delta_window = [abs(x) for x in adam_delta_window if x < 0]
                
                threshold_pos = 0
                threshold_neg = 0
                        
                if positive_adam_delta_window:
                    threshold_pos = sum(positive_adam_delta_window) / len(positive_adam_delta_window)
                if negative_adam_delta_window:
                    threshold_neg = sum(negative_adam_delta_window) / len(negative_adam_delta_window)
                
                # For debugging of the estimators
                #print('The positive threshold is:')
                #print(threshold_pos)
                #print('The negative threshold is:')
                #print(threshold_neg)
                
                    
                if adam_delta > 0:
                    if adam_delta > threshold_pos:
                        self.pos_high +=1
                    else:
                        self.pos_low +=1
                else:
                    if adam_delta < -threshold_neg:
                        self.neg_high +=1
                    else:
                        self.neg_low +=1
                    
                self.step = state['step']
                
                # For debugging of the estimators
                #print('Counters:')
                #print(self.pos_high)
                #print(self.pos_low)
                #print(self.neg_high)
                #print(self.neg_low)
                
                #print('The window size:')
                #print(self.windowStep)
                # TODO: I think I have a bug with the progression of the window size
                
                
                # We compute an estimator and store it only every est_interval
                if self.step%self.est_interval == 0:
                    
                    # For debugging of the estimators
                    #print('We are in and the current step is:')
                    #print(self.step)
                    

                    #adam_ratio = (self.alpha**2)*(m1**2) / (m2 + 1e-8)  #
                    #adam_diff = m2 - m1**2
                    #adam_delta = m1 / torch.sqrt((m2 + 1e-8))
                    
                    # Extracting the current weight:
                    if self.track_bias:
                        val = self.linear.bias.view(-1)[self.tracked_index]
                    else:
                        val = self.linear.weight.view(-1)[self.tracked_index]
                        
                    
                    # Bias correction
                    beta_pow1 = self.beta1 ** self.step
                    beta_pow2 = self.beta2 ** self.step
                    
                    #T = state['step']
                    T_internal = self.windowStep.item() # TODO: check if we need +1 here?? 
                    T = T_internal # With the full T we get worse estimators... 
                    
                    # printing the T and positive values:
                    #print('T and positive values:')
                    #print(T)
                    #print(self.positive)
                    #print(2*self.positive - T)
                            
                    if T <= 0:
                        adam_mean_est1 = val
                        adam_mean_est2 = val
                        adam_var_est1 = torch.zeros(1)
                        adam_var_est2 = torch.zeros(1)
                        adam_var_est3 = torch.zeros(1)
                    else:              
                        adam_mean_est1 = ((T+1)/T * val + (T+1)/2 * (self.alpha)*adam_delta * (2*self.positive - T)/T) 
                        adam_mean_est2 = ((T+1)/T * val + (T+1)/2 * (self.alpha)*(threshold_pos+threshold_neg)/2 * (2*self.positive - T_internal)/T_internal) 
                        adam_mean_est3 = ((T+1)/T * val + (T+1)/2 * (self.alpha)*( self.pos_high * self.a1*threshold_pos + self.pos_low* self.a2*threshold_pos - self.neg_high * self.a1*threshold_neg - self.neg_low *self.a2*threshold_neg )/T) 
                        adam_var_est1 = (T+1)/T * (val**2) - adam_mean_est1**2 
                        adam_var_est2 = (T+1)/T * (val**2) + self.alpha* val * adam_delta * (T+1) * (2*self.positive - T)/T + (self.alpha**2) * (adam_delta**2) * ( 3*T - 1)/2  - adam_mean_est1**2 
                        adam_var_est3 = (T+1)/T * (val**2) + self.alpha* val * (T+1) * ( self.pos_high * self.a1*threshold_pos + self.pos_low* self.a2*threshold_pos - self.neg_high * self.a1*threshold_neg - self.neg_low *self.a2*threshold_neg )/T \
                            + (self.alpha**2) *(T+1)/2 *(self.pos_high * (self.a1**2) * (threshold_pos**2) + self.pos_low* (self.a2**2)*(threshold_pos**2) + self.neg_high * (self.a1**2)*(threshold_neg**2) + self.neg_low *(self.a2**2)*threshold_neg**2 )/T \
                            + (self.alpha**2) *(self.pos_high*(self.pos_high-1) * (self.a1**2) * (threshold_pos**2) + self.pos_low*(self.pos_low-1)* (self.a2**2)* (threshold_pos**2) + self.neg_high*(self.neg_high-1) * (self.a1**2) *(threshold_neg**2) + self.neg_low*(self.neg_low-1) *(self.a2**2) * (threshold_neg**2))/T \
                            + 2*(self.alpha**2) * ( self.pos_high*self.pos_low* self.a1*self.a2 * (threshold_pos**2) - self.pos_high*self.neg_high*(self.a1**2)*threshold_pos*threshold_neg - self.pos_high*self.neg_low*self.a1*self.a2*threshold_pos*threshold_neg - self.pos_low*self.neg_high*self.a1*self.a2*threshold_pos*threshold_neg - self.pos_low*self.neg_low*(self.a2**2)*threshold_pos*threshold_neg + self.neg_high*self.neg_low*self.a1*self.a2*(threshold_neg**2))/T \
                                - adam_mean_est2**2 
                        adam_var_est4 = (T+1)/T * (val**2) + self.alpha* val * adam_delta * (T+1) * (2*self.positive - T)/T + (self.alpha**2) * (adam_delta**2) * ( 3*T - 1)/2  - adam_mean_est2**2 
                        # This version of adam_var_est5 is zero throughout... not sure why
                        #adam_var_est5 = (T+1)/T * (val**2) + self.alpha* val * (threshold_pos+threshold_neg)/2 * (T+1) * (2*self.positive - T)/T + (self.alpha**2) * ((threshold_pos+threshold_neg)**2/4) * ( 3*T - 1)/2  - adam_mean_est2**2 
                        adam_var_est5 = (T+1)/T * (val**2) + self.alpha* val * (threshold_pos+threshold_neg)/2 * (T+1) * (2*self.positive - T)/T + (self.alpha**2) * (((threshold_pos+threshold_neg)/2)**2) * ( 3*T - 1)/2  - adam_mean_est2**2 
                        #adam_var_est3 = (T+1)/T * (val**2) + self.alpha* val * adam_delta * (T+1) * (2*self.positive - T_internal)/T_internal + (self.alpha**2) * (adam_delta**2) * ( 3*T - 1)/2  - adam_mean_est2**2 
                        if adam_var_est2 < 0:
                            if adam_var_est1 >= 0:
                                adam_var_est2 = adam_var_est1
                            else:
                                adam_var_est2 = torch.ones(1)*float('nan')
                                
                        if adam_var_est3 < 0:
                            #if adam_var_est1 >= 0:
                            #    adam_var_est3 = adam_var_est1
                            #else:
                            adam_var_est3 = torch.ones(1)*float('nan')
                        
                        if adam_var_est4 < 0:
                            #if adam_var_est1 >= 0:
                            #    adam_var_est4 = adam_var_est1
                            #else:
                            adam_var_est4 = torch.ones(1)*float('nan')
                            
                        if adam_var_est5 < 0:
                            #if adam_var_est1 >= 0:
                            #    adam_var_est4 = adam_var_est1
                            #else:
                            adam_var_est5 = torch.ones(1)*float('nan')
                        # adam_var_est2 = ((T+1)/T * (val**2) + 3*(T-1)/2 * ((self.alpha)**2) * (adam_delta**2) +  adam_delta * ( self.alpha*val*(T-1) + 2*(self.alpha**2)/T)* (2*self.positive - T)/T) - adam_mean_est1**2 
                            
                    
                    # Saving for plots:
                    #self.adam_ratio_history.append(adam_ratio.item())
                    #self.adam_diff_history.append(adam_diff.item())
                    self.adam_delta_history.append(adam_delta.item())
                    #self.adam_mean_est1_history.append(adam_mean_est1.item())
                    self.adam_mean_est2_history.append(adam_mean_est2.item())
                    self.adam_mean_est3_history.append(adam_mean_est3.item())
                    #self.adam_var_est1_history.append(adam_var_est1.item())
                    self.adam_var_est2_history.append(adam_var_est2.item())
                    self.adam_var_est3_history.append(adam_var_est3.item())
                    self.adam_var_est4_history.append(adam_var_est4.item())
                    self.adam_var_est5_history.append(adam_var_est5.item())
                
                
                if self.windowStep.item() == int(self.T_window/2): 
                    self.init_counter = self.positive
                    self.init_pos_high = self.pos_high 
                    self.init_pos_low = self.pos_low 
                    self.init_neg_high = self.neg_high 
                    self.init_neg_low = self.neg_low 
                
                if self.windowStep.item() == self.T_window:
                    self.windowStep = torch.tensor(int(self.T_window/2) - 1)
                    self.positive = self.positive - self.init_counter
                    # Currently we do not have init counters for the higher resolution counters
                    self.pos_high = self.pos_high - self.init_pos_high
                    self.pos_low = self.pos_low - self.init_pos_low
                    self.neg_high = self.neg_high - self.init_neg_high
                    self.neg_low = self.neg_low - self.init_neg_low
                    
                self.windowStep += 1
            
        self.tracked_tensor.register_hook(grad_hook)
        
    def register_gradient_hook_std(self):
        
        self.tracked_tensor.retain_grad()

        def grad_hook_std(_):
            
            with torch.no_grad():
                state = self.optimizer.state.get(self.tracked_tensor)
                if state is None:
                    return

                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None or exp_avg_sq is None:
                    return

                m1 = exp_avg.view(-1)
                m2 = exp_avg_sq.view(-1)
                
                # Updating a general positive delta counter:
                mask = m1 > 0
                self.pos_counter[mask] += 1
                    
                adam_deltas = m1 / torch.sqrt((m2 + 1e-8))
                
                self.deltas_ma = (1 - self.ma_d) * self.deltas_ma + (self.ma_d)*adam_deltas
                self.delta_single_ma = self.deltas_ma.mean()
                    
                self.step = state['step']
                
                mean_interval = int(self.windowStep.item()/self.est_interval * 1)
                # We compute an estimator and store it only every est_interval
                if self.step%self.est_interval == 0:
                                                          
                    # Extracting the current weight:
                    if self.track_bias:
                        vals = self.linear.bias.view(-1)
                    else:
                        vals = self.linear.weight.view(-1)
                    
                    #T = state['step']
                    T_internal = self.windowStep.item() # TODO: check if we need +1 here?? 
                    T = T_internal # With the full T we get worse estimators... 
                
                            
                    if T <= 0:
                        #adam_mean_est1 = vals
                        adam_mean_est2 = vals
                        adam_mean_est4 = vals
                    else:              
                        #adam_mean_est1 = ((T+1)/T * val + (T+1)/2 * (self.alpha)*adam_delta * (2*self.positive - T)/T) 
                        adam_mean_est2 = ((T+1)/T * vals + (T+1)/2 * (self.alpha)*self.deltas_ma * (2*self.pos_counter - T_internal)/T_internal) 
                        adam_mean_est4 = ((T+1)/T * vals + (T+1)/2 * (self.alpha)* self.delta_single_ma * (2*self.pos_counter - T_internal)/T_internal) 
                        #adam_mean_est3 = ((T+1)/T * val + (T+1)/2 * (self.alpha)*( self.pos_high * self.a1*threshold_pos + self.pos_low* self.a2*threshold_pos - self.neg_high * self.a1*threshold_neg - self.neg_low *self.a2*threshold_neg )/T) 
                        
                    # These are now tensors so we cannot plot this.... 
                    # Saving for plots:
                    #self.adam_mean_est1_history.append(adam_mean_est1.item())
                    #self.adam_mean_est2_history.append(adam_mean_est2.item())
                    #self.adam_mean_est3_history.append(adam_mean_est3.item())
                    #self.adam_mean_est4_history.append(adam_mean_est4.item())
                    
                    print('Now we create a histogram of the values in the buffer')
                    data = adam_mean_est4.cpu().numpy()
                    abs_data= abs(data)
                    # Histogram
                    count, bins, ignored = plt.hist(abs_data, bins=30, density=True, alpha=0.6, color='g')

                    # fit halfnormal distributions:
                    loc, scale = halfnorm.fit(abs_data)
                    std = scale*np.sqrt( 1- 2 / np.pi)
                    x = np.linspace(0, abs_data.max(), 100)
                    pdf = halfnorm.pdf(x, loc, scale)

                    plt.hist(abs_data, bins=30, density=True, alpha=0.6, color='g')
                    plt.plot(x, pdf, 'r', linewidth=2, label=f"σ={std:.5f}")
                    plt.legend()
                    plt.title(f"Hist of |data| with Half-Normal Fit: {self.full_name} Iter: {str(int(self.step.item()))}")

                    self.est_std_values.append(std)
                    
                    filename = f"{self.directory}{self.full_name}_{str(int(self.step.item()))}_expectation_est_hist.png"
                    plt.tight_layout()
                    plt.savefig(filename)
                    plt.close()
                    print(f"Saved plot for layer expectation estimated hist {filename}")    
                    
                
                if self.windowStep.item() == int(self.T_window/2): 
                    self.init_counter = self.pos_counter
                    #self.init_pos_high = self.pos_high 
                    #self.init_pos_low = self.pos_low 
                    #self.init_neg_high = self.neg_high 
                    #self.init_neg_low = self.neg_low 
                
                if self.windowStep.item() == self.T_window:
                    self.windowStep = torch.tensor(int(self.T_window/2) - 1)
                    self.pos_counter = self.pos_counter - self.init_counter
                    # Currently we do not have init counters for the higher resolution counters
                    #self.pos_high = self.pos_high - self.init_pos_high
                    #self.pos_low = self.pos_low - self.init_pos_low
                    #self.neg_high = self.neg_high - self.init_neg_high
                    #self.neg_low = self.neg_low - self.init_neg_low
                    
                    
                self.windowStep += 1
            
        self.tracked_tensor.register_hook(grad_hook_std)

    def forward_layer_deep_dive(self, input):
        
        if torch.is_grad_enabled() and input.requires_grad:
            
                
            if self.track_bias:
                vals = self.linear.bias.view(-1)
            else:
                vals = self.linear.weight.view(-1)
            
            buffer_name = "moments1"
            buffer = getattr(self, buffer_name)    
            
            with torch.no_grad():
                buffer.mul_(self.beta1).add_((1 - self.beta1) * vals)
            
                if self.optimizer.state:
                    step = self.optimizer.state[next(iter(self.optimizer.state))].get('step',None)
                else:
                    step = None 
                
                #if self.step and self.step%self.est_interval == 0:
                if step != None and step and step%self.est_interval == 0:
                    print('Now we create a histogram of the values in the buffer')
                    data = buffer.cpu().numpy()
                    abs_data= abs(data)
                    # Histogram
                    count, bins, ignored = plt.hist(abs_data, bins=30, density=True, alpha=0.6, color='g')

                    """
                    # Fit a normal distribution to the data
                    mu, std = norm.fit(data)

                    # Plot the PDF
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, mu, std)
                    plt.plot(x, p, 'k', linewidth=2, label=f"Fit: μ={mu:.5f}, σ={std:.5f}")
                    plt.legend()
                    plt.title(f"Hist with Fitted Normal Dist: {layer_name} Iter: {str(step.item())}")
                    """
                    # fit halfnormal distributions:
                    loc, scale = halfnorm.fit(abs_data)
                    std = scale*np.sqrt( 1- 2 / np.pi)
                    x = np.linspace(0, abs_data.max(), 100)
                    pdf = halfnorm.pdf(x, loc, scale)

                    plt.hist(abs_data, bins=30, density=True, alpha=0.6, color='g')
                    plt.plot(x, pdf, 'r', linewidth=2, label=f"σ={std:.5f}")
                    plt.legend()
                    plt.title(f"Hist of |data| with Half-Normal Fit: {self.full_name} Iter: {str(int(step.item()))}")
    
                    self.std_values.append(std)
                    """
                    # Fit Laplace distribution
                    loc, scale = laplace.fit(data)  # loc = mean, scale = diversity (b)

                    # Plot the PDF
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    pdf = laplace.pdf(x, loc, scale)

                    plt.plot(x, pdf, 'k', linewidth=2, label=f"Laplace Fit: μ={loc:.5f}, b={scale:.5f}")
                    plt.legend()
                    plt.title(f"Hist with Fitted Laplace Dist: {layer_name} Iter: {str(step.item())}")
                    """
                    
                    filename = f"{self.directory}{self.full_name}_{str(int(step.item()))}_expectation_hist.png"
                    plt.tight_layout()
                    plt.savefig(filename)
                    plt.close()
                    print(f"Saved plot for layer expectation hist {filename}")     

  
    def forward_optimizer_est(self, input):
        
        if torch.is_grad_enabled() and input.requires_grad:
            # Get value of the tracked element
            if self.track_bias:
                val = self.linear.bias.view(-1)[self.tracked_index]
            else:
                val = self.linear.weight.view(-1)[self.tracked_index]
                
            # I took this out to improve our baseline estimator. If we take into consideration all the values we should get a better estimator (even if we actually calculate it every self.est_interval iterations...)
            # Update EMA stats
            #self.moment1 = self.beta1 * self.moment1 + (1 - self.beta1) * val
            #self.moment2 = self.beta2 * self.moment2 + (1 - self.beta2) * (val ** 2)
            with torch.no_grad():
                self.moment1.mul_(self.beta1).add_((1 - self.beta1) * val)
                self.moment2.mul_(self.beta2).add_((1 - self.beta2) * val ** 2)
            
                # We create the baseline of the mean and variance only every est_interval: 
                if self.step and self.step%self.est_interval == 0:

                    # Bias correction
                    beta_pow1 = self.beta1 ** self.step
                    beta_pow2 = self.beta2 ** self.step
                    corrected_m1 = self.moment1 / (1 - beta_pow1)
                    corrected_m2 = self.moment2 / (1 - beta_pow2)
                    
                    # Compute variance estimator
                    variance_est = corrected_m2 - corrected_m1**2
                    self.variance_est_history.append(variance_est.item())

                    # Record history
                    self.value_history.append(val.item())
                    self.moment1_history.append(corrected_m1.item())
                    self.moment2_history.append(corrected_m2.item())
                    
                    #self.step2 += 1
        

    def forward(self, input):
        
        if self.mode == 'optimizer_est':
            self.forward_optimizer_est(input)
        
        if self.mode == 'layer_deep_dive':
            self.forward_layer_deep_dive(input)           
        

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
    
    def fill_nan_with_nearest(self, arr):
        arr = np.asarray(arr, dtype=np.float32).copy()
        isnan = np.isnan(arr)

        if not isnan.any():
            return arr

        indices = np.arange(len(arr))
        valid_indices = indices[~isnan]
        valid_values = arr[~isnan]

        if len(valid_values) > 0:
            for i in indices[isnan]:
                # Find the index of the closest valid (non-NaN) value
                closest_idx = np.abs(valid_indices - i).argmin()
                arr[i] = valid_values[closest_idx]
        else:
            return []

        return arr
    
    def get_statistics(self):
        """Return the latest statistics."""
        if self.step == 0:
            return None
        return {
            "mean": self.moment1_history[-1],
            "second_moment": self.moment2_history[-1],
            "raw_value": self.value_history[-1]
        }

    def get_history(self):
        """Return full history of tracked statistics."""
        return {
            "values": self.value_history,
            "mean": self.moment1_history,
            "second_moment": self.moment2_history
        }
        
    def get_history_2(self):
        """Return full history of tracked statistics."""
        return {
            "direct_estimate_var": self.variance_est_history,
            "direct_estimate_expectation": self.moment1_history,
            #"adam_estimate": self.adam_ratio_history,
            #"adam_diff_estimate": self.adam_diff_history,
            "adam_delta_estimate": self.adam_delta_history,
            "adam_mean_est1": self.adam_mean_est1_history,
            "adam_mean_est2": self.adam_mean_est2_history,
            "adam_mean_est3": self.adam_mean_est3_history,
            "adam_var_est1": self.fill_nan_with_nearest(self.adam_var_est1_history),
            "adam_var_est2": self.fill_nan_with_nearest(self.adam_var_est2_history),
            "adam_var_est3": self.fill_nan_with_nearest(self.adam_var_est3_history),
            "adam_var_est4": self.fill_nan_with_nearest(self.adam_var_est4_history),
            "adam_var_est5": self.fill_nan_with_nearest(self.adam_var_est5_history)
        }

# Filter + sort modules by number of parameters (descending)
def sorted_named_children(items, module, key_func, reverse=False, filter_func=None):
    
    for name, child in module.named_children():
        if (isinstance(child, DropoutWrapper) or isinstance(child, StatWrapper)):
            if filter_func is None or filter_func(child):
                attr = key_func(child)
                items.append((name, child, attr))
        else:
            items = sorted_named_children(items, child, key_func, filter_func=filter_func, reverse=reverse)
    # Sort by the extracted attribute
    sorted_items = sorted(items, key=lambda x: x[2], reverse=reverse)
    return sorted_items 

# Adding this to support changing the run_flag. Currently this is simplified. 
# We recieve an updated value for the run_flag, but we can also have something that receives a method that will automatically change the run_flag
# even for each layer of the model... we want to generalize these.
def update_precision(module, func, sort_func, filter_func, iter_num, parent_name=None, update=False): #, newVal=None):
    
    if update:
        items = []
        items = sorted_named_children(items, module, sort_func, filter_func=filter_func)
        for name, child, _ in items: #module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if (isinstance(child, DropoutWrapper) or isinstance(child, StatWrapper)):
                child.weight_q, child.input_q, child.output_q, child.gradient_q = func(child, iter_num, full_name)
                child.quantizer_w = map_quantizer(child.weight_q, True)
                child.quantizer_input = map_quantizer(child.input_q, False)
                child.quantizer_output = map_quantizer(child.output_q, False)
                
            else:
                update_precision(child, func, sort_func, filter_func, iter_num, parent_name=full_name, update=update)
                
    """    
    if func == None and newVal == None:
        print('No new update')
    else:
        if newVal:
            self.run_flag = newVal
        if func:
            # Just for testing to see that we can use func. The idea here is to go over the model layer by layer and determine whether to change the precision, either to increase or reduce... 
            layer_var = 0.3
            var_threshold = 0.4
            current_run_flag = self.run_flag
            self.run_flag = func(layer_var, current_run_flag, var_threshold)
    """



# TODO: this should be extended also to LayerNorms (what about embeddings??). It should receive instead of a run_flag an instructor that will tell us how to define each layer
def wrap_linear_layers(module, wrapper_cls, wrapper_cls_quantize, optimizer, est_interval, instructions, parent_name = None, **kwargs):
    """
    Recursively replace all nn.Linear layers in `module` with instances of `wrapper_cls`.

    Args:
        module (nn.Module): The model or submodule to modify.
        wrapper_cls (type): A class like TrackedLinear that wraps nn.Linear.
        est_interval: the interval used for creating an estimator
        **kwargs: Extra arguments to pass to the wrapper_cls (like beta).
    """
    # This boolean is used when we want only a single layer to be wrapped or even control whether to have a wrapper at all... 
    single_wrapper = False
    
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        #if 'lm_head' in name:
        #    print('Stop for debug')
        
        # TODO: This does not wrap the LayerNorm which is not nn.LayerNorm but locally defined!
        if (isinstance(child, nn.Linear) or isinstance(child, nn.LayerNorm)) and not single_wrapper:
            bias = child.bias is not None

            # We want to get from a predefined instructor how to initialize the given layer:
            weight_q, input_q, output_q, gradient_q = instructions.get_quantization_config(full_name)
            # We also want to examine how this layer should behave throughout the run. For this we examine whether a mode and layers_of_interest were defined for the layer:  
            if "mode" in kwargs and kwargs["mode"]=="layer_deep_dive" and "layers_of_interest" in kwargs and kwargs["layers_of_interest"] != None:
                wrap_flag=False
                for layer_name in kwargs['layers_of_interest']:
                    if layer_name in full_name:
                        wrap_flag=True
                if wrap_flag==False:
                    kwargs['mode']="nothing"        
            
            wrapped = wrapper_cls(child, optimizer, est_interval, weight_q, input_q, output_q, gradient_q, bias=bias, parent_name=full_name, **kwargs)
            single_wrapper = False
            
            # Replace the linear layer in the parent module
            setattr(module, name, wrapped)
        elif isinstance(child, nn.Dropout) and not single_wrapper:
            
            weight_q, input_q, output_q, gradient_q = instructions.get_quantization_config(full_name)
            #if weight_q !=DEFAULT_Q_VALUE or input_q !=DEFAULT_Q_VALUE or output_q !=DEFAULT_Q_VALUE or gradient_q !=DEFAULT_Q_VALUE:
            #    quantize = True
            #else:
            #    quantize = False
            
            wrapped = wrapper_cls_quantize(child, optimizer, est_interval, weight_q, input_q, output_q, gradient_q, bias=None, **kwargs)
            single_wrapper = False
            
            # Replace the linear layer in the parent module
            setattr(module, name, wrapped)
            
        else:
            # Recursively wrap submodules
            wrap_linear_layers(child, wrapper_cls, wrapper_cls_quantize, optimizer, est_interval, instructions, parent_name=full_name, **kwargs)
            
    return module

# Recursively collect all TrackedLinear layers
def collect_tracked_layers(module, tracked_layers=[], prefix=''):
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, StatWrapper) or isinstance(child, nn.LayerNorm):  # TODO: should we add additional wrappers here??? For example the DropourWrapper???  What about LayerNorm. The nn.LayerNorm is not relevant!!
            tracked_layers.append((full_name, child))
        else:
            collect_tracked_layers(child, tracked_layers, full_name)
    return tracked_layers

# By adding the attribute 'reduced' we plot reduced std and not reduced in two seperate files. 
def plot_std_across_iterations(model, run_prefix, est_interval, now=None, output_dir="stat_output"):
    
    tracked_layers=collect_tracked_layers(model, tracked_layers=[])

    plt.figure(1, figsize=(10, 5))
    plt.figure(2, figsize=(10, 5))
    
    for i, (name, layer) in enumerate(tracked_layers):
        
        #if hasattr(layer, "mode") and layer.mode == "layer_deep_dive": 
        if hasattr(layer, "std_values"): 
            if hasattr(layer, 'reduced'):
                plt.figure(1)
                random_color = np.random.rand(3,)
                data = layer.std_values
                plt.plot(range(0, len(data) * est_interval, est_interval), data, color=random_color, label=name) 
                data = layer.est_std_values
                plt.plot(range(0, len(data) * est_interval, est_interval), data, color=random_color, label=name, linestyle="--") 
            else:
                plt.figure(2)
                random_color = np.random.rand(3,)
                data = layer.std_values
                plt.plot(range(0, len(data) * est_interval, est_interval), data, color=random_color, label=name) 
                data = layer.est_std_values
                plt.plot(range(0, len(data) * est_interval, est_interval), data, color=random_color, label=name, linestyle="--")

    plt.figure(1)
    plt.title(f"Reduced: Std values across iterations: {run_prefix}")
    plt.xlabel("Training Step")
    plt.ylabel("std")
    plt.legend(loc='upper right')
    plt.grid(True)

    filename = f"{output_dir}/{now}_{run_prefix}_reduced_std.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot for all layers variance to {filename}")   
    
    plt.figure(2)
    plt.title(f"Not reduced: Std values across iterations: {run_prefix}")
    plt.xlabel("Training Step")
    plt.ylabel("std")
    plt.legend(loc='upper right')
    plt.grid(True)

    filename = f"{output_dir}/{now}_{run_prefix}_not_reduced_std.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot for all layers variance to {filename}")   
        
# We want to plot the baselines of specific layers. 
# This method will receive a layer and for that layer will print atention variances in one color and MLP variances in another color
def plot_baseline_var_attn_vs_mlp(model, layer_prefix, est_interval, now=None, output_dir="stat_output"):
    
    tracked_layers=collect_tracked_layers(model, tracked_layers=[])
    
    plt.figure(1, figsize=(10, 5))
    plt.figure(2, figsize=(10, 5))
        
    color_attn_c = 'b'
    color_attn_proj = 'r'
    color_mlp_fc = 'g'
    color_mlp_proj = 'm'
    color = 'k'
    
    for i, (name, layer) in enumerate(tracked_layers):
        if layer_prefix in name:
            stats = layer.get_history_2()
            if 'c_attn' in name:
                color = color_attn_c
            if 'attn.c_proj' in name:
                color = color_attn_proj
            if 'mlp.c_fc' in name:
                color = color_mlp_fc
            if 'mlp.c_proj' in name:
                color = color_mlp_proj    
            plt.figure(1)
            plt.plot(range(0, len(stats['direct_estimate_var']) * est_interval, est_interval), stats['direct_estimate_var'], color, label=name) 
            plt.figure(2)
            plt.plot(range(0, len(stats['direct_estimate_expectation']) * est_interval, est_interval), stats['direct_estimate_expectation'], color, label=name) 
            #plt.plot(range(mid_point* est_interval, len(stats['direct_estimate_var']) * est_interval, est_interval), stats['direct_estimate_var'][mid_point:], color = random_color, linestyle="-", label=name) 

    plt.figure(1)
    plt.title(f"Comparison of variance of attn and mlp in layer: {layer_prefix}")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.legend(loc='upper right')
    plt.grid(True)

    filename = f"{output_dir}/{now}_{layer_prefix}_attn_mlp_var_comparison.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot for all layers variance to {filename}")   
    
    plt.figure(2)
    plt.title(f"Comparison of mean of attn and mlp in layer: {layer_prefix}")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.legend(loc='upper right')
    plt.grid(True)

    filename = f"{output_dir}/{now}_{layer_prefix}_attn_mlp_mean_comparison.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot for all layers variance to {filename}")     
    
    

def plot_tracked_linear_stats(model, est_interval, now=None, output_dir="stat_output"):
    """
    Finds all TrackedLinear layers in the model and saves a plot of their tracked statistics.

    Args:
        model (nn.Module): The model containing TrackedLinear layers.
        output_dir (str): Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    if now == None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracked_layers = []

    collect_tracked_layers(model, tracked_layers=[])

    if not tracked_layers:
        print("No TrackedLinear layers found.")
        return

    for i, (name, layer) in enumerate(tracked_layers):
        stats = layer.get_history()

        plt.figure(figsize=(10, 5))
        plt.plot(range(0, len(stats['values']) * est_interval, est_interval), stats['values'], label="Raw Value")
        plt.plot(range(0, len(stats['mean']) * est_interval, est_interval), stats['mean'], label="EMA Mean")
        plt.plot(range(0, len(stats['second_moment']) * est_interval, est_interval), stats['second_moment'], label="EMA 2nd Moment")
        plt.title(f"StatWrapper Statistics: {name}")
        plt.xlabel("Training Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        filename = f"{output_dir}/{now}_{name.replace('.', '_')}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot for '{name}' to {filename}")
        
            
    for i, (name, layer) in enumerate(tracked_layers):
        stats = layer.get_history_2()

        # A figure for each layer for the mean:
        plt.figure(figsize=(10, 5))
        #plt.plot(stats['direct_estimate_var'], label="Direct Var Eastimate") 
        plt.plot(range(0, len(stats['direct_estimate_expectation']) * est_interval, est_interval), stats['direct_estimate_expectation'], label="Direct Expectation Eastimate")
        #plt.plot(stats['adam_estimate'], label="ADAM estimate") 
        #plt.plot(stats['adam_diff_estimate'], label="ADAM diff estimate") 
        #plt.plot(stats['adam_delta_estimate'], label="ADAM delta") 
        #plt.plot(range(0, len(stats['adam_mean_est1']) * est_interval, est_interval), stats['adam_mean_est1'], label="ADAM mean est1")
        plt.plot(range(0, len(stats['adam_mean_est2']) * est_interval, est_interval), stats['adam_mean_est2'], label="ADAM mean est2")
        plt.plot(range(0, len(stats['adam_mean_est3']) * est_interval, est_interval), stats['adam_mean_est3'], label="ADAM mean est3")
        plt.title(f"Mean - Direct Vs. ADAM: {name}")
        plt.xlabel("Training Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        filename = f"{output_dir}/{now}_{name.replace('.', '_')}_estimateMean.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot for '{name}' to {filename}")
        
        # A figure for each layer for the mean:
        plt.figure(figsize=(10, 5))
        #plt.plot(stats['direct_estimate_expectation'], label="Direct Expectation Eastimate")
        #plt.plot(stats['adam_estimate'], label="ADAM estimate") 
        #plt.plot(stats['adam_diff_estimate'], label="ADAM diff estimate") 
        #plt.plot(stats['adam_var_est1'], label="ADAM var est1")
        #plt.plot(range(0, len(stats['adam_var_est2']) * est_interval, est_interval), stats['adam_var_est2'], label="ADAM var est2")
        #plt.plot(range(0, len(stats['adam_var_est3']) * est_interval, est_interval), stats['adam_var_est3'], label="ADAM var est3")
        #plt.plot(range(0, len(stats['adam_var_est4']) * est_interval, est_interval), stats['adam_var_est4'], label="ADAM var est4")
        plt.plot(range(0, len(stats['adam_var_est5']) * est_interval, est_interval), stats['adam_var_est5'], label="ADAM var est5")
        plt.plot(range(0, len(stats['direct_estimate_var']) * est_interval, est_interval), stats['direct_estimate_var'], label="Direct Var Eastimate") 
        #plt.plot(stats['adam_delta_estimate'], label="ADAM delta") 
        plt.title(f"Var - Direct Vs. ADAM: {name}")
        plt.xlabel("Training Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        filename = f"{output_dir}/{now}_{name.replace('.', '_')}_estimateVar.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot for '{name}' to {filename}")
    
    # Creating two graphs of number_elem_in_fig variance estimator in each. The layers are randomly chosen each time to have a mixture. 
    number_elem_in_fig = 5
    indices4fig1 = random.sample(range(len(tracked_layers)),number_elem_in_fig)
    indices4fig2 = random.sample(range(len(tracked_layers)),number_elem_in_fig)
    
    plt.figure(1)
    plt.figure(2)
    for i in indices4fig1:
        
        name, layer = tracked_layers[i]
        stats = layer.get_history_2()
        random_color = np.random.rand(3,)
        plt.figure(1)
        
        #plt.plot(stats['direct_estimate_expectation'], label="Direct Expectation Eastimate")
        #plt.plot(stats['adam_estimate'], label="ADAM estimate") 
        #plt.plot(stats['adam_diff_estimate'], label="ADAM diff estimate") 
        #plt.plot(stats['adam_var_est1'], label="ADAM var est1")
        #plt.plot(range(0, len(stats['adam_var_est2']) * est_interval, est_interval), stats['adam_var_est2'], label="ADAM var est2")
        #plt.plot(range(0, len(stats['adam_var_est3']) * est_interval, est_interval), stats['adam_var_est3'], color = random_color, linestyle="--")
        #plt.plot(range(0, len(stats['adam_var_est4']) * est_interval, est_interval), stats['adam_var_est4'], label="ADAM var est4")
        plt.plot(range(0, len(stats['adam_var_est5']) * est_interval, est_interval), stats['adam_var_est5'], color = random_color, linestyle="-.")
        plt.plot(range(0, len(stats['direct_estimate_var']) * est_interval, est_interval), stats['direct_estimate_var'], color = random_color, linestyle="-", label=name) 
           
        plt.figure(2)   
        
        if len(stats['adam_var_est5']) > 0:
            mid_point = int(len(stats['adam_var_est5']) / 2)
              
            #plt.plot(stats['direct_estimate_expectation'], label="Direct Expectation Eastimate")
            #plt.plot(stats['adam_estimate'], label="ADAM estimate") 
            #plt.plot(stats['adam_diff_estimate'], label="ADAM diff estimate") 
            #plt.plot(stats['adam_var_est1'], label="ADAM var est1")
            #plt.plot(range(0, len(stats['adam_var_est2']) * est_interval, est_interval), stats['adam_var_est2'], label="ADAM var est2")
            #plt.plot(range(mid_point* est_interval, len(stats['adam_var_est3']) * est_interval, est_interval), stats['adam_var_est3'][mid_point:], color = random_color, linestyle="--")
            #plt.plot(range(0, len(stats['adam_var_est4']) * est_interval, est_interval), stats['adam_var_est4'], label="ADAM var est4")
            plt.plot(range(mid_point* est_interval, len(stats['adam_var_est5']) * est_interval, est_interval), stats['adam_var_est5'][mid_point:], color = random_color, linestyle="--")
            plt.plot(range(mid_point* est_interval, len(stats['direct_estimate_var']) * est_interval, est_interval), stats['direct_estimate_var'][mid_point:], color = random_color, linestyle="-", label=name) 
           
    # Finilizing figure 1   
    plt.figure(1)
    plt.title(f"Var - Direct Vs. ADAM: five layers")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    filename = f"{output_dir}/{now}_1_allLayers_estimateVar.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot for all layers variance to {filename}")    
    
    # Finilizing figure 2   
    plt.figure(2)
    plt.title(f"Var - Direct Vs. ADAM: five layers (second half)")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    filename = f"{output_dir}/{now}_2_allLayers_estimateVar.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot for all layers variance to {filename}")       
     
    plt.figure(3)
    plt.figure(4)
    for i in indices4fig2:
        
        name, layer = tracked_layers[i]
        stats = layer.get_history_2()
        random_color = np.random.rand(3,)
        plt.figure(3)
        
        #plt.plot(stats['direct_estimate_expectation'], label="Direct Expectation Eastimate")
        #plt.plot(stats['adam_estimate'], label="ADAM estimate") 
        #plt.plot(stats['adam_diff_estimate'], label="ADAM diff estimate") 
        #plt.plot(stats['adam_var_est1'], label="ADAM var est1")
        #plt.plot(range(0, len(stats['adam_var_est2']) * est_interval, est_interval), stats['adam_var_est2'], label="ADAM var est2")
        #plt.plot(range(0, len(stats['adam_var_est3']) * est_interval, est_interval), stats['adam_var_est3'], color = random_color, linestyle="--")
        #plt.plot(range(0, len(stats['adam_var_est4']) * est_interval, est_interval), stats['adam_var_est4'], label="ADAM var est4")
        plt.plot(range(0, len(stats['adam_var_est5']) * est_interval, est_interval), stats['adam_var_est5'], color = random_color, linestyle="-.")
        plt.plot(range(0, len(stats['direct_estimate_var']) * est_interval, est_interval), stats['direct_estimate_var'], color = random_color, linestyle="-", label=name) 
        
        plt.figure(4)
        if len(stats['adam_var_est5']) > 0:
            mid_point = int(len(stats['adam_var_est5']) / 2)
            
            #plt.plot(stats['direct_estimate_expectation'], label="Direct Expectation Eastimate")
            #plt.plot(stats['adam_estimate'], label="ADAM estimate") 
            #plt.plot(stats['adam_diff_estimate'], label="ADAM diff estimate") 
            #plt.plot(stats['adam_var_est1'], label="ADAM var est1")
            #plt.plot(range(0, len(stats['adam_var_est2']) * est_interval, est_interval), stats['adam_var_est2'], label="ADAM var est2")
            plt.plot(range(mid_point* est_interval, len(stats['adam_var_est3']) * est_interval, est_interval), stats['adam_var_est3'][mid_point:], color = random_color, linestyle="--")
            #plt.plot(range(0, len(stats['adam_var_est4']) * est_interval, est_interval), stats['adam_var_est4'], label="ADAM var est4")
            plt.plot(range(mid_point* est_interval, len(stats['adam_var_est5']) * est_interval, est_interval), stats['adam_var_est5'][mid_point:], color = random_color, linestyle="--")
            plt.plot(range(mid_point* est_interval, len(stats['direct_estimate_var']) * est_interval, est_interval), stats['direct_estimate_var'][mid_point:], color = random_color, linestyle="-", label=name) 
           
    plt.figure(3)
    plt.title(f"Var - Direct Vs. ADAM: five layers")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    filename = f"{output_dir}/{now}_3_allLayers_estimateVar.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot for all layers variance to {filename}")
    
    plt.figure(4)
    plt.title(f"Var - Direct Vs. ADAM: five layers (zoom)")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    filename = f"{output_dir}/{now}_4_allLayers_estimateVar.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot for all layers variance to {filename}")