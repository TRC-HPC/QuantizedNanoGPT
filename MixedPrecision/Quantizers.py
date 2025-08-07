import torch
import abc


QUANTIZATION_NECESSITY_THRESHOLD = 16

        
# We generalized this method to work with any quantizer    
def quantize_block(block, quantizer):
    if hasattr(block, "weight"):
        with torch.no_grad():
            quantizer.find_params(block.weight.data)      
        block.weight = torch.nn.Parameter(quantizer(block.weight))
        quantizer.free()
    else:
        for child in block.children():
            quantize_block(child, quantizer=quantizer)
    return block


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantize_fn):
        return quantize_fn(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None

class IntQuantizations:
    class SymSTEQuantize(STEQuantize):
        @staticmethod
        def forward(ctx, x, scale, maxq):
            scale = scale.to(x.device)
            q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
            return scale * q

    class AsymSTEQuantize(STEQuantize):
        @staticmethod
        def forward(ctx, x, scale, zero, maxq):
            scale = scale.to(x.device)
            zero = zero.to(x.device)
            q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
            return scale * (q - zero)


BITS_TO_DTYPE_MAP = {
    32: torch.float32,
    16: torch.float16,
    8:  torch.float8_e4m3fnuz
}

class FloatQuantizer(torch.nn.Module):
    
    def __init__(self, precision) -> None:
        super(FloatQuantizer, self).__init__()
        self.quant_dtype = BITS_TO_DTYPE_MAP[precision]

    def forward(self, x):
        return STEQuantize.apply(x, self.quantize)
        
    def quantize(self, x):
        return x.to(self.quant_dtype).to(x.dtype)
        
    def free(self) -> None:
        pass
    
    def find_params(self, x) -> None:
        pass

class IntActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self, precision) -> None:
        super(IntActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = precision

    def free(self) -> None:
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == torch.finfo(x.dtype).bits:
            return x
        elif self.sym:
            return IntQuantizations.SymSTEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return IntQuantizations.AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)
       
    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0
    ) -> None:
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(self, x) -> None:
        init_shape = x.shape
        #if x.shape[-2] == 315:
        #    print('Stop for debug')
        dividing = (x.shape[0]*x.shape[2]) / self.groupsize
        if dividing % (x.shape[-1] // self.groupsize) == 0: 
            reshaped_x = x.reshape(
                -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
            )
        else:
            reshaped_x = x.reshape(
                1, x.shape[-2], int(dividing), self.groupsize
            )

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x) -> None:
        if self.bits >= QUANTIZATION_NECESSITY_THRESHOLD:  # Quantization unnecessary
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        # print the name of the layer
        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = (
                self.scale.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )
            self.zero = (
                self.zero.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )

class IntWeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(IntWeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        weight_groupsize: int = -1,
    ) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params_weight_groupwise(self, x) -> None:
        init_shape = x.shape
        x = x.reshape(
            x.shape[-2], x.shape[-1] // self.weight_groupsize, self.weight_groupsize
        )

        xmax = torch.amax(x, dim=-1, keepdim=True)
        xmin = torch.amin(x, dim=-1, keepdim=True)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, self.weight_groupsize)
        self.zero = self.zero.repeat(1, 1, self.weight_groupsize)

        if self.mse:
            best = torch.full(
                [x.shape[0], x.shape[1]], float("inf"), device=x.device
            ).type_as(x)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = sym_quant_dequant(x, scale1, self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = asym_quant_dequant(x, scale1, zero1, self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, -1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        self.scale = self.scale.reshape(init_shape)
        self.zero = self.zero.reshape(init_shape)

    def find_params(self, x) -> None:
        if self.bits >= QUANTIZATION_NECESSITY_THRESHOLD:  # Quantization unnecessary
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape

        if self.weight_groupsize > 0:
            # group-wise per-token quantization
            self.find_params_weight_groupwise(x)
            # utils.cleanup_memory(verbos=False)
            return
        elif self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def forward(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < QUANTIZATION_NECESSITY_THRESHOLD:
            if self.sym:
                return IntQuantizations.SymSTEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            return IntQuantizations.AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
                x_dtype
            )
        return x

    # Return int value and scale in addtional to fake quantized weight
    def fake_quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < QUANTIZATION_NECESSITY_THRESHOLD:
            scale = self.scale.to(x.device)
            q = torch.clamp(torch.round(x / scale), -(self.maxq + 1), self.maxq)
            return (scale * q).to(x_dtype), q, scale
        else:
            return None, None, None

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

    def free(self):
        pass
