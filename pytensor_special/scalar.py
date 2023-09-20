import pytensor
import scipy
from pytensor.scalar.basic import BinaryScalarOp, ScalarOp
from pytensor.scalar.basic import (
    upgrade_to_float_no_complex, float_types
)

class GammaIncInv(BinaryScalarOp):
    """
    Inverse of regularized lower incomplete gamma function
    """
    nin = 2
    nfunc_spec = ('scipy.special.gammaincinv', 2, 1)

    def impl(self, a, z):
        return scipy.special.gammaincinv(a, z)
    
    def grad(self, inp, grads):
        #raise NotImplementedError()
        a, z = inp
        (gz, ) = grads
        
        x = GammaIncInv(upgrade_to_float_no_complex, name='gammaincinv_wrtx_nograd')(a, z)

        dx_dz = pytensor.scalar.math.gamma(a) / pytensor.scalar.basic.pow(x, a-1) * pytensor.scalar.basic.exp(-x)
        dz_da = pytensor.scalar.math.gammainc_grad(a, z)
        dx_da = dx_dz * dz_da

        return [
            gz * dx_da,
            gz * dx_dz
        ]
    
    def c_code(self, node, name, inputs, outputs, sub):
        raise NotImplementedError()
        # (a, z,) = inputs
        # (x,) = outputs
        # if node.inputs[0].type in float_types:
        #     return """
                

        #     """ % (locals())
        # raise NotImplementedError("only floating point is implemented")
    
gammaincinv = GammaIncInv(upgrade_to_float_no_complex, name='gammaincinv')

