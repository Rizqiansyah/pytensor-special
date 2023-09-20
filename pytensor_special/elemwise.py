from pytensor.tensor.elemwise import Elemwise
from pytensor.scalar.basic import transfer_type

def scalar_elemwise(*symbol, nfunc=None, nin=None, nout=None, symbolname=None):
    """Replace a symbol definition with an `Elemwise`-wrapped version of the corresponding scalar `Op`.

    If it is not ``None``, the `nfunc` argument should be a string such that
    ``getattr(numpy, nfunc)`` implements a vectorized version of the `Elemwise`
    operation.  `nin` is the number of inputs expected by that function, and nout
    is the number of **destination** inputs it takes.  That is, the function
    should take nin + nout inputs. `nout == 0` means that the numpy function does
    not take a NumPy array argument to put its result in.

    """
    from . import scalar as scalar

    def construct(symbol):
        nonlocal symbolname

        symbolname = symbolname or symbol.__name__

        if symbolname.endswith("_inplace"):
            base_symbol_name = symbolname[: -len("_inplace")]
            scalar_op = getattr(scalar, base_symbol_name)
            inplace_scalar_op = scalar_op.__class__(transfer_type(0))
            rval = Elemwise(
                inplace_scalar_op,
                {0: 0},
                nfunc_spec=(nfunc and (nfunc, nin, nout)),
            )
        else:
            scalar_op = getattr(scalar, symbolname)
            rval = Elemwise(scalar_op, nfunc_spec=(nfunc and (nfunc, nin, nout)))

        if getattr(symbol, "__doc__"):
            rval.__doc__ = symbol.__doc__ + "\n\n    " + rval.__doc__

        # for the meaning of this see the ./epydoc script
        # it makes epydoc display rval as if it were a function, not an object
        rval.__epydoc_asRoutine = symbol
        rval.__module__ = symbol.__module__

        return rval

    if symbol:
        return construct(symbol[0])
    else:
        return construct