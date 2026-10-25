# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``quant_method`` name of HI2 checkpoints, in one place.

The final name is not settled (the format carries INT2, INT4 and INT8 rows).
Change ``HI2_QUANT_METHOD`` to rename it; keep every earlier name in
``HI2_QUANT_METHOD_ALIASES`` so checkpoints written with it keep loading.
No imports here: the quantization registry imports this module at startup.
"""

HI2_QUANT_METHOD = "hadamard_int2"
HI2_QUANT_METHOD_ALIASES: tuple[str, ...] = ()
HI2_QUANT_METHOD_NAMES: tuple[str, ...] = (HI2_QUANT_METHOD, *HI2_QUANT_METHOD_ALIASES)
