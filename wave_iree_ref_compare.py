import torch
import math
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_zeros,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
import numpy as np

def main():
    # Load inputs
    q = torch.from_numpy(np.load("data/fp8/q.npy")).to("cuda", dtype=torch.float16)
    k = torch.from_numpy(np.load("data/fp8/k.npy")).to("cuda", dtype=torch.float16)
    v = torch.from_numpy(np.load("data/fp8/v.npy")).to("cuda", dtype=torch.float16)
    ref = torch.from_numpy(np.load("data/fp8/o.npy")).to("cuda", dtype=torch.float32)

    # Load quantization scaling
    q_scale = float(np.load("data/fp8/qscale.npy"))
    k_scale = float(np.load("data/fp8/kscale.npy"))
    v_scale = float(np.load("data/fp8/vscale.npy"))

    # Order of shapes: (B, M, N, K1, K2)
    # mfma_variant = (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8)
    mfma_variant = (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8)
    input_shape = (1, 4096, 64, 64, 4096)
    shape = AttentionShape(
        num_query_heads=input_shape[0],
        num_kv_heads=input_shape[0],
        query_seq_len=input_shape[1],
        head_size_kv=input_shape[2],
        head_size=input_shape[3],
        kv_seq_len=input_shape[4],
    )
    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_brevitas_pertensor_fp8_attention_kernel(shape, mfma_variant, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)
    o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )

    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention)

    torch.manual_seed(0)
    wave_output = device_zeros(o_shape, dtype=torch.float32)
    # TODO: Add scaling of QK as part of kernel.
    base_attention(q, k, v, wave_output)
    rmse = torch.sqrt(torch.mean(torch.square(ref - wave_output)))
    assert rmse < 0.003
    print("PASS")

    # res.npy is generated from `main_compile.sh attention_fp8.mlir && run.sh`
    iree_output = torch.from_numpy(np.load("res.npy")).to(wave_output.device)

    print("ABSMAX(IREE, Wave):", torch.max(torch.abs(iree_output - wave_output)))
    print("ABSMAX(IREE, Ref):", torch.max(torch.abs(iree_output - ref)))
    print("ABSMAX(Wave, Ref):", torch.max(torch.abs(wave_output - ref)))
    print("\n\n")
    print("RMSE(IREE, Wave):", torch.sqrt(torch.mean(torch.square(iree_output - wave_output))))
    print("RMSE(IREE, Ref):", torch.sqrt(torch.mean(torch.square(iree_output - ref))))
    print("RMSE(Wave, Ref):", torch.sqrt(torch.mean(torch.square(wave_output - ref))))

if __name__ == "__main__":
    main()
