
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func private @scale(%arg0 : tensor<1x4096x64xf32>, %arg1 : f32) -> tensor<1x4096x64xf32> {
    %empty = tensor.empty() : tensor<1x4096x64xf32>
    %generic = linalg.generic  {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel", "parallel"]}      
      ins(%arg0 : tensor<1x4096x64xf32>)
      outs(%empty : tensor<1x4096x64xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.mulf %b0, %arg1 : f32
        linalg.yield %0 : f32
      } -> tensor<1x4096x64xf32>

    return %generic : tensor<1x4096x64xf32>
}

func.func @main(
    %q : tensor<1x4096x64xf32>,
    %k : tensor<1x4096x64xf32>,
    %v : tensor<1x4096x64xf32>,
    %scale : tensor<f32>,
    %qscale : tensor<f32>,
    %kscale : tensor<f32>,
    %vscale : tensor<f32>) -> tensor<1x4096x64xf32> {

    %scalef32 = tensor.extract %scale[] : tensor<f32>
    %qscalef32 = tensor.extract %qscale[] : tensor<f32>
    %kscalef32 = tensor.extract %kscale[] : tensor<f32>
    %vscalef32 = tensor.extract %vscale[] : tensor<f32>

    %qf8 = arith.truncf %q : tensor<1x4096x64xf32> to tensor<1x4096x64xf16>
    %kf8 = arith.truncf %k : tensor<1x4096x64xf32> to tensor<1x4096x64xf16>
    %vf8 = arith.truncf %v : tensor<1x4096x64xf32> to tensor<1x4096x64xf16>

    %qk = arith.mulf %qscalef32, %kscalef32 : f32
    %qks = arith.mulf %qk, %scalef32 : f32

    %empty = tensor.empty() : tensor<1x4096x64xf16>
    %c0 = arith.constant 0.0 : f16
    %fill = linalg.fill ins(%c0 : f16) outs(%empty : tensor<1x4096x64xf16>)  -> tensor<1x4096x64xf16>
    %atten = iree_linalg_ext.attention ins(%qf8, %kf8, %vf8, %qks : tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f32) outs(%fill : tensor<1x4096x64xf16>) -> tensor<1x4096x64xf16>

    %attenf32 = arith.extf %atten : tensor<1x4096x64xf16> to tensor<1x4096x64xf32>

    %atten_scale = call @scale(%attenf32, %vscalef32) : (tensor<1x4096x64xf32>, f32) -> tensor<1x4096x64xf32>

    return %atten_scale : tensor<1x4096x64xf32>
}