
#include "deform_psroi_pooling.h"
#include "deform_conv.h"
#include "modulated_deform_conv.h"
#include "depth_deform_conv.h"
#include "depthaware_conv.h"
#include "sample_conv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward", &deform_conv_backward, "deform_conv_backward");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");
  m.def("depth_deform_conv_forward", &depth_deform_conv_forward, "depth_deform_conv_forward");
  m.def("depth_deform_conv_backward", &depth_deform_conv_backward, "depth_deform_conv_backward");
  m.def("deform_psroi_pooling_forward", &deform_psroi_pooling_forward, "deform_psroi_pooling_forward");
  m.def("deform_psroi_pooling_backward", &deform_psroi_pooling_backward, "deform_psroi_pooling_backward");
  m.def("depthaware_conv_forward", &depthaware_conv_forward, "depthaware_conv_forward");
  m.def("depthaware_conv_backward", &depthaware_conv_backward, "depthaware_conv_backward");
  m.def("sample_conv_forward", &sample_conv_forward, "sample_conv_forward");
  m.def("sample_conv_backward", &sample_conv_backward, "sample_conv_backward");
}
