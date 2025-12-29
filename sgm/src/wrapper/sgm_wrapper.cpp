// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include<pybind11/numpy.h>
#include <thread>

#include "sgm_gpu/sgm_gpu.h"

namespace py = pybind11;
using namespace sgm_gpu;

cv::Mat numpyToMat_Gray(py::array_t<unsigned char>& img){
	if (img.ndim() != 2)
		throw std::runtime_error("1-channel image must be 2 dims ");
 
	py::buffer_info buf = img.request();
	cv::Mat mat(static_cast<int>(buf.shape[0]), static_cast<int>(buf.shape[1]),
		CV_8UC1, (unsigned char*)buf.ptr);
 
	return mat;
}

PYBIND11_MODULE(sgm_wrapper, m) {
  py::class_<SgmGpu, std::shared_ptr<SgmGpu>>(m, "SgmGpu")
    .def(py::init<const unsigned short, const unsigned short>())
    .def("computeDisparity", [](SgmGpu& sgm, py::array_t<unsigned char> left_py, py::array_t<unsigned char> right_py) {
      cv::Mat left_cv, right_cv;
      left_cv = numpyToMat_Gray(left_py);
      right_cv = numpyToMat_Gray(right_py);
      cv::Mat disp = cv::Mat(left_cv.rows, left_cv.cols, CV_8UC1);
      // show the image
      // cv::imshow("left", left_cv);
      // cv::imshow("right", right_cv);
      sgm.computeDisparity(left_cv, right_cv, &disp);
      // cv::imshow("disp", disp);
      // cv::waitKey(1);
      // disp.convertTo(disp, CV_32FC1);
      py::array_t<unsigned char> disp_py = py::array_t<unsigned char>({ left_cv.rows,left_cv.cols}, disp.data);
      return disp_py;
    });

}
