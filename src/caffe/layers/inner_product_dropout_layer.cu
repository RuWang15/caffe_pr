#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) { 
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) { 
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    //LOG(INFO) << "IPD: Forward_gpu start.";
    // pseudo-random dropout
    T_ = rand_seq_[seq_idx_];
    seq_idx_ = (seq_idx_ + 1)%rand_seq_.size();
    if (T_ == 1) {// no optimization
        caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    } else {
      start_ = rand() % T_;
      //LOG(INFO) << "IPD: T=" << T_ << ", seq_idx=" << seq_idx_ << ", start=" << start_;
      //LOG(INFO) << "Dropout: K=" << K_ << ", K'= " << 1+(K_-start_-1)/T_;
      caffe_gpu_gemm_ipdf<Dtype>(N_, M_, K_, weight, bottom_data, top_data, transpose_, T_, start_);
      //LOG(INFO) << "IPD: Forward_gpu finish.";
    }
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //LOG(INFO) << "IPD: Backward_gpu start.";
  //LOG(INFO) << "T=" << T_ << ", start=" << start_;
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (T_ != 1) {
      caffe_gpu_gemm_ipdb_w<Dtype>(N_, M_, K_, top_diff, bottom_data, 
        this->blobs_[0]->mutable_gpu_diff(), transpose_, T_, start_);
    } else {
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            K_, N_, M_,
            (Dtype)1., bottom_data, top_diff,
            (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
      } else {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
            N_, K_, M_,
            (Dtype)1., top_diff, bottom_data,
            (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
      }
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (T_ != 1) {
      caffe_gpu_gemm_ipdb_bd<Dtype>(N_, M_, K_, top_diff, this->blobs_[0]->gpu_data(), 
        bottom[0]->mutable_gpu_diff(), transpose_, T_, start_);
    } else {
      if (transpose_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
            M_, K_, N_,
            (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
            (Dtype)0., bottom[0]->mutable_gpu_diff());

      } else {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
            M_, K_, N_,
           (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
           (Dtype)0., bottom[0]->mutable_gpu_diff());
      }
    }
  }
  //LOG(INFO) << "IPD: Backward_gpu finish.";
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductDropoutLayer);

}  // namespace caffe

