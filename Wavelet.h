/*
* *
* *Filename: Wavelet.h
* *
* *Description:
* * This file contains the implementation of the WaveletOps class, which provides
* * various wavelet-based operations for signal processing, including multi-level
* * discrete wavelet transform (DWT) and continuous wavelet transform (CWT) with
* * thresholding for denoising, as well as a method for splitting a signal into transient
* *  and tonal components. The class supports multiple wavelet types and thresholding methods.
* *
* *
* * Author:
* *  JEP, J.Enrique Peraza
* *
* *
*/
#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <complex>
#include <thread>
#include <future>
#include <chrono>
#include <stdexcept>
#include <optional>
#include "DSPWindows.h"

namespace sig::spectral
{
  using namespace std;
 
// Class template for spectral operations
template<typename T>
class WaveletOps
{
  public:
  // The wavelets we know.
  enum class WaveletType
  {
    Haar,                            // The Haar Wavelet type.
    Db1,                             // Daubechies wavelet type 1.
    Db6,                             // Daubechies wavelet type 6.
    Sym5,                            // Symlet type 5
    Sym8,                            // Symlet type 8.
    Coif5,                           // Coiflet type 5
    Morlet,                          // Morlet wavelet.
    MexicanHat,                      // Mexican Hat wavelet.
    Meyer,                           // Meyer wavelet.
    Gaussian                         // Gaussian wavelet. 
  };
  enum class ThresholdType
  {
    Hard,                           // Hard thresholding.
    Soft                            // Soft thresholding.
  };
  public:
    // Constructors
  explicit WaveletOps(          // Constructor
      WaveletType wt=WaveletType::Haar, // Our default wavelet.
      size_t levels=1,            // Decomposition level.
      double threshold=0.0f,      // Denoising threshold.
      ThresholdType tt=ThresholdType::Hard)// The denoising type.
   : wType{wt},tType{tt},levels{levels},threshold{threshold} {}
    // ---------------------------- //
    // Denoise: Apply multi-level DWT denoising and reconstruct signal.
    // ---------------------------- //
    std::vector<double> Denoise(const std::vector<double>& signal)
    {                               // --------- Denoise --------------- //
      // -------------------------- //
      // 1. Zero-Pad the signal to make sure the operation is possible.
      // -------------------------- //   
      auto padded=pad_to_pow2(signal);// Zero-pad the signal.
      // -------------------------- //
      // 2. Now forward DWT+threshold.
      // -------------------------- //
      auto coeffs=dwt_multilevel(padded,selectForward(),
        this->levels,this->threshold,tTypeToString());
      // -------------------------- //
      // 3. Now perform the inverse DWT.
      // ------------------------- //
      auto recon=idwt_multilevel(coeffs,selectInverse());
      // ------------------------- //
      // 4. Remove the zero padding from the signal.
      // ------------------------- //
      return remove_padding(recon,signal.size());
    }                              // ---------- Denoise ----------- 
    // ---------- SplitTransientTonal: wavelet-based decomposition into transient/tonal layers ---------- //
// We run a multi-level DWT. The sum of details across levels behaves like "transient/noisy"
// content (fast changes), while the final approximation behaves like "tonal/sustained".
// You can process each independently (e.g., saturate transients, chorus tonals) and mix back.
inline pair<vector<double>,vector<double>> SplitTransientTonal(
  const vector<double>& signal_in)
{
  vector<double> x=signal_in;
  auto coeffs=dwt_multilevel(x,selectForward(),this->levels,this->threshold,tTypeToString());
  // Reconstruct tonal from approximation only
  vector<pair<vector<double>,vector<double>>> tonal_coeffs=coeffs;
  for(auto& p:tonal_coeffs)p.second=vector<double>(p.second.size(),0.0);
  vector<double> tonal=idwt_multilevel(tonal_coeffs,selectInverse());
  // Transient layer is residual detail sum
  vector<double> trans(signal_in.size(),0.0);
  for(size_t i=0;i<signal_in.size();++i)trans[i]=signal_in[i]-tonal[i];
  return {trans,tonal};
}
  inline void setMorletCentralFrequency(double w0){ morlet_w0=w0; }
  inline void setMexicanHatWidth(double a){ mexhat_a=a; }
  inline void setGaussianSlope(double a){ gaussian_a=a; }
public:
  inline vector<double> pad_to_pow2(const vector<double>& signal) 
  {
    size_t original_length=signal.size();
    size_t padded_length=static_cast<size_t>(next_power_of_2(original_length));
    vector<double> padded_signal(padded_length);

    copy(signal.begin(), signal.end(), padded_signal.begin());
    fill(padded_signal.begin()+original_length, padded_signal.end(), 0);

    return padded_signal;
  }                                                     
/// Remove padding back to the original length
  inline vector<double> remove_padding(const vector<double>& signal, size_t original_length) 
  {
    return vector<double>(signal.begin(), signal.begin()+original_length);
  }
  
// Normalization.
inline vector<double> normalize_minmax(const vector<double>& data) 
{
    double min_val=*min_element(data.begin(), data.end());
    double max_val=*max_element(data.begin(), data.end());
    vector<double> normalized_data(data.size());

    transform(data.begin(), data.end(), normalized_data.begin(),
        [min_val, max_val](double x) { return (x-min_val) / (max_val-min_val); });

    return normalized_data;
}

inline vector<double> normalize_zscore(const vector<double>& data) 
{
    double mean_val=accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sq_sum=inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double std_val=sqrt(sq_sum / data.size()-mean_val*mean_val);
    vector<double> normalized_data(data.size());

    transform(data.begin(), data.end(), normalized_data.begin(),
        [mean_val, std_val](double x) { return (x-mean_val) / std_val; });

    return normalized_data;
}

inline vector<double> awgn(const vector<double>& signal, double desired_snr_db) 
{
    double signal_power=accumulate(signal.begin(), signal.end(), 0.0,
        [](double sum, double val) { return sum+val*val; }) / signal.size();
    double desired_noise_power=signal_power / pow(10, desired_snr_db / 10);
    vector<double> noise(signal.size());

    generate(noise.begin(), noise.end(), [desired_noise_power]() {
        return sqrt(desired_noise_power)*((double)rand() / RAND_MAX*2.0-1.0);
    });

    vector<double> noisy_signal(signal.size());
    transform(signal.begin(), signal.end(), noise.begin(), noisy_signal.begin(), plus<double>());

    return noisy_signal;
}
/// Multi-level discrete wavelet transform with hard/soft thresholding
inline vector<pair<vector<double>, vector<double>>> 
  dwt_multilevel (
    vector<double>& signal, 
    function<pair<vector<double>,
    vector<double>>(const vector<double>&)> wavelet_func, 
    size_t levels,
    double threshold,
    const string& threshold_type="hard") 
    {

      size_t n=signal.size();
      size_t n_pad=static_cast<size_t>(next_power_of_2(n));
      if (n_pad != n)
        signal.resize(n_pad, 0.0);
      vector<pair<vector<double>, vector<double>>> coeffs;
      vector<double> current_signal=signal;
      for (size_t i=0; i < levels; ++i) 
      {
         auto [approx, detail]=wavelet_func(current_signal);
        // Apply thresholding to the detail coefficients
        if (threshold_type== "hard") 
          detail=hard_threshold(detail, threshold);
        else if (threshold_type== "soft")
          detail=soft_threshold(detail, threshold);
        coeffs.emplace_back(approx, detail);
        current_signal=approx;

        if (current_signal.size() < 2)
          break;
    }
    return coeffs;
}                                                                                                                         
/// Inverse multi-level DWT
inline vector<double> 
idwt_multilevel(vector<pair<vector<double>, vector<double>>>& coeffs, function<vector<double>(const vector<double>&, const vector<double>&)> wavelet_func) 
{                                       // ~~~~~~~~~~~ idwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~
    vector<double> signal=coeffs[0].first;
    for (int i=coeffs.size()-1;i>=0;--i)// For each level in decomposition algorithm... 
    {                                   // Get the produced approx (LP response) and detail (HP response)...
      auto [approx,detail]=coeffs[i];   // The output of the wavelet multilevel Filter bank.
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Recompose the original signal, now that it has been compressed
      // and maybe denoised...
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      signal=wavelet_func(approx,detail);// Perfect Reconstruction (PR) Filter Bank output.
    }                                   // Done reconstructing signal from approx and details.
    return signal;                      // Return Perfectly Reconstructed signal.
}                                       // ~~~~~~~~~~~ idwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~


/// Forward wavelet mother waveforms
inline pair<vector<double>, vector<double>> haar (
  const vector<double>& signal) const
{                                       // ~~~~~~~~~~~~~~~~~ Haar ~~~~~~~~~~~~~~~~~~~~~~~~
    const vector<double> h={ 1.0/sqrt(2.0),1.0/sqrt(2.0)};// The LP coeffs of the Haar wavelet PR Filter Bank
    const vector<double> g={1.0/sqrt(2.0),-1.0 / sqrt(2.0)};// The HP coeffs of the Haar wavelet PR Filter Bank
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Our integration limit to decompose the signal into two complementary parts
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    size_t n=signal.size()/2;           // Integration limit (see Wavelet equation).
    vector<double> approx(n),detail(n); // Where to store LP & HP output samples.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Perform the convolution for each pair of input samples
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    for (size_t i=0;i<n;++i)            // For half the length of the signal....
    {                                   // Convolute with the filter bank.
        approx[i]=h[0]*signal[2*i]+h[1]*signal[2*i+1];// LP
        detail[i]=g[0]*signal[2*i]+g[1]*signal[2*i+1];// HP
    }                                   // Done producing approx and detail coeffs.
    return make_pair(approx,detail);    // Return both produced signals.
}                                       // ~~~~~~~~~~~~~~~~~ Haar ~~~~~~~~~~~~~~~~~~~~~~~~
// Daubechies' Mother Wavelet Type 1
inline pair<vector<double>, vector<double>> db1(const vector<double>& signal) const
{                                        // ~~~~~~~~~~~~~~ Db1 ~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Daubechies' Mother Wavelet Type 1 filters H is LP and G is HP
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    const vector<double> h={(1.0+sqrt(3.0))/4.0,(3.0+sqrt(3.0))/4.0,(3.0-sqrt(3.0))/4.0,(1.0-sqrt(3.0))/4.0};
    const vector<double> g={h[3],-h[2],h[1],-h[0]};
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Our integration limit to decompose the signal into two complementary parts
    // We actually downsample in the Forward wavelet algorithm, and then
    // upsample in the Inverse wavelet algorithm.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    size_t n=signal.size()/2;           // Integration limit (see Wavelet equation).
    vector<double> approx(n),detail(n); // Resize our output buffers.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // Perform the convolution for each pair of input samples
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //    
    for (size_t i=0;i<n;++i)            // For the integration limit.... 
    {                                   // Apply the Daubechies PR Type 1 filters
      approx[i]=0.0;                    // Init Lowpass accumulator to zero
      detail[i]=0.0;                    // Init Highpass accumulator to zero
      for (size_t k=0;k<h.size();++k)   // For the number of coefficients in the filters...
      {                                 // Wavelet decompose....
        size_t index=2*i+k-h.size()/2+1;// The index for the current sample
        if (index<signal.size())        // Is our index within the integration limit?
        {                               // Yes
          approx[i]+=h[k]*signal[index];// Convolute and accumulate with LP
          detail[i]+=g[k]*signal[index];// Convolute and accumulate with HP
        }                               // Done applying forward wavelet filters once
      }                                 // Done applying traversing through filter coefficients
    }                                   // Done applying forward wavelet algorithm
    return make_pair(approx, detail);   // Return both produced signals
}                                       // ~~~~~~~~~~~~~~ Db1 ~~~~~~~~~~~~~~~~~~~~~~~~~~
inline pair<vector<double>, vector<double>> db6(const vector<double>& signal) const
{
    const vector<double> h={
        -0.001077301085308,
        0.0047772575109455,
        0.0005538422011614,
        -0.031582039318486,
        0.027522865530305,
        0.097501605587322,
        -0.129766867567262,
        -0.226264693965440,
        0.315250351709198,
        0.751133908021095,
        0.494623890398453,
        0.111540743350109
    };
    const vector<double> g={
        h[11], -h[10], h[9], -h[8], h[7], -h[6], h[5], -h[4], h[3], -h[2], h[1], -h[0]
    };

    size_t n=signal.size() / 2;
    vector<double> approx(n), detail(n);

    for (size_t i=0; i < n; ++i) {
        approx[i]=0;
        detail[i]=0;
        for (size_t k=0; k < h.size(); ++k) {
            size_t index=2*i+k-h.size() / 2+1;
            if (index < signal.size()) {
                approx[i] += h[k]*signal[index];
                detail[i] += g[k]*signal[index];
            }
        }
    }

    return make_pair(approx, detail);
}                                     
inline pair<vector<double>, vector<double>> sym5(const vector<double>& signal) const
{
    const vector<double> h={
        0.027333068345078, 0.029519490925774, -0.039134249302383,
        0.199397533977394, 0.723407690402421, 0.633978963458212,
        0.016602105764522, -0.175328089908450, -0.021101834024759,
        0.019538882735287
    };
    const vector<double> g={
        h[9], -h[8], h[7], -h[6], h[5], -h[4], h[3], -h[2], h[1], -h[0]
    };

    size_t n=signal.size() / 2;
    vector<double> approx(n), detail(n);

    for (size_t i=0; i < n; ++i) {
        approx[i]=0;
        detail[i]=0;
        for (size_t k=0; k < h.size(); ++k) {
            size_t index=2*i+k-h.size() / 2+1;
            if (index < signal.size()) {
                approx[i] += h[k]*signal[index];
                detail[i] += g[k]*signal[index];
            }
        }
    }

    return make_pair(approx, detail);
}
                                    
inline pair<vector<double>, vector<double>> sym8(const vector<double>& signal) const
{
    const vector<double> h={
        -0.003382415951359, -0.000542132331635, 0.031695087811492,
        0.007607487325284, -0.143294238350809, -0.061273359067908,
        0.481359651258372, 0.777185751700523, 0.364441894835331,
        -0.051945838107709, -0.027219029917056, 0.049137179673476,
        0.003808752013890, -0.014952258336792, -0.000302920514551,
        0.001889950332900
    };
    const vector<double> g={
        h[15], -h[14], h[13], -h[12], h[11], -h[10], h[9], -h[8],
        h[7], -h[6], h[5], -h[4], h[3], -h[2], h[1], -h[0]
    };

    size_t n=signal.size() / 2;
    vector<double> approx(n), detail(n);

    for (size_t i=0; i < n; ++i) {
        approx[i]=0;
        detail[i]=0;
        for (size_t k=0; k < h.size(); ++k) {
            size_t index=2*i+k-h.size() / 2+1;
            if (index < signal.size()) {
                approx[i] += h[k]*signal[index];
                detail[i] += g[k]*signal[index];
            }
        }
    }

    return make_pair(approx, detail);
}

inline pair<vector<double>, vector<double>> coif5(const vector<double>& signal) const
{
    const vector<double> h={
        -0.000720549445364, -0.001823208870703, 0.005611434819394,
        0.023680171946334, -0.059434418646456, -0.076488599078311,
        0.417005184421393, 0.812723635445542, 0.386110066821162,
        -0.067372554721963, -0.041464936781959, 0.016387336463522
    };
    const vector<double> g={
        h[11], -h[10], h[9], -h[8], h[7], -h[6], h[5], -h[4],
        h[3], -h[2], h[1], -h[0]
    };

    size_t n=signal.size() / 2;
    vector<double> approx(n), detail(n);

    for (size_t i=0; i < n; ++i) {
        approx[i]=0;
        detail[i]=0;
        for (size_t k=0; k < h.size(); ++k) {
            size_t index=2*i+k-h.size() / 2+1;
            if (index < signal.size()) {
                approx[i] += h[k]*signal[index];
                detail[i] += g[k]*signal[index];
            }
        }
    }

    return make_pair(approx, detail);
}

// Inverse wavelet reconstruction
inline vector<double> inverse_haar(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv={ 0.7071067811865476, 0.7071067811865476 };
    const vector<double> g_inv={ -0.7071067811865476, 0.7071067811865476 };

    vector<double> reconstructed_signal;
    for (size_t i=0; i < approx.size(); ++i) {
        reconstructed_signal.push_back(approx[i]*h_inv[0]+detail[i]*g_inv[0]);
        reconstructed_signal.push_back(approx[i]*h_inv[1]+detail[i]*g_inv[1]);
    }

    return reconstructed_signal;
}
inline vector<double> inverse_db1(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv={(1.0 +sqrt(3.0))/4.0,(3.0+sqrt(3.0))/4.0,(3.0-sqrt(3.0))/4.0,(1.0-sqrt(3.0))/4.0};
    const vector<double> g_inv={ h_inv[3],-h_inv[2],h_inv[1],-h_inv[0] };

    vector<double> reconstructed_signal(2*approx.size(), 0.0);
    for (size_t i=0; i < approx.size(); ++i) 
    {
        for (size_t k=0; k < h_inv.size(); ++k)
        {
          size_t index=(2*i+k-h_inv.size()/2+1);
          if (index < reconstructed_signal.size())
            reconstructed_signal[index] += approx[i]*h_inv[k]+detail[i]*g_inv[k];
        }
    }
    return reconstructed_signal;
}
inline vector<double> inverse_db6(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv={
        0.111540743350109, 0.494623890398453, 0.751133908021095,
        0.315250351709198, -0.226264693965440, -0.129766867567262,
        0.097501605587322, 0.027522865530305, -0.031582039318486,
        0.0005538422011614, 0.0047772575109455, -0.001077301085308
    };
    const vector<double> g_inv={
        h_inv[11], -h_inv[10], h_inv[9], -h_inv[8], h_inv[7], -h_inv[6],
        h_inv[5], -h_inv[4], h_inv[3], -h_inv[2], h_inv[1], -h_inv[0]
    };

    vector<double> reconstructed_signal(2*approx.size(), 0.0);
    for (size_t i=0; i < approx.size(); ++i) {
        for (size_t k=0; k < h_inv.size(); ++k) {
            size_t index=(2*i+k-h_inv.size() / 2+1);
            if (index < reconstructed_signal.size()) {
                reconstructed_signal[index] += approx[i]*h_inv[k]+detail[i]*g_inv[k];
            }
        }
    }

    return reconstructed_signal;
}

inline vector<double> inverse_sym5(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv={
        0.019538882735287, -0.021101834024759, -0.175328089908450,
        0.016602105764522, 0.633978963458212, 0.723407690402421,
        0.199397533977394, -0.039134249302383, 0.029519490925774,
        0.027333068345078
    };
    const vector<double> g_inv={
        h_inv[9], -h_inv[8], h_inv[7], -h_inv[6], h_inv[5], -h_inv[4],
        h_inv[3], -h_inv[2], h_inv[1], -h_inv[0]
    };

    vector<double> reconstructed_signal(2*approx.size(), 0.0);
    for (size_t i=0; i < approx.size(); ++i) {
        for (size_t k=0; k < h_inv.size(); ++k) {
            size_t index=(2*i+k-h_inv.size() / 2+1);
            if (index < reconstructed_signal.size()) {
                reconstructed_signal[index] += approx[i]*h_inv[k]+detail[i]*g_inv[k];
            }
        }
    }

    return reconstructed_signal;
}

inline vector<double> inverse_sym8(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv={
        0.001889950332900, -0.000302920514551, -0.014952258336792,
        0.003808752013890, 0.049137179673476, -0.027219029917056,
        -0.051945838107709, 0.364441894835331, 0.777185751700523,
        0.481359651258372, -0.061273359067908, -0.143294238350809,
        0.007607487325284, 0.031695087811492, -0.000542132331635,
        -0.003382415951359
    };
    const vector<double> g_inv={
        h_inv[15], -h_inv[14], h_inv[13], -h_inv[12], h_inv[11], -h_inv[10],
        h_inv[9], -h_inv[8], h_inv[7], -h_inv[6], h_inv[5], -h_inv[4],
        h_inv[3], -h_inv[2], h_inv[1], -h_inv[0]
    };

    vector<double> reconstructed_signal(2*approx.size(), 0.0);
    for (size_t i=0; i < approx.size(); ++i) {
        for (size_t k=0; k < h_inv.size(); ++k) {
            size_t index=(2*i+k-h_inv.size() / 2+1);
            if (index < reconstructed_signal.size()) {
                reconstructed_signal[index] += approx[i]*h_inv[k]+detail[i]*g_inv[k];
            }
        }
    }

    return reconstructed_signal;
}

inline vector<double> inverse_coif5(const vector<double>& approx, const vector<double>& detail) const
{
    const vector<double> h_inv={
        0.016387336463522, -0.041464936781959, -0.067372554721963,
        0.386110066821162, 0.812723635445542, 0.417005184421393,
        -0.076488599078311, -0.059434418646456, 0.023680171946334,
        0.005611434819394, -0.001823208870703, -0.000720549445364
    };
    const vector<double> g_inv={
        h_inv[11], -h_inv[10], h_inv[9], -h_inv[8], h_inv[7], -h_inv[6],
        h_inv[5], -h_inv[4], h_inv[3], -h_inv[2], h_inv[1], -h_inv[0]
    };

    vector<double> reconstructed_signal(2*approx.size(), 0.0);
    for (size_t i=0; i < approx.size(); ++i) {
        for (size_t k=0; k < h_inv.size(); ++k) {
            size_t index=(2*i+k-h_inv.size() / 2+1);
            if (index < reconstructed_signal.size()) {
                reconstructed_signal[index] += approx[i]*h_inv[k]+detail[i]*g_inv[k];
            }
        }
    }

    return reconstructed_signal;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Continuous Wavelet Transforms (CWT)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// CWT multilevel with hard/soft thresholding
inline vector<vector<double>> cwt_multilevel (
  const vector<double>& signal, 
  const vector<double>& scales,
  function<vector<double>(const vector<double>&, double)> wavelet_func,
  double threshold,
  const string& threshold_type="hard") 
{                                       // ~~~~~~~~~~~ cwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~
    size_t n=signal.size();             // Original signal length
    size_t n_pad=static_cast<size_t>(next_power_of_2(n));// Next power of 2
    vector<double> padded_signal=signal;// Copy original signal
    if (n_pad!=n)                       // Need to pad?
      padded_signal.resize(n_pad,0.0);  // Yes, zero-pad
    vector<vector<double>> coeffs;      // CWT coefficients for each scale
    for (const auto& scale:scales)      // For each scale 
    {                                   // Compute CWT coefficients
      auto cwt_coeffs=wavelet_func(padded_signal,scale);
      // Apply thresholding to the CWT coefficients
      if (threshold_type== "hard") 
        cwt_coeffs=hard_threshold(cwt_coeffs, threshold);
      else if (threshold_type== "soft")
        cwt_coeffs=soft_threshold(cwt_coeffs, threshold);
      coeffs.push_back(cwt_coeffs);
    }
    return coeffs;
}                                       // ~~~~~~~~~~~ cwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~

// ICWT multilevel reconstruction
inline vector<double> icwt_multilevel (
  const vector<vector<double>>& coeffs, // CWT coefficients for each scale
  function<vector<double>(const vector<double>&, double)> wavelet_func, // Wavelet function
  const vector<double>& scales)         // Corresponding scales
{                                       // ~~~~~~~~~~~ icwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~
    if (coeffs.size()!=scales.size())   // Mismatched sizes?
      throw invalid_argument("Coeffs and scales size mismatch");
    size_t n=coeffs[0].size();          // Length of each coeff vector
    vector<double> signal(n,0.0);       // Reconstructed signal
    for (size_t i=0;i<coeffs.size();++i)// For each scale
    {                                   // Reconstruct signal
      auto recon=wavelet_func(coeffs[i],scales[i]);
      transform(signal.begin(),signal.end(),recon.begin(),signal.begin(),plus<double>());
    }                                   // Done reconstructing signal
    return signal;                      // Return reconstructed signal
}                                       // ~~~~~~~~~~~ icwt_multilevel ~~~~~~~~~~~~~~~~~~~~~~~~

// Scalogram: CWT magnitude squared
inline vector<vector<double>> Scalogram (
  const vector<vector<double>>& cwt_coeffs) // CWT coefficients
{                                       // ~~~~~~~~~~~ Scalogram ~~~~~~~~~~~~~~~~~~~~~~~~
    vector<vector<double>> scalogram;   // Scalogram output
    for (const auto& coeffs:cwt_coeffs) // For each scale's coefficients
    {                                   // Compute magnitude squared
      vector<double> mag_sq(coeffs.size());// Magnitude squared
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      // Square of each coefficient (magnitude squared)
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
      transform(coeffs.begin(),coeffs.end(),mag_sq.begin(),
        [](double x){return x*x;});
      scalogram.push_back(mag_sq);      // Add to scalogram
    }                                   // Done all scales
    return scalogram;                   // Return scalogram
}                                       // ~~~~~~~~~~~ Scalogram ~~~~~~~~~~~~~~~~~~~~~~~~

// Select the forward continuous wavelet function
inline function<vector<double>(const vector<double>&, double)> selectCWTForward (void) const
{
  return [this](const vector<double>& signal, double scale)
  {
    return this->cwt_forward(signal, scale);
  };
}

// --- Scalar mother-wavelet kernels for CWT (psi) --- //
inline double MorletPsi(double x, double w0) const
{
  // Standard real Morlet (unnormalized amplitude as in comments above)
  // psi(x)=exp(-x^2/2)*cos(w0*x)
  return std::exp(-0.5*x*x)*std::cos(w0*x);
}

inline double MexicanHatPsi(double x, double /*a*/) const
{
  // Ricker (Mexican hat): (1-x^2)*exp(-x^2/2)
  const double x2=x*x;
  return (1.0-x2)*std::exp(-0.5*x2);
}

inline double MeyerPsi(double x) const
{
  // Piecewise Meyer per comment (real part only)
  const double ax=std::fabs(x);
  if (ax < 1.0/3.0)
    return 1.0;
  else if (ax < 2.0/3.0)
    return std::cos(3.0*M_PI/2.0*(ax-1.0/3.0));
  else
    return 0.0;
}

inline double GaussianPsi(double x, double a) const
{
  // Gaussian derivative (first order)
  return -2.0*a*a*x*std::exp(-a*a*x*x);
}

// cwt_forward
inline vector<double> cwt_forward (
  const vector<double>& s,
  double scale) const
{                                       // ~~~~~~~~~~~~~~ cwt_forward ~~~~~~~~~~~~~~~~~~~~~~ //
  if(scale==0.0)
    throw invalid_argument("cwt_forward: scale must be non-zero");
  size_t n=s.size();                    // Length of input signal.
  vector<double> coeffs(n);             // Where to store CWT output samples.
  double scale_inv=1.0/scale;           // Scale inverse.
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Convolute with selected wavelet.
    double t=static_cast<double>(i)-static_cast<double>(n)/2.0; // Center time around zero.
    double wavelet_val=0.0;             // Wavelet value at this time and scale
    switch(this->wType)
    {
      case WaveletType::Morlet:
        wavelet_val=MorletPsi(t*scale_inv,this->morlet_w0);
        break;
      case WaveletType::MexicanHat:
        wavelet_val=MexicanHatPsi(t*scale_inv,this->mexhat_a);
        break;
      case WaveletType::Meyer:
        wavelet_val=MeyerPsi(t*scale_inv);
        break;
      case WaveletType::Gaussian:
        wavelet_val=GaussianPsi(t*scale_inv,this->gaussian_a);
        break;
      default:
        wavelet_val=MorletPsi(t*scale_inv,this->morlet_w0);
        break;
    }
    coeffs[i]=s[i]*wavelet_val;         // CWT coefficient at this scale
  }                                     // Done producing CWT coefficients.
  return coeffs;                       // Return produced CWT coefficients.
}                                       // ~~~~~~~~~~~~~~ cwt_forward ~~~~~~~~~~~~~~~~~~~~~~ //

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Morlet wavelet decomposition
// PSI(x)=e^-x/2*cos(5*x)
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline pair<vector<double>,vector<double>> Morlet (
  const vector<double>& s,
  double w0=5.0)
{                                       // ~~~~~~~~~~~~~ Morlet ~~~~~~~~~~~~~~~~~ //
  size_t n=s.size();                    // Length of input signal.
  vector<double> approx(n),detail(n);   // Where to store LP & HP output samples.
  double A=1.0/sqrt(2.0*M_PI);          // Normalization constant.
  double s_inv=1.0/w0;                  // Scale inverse.
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Convolute with Morlet wavelet.
    double t=i-n/2;                     // Center time around zero.
    double morlet=A*exp(-t*t/(2*s_inv*s_inv))*cos(w0*t/s_inv);// Morlet wavelet
    approx[i]=s[i]*morlet;              // Lowpass output (approx)
    detail[i]=s[i]*(1.0-morlet);        // Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return make_pair(approx,detail);      // Return both produced signals.
}                                       // ~~~~~~~~~~~~~ Morlet ~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Mexican hat wavelet decomposition
// PSI(x)=(1-x^2)*e^-x/2
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline pair<vector<double>,vector<double>> MexicanHat (
  const vector<double>& s,
  double a=1.0)
{                                       // ~~~~~~~~~~ MexicanHat ~~~~~~~~~~~~~~~~~~ //
  size_t n=s.size();                    // Length of input signal.
  vector<double> approx(n),detail(n);   // Where to store LP & HP output samples.
  double A=2.0/(sqrt(3.0*a)*pow(M_PI,0.25));// Normalization constant.
  double a2=a*a;                        // a^2
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Convolute with Mexican Hat wavelet.
    double t=i-n/2;                     // Center time around zero.
    double t2=t*t;                      // t^2
    double mexhat=A*(1.0-t2/a2)*exp(-t2/(2*a2));// Mexican Hat wavelet
    approx[i]=s[i]*mexhat;              // Lowpass output (approx)
    detail[i]=s[i]*(1.0-mexhat);        // Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return make_pair(approx,detail);      // Return both produced signals.
}                                       // ~~~~~~~~~~ MexicanHat ~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Meyer wavelet decomposition
// PSI(x)=sin(pi/2*v(3|x|-1))*e^(j*pi*x) for 1/3<=|x|<=2/3
//      =1 for |x|<1/3
//      =0 for |x|>2/3
// v(x) =0 for x<0
//      =x^3(35/32-35/16*x+21/16*x^2-5/8*x^3) for 0<=x<1
//      =1 for x>=1
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline pair<vector<double>,vector<double>> Meyer (
  const vector<double>& s)
{                                       // ~~~~~~~~~~~~~ Meyer ~~~~~~~~~~~~~~~~~~ //
  size_t n=s.size();                    // Length of input signal.
  vector<double> approx(n),detail(n);   // Where to store LP & HP output samples.
  for (size_t i=0;i<n;++i)              // For each input sample
  {                                     // Convolute with Meyer wavelet.
    double t=i-n/2;                     // Center time around zero.
    double t_abs=fabs(t);               // |t|
    double meyer=0.0;                   // Meyer wavelet
    if (t_abs<1.0/3.0)                  // |t|<1/3
      meyer=1.0;                        // Psi=1
    else if (t_abs>=1.0/3.0&&t_abs<=2.0/3.0)// 1/3<=|t|<=2/3
      meyer=sin(M_PI/2.0*MeyersVx(3.0*t_abs-1.0))*cos(M_PI*t);// Psi
    else                                // |t|>2/3
      meyer=0.0;                        // Psi=0
    approx[i]=s[i]*meyer;               // Lowpass output (approx)
    detail[i]=s[i]*(1.0-meyer);         // Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return make_pair(approx,detail);      // Return both produced signals.
}                                       // ~~~~~~~~~~~~~ Meyer ~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// v(x) function used in Meyer wavelet
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
inline double MeyersVx (
  double x) const                       // v(x) function
{                                       // ~~~~~~~~~~~~~ MeyersVx ~~~~~~~~~~~~~~~~~~ //
  if (x<0.0)                            // Outside support
    return 0.0;                         // v(x)=0
  else if (x>=0.0&&x<1.0)               // Within compact support grid?
    return x*x*x*(35.0/32.0-35.0/16.0*x+21.0/16.0*x*x-5.0/8.0*x*x*x);
  else                                  // Outside support
    return 1.0;                         // v(x)=1
}                                       // ~~~~~~~~~~~~~ MeyersVx ~~~~~~~~~~~~~~~~~~ //
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Gaussian wavelet decomposition
// PSI(x)=(1/sqrt(a))*pi^(-1/4)*e^(-x^2/2a^2)*(e^(j*sqrt(2pi/a)x)-e^(-a/2))
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
 inline pair<vector<double>,vector<double>> GaussianWavelet (
  const vector<double>& s,              // Input signal
  double a=1.0)                         // Scale parameter
{                                       // ~~~~~~~~~~ GaussianWavelet ~~~~~~~~~~~~~~~~~~ //
  size_t n=s.size();                    // Length of input signal.
  vector<double> approx(n),detail(n);   // Where to store LP & HP output samples.
  double A=1.0/(sqrt(a)*pow(M_PI,0.25)); // Normalization constant.
  double a2=a*a;                        // a^2
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Convolute with Gaussian wavelet.
    double t=i-n/2;                     // Center time around zero.
    double t2=t*t;                      // t^2
    double gauss=A*exp(-t2/(2*a2))*(cos(sqrt(2.0*M_PI/a)*t)-exp(-a/2.0));// Gaussian wavelet
    approx[i]=s[i]*gauss;               // Lowpass output (approx)
    detail[i]=s[i]*(1.0-gauss);         // Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return make_pair(approx,detail);      // Return both produced signals.
}                                       // ~~~~~~~~~~ GaussianWavelet ~~~~~~~~~~~~~~~~~~ //
// Inverse Morlet wavelet reconstruction
inline vector<double> InverseMorlet (
  const vector<double>& approx,
  const vector<double>& detail,
  double w0=6.0)
{                                       // ~~~~~~~~~~ InverseMorlet ~~~~~~~~~~~~~~~~~~ //
  size_t n=approx.size();               // Length of input signal.
  vector<double> reconstructed_signal(2*n,0.0);// Where to store reconstructed signal.
  double A=1.0/sqrt(2.0*M_PI);          // Normalization constant.
  double s_inv=1.0/w0;                  // Scale inverse.
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Reconstruct with Morlet wavelet.
    double t=2*i-n;                     // Center time around zero.
    double morlet=A*exp(-t*t/(2*s_inv*s_inv))*cos(w0*t/s_inv);// Morlet wavelet
    reconstructed_signal[2*i]+=approx[i]*morlet;// Lowpass output (approx)
    reconstructed_signal[2*i+1]+=detail[i]*(1.0-morlet);// Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return reconstructed_signal;          // Return reconstructed signal.
}                                       // ~~~~~~~~~~ InverseMorlet ~~~~~~~~~~~~~~~~~~ //
// Inverse Mexican Hat wavelet reconstruction
inline vector<double> InverseMexicanHat (
  const vector<double>& approx,         // Low frequency coefficients
  const vector<double>& detail,         // High frequency coefficients 
  double a=1.0)                         // Scale parameter
{                                       // ~~~~~~~~ InverseMexicanHat ~~~~~~~~~~~~~~~~~~ //
  size_t n=approx.size();               // Length of input signal.
  vector<double> reconstructed_signal(2*n,0.0);// Where to store reconstructed signal.
  double A=2.0/(sqrt(3.0*a)*pow(M_PI,0.25));// Normalization constant.
  double a2=a*a;                        // a^2
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Reconstruct with Mexican Hat wavelet.
    double t=2*i-n;                     // Center time around zero.
    double t2=t*t;                      // t^2
    double mexhat=A*(1.0-t2/a2)*exp(-t2/(2*a2));// Mexican Hat wavelet
    reconstructed_signal[2*i]+=approx[i]*mexhat;// Lowpass output (approx)
    reconstructed_signal[2*i+1]+=detail[i]*(1.0-mexhat);// Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return reconstructed_signal;          // Return reconstructed signal.
}                                       // ~~~~~~~~ InverseMexicanHat ~~~~~~~~~~~~~~~~~~ 
// Inverse Meyer wavelet Perfect Reconstruction Filter Bank
inline vector<double> InverseMeyer (
  const vector<double>& approx,         // Low frequency coefficients
  const vector<double>& detail)         // High frequency coefficients
{                                       // ~~~~~~~~~~ InverseMeyer ~~~~~~~~~~~~~~~~~~ //
  size_t n=approx.size();               // Length of input signal.
  vector<double> reconstructed_signal(2*n,0.0);// Where to store reconstructed signal.
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Reconstruct with Meyer wavelet.
    double t=2*i-n;                     // Center time around zero.
    double t_abs=fabs(t);               // |t|
    double meyer=0.0;                   // Meyer wavelet
    if (t_abs<1.0/3.0)                  // |t|<1/3
      meyer=1.0;                        // Psi=1
    else if (t_abs>=1.0/3.0&&t_abs<=2.0/3.0)// 1/3<=|t|<=2/3
      meyer=sin(M_PI/2.0*MeyersVx(3.0*t_abs-1.0))*cos(M_PI*t);// Psi
    else                                // |t|>2/3
      meyer=0.0;                        // Psi=0
    reconstructed_signal[2*i]+=approx[i]*meyer;// Lowpass output (approx)
    reconstructed_signal[2*i+1]+=detail[i]*(1.0-meyer);// Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return reconstructed_signal;          // Return reconstructed signal.
}                                       // ~~~~~~~~~~ InverseMeyer ~~~~~~~~~~~~~~~~~~ //

// Inverse Gaussian wavelet reconstruction
inline vector<double> InverseGaussianWavelet (
  const vector<double>& approx,         // Low frequency coefficients
  const vector<double>& detail,         // High frequency coefficients
  double a=1.0)                         // Scale parameter
{                                       // ~~~~~~~ InverseGaussianWavelet ~~~~~~~~~~~~~~~~~~ //
  size_t n=approx.size();               // Length of input signal.
  vector<double> reconstructed_signal(2*n,0.0);// Where to store reconstructed signal.
  double A=1.0/(sqrt(a)*pow(M_PI,0.25)); // Normalization constant.
  double a2=a*a;                        // a^2
  for (size_t i=0;i<n;++i)              // For each input sample...
  {                                     // Reconstruct with Gaussian wavelet.
    double t=2*i-n;                     // Center time around zero.
    double t2=t*t;                      // t^2
    double gauss=A*exp(-t2/(2*a2))*(cos(sqrt(2.0*M_PI/a)*t)-exp(-a/2.0));// Gaussian wavelet
    reconstructed_signal[2*i]+=approx[i]*gauss;// Lowpass output (approx)
    reconstructed_signal[2*i+1]+=detail[i]*(1.0-gauss);// Highpass output (detail)
  }                                     // Done producing approx and detail coeffs.
  return reconstructed_signal;          // Return reconstructed signal.
}                                       // ~~~~~~~ InverseGaussianWavelet ~~~~~~~~~~~~~~~~~~ //


inline vector<double> hard_threshold(const vector<double>& detail, double threshold) 
{
    vector<double> result(detail.size());
    transform(detail.begin(), detail.end(), result.begin(), [threshold](double coeff) {
        return abs(coeff) < threshold ? 0.0 : coeff;
    });
    return result;
}

inline vector<double> soft_threshold(const vector<double>& detail, double threshold) 
{
    vector<double> result(detail.size());
    transform(detail.begin(), detail.end(), result.begin(), [threshold](double coeff) {
        return signbit(coeff) ? -max(0.0, abs(coeff)-threshold) : max(0.0, abs(coeff)-threshold);
    });
    return result;
}  
  
private:
  WaveletType wType{WaveletType::Haar};       // The wavelet of choice.
  ThresholdType tType{ThresholdType::Hard};     // The type of thresholding.
  size_t levels{1};              // The wavelet decomposition level.
  double threshold{0.0f};        // The threshold of when to cancel.
  double morlet_w0{5.0};         // Morlet central frequency for CWT.
  double mexhat_a{1.0};          // Mexican hat width parameter.
  double gaussian_a{1.0};        // Gaussian derivative slope parameter.
// Method to choose the correct forward wavelet function.
inline std::function<std::pair<std::vector<double>, std::vector<double>>(const std::vector<double>&)> selectForward(void) const
  {                              // -------- selectForward --------
    switch (wType)               // Act according to the type.
    {                            //
      case WaveletType::Haar:  return [this](const std::vector<double>& v){ return this->haar(v); };
      case WaveletType::Db1:   return [this](const std::vector<double>& v){ return this->db1(v); };
      case WaveletType::Db6:   return [this](const std::vector<double>& v){ return this->db6(v); };
      case WaveletType::Sym5:  return [this](const std::vector<double>& v){ return this->sym5(v); };
      case WaveletType::Sym8:  return [this](const std::vector<double>& v){ return this->sym8(v); };
      case WaveletType::Coif5: return [this](const std::vector<double>& v){ return this->coif5(v); };
    }                           // Done acting according to wlet typ.
    return [this](const std::vector<double>& v){ return this->haar(v); };
  }                             // -------- selectForward --------
// Chooses the correct inverse wavelet reconstruction
  inline std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)>
  selectInverse(void) const   // Select the correct reconstruct wave.
  {                           // -------- selectInverse --------
    switch(wType)             // Act according to the wave type
    {                         // Select the recon wavelet.
      case WaveletType::Haar:  return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_haar(a,d); };
      case WaveletType::Db1:   return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_db1(a,d); };
      case WaveletType::Db6:   return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_db6(a,d); };
      case WaveletType::Sym5:  return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_sym5(a,d); };
      case WaveletType::Sym8:  return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_sym8(a,d); };
      case WaveletType::Coif5: return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_coif5(a,d); };
    }                          // Done acting according to wtype.
    return [this](const std::vector<double>& a, const std::vector<double>& d){ return this->inverse_haar(a,d); };
  }                            // -------- selectInverse --------
  // Convert enum to the string the denoiser expects.
  // Expose enum string helpers for UI
  public:
  inline std::string tTypeToString(void) const
  {
    return (this->tType==ThresholdType::Hard?"hard":"soft");
  }
  // String helpers for UI/debug
  static inline std::string WaveletTypeToStr(WaveletType wt)
  {
    switch (wt)
    {
      case WaveletType::Haar:       return "Haar";
      case WaveletType::Db1:        return "Db1";
      case WaveletType::Db6:        return "Db6";
      case WaveletType::Sym5:       return "Sym5";
      case WaveletType::Sym8:       return "Sym8";
      case WaveletType::Coif5:      return "Coif5";
      case WaveletType::Morlet:     return "Morlet";
      case WaveletType::MexicanHat: return "MexicanHat";
      case WaveletType::Meyer:      return "Meyer";
      case WaveletType::Gaussian:   return "Gaussian";
      default:                      return "Unknown";
    }
  }
  static inline std::string ThresholdTypeToStr(ThresholdType tt)
  {
    switch (tt)
    {
      case ThresholdType::Hard: return "Hard";
      case ThresholdType::Soft: return "Soft";
      default:                  return "Unknown";
    }
  }
protected:
/// Pad a signal up to the next power of two
  inline double next_power_of_2(double x) 
  {
    return x== 0 ? 1 : pow(2, ceil(log2(x)));
  }
  
};

} // namespace sig::spectral
