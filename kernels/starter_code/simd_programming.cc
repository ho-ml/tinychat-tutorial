#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif

namespace matmul {
void MatmulOperator::mat_mul_simd_programming(struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;  // block_size = 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    int m = C->row, n = C->column, k = A->column;
    // A: m x k; B: n x k; C: m x n
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
#ifdef QM_ARM
            // order of weights with QM_ARM:
            // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w30,w31)
            // QM_ARM order: (w0,w16),(w1,w17),(w2,w18),(w3,w19),(w4, w20),... (w15,w31)
            //               |--|
            //               4 bits
            //               |------|
            //               8 bits (byte)
            //            low|----------------------------------------------------------|high
            //               0                         128 bit                         127
            // scalar 0. to vetctor [0., 0., 0., 0.]
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            
            // pointer of the int4 weights
            const uint8_t *w_start = &B->int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const int8_t *a_start = &A->int8_data_ptr[row * k];
            // scale of activation
            float *s_a = &params->A_scales[row * k / 32];
            // scale of weight
            float *s_w = &params->scales[col * k / 32];

            const int num_block = k / block_size;
            // lowbit mask
            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            // offsets
            const int8x16_t offsets = vdupq_n_s8(8);

            // Compute each block
            for (int q = 0; q < num_block; q++) {
                // load 32x4bit (16 bytes) weight
                const uint8x16_t w0 = vld1q_u8(w_start);
                w_start += 16;

                /*
                   We will accelerate the program using ARM Intrinsics. You can check the documentation of operations
                   at: https://developer.arm.com/architectures/instruction-sets/intrinsics
                */            
                // unpack the weights using lowbit mask
                // `vshrq_n_u8`: right shift operation
                // `vreinterpretq_s8_u8`: convert uint8x16_t to int8x16_t
                int8x16_t w_de_0 = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w_de_16 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(w0, 4), mask_low4bit));

                // apply zero_point to weights using offsets
                w_de_0 = vsubq_s8(w_de_0, offsets);
                w_de_16 = vsubq_s8(w_de_16, offsets);

                // load 32 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                a_start += 32;

                // int32x4 vector to store intermediate sum
                int32x4_t int_sum0 = vdupq_n_s32(0);
                
                // dot product
                // `vdotq_s32`: dot product and accumulate result into destination register
                int_sum0 = vdotq_s32(int_sum0, a0, w_de_0);
                int_sum0 = vdotq_s32(int_sum0, a1, w_de_16);
                
                // scaling and accumulation
                // `vmlaq_n_f32`: vector mac with scalar
                // `vcvtq_f32_s32`: convert int32x4_t to float32x4_t
                float s_0 = *s_a++ * *s_w++;
                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
            }

            C->data_ptr[row * n + col] = vaddvq_f32(sumv0);
#endif
#ifdef QM_x86
            // order of weights with QM_x86:
            // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
            // QM_ARM order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
            //               |--|
            //               4 bits
            //               |------|
            //               8 bits (byte)
            //            low|----------------------------------------------------------|high
            //               0                         256 bit
            __m256 acc0 = _mm256_setzero_ps();
            // pointer of the int4 weights
            const __m256i *w_start = (__m256i *)&B->int4_data_ptr[col * k / 2];
            // pointer of the int8 activation
            const __m256i *a_start = (__m256i *)&A->int8_data_ptr[row * k];
            // scale of weight
            float *s_ptr = &params->scales[col * k / 32];
            // scale of activation
            float *sa_ptr = &params->A_scales[row * k / 32];

            const int num_block = k / block_size;
            // lowbit mask
            const __m256i lowMask = _mm256_set1_epi8(0xF);
            // zero point (offset)
            const __m256i zero_point = _mm256_set1_epi8(8);
            // vector which is filled with 1
            const __m256i ones = _mm256_set1_epi16(1);
            
            // Compute two blocks in each iteration
            for (int q = 0; q < num_block; q += 2) {
                /*
                   We will accelerate the program using x86 Intrinsics. You can check the documentation of operations
                   at: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avxnewtechs=AVX2
                */
                // load 256 bit from w_strat
                __m256i raw_w = _mm256_loadu_si256(w_start);
                
                // unpack the weights using lowbit mask
                // `_mm256_srli_epi16`: right shift operation
                __m256i w_0, w_128;
                w_0 = _mm256_and_si256(raw_w, lowMask);
                w_128 = _mm256_and_si256(_mm256_srli_epi16(raw_w, 4), lowMask);

                // apply zero_point to weights
                w_0 = _mm256_sub_epi8(w_0, zero_point);
                w_128 = _mm256_sub_epi8(w_128, zero_point);

                // Perform int8 dot product with _mm256_maddubs_epi16
                // `__m256i _mm256_maddubs_epi16(__m256i s1, __m256i s2)`:
                // (1) multiplies vertically s1(unsigned) with the corresponding s2(signed)
                // (2) add each adjacent pair of signed words
                // (3) pack the saturated result to the destination vector
                // 
                // utilize _mm256_maddubs_epi16 which only takes unsigned s1:
                // A x W = (A x sign(W)) x abs(W)
        
                // __m256 vector to store lower and upper halves sum
                __m256i dot, dot2;
                
                // Get absolute values of weights
                const __m256i uw = _mm256_sign_epi8(w_0, w_0);
                const __m256i uw2 = _mm256_sign_epi8(w_128, w_128);

                // Load activation
                __m256i activation = a_start[0];
                __m256i activation2 = a_start[1];
                
                // Change the sign of activation depending on the sign of corresponding weights
                const __m256i sa = _mm256_sign_epi8(activation, w_0);
                const __m256i sa2 = _mm256_sign_epi8(activation2, w_128);
                
                // int8 dot product
                dot = _mm256_maddubs_epi16(uw, sa);
                dot2 = _mm256_maddubs_epi16(uw2, sa2);

                // Convert int32 vectors to floating point vectors
                const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
                const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);
                __m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);
                __m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);

                // Create vectors for scales
                __m256 v_s = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);
                __m256 v_s2 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);

                // apply scales to intermediate results
                acc0 = _mm256_fmadd_ps(intermediate, v_s, acc0);
                acc0 = _mm256_fmadd_ps(intermediate2, v_s2, acc0);

                // move pointer
                s_ptr += 2;
                sa_ptr += 2;
                w_start += 1;
                a_start += 2;
            }

            float *ptr = (float *)&acc0;
            C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
#endif
        }
    }
};
}  // namespace matmul
