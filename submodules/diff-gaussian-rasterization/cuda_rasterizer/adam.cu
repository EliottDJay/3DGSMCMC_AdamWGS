#include "auxiliary.h"
#include "adam.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// step on a grid of size (N, M)
// N is always number of gaussians
__global__
void adamUpdateCUDA(
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    int* __restrict__ prim_step,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const int reg,
    const float reg_scale,
    const uint32_t N,
    const uint32_t M) {

	auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / M;
    if (g_idx >= N) return;
    if (tiles_touched[g_idx]) {
        float Register_param = param[p_idx];
        float Register_param_grad = param_grad[p_idx];
        float Register_exp_avg = exp_avg[p_idx];
        float Register_exp_avg_sq = exp_avg_sq[p_idx];
        int asym_step = prim_step[g_idx];

        // bias correction : float bias_correction1 = 1-b1^primitive_step
        //float bias_correction1 = 1.0f - powf(b1, primitive_step);
        float inv_bias_correction1 = 1.0f / (1.0f - powf(b1, asym_step)); 
        // bias correction : float bias_correction2 = 1-b2^primitive_step
        //float bias_correction2 = 1.0f - powf(b2, primitive_step);
        float inv_bias_correction2 = 1.0f / (1.0f - powf(b2, asym_step));


        Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
        Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;

        float denom = sqrt(Register_exp_avg_sq * inv_bias_correction2) + eps;

        float reg_term = 0.0f;
        if (reg == 1)
        {
            float opacity = 0.5f * (1.0f + tanh(0.5f * Register_param));
            float opacity_decay = reg_scale * opacity * (1.0f - opacity) / denom;
            reg_term = fminf(opacity_decay, 10.0f);
        }
        else if (reg == 2)
        {
            // scale regularization:
            float scale_decay = reg_scale * exp(Register_param) / denom;
            reg_term = fminf(scale_decay, 10.0f);
        }

        //float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);
        float step = -lr * (Register_exp_avg * inv_bias_correction1 / denom + reg_term);

        // param[p_idx] += step;
        param[p_idx] = Register_param + step;
        exp_avg[p_idx] = Register_exp_avg;
        exp_avg_sq[p_idx] = Register_exp_avg_sq;
    }
}


void ADAM::adamUpdate(
    float* param,
    const float* param_grad,
    float* exp_avg,
    float* exp_avg_sq,
    int* prim_step,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const int reg,
    const float reg_scale,
    const uint32_t N,
    const uint32_t M) {

    const uint32_t cnt = N * M;
    adamUpdateCUDA<<<(cnt + 255) / 256, 256>>> (
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        prim_step,
        tiles_touched,
        lr,
        b1,
        b2,
        eps,
        reg,
        reg_scale,
        N,
        M
    );
}