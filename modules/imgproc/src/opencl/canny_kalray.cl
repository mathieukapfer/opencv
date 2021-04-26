/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2021-2023, Kalray Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// ============================================================================
#ifdef KALRAY_DEVICE
// ============================================================================

#define _CONCAT(x, y) x ## y
#define CONCAT(x, y) _CONCAT(x, y)

#define TG22 0.4142135623730950488016887242097f
#define TG67 2.4142135623730950488016887242097f

#define TG22D ( 4142 / 2)
#define TG67D (24142 / 2)

// ============================================================================
// Kalray async block_2D2D extension
// NOTE: These functions are to be removed in later SDK version, once they have
//       been intergrated into OpenCL runtime
// ============================================================================
#define IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE, ELEM_FMT)                    \
__attribute__((overloadable))                                                        \
void print_mat(__local GENTYPE *mat, int rows, int cols, int row_stride,             \
    __constant char* name)                                                           \
{                                                                                    \
    printf("%s:\n", name);                                                           \
    for(int j = 0; j < rows; j++) {                                                  \
        for(int i = 0; i < cols; i++) {                                              \
            printf("  "ELEM_FMT, mat[i + j * row_stride]);                           \
        }                                                                            \
        printf("\n");                                                                \
    }                                                                                \
}                                                                                    \
                                                                                     \
__attribute__((overloadable))                                                        \
event_t async_work_group_copy_block_2D2D(__local GENTYPE *dst,                       \
                                          const __global GENTYPE *src,               \
                                          size_t num_gentype_per_elem,               \
                                          int2 block_size,                           \
                                          int4 local_point,                          \
                                          int4 global_point,                         \
                                          event_t event)                             \
{                                                                                    \
    event_t ret = event;                                                             \
    const size_t local_stride  = (local_point.z - block_size.x) *                    \
                                 num_gentype_per_elem * sizeof(GENTYPE);             \
    const size_t global_stride = (global_point.z - block_size.x) *                   \
                                 num_gentype_per_elem * sizeof(GENTYPE);             \
    const size_t local_offset  = mad24(local_point.y, local_point.z,                 \
                                       local_point.x);                               \
    const size_t global_offset = mad24(global_point.y, global_point.z,               \
                                       global_point.x);                              \
    __local uchar        *dst_used = (__local uchar        *)(dst) +                 \
                                     (local_offset * num_gentype_per_elem *          \
                                     sizeof(GENTYPE));                               \
    const __global uchar *src_used = (const __global uchar *)(src) +                 \
                                     (global_offset * num_gentype_per_elem           \
                                     * sizeof(GENTYPE));                             \
    const size_t num_elements_per_line = block_size.x * num_gentype_per_elem *       \
                                         sizeof(GENTYPE);                            \
    ret = async_work_group_copy_2D2D(dst_used,                                       \
                                     src_used,                                       \
                                     num_elements_per_line,                          \
                                     block_size.y,  /* num_lines */                  \
                                     global_stride, /* src_stride */                 \
                                     local_stride,  /* dst_stride */                 \
                                     event          /* event */                      \
                                     );                                              \
    return ret;                                                                      \
}                                                                                    \
                                                                                     \
__attribute__((overloadable))                                                        \
event_t async_work_group_copy_block_2D2D(__global GENTYPE *dst,                      \
                                          const __local GENTYPE *src,                \
                                          size_t num_gentype_per_elem,               \
                                          int2 block_size,                           \
                                          int4 local_point,                          \
                                          int4 global_point,                         \
                                          event_t event)                             \
{                                                                                    \
    event_t ret = event;                                                             \
    const size_t local_stride  = (local_point.z - block_size.x) *                    \
                                 num_gentype_per_elem * sizeof(GENTYPE);             \
    const size_t global_stride = (global_point.z - block_size.x) *                   \
                                 num_gentype_per_elem * sizeof(GENTYPE);             \
    const size_t local_offset  = mad24(local_point.y, local_point.z,                 \
                                       local_point.x);                               \
    const size_t global_offset = mad24(global_point.y, global_point.z,               \
                                       global_point.x);                              \
    __global uchar      *dst_used = (__global uchar      *)(dst) +                   \
                                    (global_offset * num_gentype_per_elem *          \
                                     sizeof(GENTYPE));                               \
    const __local uchar *src_used = (const __local uchar *)(src) +                   \
                                    (local_offset * num_gentype_per_elem *           \
                                     sizeof(GENTYPE));                               \
    const size_t num_elements_per_line = block_size.x * num_gentype_per_elem *       \
                                         sizeof(GENTYPE);                            \
    ret = async_work_group_copy_2D2D(dst_used,                                       \
                                     src_used,                                       \
                                     num_elements_per_line,                          \
                                     block_size.y,  /* num_lines */                  \
                                     local_stride,  /* src_stride */                 \
                                     global_stride, /* dst_stride */                 \
                                     event          /* event */                      \
                                     );                                              \
    return ret;                                                                      \
}

#define IMPLEMENT_ASYNC_COPY_2D2D_FUNCS(GENTYPE,      ELEM_FMT)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE,     ELEM_FMT)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##2,  ELEM_FMT)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##3,  ELEM_FMT)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##4,  ELEM_FMT)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##8,  ELEM_FMT)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##16, ELEM_FMT)

IMPLEMENT_ASYNC_COPY_2D2D_FUNCS(uchar, "%3d");
IMPLEMENT_ASYNC_COPY_2D2D_FUNCS(int,   "%3d");
IMPLEMENT_ASYNC_COPY_2D2D_FUNCS(float, "%.3f");


#ifdef WITH_SOBEL

#if cn == 1
#define loadpix(addr) convert_floatN(*(__global const TYPE *)(addr))
#define TYPEN TYPE
#define CONVERT_TO(type, n) CONCAT(convert_, type)
#else
#define loadpix(addr) convert_floatN(vload3(0, (__global const TYPE *)(addr)))
#define TYPEN CONCAT(TYPE, cn)
#define CONVERT_TO(type, n) CONCAT(CONCAT(convert_, type), n)
#endif
#define storepix(value, addr) *(__global int *)(addr) = (int)(value)

#define NMS_HALO_SIZE    (1)
#define SOBEL_HALO_SIZE  (APRT/2)  // (APRT always odd, if 3 returns 1, if 5 returns 2)

#define IN_WIDTH  (GRP_SIZEX + 2*NMS_HALO_SIZE + 2*SOBEL_HALO_SIZE)
#define IN_HEIGHT (GRP_SIZEY + 2*NMS_HALO_SIZE + 2*SOBEL_HALO_SIZE)
#define IN_OFFSET_Y(cur, dist) ((cur) + (dist) * IN_WIDTH)

#define MAG_WIDTH  (GRP_SIZEX + 2*NMS_HALO_SIZE)
#define MAG_HEIGHT (GRP_SIZEY + 2*NMS_HALO_SIZE)
#define MAG_OFFSET_Y(cur, dist) ((cur) + (dist) * MAG_WIDTH)

#define OUT_WIDTH  (GRP_SIZEX + 0)
#define OUT_HEIGHT (GRP_SIZEY + 0)
#define OUT_OFFSET_Y(cur, dist) ((cur) + (dist) * OUT_WIDTH)

#ifdef L2GRAD
#define dist(x, y) ((int)(x) * (x) + (int)(y) * (y))
#define distn(x, y) (convert_int4(x) * convert_int4(x) + convert_int4(y) * convert_int4(y))
#else
#define dist(x, y) (abs((int)(x)) + abs((int)(y)))
#define distn(x, y) convert_int4(abs(x) + abs(y))
#endif

/*
    stage1_with_sobel:
        Sobel operator
        Calc magnitudes
        Non maxima suppression
        Double thresholding
*/

inline int3 sobel_3x3(int idx, __local const shortN *smem)
{
    // result: x, y, mag
    int3 res;
    // pixels around the one we are computing.
    // Naming: x-ffoset,y-offset with the offset being one of:
    // m (minus), e (equal), p (plus)
#if cn == 1
    shortN mm = smem[IN_OFFSET_Y(idx + 0, 0)];
    shortN em = smem[IN_OFFSET_Y(idx + 1, 0)];
    shortN pm = smem[IN_OFFSET_Y(idx + 2, 0)];
    shortN me = smem[IN_OFFSET_Y(idx + 0, 1)];
    shortN pe = smem[IN_OFFSET_Y(idx + 2, 1)];
    shortN mp = smem[IN_OFFSET_Y(idx + 0, 2)];
    shortN ep = smem[IN_OFFSET_Y(idx + 1, 2)];
    shortN pp = smem[IN_OFFSET_Y(idx + 2, 2)];
#else
    shortN mm = vload3(IN_OFFSET_Y(idx + 0, 0), (__local short *)smem);
    shortN em = vload3(IN_OFFSET_Y(idx + 1, 0), (__local short *)smem);
    shortN pm = vload3(IN_OFFSET_Y(idx + 2, 0), (__local short *)smem);
    shortN me = vload3(IN_OFFSET_Y(idx + 0, 1), (__local short *)smem);
    shortN pe = vload3(IN_OFFSET_Y(idx + 2, 1), (__local short *)smem);
    shortN mp = vload3(IN_OFFSET_Y(idx + 0, 2), (__local short *)smem);
    shortN ep = vload3(IN_OFFSET_Y(idx + 1, 2), (__local short *)smem);
    shortN pp = vload3(IN_OFFSET_Y(idx + 2, 2), (__local short *)smem);
#endif

    // Convolution
    // -1 0 1
    // -2 0 2
    // -1 0 1
    shortN dx = (shortN)2 * (pe - me) + pm - mm + pp - mp;

    // Convolution
    //  1  2  1
    //  0  0  0
    // -1 -2 -1
    shortN dy = (shortN)2 * (em - ep) + pm - pp + mm - mp;

#ifdef L2GRAD
    intN _dx = CONVERT_TO(int, cn)(dx);
    intN _dy = CONVERT_TO(int, cn)(dy);
    intN magN = _dx * _dx + _dy * _dy;
#else
    shortN magN = CONVERT_TO(short, cn)(abs(dx) + abs(dy));
#endif
#if cn == 1
    res.z = magN;
    res.x = dx;
    res.y = dy;
#else
    res.z = max(magN.x, max(magN.y, magN.z));
    if (res.z == magN.y)
    {
        res.x = dx.y;
        res.y = dy.y;
    }
    else if (res.z == magN.z)
    {
        res.x = dx.z;
        res.y = dy.z;
    }
    else
    {
        res.x = dx.x;
        res.y = dy.x;
    }
#endif

    return res;
}

static inline void stage1_singlepass_sobel_compute_block(
    __local shortN *smem,
    __local short *magx,
    __local short *magy,
    __local int *magz)
{
    // Sobel, Magnitude
    int nb_pe = get_local_size(1) * get_local_size(0);
    int cur_pe = get_local_id(0) * get_local_size(1) + get_local_id(1);

    int startmagrow = MAG_HEIGHT * cur_pe / nb_pe;
    int endmagrow = MAG_HEIGHT * (cur_pe + 1) / nb_pe;

    for (int y = startmagrow; y < endmagrow; y++) {
        int mag_idx = MAG_OFFSET_Y(0, y);
        int in_idx = IN_OFFSET_Y(0, y);
        int3 s;

        for (int x = 0; x < MAG_WIDTH / 16; x++) {
            #pragma unroll
            for (int iter = 0; iter < 16; iter++) {
                s = sobel_3x3(in_idx, smem);
                magx[mag_idx] = s.x;
                magy[mag_idx] = s.y;
                magz[mag_idx] = s.z;
                mag_idx++;
                in_idx++;
            }
        }
        for (int x = 0; x < MAG_WIDTH % 16; x++) {
            s = sobel_3x3(in_idx, smem);
            magx[mag_idx] = s.x;
            magy[mag_idx] = s.y;
            magz[mag_idx] = s.z;
            mag_idx++;
            in_idx++;
        }
    }
}

inline void sobel_sep_row(int in_idx, int out_idx, __local const shortN *smem, __local short *half_dx,__local short *half_dy, const short *kx, const short *ky)
{
// cn==3 not supported
#if cn == 1
    short sum_x = 0, sum_y = 0;
    int start_x = in_idx - SOBEL_HALO_SIZE;
    #pragma unroll
    for (int i=0; i<APRT; i++)
    {
        sum_x += smem[start_x + i] * kx[i];
        sum_y += smem[start_x + i] * ky[i];
    }
    half_dx[out_idx] = sum_x;
    half_dy[out_idx] = sum_y;
#endif
}

static inline void stage1_separable_sobel_compute_block(
    __local shortN *smem,
    __local short *half_dx,
    __local short *half_dy,
    __local short *dx,
    __local short *dy,
    __local int *magz)
{
    // Sobel, Magnitude
    int nb_pe = get_local_size(1) * get_local_size(0);
    int cur_pe = get_local_id(0) * get_local_size(1) + get_local_id(1);

    int start_in_row = IN_HEIGHT * cur_pe / nb_pe;
    int end_in_row = IN_HEIGHT * (cur_pe + 1) / nb_pe;

    // TODO generate kernel
    const short kx_x[5] = {-1,-2,0,2,1};
    const short ky[5] = {1,4,6,4,1};

    for (int y = start_in_row; y < end_in_row; y++) {
        int out_idx = MAG_OFFSET_Y(0, y);
        int in_idx = IN_OFFSET_Y(SOBEL_HALO_SIZE, y);
        for(int x=0;x<MAG_WIDTH/16;x++) {
            #pragma unroll
            for (int iter = 0; iter < 16; iter++) {
                sobel_sep_row(in_idx, out_idx, smem, half_dx, half_dy, kx_x, ky);
                in_idx++;
                out_idx++;
            }
        }
        for(int x=0;x<MAG_WIDTH%16;x++) {
            sobel_sep_row(in_idx, out_idx, smem, half_dx, half_dy, kx_x, ky);
            in_idx++;
            out_idx++;
        }
    }

    // wait for the horizontal part of separable filters (dx and dy) to be complete
    barrier(CLK_LOCAL_MEM_FENCE);

    int start_mag_row = MAG_HEIGHT * cur_pe / nb_pe;
    int end_mag_row = MAG_HEIGHT * (cur_pe + 1) / nb_pe;
    const short kx_y[5] = {1,2,0,-2,-1};
    // TODO compare horizontal versus vertical iteration for vertical filtering.
    // Which one maximize cache hit ? (probably depends on aperture size)
    for (int y = start_mag_row; y < end_mag_row; y++) {
        for(int x=0;x<MAG_WIDTH/4;x++) {
            int row_offset = MAG_OFFSET_Y(0, y);
            short4 sum_x = vload4(x, &(half_dx[row_offset])) * ky[0];
            short4 sum_y = vload4(x, &(half_dy[row_offset])) * kx_y[0];
            #pragma unroll
            for (int i=1; i<APRT; i++)
            {
                sum_x += vload4(x, &(half_dx[MAG_OFFSET_Y(0, y+i)])) * ky[i];
                sum_y += vload4(x, &(half_dy[MAG_OFFSET_Y(0, y+i)])) * kx_y[i];
            }
            vstore4(sum_x, x, &(dx[row_offset]));
            vstore4(sum_y, x, &(dy[row_offset]));
            vstore4(distn(sum_x, sum_y), x, &(magz[row_offset]));
        }
        int idx = MAG_WIDTH - (MAG_WIDTH % 4);
        for(int x=0;x<MAG_WIDTH%4;x++) {
            int offset = MAG_OFFSET_Y(idx, y);
            short sum_x = half_dx[offset] * ky[0];
            short sum_y = half_dy[offset] * kx_y[0];
            #pragma unroll
            for (int i=1; i<APRT; i++)
            {
                sum_x += half_dx[MAG_OFFSET_Y(idx, y+i)] * ky[i];
                sum_y += half_dy[MAG_OFFSET_Y(idx, y+i)] * kx_y[i];
            }
            dx[offset] = sum_x;
            dy[offset] = sum_y;
            magz[offset] = dist(sum_x, sum_y);
            idx++;
        }
    }
}

static inline void stage1_nms_compute_block(
    __local short *magx,
    __local short *magy,
    __local int *magz,
    __local uchar *map_out,
    int rows, float low_thr, float high_thr)
{
    int nb_pe = get_local_size(1) * get_local_size(0);
    int cur_pe = get_local_id(0) * get_local_size(1) + get_local_id(1);

    // NMS work split computation.
    int remaining_rows = min((int)(rows - get_group_id(1) * OUT_HEIGHT), (int)OUT_HEIGHT);
    int startrow = remaining_rows * cur_pe / nb_pe;
    int endrow = remaining_rows * (cur_pe + 1) / nb_pe;

    // Magnitudes are always integer so we can safely round the thresholds.
    short low_thr_short = round(low_thr);
    short high_thr_short = round(high_thr);

    for (int cur_y = startrow; cur_y < endrow; cur_y++) {
        int mag_idx_y = cur_y + 1;
        int magi = MAG_OFFSET_Y(1, mag_idx_y);

        for (int cur_x = 0; cur_x < OUT_WIDTH; cur_x += 4) {
            int4 magis = (int4)magi + (int4)(0, 1, 2, 3);
            int4 cur_xs = (int4)(cur_x) + (int4)(0, 1, 2, 3);
            int4 mag_idx_x = cur_xs + (int4)1;

            int4 mag0s;
            #pragma unroll
            for (int idx = 0; idx < 4; idx++) {
                mag0s[idx] = magz[magis[idx]];
            }

            int4 value = (int4)1;
            if (mag0s.s0 > low_thr_short ||
                mag0s.s1 > low_thr_short ||
                mag0s.s2 > low_thr_short ||
                mag0s.s3 > low_thr_short)
            {
                short4 x;
                #pragma unroll
                for (int idx = 0; idx < 4; idx++) {
                    x[idx] = magx[magis[idx]];
                }
                short4 y;
                #pragma unroll
                for (int idx = 0; idx < 4; idx++) {
                    y[idx] = magy[magis[idx]];
                }
                ushort4 x_ = abs(x);
                ushort4 y_ = abs(y);

                // TG22D and TG67D are 5000x-scaled versions of TG22 and TG67.
                int4 x_5000 = convert_int4(x_) * (int4)5000;
                int4 y_22 = convert_int4(y_) * (int4)TG22D;
                int4 y_67 = convert_int4(y_) * (int4)TG67D;

                int4 a_ = x_5000 - y_22;
                int4 b_ = x_5000 - y_67;
                int4 a = select((int4)1, (int4)2, a_);
                int4 b = select((int4)0, (int4)1, b_);

                //  a = { 1, 2 }
                //  b = { 0, 1 }
                //  a * b = { 0, 1, 2 } - directions that we need ( + 3 if x ^ y < 0)

                int4 dir3 = convert_int4(a * b) & ((convert_int4(x ^ y) & (int4)0x8000) >> (int4)15); // if a = 1, b = 1, dy ^ dx < 0
                int4 dir = convert_int4(a * b) + (int4)2 * dir3;

                /*
                    Now we have the direction of the gradient, we retrieve the
                    elements alongside it so we could check whether the current
                    value is bigger.
                    prev and next store the offsets to apply. Since they
                    represent offset for the previous and next pixels in a line,
                    next = -prev.
                */

                int4 next_x = ((dir + (int4)2) % (int4)3) - (int4)1;
                int4 next_y = (dir != (int4)0) & (int4)0x1;
                int4 prevpos = MAG_OFFSET_Y(mag_idx_x - next_x, mag_idx_y - next_y);
                int4 nextpos = MAG_OFFSET_Y(mag_idx_x + next_x, mag_idx_y + next_y);

                int4 prev_mag;
                int4 next_mag;
                #pragma unroll
                for (int idx = 0; idx < 4; idx++) {
                    prev_mag[idx] = magz[prevpos[idx]];
                    next_mag[idx] = magz[nextpos[idx]] + (dir[idx] & 1);
                    if (mag0s[idx] > prev_mag[idx] && mag0s[idx] >= next_mag[idx] && mag0s[idx] > low_thr_short)
                    {
                        value[idx] = (mag0s[idx] > high_thr_short) ? 2 : 0;
                    }
                }
            }

            #pragma unroll
            for (int idx = 0; idx < 4; idx++) {
                map_out[OUT_OFFSET_Y(cur_x + idx, cur_y)] = value[idx];
            }
            magi += 4;
        }
    }
}

// Blocksize, copy position and halo pruning algorithm
// https://hal.univ-grenoble-alpes.fr/hal-01652614/document
static inline int local_offset(const int iblock, const int num_blocks)
{
    return (iblock == 0 ? (NMS_HALO_SIZE + SOBEL_HALO_SIZE) : 0);
}
static inline int remote_offset(const int iblock, const int num_blocks)
{
    return (iblock == 0 ? 0 : -(NMS_HALO_SIZE + SOBEL_HALO_SIZE));
}
static inline int halo_cutoff(const int iblock, const int num_blocks)
{
    return ((iblock > 0 && iblock < (num_blocks - 1)) ? 0 : -(NMS_HALO_SIZE + SOBEL_HALO_SIZE));
}

__kernel void stage1_with_sobel(__global const uchar *src, int src_step, int src_offset, int rows, int cols,
                                __global uchar *map, int map_step, int map_offset,
                                float low_thr, float high_thr)
{
    //                                GRP_SIZEX + 2*NMS_HALO_SIZE +
    //                                            2*SOBEL_HALO_SIZE
    //                         +--------------------------------+
    //                         |                                |
    //                         |    +----------------------+    |
    //                         |    |      GRP_SIZEX       |    |
    //                         |    |                      |    |
    //                         |    |                      |    |
    //     GRP_SIZEY +         |    | GRP_SIZEY            |    |
    //     2*NMS_HALO_SIZE +   |    |                      |    |
    //     2*SOBEL_HALO_SIZE   |    |                      |    |
    //                         |    +----------------------+    |
    //                         |                                |
    //                         +--------------------------------+
    //

    // input double buffer
    __local TYPEN  smem_src[2][IN_WIDTH * IN_HEIGHT];
    __local shortN smem [IN_WIDTH * IN_HEIGHT];

    // intermediate buffers
    __local short  magx[MAG_WIDTH * MAG_HEIGHT];
    __local short  magy[MAG_WIDTH * MAG_HEIGHT];
#if APRT > 3
    // TODO separable sobel only supports cn==1
    // separable sobel buffers
    __local short half_dx[MAG_WIDTH * IN_HEIGHT];
    __local short half_dy[MAG_WIDTH * IN_HEIGHT];
#endif
    __local int magz[MAG_WIDTH * MAG_HEIGHT];

    // output double buffers
    __local uchar map_out[2][GRP_SIZEX * GRP_SIZEY];

    const int lsizex = get_local_size(0);
    const int lsizey = get_local_size(1);
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);

    // linearized workitem id in workgroup
    const int wid = lidx + lidy * lsizex;

    const int num_groups = get_num_groups(0) * get_num_groups(1);
    const int group_id = get_group_id(0) + (get_group_id(1) * get_num_groups(0));

    const int num_blocks_x = (int)ceil(((float)cols) / GRP_SIZEX);
    const int num_blocks_y = (int)ceil(((float)rows) / GRP_SIZEY);

    const int num_blocks_per_group = (num_blocks_x * num_blocks_y) / num_groups;
    const int num_blocks_trailing  = (num_blocks_x * num_blocks_y) % num_groups;

    const int iblock_begin = group_id * num_blocks_per_group + min(group_id, num_blocks_trailing);
    const int iblock_end   = iblock_begin + num_blocks_per_group + ((group_id < num_blocks_trailing) ? 1 : 0);

    int iblock_x_next = iblock_begin % num_blocks_x;
    int iblock_y_next = iblock_begin / num_blocks_x;

    event_t event_read[2]  = {0, 0};
    event_t event_write[2] = {0, 0};

    // block to copy
    // clamp if any last block exceeds the remaining size
    // Assumption: there are at least >= 2 blocks in each row and col dimension
    const int2 block_output_first = {clamp(GRP_SIZEX, GRP_SIZEX, cols - (GRP_SIZEX * iblock_x_next)),
                                     clamp(GRP_SIZEY, GRP_SIZEY, rows - (GRP_SIZEY * iblock_y_next))};
    int2 block_size   = {(block_output_first.x + 2*NMS_HALO_SIZE + 2*SOBEL_HALO_SIZE) +
                          halo_cutoff(iblock_x_next, num_blocks_x),
                         (block_output_first.y + 2*NMS_HALO_SIZE + 2*SOBEL_HALO_SIZE) +
                          halo_cutoff(iblock_y_next, num_blocks_y)};

    // local write position and dimension
    int4 local_point  = {0 + local_offset(iblock_x_next, num_blocks_x),
                         0 + local_offset(iblock_y_next, num_blocks_y),
                         IN_WIDTH,
                         IN_HEIGHT};

    // global read position and dimension
    int4 global_point = {(GRP_SIZEX * iblock_x_next) + remote_offset(iblock_x_next, num_blocks_x),
                         (GRP_SIZEY * iblock_y_next) + remote_offset(iblock_y_next, num_blocks_y),
                         src_step / (cn*sizeof(TYPE)),
                         rows};

    // copy the 2D block from src image into smem_src
    event_read[iblock_begin & 1] = async_work_group_copy_block_2D2D(
        (__local TYPE *)(smem_src[iblock_begin & 1]),
        (const __global TYPE *)(src + src_offset),
        cn,             // num_gentypes_per_pixel
        block_size,    // block_size
        local_point,   // local_point
        global_point,  // global_point
        0              // event
    );

    for (int iblock = iblock_begin; iblock < iblock_end; iblock++)
    {
        // =========================================================
        // current block to be processed
        // =========================================================
        const int iblock_x = iblock_x_next;
        const int iblock_y = iblock_y_next;
        const int iblock_parity = iblock & 1;

        // start coordinates of output block in the output image
        const int start_x = GRP_SIZEX * iblock_x;
        const int start_y = GRP_SIZEY * iblock_y;

        // =========================================================
        // prefetch next block (if any)
        // =========================================================
        const int iblock_next = iblock + 1;
        iblock_x_next = iblock_next % num_blocks_x;
        iblock_y_next = iblock_next / num_blocks_x;

        // clamp if any last block exceeds the remaining size
        // Assumption: there are at least >= 2 blocks in each row and col dimension
        const int2 block_output_next = {clamp(GRP_SIZEX, GRP_SIZEX, cols - (GRP_SIZEX * iblock_x_next)),
                                        clamp(GRP_SIZEY, GRP_SIZEY, rows - (GRP_SIZEY * iblock_y_next))};
        const int2 block_size_next = {(block_output_next.x + 2*NMS_HALO_SIZE + 2*SOBEL_HALO_SIZE) +
                                       halo_cutoff(iblock_x_next, num_blocks_x),
                                      (block_output_next.y + 2*NMS_HALO_SIZE + 2*SOBEL_HALO_SIZE) +
                                       halo_cutoff(iblock_y_next, num_blocks_y)};
        const int2 local_pos_next  = {0 + local_offset(iblock_x_next, num_blocks_x),
                                      0 + local_offset(iblock_y_next, num_blocks_y)};
        const int2 global_pos_next = {(GRP_SIZEX * iblock_x_next) + remote_offset(iblock_x_next, num_blocks_x),
                                      (GRP_SIZEY * iblock_y_next) + remote_offset(iblock_y_next, num_blocks_y)};
        // only prefetch if we still have work
        if (iblock_next < iblock_end)
        {
            const int iblock_next_parity = iblock_next & 1;

            int4 local_point_next  = {local_pos_next.x, local_pos_next.y,
                                      local_point.z, local_point.w};
            int4 global_point_next = {global_pos_next.x, global_pos_next.y,
                                      global_point.z, global_point.w};

            event_read[iblock_next_parity] = async_work_group_copy_block_2D2D(
                                    (__local TYPE *)(smem_src[iblock_next_parity]),
                                    (const __global TYPE *)(src + src_offset),
                                    cn,                  // num_gentypes_per_pixel
                                    block_size_next,    // block_size
                                    local_point_next,   // local_point
                                    global_point_next,  // global_point
                                    0                   // event
                                 );
        }

        // =========================================================
        // wait for current block
        // =========================================================
        wait_group_events(1, &event_read[iblock_parity]);

        // =========================================================
        // padding: cast uchar to float and pad halo pixels
        // =========================================================
        // each WI will read a row of smem_src[(IN_HEIGHT)][(IN_WIDTH)],
        // cast to float and write to     smem[(IN_HEIGHT)][(IN_WIDTH)]
        for(int irow = wid; irow < IN_HEIGHT; irow += (lsizex * lsizey))
        {
            // clamp to copy the valid border, similarly to copyMakeBorder(BORDER_REPLICATE)
            const int irow_src = clamp(irow, local_point.y, (local_point.y + block_size.y) - 1);
            for (int icol = 0; icol < IN_WIDTH; icol++)
            {
                const int icol_src = clamp(icol, local_point.x, (local_point.x + block_size.x) - 1);
#if cn == 1
                TYPEN pixel_src = smem_src[iblock_parity][irow_src * IN_WIDTH + icol_src];
                smem[irow * IN_WIDTH + icol] = CONVERT_TO(short, cn)(pixel_src);
#else
                TYPEN pixel_src = vload3(mad24(irow_src, IN_WIDTH, icol_src),
                                         (__local TYPE *)(smem_src[iblock_parity]));
                vstore3(CONVERT_TO(short, cn)(pixel_src),
                        mad24(irow, IN_WIDTH, icol),
                        (__local short *)smem);
#endif
            }
        }
        // sync to gather copy from all WI into smem[] before the compute step
        barrier(CLK_LOCAL_MEM_FENCE);

        // after padding & barrier, update async copy info to the next block,
        // used by the padding of next iteration
        block_size      = block_size_next;
        local_point.xy  = local_pos_next;
        global_point.xy = global_pos_next;

        // =========================================================
        // wait for previous put of the 2D block from local to global to avoid
        // data race: writing result to a being-put buffer
        // =========================================================
        if (iblock - iblock_begin > 1) {
            wait_group_events(1, &event_write[iblock_parity]);
        }

        // =========================================================
        // now compute the current block
        // =========================================================

#if APRT == 3
        stage1_singlepass_sobel_compute_block(smem, magx, magy, magz);
#else
        stage1_separable_sobel_compute_block(smem, half_dx, half_dy, magx, magy, magz);
#endif

        // Wait for all magnitudes to be computed. Needed for the first/last
        // lines for the WIs.
        barrier(CLK_LOCAL_MEM_FENCE);

        stage1_nms_compute_block(magx, magy, magz, map_out[iblock_parity], rows, low_thr, high_thr);


        // sync to gather output from all WI into map_out[] before putting to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // =========================================================
        // put result to global memory
        // =========================================================
        event_write[iblock_parity] = async_work_group_copy_block_2D2D(
                                (__global uchar *)(map + map_offset),
                                (const __local uchar *)(map_out[iblock_parity]),
                                1,                      // num_gentypes_per_pixel
                                (int2){clamp(GRP_SIZEX, GRP_SIZEX, (cols - start_x)),
                                       clamp(GRP_SIZEY, GRP_SIZEY, (rows - start_y))}, // block_size
                                (int4){0, 0, GRP_SIZEX, GRP_SIZEY},  // local_point
                                (int4){iblock_x * GRP_SIZEX,
                                       iblock_y * GRP_SIZEY,
                                       (map_step),
                                       rows},                        // global_point
                                0                                    // event
                             );
    }  // for (int iblock = iblock_begin; iblock < iblock_end; iblock++)

    async_work_group_copy_fence(CLK_GLOBAL_MEM_FENCE);
}



#elif defined WITHOUT_SOBEL

/*
    stage1_without_sobel:
        Calc magnitudes
        Non maxima suppression
        Double thresholding
*/

#define loadpix(addr) (__global short *)(addr)
#define storepix(val, addr) *(__global int *)(addr) = (int)(val)

#ifdef L2GRAD
#define dist(x, y) ((int)(x) * (x) + (int)(y) * (y))
#else
#define dist(x, y) (abs((int)(x)) + abs((int)(y)))
#endif

__constant int prev[4][2] = {
    { 0, -1 },
    { -1, -1 },
    { -1, 0 },
    { -1, 1 }
};

__constant int next[4][2] = {
    { 0, 1 },
    { 1, 1 },
    { 1, 0 },
    { 1, -1 }
};

__kernel void stage1_without_sobel(__global const uchar *dxptr, int dx_step, int dx_offset,
                                   __global const uchar *dyptr, int dy_step, int dy_offset,
                                   __global uchar *map, int map_step, int map_offset, int rows, int cols,
                                   int low_thr, int high_thr)
{
    int start_x = get_group_id(0) * GRP_SIZEX;
    int start_y = get_group_id(1) * GRP_SIZEY;

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local int mag[(GRP_SIZEX + 2) * (GRP_SIZEY + 2)];
    __local short2 sigma[(GRP_SIZEX + 2) * (GRP_SIZEY + 2)];

#pragma unroll
    for (int i = lidx + lidy * GRP_SIZEX; i < (GRP_SIZEX + 2) * (GRP_SIZEY + 2); i += GRP_SIZEX * GRP_SIZEY)
    {
        int x = clamp(start_x - 1 + i % (GRP_SIZEX + 2), 0, cols - 1);
        int y = clamp(start_y - 1 + i / (GRP_SIZEX + 2), 0, rows - 1);

        int dx_index = mad24(y, dx_step, mad24(x, cn * (int)sizeof(short), dx_offset));
        int dy_index = mad24(y, dy_step, mad24(x, cn * (int)sizeof(short), dy_offset));

        __global short *dx = loadpix(dxptr + dx_index);
        __global short *dy = loadpix(dyptr + dy_index);

        int mag0 = dist(dx[0], dy[0]);
#if cn > 1
        short cdx = dx[0], cdy = dy[0];
#pragma unroll
        for (int j = 1; j < cn; ++j)
        {
            int mag1 = dist(dx[j], dy[j]);
            if (mag1 > mag0)
            {
                mag0 = mag1;
                cdx = dx[j];
                cdy = dy[j];
            }
        }
        dx[0] = cdx;
        dy[0] = cdy;
#endif
        mag[i] = mag0;
        sigma[i] = (short2)(dx[0], dy[0]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    if (gidx >= cols || gidy >= rows)
        return;

    lidx++;
    lidy++;

    int mag0 = mag[lidx + lidy * (GRP_SIZEX + 2)];
    short x = (sigma[lidx + lidy * (GRP_SIZEX + 2)]).x;
    short y = (sigma[lidx + lidy * (GRP_SIZEX + 2)]).y;

    int value = 1;
    if (mag0 > low_thr)
    {
        float x_ = abs(x);
        float y_ = abs(y);

        int a = (y_ * TG22 >= x_) ? 2 : 1;
        int b = (y_ * TG67 >= x_) ? 1 : 0;

        int dir3 = (a * b) & (((x ^ y) & 0x80000000) >> 31);
        int dir = a * b + 2 * dir3;
        int prev_mag = mag[(lidy + prev[dir][0]) * (GRP_SIZEX + 2) + lidx + prev[dir][1]];
        int next_mag = mag[(lidy + next[dir][0]) * (GRP_SIZEX + 2) + lidx + next[dir][1]] + (dir & 1);

        if (mag0 > prev_mag && mag0 >= next_mag)
        {
            value = (mag0 > high_thr) ? 2 : 0;
        }
    }

    map[mad24(gidy, map_step, gidx + map_offset)] = value;
}

#undef TG22
#undef CANNY_SHIFT

#elif defined STAGE2
/*
    stage2:
        hysteresis (add edges labeled 0 if they are connected with an edge labeled 2)
*/

#define loadpix(addr) *(__global int *)(addr)
#define storepix(val, addr) *(__global int *)(addr) = (int)(val)
#define ZONE_SIZE_X 120
#define ZONE_SIZE_Y 30
#define LOCAL_TOTAL ((ZONE_SIZE_X + 2) * (ZONE_SIZE_Y + 2))
#define stack_size (4*LOCAL_TOTAL)
//#define total_stack_size (16*stack_size)

__constant short move_dir[2][8] = {
    { -1, -1, -1, 0, 0, 1, 1, 1 },
    { -1, 0, 1, -1, 1, -1, 0, 1 }
};

__kernel void stage2_hysteresis_block(__global uchar *map_ptr, int map_step, int map_offset, int rows, int cols,
                                      int startx, int starty, int endx, int endy, __local ushort2 local_stack[NB_PE][stack_size])
{
    int pe_id = get_local_id(0) * get_local_size(1) + get_local_id(1);
    __local ushort2 *stack = local_stack[pe_id];
    int counter = 0;
    for (int y = starty; y < endy; y++) {
        #pragma unroll
        for (int x = startx; x < endx; x++) {
            __global uchar* map = map_ptr + mad24(y, map_step, x);
            int type = *map;//loadpix(map);
            if (type == 2)
            {
                stack[counter++] = (ushort2)(x, y);
            }
        }
    }
    while (counter != 0) {
        ushort2 pos = stack[--counter];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            ushort posx = pos.x + move_dir[0][j];
            ushort posy = pos.y + move_dir[1][j];
            if (posx < 0 || posy < 0 || posx >= cols || posy >= rows)
                continue;
            __global uchar *addr = map_ptr + mad24(posy, map_step, posx);
            int type = *addr;//loadpix(addr);
            if (type == 0)
            {
                stack[counter++] = (ushort2)(posx, posy);
                *addr = 2;//storepix(2, addr);
            }
        }
    }
}

__kernel void stage2_hysteresis(__global uchar *map_ptr, int map_step, int map_offset, int rows, int cols)
{
    map_ptr += map_offset;

    int startx = get_global_id(0) * SIZE_X;
    int starty = get_global_id(1) * SIZE_Y;
    int endx = startx + SIZE_X;
    int endy = starty + SIZE_Y;
    if (endx > cols) {
        endx = cols;
    }
    if (endy > rows) {
        endy = rows;
    }

    __local ushort2 local_stack[NB_PE][stack_size];

    int nb_x_blocks = (endx - startx + 1) / ZONE_SIZE_X;
    int nb_y_blocks = (endy - starty + 1) / ZONE_SIZE_Y;
    for (int y = starty; y < endy; y += ZONE_SIZE_Y) {
        for (int x = startx; x < endx; x += ZONE_SIZE_X) {
            int local_y_end = y + ZONE_SIZE_Y > endy ? endy : y + ZONE_SIZE_Y;
            int local_x_end = x + ZONE_SIZE_X > endx ? endx : x + ZONE_SIZE_X;
            stage2_hysteresis_block(map_ptr, map_step, map_offset, rows, cols, x, y,
                                    local_x_end, local_y_end, local_stack);
        }
    }
}

#elif defined GET_EDGES

#define SIZEX 256
#define SIZEY 128


// Get the edge result. edge type of value 2 will be marked as an edge point and set to 255. Otherwise 0.
// map      edge type mappings
// dst      edge output

__kernel void getEdgesBlock(__local const uchar *map, __local uchar *dst)
{
    int nb_pe = get_local_size(1) * get_local_size(0);
    int cur_pe = get_local_id(0) * get_local_size(1) + get_local_id(1);

    int startx = 0;
    int endx = SIZEX;

    int starty = cur_pe * SIZEY / nb_pe;
    int endy = (cur_pe + 1) * SIZEY / nb_pe;

    if (endy > SIZEY) {
        endy = SIZEY;
    }

    // Main loop. Transform 2 to 255, and 1 to 0.
    for (int y = starty; y < endy; y++) {
        int index = y * SIZEX;
        int x = 0;
        #pragma unroll
        // Perform the computation on 8 items at a time.
        // Step 1:  Read 8 elements at once from the input.
        //          Example: 0x0201010002020101
        // Step 2a: Shift by 1 to the left. 2 becomes 1, and 1 add 0xf0 to the
        //          next byte.
        //          Example: 0x0100f0f0010100f0
        // Step 2b: Elimination of the 1 that have slipped to the next byte.
        //          This is done by masking on 0x01 for each byte since this
        //          is the only bit we care about.
        //          Example: 0x0100000001010000
        // Step 3:  Filling of the 0x01 bytes. This is done by multiplying them
        //          by 0xff. 0x00 bytes are unchanged by this computation.
        //          Example: 0xff000000ffff0000
        // Step 4:  Store the resulting value.
        for (; x < endx - 8; x += 8) {
            *(__local ulong *)(dst + index) = (((*(__local ulong *)(map + index)) >> 1) & 0x0101010101010101llu) * 0xffllu;
            index += 8;
        }

        #pragma unroll
        // Loop on the last elements and do the calcul byte per byte.
        for (; x < endx; x++) {
            dst[index] = (uchar)(-(map[index] >> 1));
            index++;
        }
    }
}

__kernel void getEdges(__global const uchar *src, int src_step, int src_offset, int rows, int cols,
                                __global uchar *map, int map_step, int map_offset)
{
    __local uchar  smem_src_even[SIZEX * SIZEY];
    __local uchar  smem_src_odd [SIZEX * SIZEY];
    __local uchar *smem_src[2] = {smem_src_even, smem_src_odd};

    __local uchar map_out_even[SIZEX * SIZEY];
    __local uchar map_out_odd [SIZEX * SIZEY];
    __local uchar *map_out[2] = {map_out_even, map_out_odd};

    const int lsizex = get_local_size(0);
    const int lsizey = get_local_size(1);
    const int lidx = get_local_id(0);
    const int lidy = get_local_id(1);

    // linearized workitem id in workgroup
    const int wid = lidx + lidy * lsizex;

    const int num_groups = get_num_groups(0) * get_num_groups(1);
    const int group_id = get_group_id(0) + (get_group_id(1) * get_num_groups(0));

    const int num_blocks_x = (int)ceil(((float)cols) / SIZEX);
    const int num_blocks_y = (int)ceil(((float)rows) / SIZEY);

    const int num_blocks_per_group = (num_blocks_x * num_blocks_y) / num_groups;
    const int num_blocks_trailing  = (num_blocks_x * num_blocks_y) % num_groups;

    const int iblock_begin = group_id * num_blocks_per_group + min(group_id, num_blocks_trailing);
    const int iblock_end   = iblock_begin + num_blocks_per_group + ((group_id < num_blocks_trailing) ? 1 : 0);

    // if (lidx == 0 && lidy == 0) {
    //     printf("Group %3d doing blocks from %3d to %3d = %3d blocks\n",
    //         group_id, iblock_begin, iblock_end, (iblock_end - iblock_begin));
    // }

    int iblock_x_next = iblock_begin % num_blocks_x;
    int iblock_y_next = iblock_begin / num_blocks_x;

    event_t event_read[2]  = {0, 0};
    event_t event_write[2] = {0, 0};

    // block to copy
    int2 block_size   = {SIZEX, SIZEY};

    // local write position and dimension
    int4 local_point  = {0, 0, SIZEX, SIZEY};

    // global read position and dimension
    int4 global_point = {(SIZEX * iblock_x_next),
                         (SIZEY * iblock_y_next),
                         src_step,
                         rows};

    // copy the 2D block from src image into smem_src
    event_read[iblock_begin & 1] = async_work_group_copy_block_2D2D(
        (smem_src[iblock_begin & 1]),
        (src + src_offset),
        1,             // num_gentypes_per_pixel
        block_size,    // block_size
        local_point,   // local_point
        global_point,  // global_point
        0              // event
    );

    for (int iblock = iblock_begin; iblock < iblock_end; iblock++)
    {
        // =========================================================
        // current block to be processed
        // =========================================================
        const int iblock_x = iblock_x_next;
        const int iblock_y = iblock_y_next;
        const int iblock_parity = iblock & 1;

        // start coordinates of output block in the output image
        const int start_x = SIZEX * iblock_x;
        const int start_y = SIZEY * iblock_y;

        // =========================================================
        // prefetch next block (if any)
        // =========================================================
        const int iblock_next = iblock + 1;
        iblock_x_next = iblock_next % num_blocks_x;
        iblock_y_next = iblock_next / num_blocks_x;

        const int2 block_size_next = {SIZEX, SIZEY};
        const int2 local_pos_next  = {0, 0};
        const int2 global_pos_next = {(SIZEX * iblock_x_next),
                                      (SIZEY * iblock_y_next)};
        // only prefetch if we still have work
        if (iblock_next < iblock_end)
        {
            const int iblock_next_parity = iblock_next & 1;

            int4 local_point_next  = {local_pos_next.x, local_pos_next.y,
                                      local_point.z, local_point.w};
            int4 global_point_next = {global_pos_next.x, global_pos_next.y,
                                      global_point.z, global_point.w};

            event_read[iblock_next_parity] = async_work_group_copy_block_2D2D(
                                    (smem_src[iblock_next_parity]),
                                    (src + src_offset),
                                    1,                  // num_gentypes_per_pixel
                                    block_size_next,    // block_size
                                    local_point_next,   // local_point
                                    global_point_next,  // global_point
                                    0                   // event
                                 );
        }

        // update async copy info to the next block
        block_size      = block_size_next;
        local_point.xy  = local_pos_next;
        global_point.xy = global_pos_next;

        // =========================================================
        // wait for current block
        // =========================================================
        wait_group_events(1, &event_read[iblock_parity]);

        // =========================================================
        // wait for previous put of the 2D block from local to global to avoid
        // data race: writing result to a being-put buffer
        // =========================================================
        if (iblock - iblock_begin > 1) {
            wait_group_events(1, &event_write[iblock_parity]);
        }

        // =========================================================
        // now compute the current block
        // =========================================================
        getEdgesBlock(smem_src[iblock_parity], map_out[iblock_parity]);
        // sync to gather output from all WI into map_out[] before putting to global memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // =========================================================
        // put result to global memory
        // =========================================================
        event_write[iblock_parity] = async_work_group_copy_block_2D2D(
                                (map + map_offset),
                                (map_out[iblock_parity]),
                                1,                      // num_gentypes_per_pixel
                                (int2){clamp(SIZEX, SIZEX, (cols - start_x)),
                                       clamp(SIZEY, SIZEY, (rows - start_y))}, // block_size
                                (int4){0, 0, SIZEX, SIZEY},  // local_point
                                (int4){iblock_x * SIZEX,
                                       iblock_y * SIZEY,
                                       (map_step),
                                       rows},                        // global_point
                                0                                    // event
                             );
    }  // for (int iblock = iblock_begin; iblock < iblock_end; iblock++)

    async_work_group_copy_fence(CLK_GLOBAL_MEM_FENCE);
}

#endif


// ============================================================================
#endif  // KALRAY_DEVICE
// ============================================================================
