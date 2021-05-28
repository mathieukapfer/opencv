// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#define DIG(a) a,
__constant float kx[] = { KERNEL_MATRIX_X };
__constant float ky[] = { KERNEL_MATRIX_Y };

#define OP(delta, y, x) (convert_float16(arr[(y + delta) * 3 + x]) * ky[y] * kx[x])

#ifndef KALRAY_BYPASS_ASYNC
// Kalray async block_2D2D extension
#define IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE)                              \
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

#define IMPLEMENT_ASYNC_COPY_2D2D_FUNCS(GENTYPE)             \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE)            \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##2)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##3)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##4)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##8)         \
  IMPLEMENT_ASYNC_COPY_2D2D_FUNCS_SINGLE(GENTYPE##16)

IMPLEMENT_ASYNC_COPY_2D2D_FUNCS(uchar);
IMPLEMENT_ASYNC_COPY_2D2D_FUNCS(int);
IMPLEMENT_ASYNC_COPY_2D2D_FUNCS(float);
#endif /* !KALRAY_BYPASS_ASYNC */

__attribute__((mppa_native))
void conv_u8_factor_u8_3x3_naive(
    __local const uchar* src, __local uchar* dst,
    __constant float *kx, __constant float *ky,
    int start_line, int end_line, int width);

__attribute__((mppa_native))
void conv_u8_factor_u8_3x3(
    __local const uchar* src, __local uchar* dst,
    __constant float *kx, __constant float *ky,
    int start_line, int end_line, int width);


// https://hal.univ-grenoble-alpes.fr/hal-01652614/document
static inline int local_offset(const int iblock, const int num_blocks)
{
    return (iblock == 0 ? 1 : 0);
}
static inline int remote_offset(const int iblock, const int num_blocks)
{
    return (iblock == 0 ? 0 : -1);
}
static inline int halo_cutoff(const int iblock, const int num_blocks)
{
    if (iblock != 0 && iblock != num_blocks - 1) {
        return 0;
    }
    if (num_blocks == 1) {
        return -2;
    }
    return -1;
}

__kernel void gaussianBlur3x3_8UC1_cols16_rows2(__global const uchar *src, int src_step,
                                                __global uchar *dst, int dst_step,
                                                int rows, int cols)
{
#ifdef HAVE_PAPI
    long start_cycle = PAPI_get_real_cyc();
#endif


    //                      SIZEX+2
    //         +--------------------------------+
    //         |  a   a b                       |
    //         |    +----------------------+    |
    //         |  a | a b    SIZEX         |    |
    //         |  c | c                    |    |
    //         |    |                      |    |
    // SIZEY+2 |    | SIZEY                |    |
    //         |    |                      |    |
    //         |    |                      |    |
    //         |    +----------------------+    |
    //         |                                |
    //         +--------------------------------+
    //

// TODO: Maybe this could be replaced with a host-set variable.
// This would however lead to several kernel versions depending
// on the configuration.
#define BUF_SIZE 250000
    __local uchar smem_src[2][BUF_SIZE];
    __local uchar smem_out[2][BUF_SIZE];

    // Warning: We need to account for the beginning and end of the
    // lines, as well as remove the two border lines.
    int fittable_lines = BUF_SIZE / (cols + 2) - 2;
    int SIZEX = 2000;
    int SIZEY = 120;

    // We do the work split vertically. We should only expand
    // horizontally if there are at lease two lines per PE.
    if (fittable_lines > 32) {
        SIZEY = fittable_lines;
        SIZEX = cols;

    }

    // If the image is too small to the point where there is not
    // enough work for each cluster, handle less lines at once to
    // share the work better.
    if (SIZEY > rows / 5) {
        SIZEY = (rows + 4) / 5;
    }

    // Ensure alignment
    SIZEX = SIZEX / 32 * 32;
    if (SIZEX == 0) {
        SIZEX = 32;
    }
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

    int iblock_x_next = iblock_begin % num_blocks_x;
    int iblock_y_next = iblock_begin / num_blocks_x;

    event_t event_read[2]  = {0, 0};
    event_t event_write[2] = {0, 0};

    // block to copy
    // clamp if any last block exceeds the remaining size
    // Assumption: there are at least >= 2 blocks in each row and col dimension
    const int2 first_block_output = {clamp(SIZEX, SIZEX, cols - (SIZEX * iblock_x_next)),
                                     clamp(SIZEY, SIZEY, rows - (SIZEY * iblock_y_next))};
    int2 block_size = {(first_block_output.x + 2) + halo_cutoff(iblock_x_next, num_blocks_x),
                       (first_block_output.y + 2) + halo_cutoff(iblock_y_next, num_blocks_y)};

    // local write position and dimension
    int4 local_point  = {0 + local_offset(iblock_x_next, num_blocks_x),
                         0 + local_offset(iblock_y_next, num_blocks_y),
                         (SIZEX + 2),
                         (SIZEY + 2)};

    // global read position and dimension
    int4 global_point = {(SIZEX * iblock_x_next) + remote_offset(iblock_x_next, num_blocks_x),
                         (SIZEY * iblock_y_next) + remote_offset(iblock_y_next, num_blocks_y),
                         src_step / (sizeof(uchar)),
                         rows};

    // Guard against images that are too small for a split between the 5
    // clusters.
    if (global_point[0] >= cols || global_point[1] >= rows) {
        return;
    }

    int start_line = wid * SIZEY / (lsizex * lsizey);
    int end_line = (wid + 1) * SIZEY / (lsizex * lsizey);
#ifndef KALRAY_BYPASS_ASYNC
    event_read[iblock_begin & 1] = async_work_group_copy_block_2D2D(
        (__local uchar *)(smem_src[iblock_begin & 1]),
        (const __global uchar *)src,
        1,             // num_gentypes_per_pixel
        block_size,    // block_size
        local_point,   // local_point
        global_point,  // global_point
        0              // event
    );
#else
    {
        int transfer_nb_lines = block_size.y;
        int transfer_start_line = wid * transfer_nb_lines / (lsizex * lsizey);
        int transfer_end_line = (wid + 1) * transfer_nb_lines / (lsizex * lsizey);
        __global uchar *cur_pe_src = src + global_point.y * global_point.z + global_point.x;
        for (int y = transfer_start_line; y < transfer_end_line; y++) {
            __global uchar *cur_pe_line_src = cur_pe_src + y * global_point.z;
            __local uchar *cur_pe_line_dst = smem_src[iblock_begin & 1] + local_point.z * (y + local_point.y) + local_point.x;
            for (int x = 0; x < block_size.x; x++) {
                cur_pe_line_dst[x] = cur_pe_line_src[x];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif

#ifdef HAVE_PAPI
    long prologue_cycles = PAPI_get_real_cyc() - start_cycle;
    long loop_start_cycle = PAPI_get_real_cyc();

    long iter_prologue_cycles = 0;
    long async_write_wait_cycles = 0;
    long async_read_wait_cycles = 0;
    long async_enqueue_cycles = 0;
    long padding_cycles = 0;
    long gauss_cycles = 0;

    long tmp_counter = PAPI_get_real_cyc();
    long prev_tmp_counter = PAPI_get_real_cyc();
#endif
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

        // clamp if any last block exceeed the remaining size
        const int2 block_output_next = {clamp(SIZEX, SIZEX, cols - (SIZEX * iblock_x_next)),
                                        clamp(SIZEY, SIZEY, rows - (SIZEY * iblock_y_next))};

        const int2 block_size_next = {(block_output_next.x + 2)
                                       + halo_cutoff(iblock_x_next, num_blocks_x),
                                      (block_output_next.y + 2)
                                       + halo_cutoff(iblock_y_next, num_blocks_y)};

        const int2 local_pos_next  = {0 + local_offset(iblock_x_next, num_blocks_x),
                                      0 + local_offset(iblock_y_next, num_blocks_y)};

        const int2 global_pos_next = {(SIZEX * iblock_x_next)
                                       + remote_offset(iblock_x_next, num_blocks_x),
                                      (SIZEY * iblock_y_next)
                                       + remote_offset(iblock_y_next, num_blocks_y)};

#ifdef HAVE_PAPI
        prev_tmp_counter = tmp_counter;
        tmp_counter = PAPI_get_real_cyc();
        iter_prologue_cycles += tmp_counter - prev_tmp_counter;
#endif
        // only prefetch if we still have work
        if (iblock_next < iblock_end)
        {
            const int iblock_next_parity = iblock_next & 1;

            int4 local_point_next  = {local_pos_next.x, local_pos_next.y,
                                      local_point.z, local_point.w};
            int4 global_point_next = {global_pos_next.x, global_pos_next.y,
                                      global_point.z, global_point.w};
#ifndef KALRAY_BYPASS_ASYNC
            event_read[iblock_next_parity] = async_work_group_copy_block_2D2D(
                                    (__local uchar *)(smem_src[iblock_next_parity]),
                                    (const __global uchar *)src,
                                    1,                  // num_gentypes_per_pixel
                                    block_size_next,    // block_size
                                    local_point_next,   // local_point
                                    global_point_next,  // global_point
                                    0                   // event
                                 );
#ifdef HAVE_PAPI
            prev_tmp_counter = tmp_counter;
            tmp_counter = PAPI_get_real_cyc();
            async_enqueue_cycles += tmp_counter - prev_tmp_counter;
#endif
#else
            {
                int transfer_nb_lines = block_size_next.y;
                int transfer_start_line = wid * transfer_nb_lines / (lsizex * lsizey);
                int transfer_end_line = (wid + 1) * transfer_nb_lines / (lsizex * lsizey);
                __global uchar *cur_pe_src = src + global_point_next.y * global_point_next.z + global_point_next.x;
                for (int y = transfer_start_line; y < transfer_end_line; y++) {
                    __global uchar *cur_pe_line_src = cur_pe_src + y * global_point_next.z;
                    __local uchar *cur_pe_line_dst = smem_src[iblock_next_parity] + local_point_next.z * (y + local_point_next.y) + local_point_next.x;
                    for (int x = 0; x < block_size_next.x; x++) {
                        cur_pe_line_dst[x] = cur_pe_line_src[x];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
#ifdef HAVE_PAPI
                prev_tmp_counter = tmp_counter;
                tmp_counter = PAPI_get_real_cyc();
                async_read_wait_cycles += tmp_counter - prev_tmp_counter;
#endif
            }
#endif
        }

        // =========================================================
        // wait for current block
        // =========================================================
        wait_group_events(1, &event_read[iblock_parity]);

#ifdef HAVE_PAPI
            prev_tmp_counter = tmp_counter;
            tmp_counter = PAPI_get_real_cyc();
            async_read_wait_cycles += tmp_counter - prev_tmp_counter;
#endif

        // padding
        if (local_point.y != 0) {
            int src_idx = local_point.y * (SIZEX + 2);
            for (int icol = 0; icol < (SIZEX + 2); icol++) {
                uchar pixel_src = smem_src[iblock_parity][src_idx + icol];
                smem_src[iblock_parity][icol] = pixel_src;
            }
        }

        if ((local_point.y + block_size.y) != (SIZEY + 2)) {
            int src_idx = ((local_point.y + block_size.y) - 1) * (SIZEX + 2);
            int dst_idx = src_idx + (SIZEX + 2);

            for (int icol = 0; icol < (SIZEX + 2); icol++) {
                uchar pixel_src = smem_src[iblock_parity][src_idx + icol];
                smem_src[iblock_parity][dst_idx + icol] = pixel_src;
            }
        }

        if (local_point.x != 0) {
            for (int irow = 0; irow < (SIZEY + 2); irow++) {
                int src_idx = irow * (SIZEX + 2) + 1;
                int dst_idx = irow * (SIZEX + 2);
                uchar pixel_src = smem_src[iblock_parity][src_idx];
                smem_src[iblock_parity][dst_idx] = pixel_src;
            }
        }

        if ((local_point.x + block_size.x) != (SIZEX + 2)) {
            for (int irow = 0; irow < (SIZEY + 2); irow++) {
                int src_idx = irow * (SIZEX + 2) + (local_point.x + block_size.x) - 1;
                int dst_idx = src_idx + 1;
                uchar pixel_src = smem_src[iblock_parity][src_idx];
                smem_src[iblock_parity][dst_idx] = pixel_src;
            }
        }

        // sync to gather copy from all WI into smem[] before the compute step
        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef HAVE_PAPI
            prev_tmp_counter = tmp_counter;
            tmp_counter = PAPI_get_real_cyc();
            padding_cycles += tmp_counter - prev_tmp_counter;
#endif

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
#ifdef HAVE_PAPI
            prev_tmp_counter = tmp_counter;
            tmp_counter = PAPI_get_real_cyc();
            async_write_wait_cycles += tmp_counter - prev_tmp_counter;
#endif
        }

        // =========================================================
        // now compute the current block
        // =========================================================
#if USE_NAIVE_CONVOLUTION
        conv_u8_factor_u8_3x3_naive(smem_src[iblock_parity], smem_out[iblock_parity], kx, ky, start_line, end_line, SIZEX);
#else
        conv_u8_factor_u8_3x3(smem_src[iblock_parity], smem_out[iblock_parity], kx, ky, start_line, end_line, SIZEX);
#endif

        // sync to gather output from all WI into smem_out[] before putting to global memory
        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef HAVE_PAPI
        prev_tmp_counter = tmp_counter;
        tmp_counter = PAPI_get_real_cyc();
        gauss_cycles += tmp_counter - prev_tmp_counter;
#endif

        // =========================================================
        // put result to global memory
        // =========================================================
#ifndef KALRAY_BYPASS_ASYNC
        if (SIZEX != cols) {
            event_write[iblock_parity] = async_work_group_copy_block_2D2D(
                (__global uchar *)dst,
                (const __local uchar *)(smem_out[iblock_parity]),
                1,
                (int2){clamp(SIZEX, SIZEX, (cols - start_x)),
                       clamp(SIZEY, SIZEY, (rows - start_y))}, // block_size
                (int4){0, 0, SIZEX, SIZEY},  // local_point
                (int4){iblock_x * SIZEX,
                       iblock_y * SIZEY,
                       (dst_step),
                       rows},                        // global_point
                0                                    // event
            );
        } else {
            event_write[iblock_parity] = async_work_group_copy(
                ((__global uchar *)dst) + SIZEY * dst_step * iblock_y,
                smem_out[iblock_parity], SIZEX * clamp(SIZEY, SIZEY, (rows - start_y)), 0
            );
        }
#ifdef HAVE_PAPI
        prev_tmp_counter = tmp_counter;
        tmp_counter = PAPI_get_real_cyc();
        async_enqueue_cycles += tmp_counter - prev_tmp_counter;
#endif
#else
        {
            __global uchar *cur_pe_dst = dst + iblock_y * SIZEY * dst_step + iblock_x * SIZEX;
            for (int y = start_line; y < clamp(end_line, end_line, (rows - start_y)); y++) {
                __global uchar *cur_pe_line_dst = cur_pe_dst + y * dst_step;
                __local uchar *cur_pe_line_src = smem_out[iblock_parity] + SIZEX * y;
                for (int x = 0; x < clamp(SIZEX, SIZEX, (cols - start_x)); x++) {
                    cur_pe_line_dst[x] = cur_pe_line_src[x];
                }
            }
        }

#ifdef HAVE_PAPI
        prev_tmp_counter = tmp_counter;
        tmp_counter = PAPI_get_real_cyc();
        async_write_wait_cycles += tmp_counter - prev_tmp_counter;
#endif
#endif
    }

    //async_work_group_copy_fence(CLK_GLOBAL_MEM_FENCE);
#ifndef KALRAY_BYPASS_ASYNC
    wait_group_events(1, &event_write[(iblock_end - 1) & 1]);
#endif


#ifdef HAVE_PAPI
    prev_tmp_counter = tmp_counter;
    tmp_counter = PAPI_get_real_cyc();
    async_enqueue_cycles += tmp_counter - prev_tmp_counter;
    long loop_cycles = tmp_counter - loop_start_cycle;
    long total_cycles = loop_cycles + prologue_cycles;
    long computation_cycles = iter_prologue_cycles + padding_cycles + gauss_cycles + prologue_cycles;
    printf("loop prologue: %-8d read: %-9d write: %-8d enqueue: %-8d padding: %-8d gauss: %-9d | init: %-6d loop: %-9d | Computation part: %d%% Gauss: %2d%% | Computation time: %3.2fms\n",
           iter_prologue_cycles, async_read_wait_cycles, async_write_wait_cycles,
           async_enqueue_cycles, padding_cycles, gauss_cycles, prologue_cycles, tmp_counter - loop_start_cycle,
           100 * computation_cycles / total_cycles,
           100 * gauss_cycles / total_cycles, computation_cycles / 1000000.0);
#endif
}
