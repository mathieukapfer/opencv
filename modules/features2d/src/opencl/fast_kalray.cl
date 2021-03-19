// OpenCL port of the FAST corner detector.
// Copyright (C) 2014, Itseez Inc. See the license at http://opencv.org

#define HALO_SIZE 3
#define IN_WIDTH (GRP_SIZEX + (2 * HALO_SIZE))
#define IN_HEIGHT (GRP_SIZEY + (2 * HALO_SIZE))
#define IN_OFFSET_Y(cur, dist) ((cur) + (dist) * IN_WIDTH)

#define CONTIGUOUS_POINTS 9

static inline void fast_compute_block(
    __local uchar *smem,
    volatile __global int* kp_loc,
    __local int* nb_kp,
    const int group_id,
    const int num_kp_groups,
    int threshold,
    int4 global_point,
    int2 block_size)
{
/**
 * FAST algorithm
 * |    |    | 15 | 0 | 1 |   |   |
 * |    | 14 |    |   |   | 2 |   |
 * | 13 |    |    |   |   |   | 3 |
 * | 12 |    |    | p |   |   | 4 |
 * | 11 |    |    |   |   |   | 5 |
 * |    | 10 |    |   |   | 6 |   |
 * |    |    | 9  | 8 | 7 |   |   |
 *
 **/
    int nb_pe = get_local_size(1) * get_local_size(0);
    int cur_pe = get_local_id(0) * get_local_size(1) + get_local_id(1);

    int startrow = (block_size.y - 2*HALO_SIZE) * cur_pe / nb_pe;
    int endrow =  (block_size.y - 2*HALO_SIZE) * (cur_pe + 1) / nb_pe;

    int halo_points[16];

    for (int roi_y = startrow; roi_y < endrow; roi_y++)
    {
        const int block_y = roi_y + HALO_SIZE;

        for (int roi_x = 0; roi_x < (block_size.x - 2*HALO_SIZE); roi_x++)
        {
            const int block_x = roi_x + HALO_SIZE;

            // Get the pixels of interest
            int p = smem[IN_OFFSET_Y(block_x,block_y)];

            int p_high = p + threshold;
            int p_low = p - threshold;

            // Get the north and south points
            halo_points[0] = smem[IN_OFFSET_Y(block_x,block_y-3)];
            halo_points[8] = smem[IN_OFFSET_Y(block_x,block_y+3)];

            // High speed tests
            // Is our pixel of interest (POI) brighter/darker than at least 2 points?
            int brighter = (0);
            int darker = (0);

            #define UPDATE_MASK(id, pixel) \
                brighter |= ((pixel > p_high) << id); \
                darker |= ((pixel < p_low) << id)

            UPDATE_MASK(0, halo_points[0]);
            UPDATE_MASK(8, halo_points[8]);

            // If our north and south points are in the range of our POI,
            // the later cannot be a border.
            if ((brighter | darker) == 0)
            {
                continue;
            }

            // Get the est and west points
            halo_points[4] = smem[IN_OFFSET_Y(block_x+3,block_y)];
            halo_points[12] = smem[IN_OFFSET_Y(block_x-3,block_y)];

            UPDATE_MASK(4, halo_points[4]);
            UPDATE_MASK(12, halo_points[12]);

            // To have CONTIGUOUS_POINTS contiguous pixels of our halo to be brighter or darker,
            // we need at least `cardinal_points` cardinal points to be all brighter or all darker.
            int cardinal_points = 2;
            if ((popcount(brighter) < cardinal_points) &&
                (popcount(darker) < cardinal_points))
            {
                continue;
            }

            halo_points[1] = smem[IN_OFFSET_Y(block_x+1,block_y-3)];
            halo_points[2] = smem[IN_OFFSET_Y(block_x+2,block_y-2)];
            halo_points[3] = smem[IN_OFFSET_Y(block_x+3,block_y-1)];
            halo_points[5] = smem[IN_OFFSET_Y(block_x+3,block_y+1)];
            halo_points[6] = smem[IN_OFFSET_Y(block_x+2,block_y+2)];
            halo_points[7] = smem[IN_OFFSET_Y(block_x+1,block_y+3)];
            halo_points[9] = smem[IN_OFFSET_Y(block_x-1,block_y+3)];
            halo_points[10] = smem[IN_OFFSET_Y(block_x-2,block_y+2)];
            halo_points[11] = smem[IN_OFFSET_Y(block_x-3,block_y+1)];
            halo_points[13] = smem[IN_OFFSET_Y(block_x-3,block_y-1)];
            halo_points[14] = smem[IN_OFFSET_Y(block_x-2,block_y-2)];
            halo_points[15] = smem[IN_OFFSET_Y(block_x-1,block_y-3)];

            UPDATE_MASK(1, halo_points[1]);
            UPDATE_MASK(2, halo_points[2]);
            UPDATE_MASK(3, halo_points[3]);
            UPDATE_MASK(5, halo_points[5]);
            UPDATE_MASK(6, halo_points[6]);
            UPDATE_MASK(7, halo_points[7]);
            UPDATE_MASK(9, halo_points[9]);
            UPDATE_MASK(10, halo_points[10]);
            UPDATE_MASK(11, halo_points[11]);
            UPDATE_MASK(13, halo_points[13]);
            UPDATE_MASK(14, halo_points[14]);
            UPDATE_MASK(15, halo_points[15]);

            int mask = 0x1FF;
            #define CHECK(shift) \
                (((brighter & (mask << shift)) == (mask << shift)) | \
                ((darker & (mask << shift)) == (mask << shift)))

            if ((popcount(brighter) >= CONTIGUOUS_POINTS) |
                (popcount(darker) >= CONTIGUOUS_POINTS))
            {
                brighter |= (brighter << 16);
                darker |= (darker << 16);

                if (CHECK(0) + CHECK(1) + CHECK(2) + CHECK(3) + CHECK(4) +
                    CHECK(5) + CHECK(6) + CHECK(7) + CHECK(8) + CHECK(9) +
                    CHECK(10) + CHECK(11) + CHECK(12) + CHECK(13) + CHECK(14) +
                    CHECK(15))
                {
                    int index = atomic_inc(nb_kp);
                    if (index > num_kp_groups)
                    {
                        return;
                    }
                    // Add the location of the keypoint
                    // N clusters can write a maximum of num_kp_groups keypoints each.
                    kp_loc[1 + 2*index + (num_kp_groups*2)*group_id] = global_point.x + block_x;
                    kp_loc[2 + 2*index + (num_kp_groups*2)*group_id] = global_point.y + block_y;
                }
            }

        } // for (int roi_x = 0; roi_x < (block_size.x - 2*HALO_SIZE); roi_x++)

    } // for (int roi_y = startrow; roi_y < endrow; roi_y++)

}

static inline int local_offset(const int iblock, const int num_blocks)
{
    // We do not handle the border of the image.
    return 0;
}

static inline int remote_offset(const int iblock, const int num_blocks)
{
    return (iblock == 0 ? 0 : -HALO_SIZE);
}

static inline int halo_cutoff(const int iblock, const int num_blocks)
{
    return ((iblock > 0 && iblock < (num_blocks - 1)) ? 0 : -HALO_SIZE);
}

__kernel
void FAST_findKeypoints(
    __global const uchar * _img, int step, int img_offset,
    int img_rows, int img_cols,
    volatile __global int* kp_loc,
    int num_kp_groups, int threshold )
{
    __local uchar smem_src_even[IN_WIDTH * IN_HEIGHT];
    __local uchar smem_src_odd[IN_WIDTH * IN_HEIGHT];
    __local uchar *smem_src[2] = {smem_src_even, smem_src_odd};

    // Number of Keypoints found by a Cluster.
    __local int nb_kp;
    nb_kp = 0;

    const int num_groups = get_num_groups(0) * get_num_groups(1);
    const int group_id = get_group_id(0) + (get_group_id(1) * get_num_groups(0));

    const int num_blocks_x = (int)ceil(((float)img_cols) / GRP_SIZEX);
    const int num_blocks_y = (int)ceil(((float)img_rows) / GRP_SIZEY);

    const int num_blocks_per_group = (num_blocks_x * num_blocks_y) / num_groups;
    const int num_blocks_trailing = (num_blocks_x * num_blocks_y) % num_groups;

    const int iblock_begin = group_id * num_blocks_per_group + min(group_id, num_blocks_trailing);
    const int iblock_end = iblock_begin + num_blocks_per_group + ((group_id < num_blocks_trailing) ? 1 : 0);

    int iblock_x_next = iblock_begin % num_blocks_x;
    int iblock_y_next = iblock_begin / num_blocks_x;

    event_t event_read[2] = {0, 0};

    // Block to copy
    // Clamp if any last block exceeds the remaining size.
    // Assumption: There are at least >= 2 blocks in each row and col dimension.
    const int2 block_output_first =
        {clamp(GRP_SIZEX, GRP_SIZEX, img_cols - (GRP_SIZEX * iblock_x_next)),
         clamp(GRP_SIZEY, GRP_SIZEY, img_rows - (GRP_SIZEY * iblock_y_next))};

    int2 block_size =
        {block_output_first.x + (2 * HALO_SIZE) + halo_cutoff(iblock_x_next, num_blocks_x),
         block_output_first.y + (2 * HALO_SIZE) + halo_cutoff(iblock_y_next, num_blocks_y)};

    int4 local_point = {0 + local_offset(iblock_x_next, num_blocks_x),
                        0 + local_offset(iblock_y_next, num_blocks_y),
                        IN_WIDTH,
                        IN_HEIGHT};

    int4 global_point = {(GRP_SIZEX * iblock_x_next) + remote_offset(iblock_x_next, num_blocks_x),
                         (GRP_SIZEY * iblock_y_next) + remote_offset(iblock_y_next, num_blocks_y),
                         step,
                         img_rows};

    // Copy the 2d block from img into smem
    event_read[iblock_begin & 1] = async_work_group_copy_block_2D2D(
        (smem_src[iblock_begin & 1]),
        (_img + img_offset),
        1,
        block_size,
        local_point,
        global_point,
        0
    );

    for (int iblock = iblock_begin; iblock < iblock_end; iblock++)
    {
        //========================================================
        // Current block to be processed
        //========================================================
        const int iblock_x = iblock_x_next;
        const int iblock_y = iblock_y_next;
        const int iblock_parity = iblock & 1;

        //========================================================
        // Prefetch next block (if any)
        //========================================================
        const int iblock_next = iblock + 1;
        iblock_x_next = iblock_next % num_blocks_x;
        iblock_y_next = iblock_next / num_blocks_x;

        // clamp if any last block exceeds the remaining size
        // Assumption: there are at least >= 2 blocks in each row and col dimension
        const int2 block_output_next =
            {clamp(GRP_SIZEX, GRP_SIZEX, img_cols - (GRP_SIZEX * iblock_x_next)),
             clamp(GRP_SIZEY, GRP_SIZEY, img_rows - (GRP_SIZEY * iblock_y_next))};
        const int2 block_size_next =
            {(block_output_next.x + (2 * HALO_SIZE)) + halo_cutoff(iblock_x_next, num_blocks_x),
             (block_output_next.y + (2 * HALO_SIZE)) + halo_cutoff(iblock_y_next, num_blocks_y)};

        const int2 local_pos_next  = {0 + local_offset(iblock_x_next, num_blocks_x),
                                      0 + local_offset(iblock_y_next, num_blocks_y)};
        const int2 global_pos_next =
            {(GRP_SIZEX * iblock_x_next) + remote_offset(iblock_x_next, num_blocks_x),
             (GRP_SIZEY * iblock_y_next) + remote_offset(iblock_y_next, num_blocks_y)};

        // Only prefetch if we still have work
        if (iblock_next < iblock_end)
        {
            const int iblock_next_parity = iblock_next & 1;

            int4 local_point_next = {local_pos_next.x, local_pos_next.y,
                                     local_point.z, local_point.w};
            int4 global_point_next = {global_pos_next.x, global_pos_next.y,
                                      global_point.z, global_point.w};

            event_read[iblock_next_parity] = async_work_group_copy_block_2D2D(
                (smem_src[iblock_next_parity]),
                (_img + img_offset),
                1,
                block_size_next,
                local_point_next,
                global_point_next,
                0
            );
        }

        //========================================================
        // Wait for current block
        //========================================================
        wait_group_events(1, &event_read[iblock_parity]);

        //========================================================
        // Compute the current block
        //========================================================
        fast_compute_block(smem_src[iblock_parity],
                           kp_loc,
                           &nb_kp,
                           group_id,
                           num_kp_groups,
                           threshold,
                           global_point,
                           block_size);

        barrier(CLK_LOCAL_MEM_FENCE);

        // After padding & barrier, update async copy info to the next block,
        // used by the padding of next iteration
        block_size = block_size_next;
        local_point.xy = local_pos_next;
        global_point.xy = global_pos_next;

    }

    if ((get_local_id(0) == get_local_id(1) == get_local_id(2) == 0) &&
        (get_group_id(0) == (num_groups - 1)))
    {
        // Set the number of key point found
        kp_loc[0] = (num_groups - 1) * num_kp_groups + nb_kp;
    }
    async_work_group_copy_fence(CLK_GLOBAL_MEM_FENCE);
}

///////////////////////////////////////////////////////////////////////////
// nonmaxSupression

inline int cornerScore(__global const uchar* img, int step)
{
    int k, tofs, v = img[0], a0 = 0, b0;
    int d[16];
    #define LOAD2(idx, ofs) \
        tofs = ofs; d[idx] = (short)(v - img[tofs]); d[idx+8] = (short)(v - img[-tofs])
    LOAD2(0, 3);
    LOAD2(1, -step+3);
    LOAD2(2, -step*2+2);
    LOAD2(3, -step*3+1);
    LOAD2(4, -step*3);
    LOAD2(5, -step*3-1);
    LOAD2(6, -step*2-2);
    LOAD2(7, -step-3);

    #pragma unroll
    for( k = 0; k < 16; k += 2 )
    {
        int a = min((int)d[(k+1)&15], (int)d[(k+2)&15]);
        a = min(a, (int)d[(k+3)&15]);
        a = min(a, (int)d[(k+4)&15]);
        a = min(a, (int)d[(k+5)&15]);
        a = min(a, (int)d[(k+6)&15]);
        a = min(a, (int)d[(k+7)&15]);
        a = min(a, (int)d[(k+8)&15]);
        a0 = max(a0, min(a, (int)d[k&15]));
        a0 = max(a0, min(a, (int)d[(k+9)&15]));
    }

    b0 = -a0;
    #pragma unroll
    for( k = 0; k < 16; k += 2 )
    {
        int b = max((int)d[(k+1)&15], (int)d[(k+2)&15]);
        b = max(b, (int)d[(k+3)&15]);
        b = max(b, (int)d[(k+4)&15]);
        b = max(b, (int)d[(k+5)&15]);
        b = max(b, (int)d[(k+6)&15]);
        b = max(b, (int)d[(k+7)&15]);
        b = max(b, (int)d[(k+8)&15]);

        b0 = min(b0, max(b, (int)d[k]));
        b0 = min(b0, max(b, (int)d[(k+9)&15]));
    }

    return -b0-1;
}

__kernel
void FAST_nonmaxSupression(
    __global const int* kp_in, volatile __global int* kp_out,
    __global const uchar * _img, int step, int img_offset,
    int rows, int cols, int counter, int max_keypoints)
{
    const int idx = get_global_id(0);

    if (idx < counter)
    {
        int x = kp_in[1 + 2*idx];
        int y = kp_in[2 + 2*idx];

        __global const uchar* img = _img + mad24(y, step, x + img_offset);

        int s = cornerScore(img, step);

        if( (x < 4 || s > cornerScore(img-1, step)) +
            (y < 4 || s > cornerScore(img-step, step)) != 2 )
            return;
        if( (x >= cols - 4 || s > cornerScore(img+1, step)) +
            (y >= rows - 4 || s > cornerScore(img+step, step)) +
            (x < 4 || y < 4 || s > cornerScore(img-step-1, step)) +
            (x >= cols - 4 || y < 4 || s > cornerScore(img-step+1, step)) +
            (x < 4 || y >= rows - 4 || s > cornerScore(img+step-1, step)) +
            (x >= cols - 4 || y >= rows - 4 || s > cornerScore(img+step+1, step)) == 6)
        {
            int new_idx = atomic_inc(kp_out);
            if( new_idx < max_keypoints )
            {
                kp_out[1 + 3*new_idx] = x;
                kp_out[2 + 3*new_idx] = y;
                kp_out[3 + 3*new_idx] = s;
            }
        }
    }
}
