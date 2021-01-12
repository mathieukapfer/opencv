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

#define TG22 0.4142135623730950488016887242097f
#define TG67 2.4142135623730950488016887242097f

#ifdef WITH_SOBEL

#if cn == 1
#define loadpix(addr) convert_floatN(*(__global const TYPE *)(addr))
#else
#define loadpix(addr) convert_floatN(vload3(0, (__global const TYPE *)(addr)))
#endif
#define storepix(value, addr) *(__global int *)(addr) = (int)(value)

/*
    stage1_with_sobel:
        Sobel operator
        Calc magnitudes
        Non maxima suppression
        Double thresholding
*/

__constant int prev[4][2] = {
    { 0, -1 },
    { -1, 1 },
    { -1, 0 },
    { -1, -1 }
};

__constant int next[4][2] = {
    { 0, 1 },
    { 1, -1 },
    { 1, 0 },
    { 1, 1 }
};

inline float3 sobel(int idx, __local const floatN *smem)
{
  // result: x, y, mag
    float3 res;

    /* if GRP_SIZEX = GRP_SIZEX = 4,
       then the tile is 8x8 including border,
       then in case idx = 0, dx coef by pixel are:
      |   | 0         | 1 | 2         | 3 | 4 | 5 | 6 | 7 |
      |---+-----------+---+-----------+---+---+---+---+---|
      | 0 | idx:   -1 |   | idx+2:  1 |   |   |   |   |   |
      | 1 | idx+8: -2 |   | idx+10: 2 |   |   |   |   |   |
      | 2 | idx+16:-1 |   | idx+18: 1 |   |   |   |   |   |
      ...
      in general case: (GRP_SIZEX + 4) means (y + 1)
     */
    floatN dx = fma((floatN)2, smem[idx + GRP_SIZEX + 6] - smem[idx + GRP_SIZEX + 4],
        smem[idx + 2] - smem[idx] + smem[idx + 2 * GRP_SIZEX + 10] - smem[idx + 2 * GRP_SIZEX + 8]);

    /* if GRP_SIZEX = GRP_SIZEX = 4,
       then the tile is 8x8 including border,
       then in case idx = 0, dy coef by pixel are:
      |   | 0         | 1         | 3         | 4 | 5 | 6 | 7 |
      |---+-----------+-----------+-----------+---+---+---+---|
      | 0 | idx:1     | idx+1:2   | idx+2:1   |   |   |   |   |
      | 1 |           |           |           |   |   |   |   |
      | 2 | idx+16:-1 | idx+17:-2 | idx+18:-1 |   |   |   |   |
      ...
      in general case: (GRP_SIZEX + 4) means (y + 1)
    */

    floatN dy = fma((floatN)2, smem[idx + 1] - smem[idx + 2 * GRP_SIZEX + 9],
        smem[idx + 2] - smem[idx + 2 * GRP_SIZEX + 10] + smem[idx] - smem[idx + 2 * GRP_SIZEX + 8]);

#ifdef L2GRAD
    floatN magN = fma(dx, dx, dy * dy);
#else
    floatN magN = fabs(dx) + fabs(dy);
#endif
#if cn == 1
    res.z = magN;
    res.x = dx;
    res.y = dy;
#else
    res.z = max(magN.x, max(magN.y, magN.z));
    if (res.z == magN.y)
    {
        dx.x = dx.y;
        dy.x = dy.y;
    }
    else if (res.z == magN.z)
    {
        dx.x = dx.z;
        dy.x = dy.z;
    }
    res.x = dx.x;
    res.y = dy.x;
#endif

    return res;
}

__kernel void stage1_with_sobel(__global const uchar *src, int src_step, int src_offset, int rows, int cols,
                                __global uchar *map, int map_step, int map_offset,
                                float low_thr, float high_thr)
{
    __local floatN smem[(GRP_SIZEX + 4) * (GRP_SIZEY + 4)];

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    int start_x = GRP_SIZEX * get_group_id(0);
    int start_y = GRP_SIZEY * get_group_id(1);

    /* this is the list of loadpix call done the loop below, in case of

         in case of
            - local size =(4,4), that means  GRP_SIZEX=4, GRP_SIZEY=4
            - image size: cols 480, rows 512
            - global_id : (0,0)

       => the tile size is 8 x 8 (border size if 2 pixels in each direction)
       => each work item have to load 4 pixels:

         lidx: 0, lidy: 0 =>smem[0] = loadpix(0)  smem[16] = loadpix(0)  smem[32] = loadpix(1280)  smem[48] = loadpix(2560)
         lidx: 1, lidy: 0 =>smem[1] = loadpix(0)  smem[17] = loadpix(0)  smem[33] = loadpix(1280)  smem[49] = loadpix(2560)
         lidx: 2, lidy: 0 =>smem[2] = loadpix(0)  smem[18] = loadpix(0)  smem[34] = loadpix(1280)  smem[50] = loadpix(2560)
         lidx: 3, lidy: 0 =>smem[3] = loadpix(1)  smem[19] = loadpix(1)  smem[35] = loadpix(1281)  smem[51] = loadpix(2561)
         lidx: 0, lidy: 1 =>smem[4] = loadpix(2)  smem[20] = loadpix(2)  smem[36] = loadpix(1282)  smem[52] = loadpix(2562)
         lidx: 1, lidy: 1 =>smem[5] = loadpix(3)  smem[21] = loadpix(3)  smem[37] = loadpix(1283)  smem[53] = loadpix(2563)
         lidx: 2, lidy: 1 =>smem[6] = loadpix(4)  smem[22] = loadpix(4)  smem[38] = loadpix(1284)  smem[54] = loadpix(2564)
         lidx: 3, lidy: 1 =>smem[7] = loadpix(5)  smem[23] = loadpix(5)  smem[39] = loadpix(1285)  smem[55] = loadpix(2565)
         lidx: 0, lidy: 2 =>smem[8] = loadpix(0)  smem[24] = loadpix(640)  smem[40] = loadpix(1920)  smem[56] = loadpix(3200)
         lidx: 1, lidy: 2 =>smem[9] = loadpix(0)  smem[25] = loadpix(640)  smem[41] = loadpix(1920)  smem[57] = loadpix(3200)
         lidx: 2, lidy: 2 =>smem[10] = loadpix(0)  smem[26] = loadpix(640)  smem[42] = loadpix(1920)  smem[58] = loadpix(3200)
         lidx: 3, lidy: 2 =>smem[11] = loadpix(1)  smem[27] = loadpix(641)  smem[43] = loadpix(1921)  smem[59] = loadpix(3201)
         lidx: 0, lidy: 3 =>smem[12] = loadpix(2)  smem[28] = loadpix(642)  smem[44] = loadpix(1922)  smem[60] = loadpix(3202)
         lidx: 1, lidy: 3 =>smem[13] = loadpix(3)  smem[29] = loadpix(643)  smem[45] = loadpix(1923)  smem[61] = loadpix(3203)
         lidx: 2, lidy: 3 =>smem[14] = loadpix(4)  smem[30] = loadpix(644)  smem[46] = loadpix(1924)  smem[62] = loadpix(3204)
         lidx: 3, lidy: 3 =>smem[15] = loadpix(5)  smem[31] = loadpix(645)  smem[47] = loadpix(1925)  smem[63] = loadpix(3205)


       <---------------------- image size (640x480)------------------------------>
       <---------------------- till (8x8) -------------------------->
       |    |    0 |    1 |    2 |    3 |    4 |    5 |    6 |    7 |
       |----+------+------+------+------+------+------+------+------|
       |  0 |    0 |    0 |    0 |    1 |    2 |    3 |    4 |    5 |
       |  8 |    0 |    0 |    0 |    1 |    2 |    3 |    4 |    5 |
       | 16 |    0 |    0 |    0 |    1 |    2 |    3 |    4 |    5 |
       | 24 |    0 |  640 |  640 |  641 |  642 |  643 |  644 |  645 |
       | 32 | 1280 | 1208 | 1280 | 1281 | 1282 | 1283 | 1284 | 1285 |
       | 40 | 1920 | 1920 | 1920 | 1921 | 1922 | 1923 | 1924 | 1925 |
       | 48 | 2560 | 2560 | 2560 | 2561 | 2562 | 2563 | 2564 | 2565 |
       | 56 | 3200 | 2320 | 3200 | 3201 | 3202 | 3203 | 3204 | 3205 |

    */

    int i = lidx + lidy * GRP_SIZEX;
    for (int j = i;  j < (GRP_SIZEX + 4) * (GRP_SIZEY + 4); j += GRP_SIZEX * GRP_SIZEY)
    {
        int x = clamp(start_x - 2 + (j % (GRP_SIZEX + 4)), 0, cols - 1);
        int y = clamp(start_y - 2 + (j / (GRP_SIZEX + 4)), 0, rows - 1);
        smem[j] = loadpix(src + mad24(y, src_step, mad24(x, cn * (int)sizeof(TYPE), src_offset)));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //// Sobel, Magnitude
    //

    __local float mag[(GRP_SIZEX + 2) * (GRP_SIZEY + 2)];

    // crop 1 pix of the border
    lidx++;
    lidy++;

    /*
       in case of local size =(4,4), that means  GRP_SIZEX=4, GRP_SIZEY=4

       => the tile size is 8 x 8  (border size if 2 pixels in each direction)
       => the sobel size is 6 x 6 (border size if 1 pixels in each direction)

       lidx: 0, lidy: 0 =>mag[ 0] = sobel( 0)  mag[30] = sobel(40)!  mag[ 0] = sobel( 0)!  mag[ 5] = sobel( 5)!  mag[ 7] = sobel( 9)
       lidx: 1, lidy: 0 =>mag[ 1] = sobel( 1)  mag[31] = sobel(41)   mag[ 6] = sobel( 8)   mag[11] = sobel(13)   mag[ 8] = sobel(10)
       lidx: 2, lidy: 0 =>mag[ 2] = sobel( 2)  mag[32] = sobel(42)   mag[12] = sobel(16)   mag[17] = sobel(21)   mag[ 9] = sobel(11)
       lidx: 3, lidy: 0 =>mag[ 3] = sobel( 3)  mag[33] = sobel(43)   mag[18] = sobel(24)   mag[23] = sobel(29)   mag[10] = sobel(12)
       lidx: 0, lidy: 1 =>mag[ 4] = sobel( 4)  mag[34] = sobel(44)   mag[24] = sobel(32)   mag[29] = sobel(37)   mag[13] = sobel(17)
       lidx: 1, lidy: 1 =>mag[ 5] = sobel( 5)  mag[35] = sobel(45)!  mag[30] = sobel(40)!  mag[35] = sobel(45)!  mag[14] = sobel(18)
       lidx: 2, lidy: 1 =>mag[15] = sobel(19)
       lidx: 3, lidy: 1 =>mag[16] = sobel(20)
       lidx: 0, lidy: 2 =>mag[19] = sobel(25)
       lidx: 1, lidy: 2 =>mag[20] = sobel(26)
       lidx: 2, lidy: 2 =>mag[21] = sobel(27)
       lidx: 3, lidy: 2 =>mag[22] = sobel(28)
       lidx: 0, lidy: 3 =>mag[25] = sobel(33)
       lidx: 1, lidy: 3 =>mag[26] = sobel(34)
       lidx: 2, lidy: 3 =>mag[27] = sobel(35)
       lidx: 3, lidy: 3 =>mag[28] = sobel(36)

            <---------------- till (8x8) ------------------------->
            <----------- sobel (6x6) ------------------>
            mag:sobel
       |   |    0  |   1 |    2 |    3 |     4 |     5 | 6   | 7   |
       |---+-------+-----+------+------+-------+-------+-----+-----|
       | 0 |  0:0  | 1:1 |  2:2 |  3:3 |   4:4 |   5:5 | :6  | :7  |
       | 1 |  6:8  | 7:9 | 8:10 | 9:11 | 10:12 | 11:13 | :14 | :15 |
       | 2 | 12:16 | ... |      |      |       |       |     |     |

       NOTES:
          - some sobel is compute twice ! (notice '!' in section above)
          - the wi work load is not the same

    */

    if (i < GRP_SIZEX + 2)
    {
        int grp_sizey = min(GRP_SIZEY + 1, rows - start_y);
        mag[i] = (sobel(i, smem)).z;
        mag[i + grp_sizey * (GRP_SIZEX + 2)] = (sobel(i + grp_sizey * (GRP_SIZEX + 4), smem)).z;
    }
    if (i < GRP_SIZEY + 2)
    {
        int grp_sizex = min(GRP_SIZEX + 1, cols - start_x);
        mag[i * (GRP_SIZEX + 2)] = (sobel(i * (GRP_SIZEX + 4), smem)).z;
        mag[i * (GRP_SIZEX + 2) + grp_sizex] = (sobel(i * (GRP_SIZEX + 4) + grp_sizex, smem)).z;
    }

    int idx = lidx + lidy * (GRP_SIZEX + 4);
    i = lidx + lidy * (GRP_SIZEX + 2);

    float3 res = sobel(idx, smem);
    mag[i] = res.z;
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = (int) res.x;
    int y = (int) res.y;

    //// Threshold + Non maxima suppression
    //

    /*
        Sector numbers

        3   2   1
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        1   2   3

        We need to determine arctg(dy / dx) to one of the four directions: 0, 45, 90 or 135 degrees.
        Therefore if abs(dy / dx) belongs to the interval
        [0, tg(22.5)]           -> 0 direction
        [tg(22.5), tg(67.5)]    -> 1 or 3
        [tg(67,5), +oo)         -> 2

        Since tg(67.5) = 1 / tg(22.5), if we take
        a = abs(dy / dx) * tg(22.5) and b = abs(dy / dx) * tg(67.5)
        we can get another intervals

        in case a:
        [0, tg(22.5)^2]     -> 0
        [tg(22.5)^2, 1]     -> 1, 3
        [1, +oo)            -> 2

        in case b:
        [0, 1]              -> 0
        [1, tg(67.5)^2]     -> 1,3
        [tg(67.5)^2, +oo)   -> 2

        that can help to find direction without conditions.

        0 - might belong to an edge
        1 - pixel doesn't belong to an edge
        2 - belong to an edge
    */

    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    if (gidx >= cols || gidy >= rows)
        return;

    float mag0 = mag[i];

    int value = 1;
    if (mag0 > low_thr)
    {
        float x_ = abs(x);
        float y_ = abs(y);

        int a = (y_ * TG22 >= x_) ? 2 : 1;
        int b = (y_ * TG67 >= x_) ? 1 : 0;

        //  a = { 1, 2 }
        //  b = { 0, 1 }
        //  a * b = { 0, 1, 2 } - directions that we need ( + 3 if x ^ y < 0)

        int dir3 = (a * b) & (((x ^ y) & 0x80000000) >> 31); // if a = 1, b = 1, dy ^ dx < 0
        int dir = a * b + 2 * dir3;

        /*
          in case of local size =(4,4), that means  GRP_SIZEX=4, GRP_SIZEY=4
          mag[x] contains sobel value

               <----------- sobel (6x6) ---------------->
                         <--- final value (4x4)----->
          |   | 0       | 1       | 2       | 3 | 4 | 5 |
          |---+---------+---------+---------+---+---+---|
          | 0 | mag(-7) | mag(-6) | mag(-5) |   |   |   |
          | 1 | mag(-1) | mag(0)  | mag(+1) |   |   |   |
          | 2 | mag(+5) | mag(+6) | mag(+7) |   |   |   |

          if lidx: 0, lidy: 0,
            and dir=0 then prev_mag = mag[-1], next_mag = mag[1]
            and dir=1 then prev_mag = mag[-5], next_mag = mag[5]
            and dir=2 then prev_mag = mag[-6], next_mag = mag[6]
            and dir=3 then prev_mag = mag[-7], next_mag = mag[7]

        */
        float prev_mag = mag[(lidy + prev[dir][0]) * (GRP_SIZEX + 2) + lidx + prev[dir][1]];
        float next_mag = mag[(lidy + next[dir][0]) * (GRP_SIZEX + 2) + lidx + next[dir][1]] + (dir & 1);

        if (mag0 > prev_mag && mag0 >= next_mag)
        {
            value = (mag0 > high_thr) ? 2 : 0;
        }
    }

    storepix(value, map + mad24(gidy, map_step, mad24(gidx, (int)sizeof(int), map_offset)));
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

    storepix(value, map + mad24(gidy, map_step, mad24(gidx, (int)sizeof(int), map_offset)));
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
#define LOCAL_TOTAL (LOCAL_X*LOCAL_Y)
#define l_stack_size (4*LOCAL_TOTAL)
#define p_stack_size 8

__constant short move_dir[2][8] = {
    { -1, -1, -1, 0, 0, 1, 1, 1 },
    { -1, 0, 1, -1, 1, -1, 0, 1 }
};

__kernel void stage2_hysteresis(__global uchar *map_ptr, int map_step, int map_offset, int rows, int cols)
{
    map_ptr += map_offset;

    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI;

    int lid = get_local_id(0) + get_local_id(1) * LOCAL_X;

    __local ushort2 l_stack[l_stack_size];
    __local int l_counter;

    if (lid == 0)
        l_counter = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < cols)
    {
        __global uchar* map = map_ptr + mad24(y, map_step, x * (int)sizeof(int));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI; ++cy)
        {
            if (y < rows)
            {
                int type = loadpix(map);
                if (type == 2)
                {
                    l_stack[atomic_inc(&l_counter)] = (ushort2)(x, y);
                }

                y++;
                map += map_step;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ushort2 p_stack[p_stack_size];
    int p_counter = 0;

    while(l_counter != 0)
    {
        int mod = l_counter % LOCAL_TOTAL;
        int pix_per_thr = l_counter / LOCAL_TOTAL + ((lid < mod) ? 1 : 0);

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < pix_per_thr; ++i)
        {
            int index = atomic_dec(&l_counter) - 1;
            if (index < 0)
               continue;
            ushort2 pos = l_stack[ index ];

            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                ushort posx = pos.x + move_dir[0][j];
                ushort posy = pos.y + move_dir[1][j];
                if (posx < 0 || posy < 0 || posx >= cols || posy >= rows)
                    continue;
                __global uchar *addr = map_ptr + mad24(posy, map_step, posx * (int)sizeof(int));
                int type = loadpix(addr);
                if (type == 0)
                {
                    p_stack[p_counter++] = (ushort2)(posx, posy);
                    storepix(2, addr);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_counter < 0)
            l_counter = 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        while (p_counter > 0)
        {
            l_stack[ atomic_inc(&l_counter) ] = p_stack[--p_counter];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#elif defined GET_EDGES

// Get the edge result. edge type of value 2 will be marked as an edge point and set to 255. Otherwise 0.
// map      edge type mappings
// dst      edge output

__kernel void getEdges(__global const uchar *mapptr, int map_step, int map_offset, int rows, int cols,
                       __global uchar *dst, int dst_step, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI;

    if (x < cols)
    {
        int map_index = mad24(map_step, y, mad24(x, (int)sizeof(int), map_offset));
        int dst_index = mad24(dst_step, y, x + dst_offset);

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI; ++cy)
        {
            if (y < rows)
            {
                __global const int * map = (__global const int *)(mapptr + map_index);
                dst[dst_index] = (uchar)(-(map[0] >> 1));

                y++;
                map_index += map_step;
                dst_index += dst_step;
            }
        }
    }
}

#endif
