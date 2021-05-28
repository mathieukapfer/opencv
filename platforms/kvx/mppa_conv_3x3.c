#include <mppa_cos.h>
#include <mppa_async.h>
#include <stdlib.h>

typedef long __attribute__((__vector_size__(4 * sizeof(long)))) v4i64;

typedef union {
    struct {
    __tca256 s0, s1, s2, s3;
    };
    struct {
        __tca512 lo, hi;
    };
    __tca1024 full;
} __tca128B;

/*  Load a line from a 3x3 convolution filter into TCA registers with the following pattern:
 *  half0:
 *    a b c 0 0 0 0 0
 *    0 a b c 0 0 0 0
 *    0 0 a b c 0 0 0
 *    0 0 0 a b c 0 0
 *  half1:
 *    0 0 a b c 0 0 0
 *    0 0 0 a b c 0 0
 *    0 0 0 0 a b c 0
 *    0 0 0 0 0 a b c
 */
#define conv_u8_3x3_tca_prepare_kernel(rhs_matrix_half0, rhs_matrix_half1, kernel) \
{\
    char kernel_H0_values[32] = {\
        kernel[0], kernel[1], kernel[2], 0, 0, 0, 0, 0,\
        0, kernel[0], kernel[1], kernel[2], 0, 0, 0, 0,\
        0, 0, kernel[0], kernel[1], kernel[2], 0, 0, 0,\
        0, 0, 0, kernel[0], kernel[1], kernel[2], 0, 0,\
    };\
    const __tca256 zero = __builtin_kvx_moveoto((v4i64){0});\
    *rhs_matrix_half0 = *(__tca256*)kernel_H0_values;\
    *rhs_matrix_half1 = __builtin_kvx_alignv(zero, *rhs_matrix_half0, 30);\
}

#define SRC_IDX(Y, X, W) ((X + 1) + ((Y + 1) * (W + 2)))
#define DST_IDX(Y, X, OUT_WIDTH) ((X) + ((Y) * (OUT_WIDTH)))

// Compute the u8 * u8 -> u8 3x3 convolution result for one pixel.
// Note: results are downscaled by a factor of 256.
void conv_u8_3x3_classic(const unsigned char *src, unsigned char *dst, const unsigned char kernel[3][3], int x, int y, int width)
{
    int acc = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int kern_val = kernel[dy + 1][dx + 1];
            int mem_val = src[SRC_IDX(y + dy, x + dx, width)];
            int val = kern_val * mem_val;
            acc += val;
        }
    }

    dst[DST_IDX(y, x, width)] = acc / 256;
}

__attribute__((noinline)) void conv_u8_3x3_tca(const unsigned char *src, unsigned char *dst, const int h, const unsigned char kernel[3][3], int width)
{
    // on current column
    // rowA: previous row
    // rowB: current row
    // rowC: next row
    // *0: shifted one element left (starting at index -1, and finishing at element n-1)
    // *1: shifted one element right (starting at index 1, and finishing at element n+1)

    // on next column
    // row_n_A: previous row
    // row_n_B: current row
    // row_n_C: next row

    // Constants initialization
    const __tca256 zero = __builtin_kvx_moveoto((v4i64){0x0});
    const __tca128B zero2 = {
        .s0 = zero,
        .s1 = zero,
        .s2 = zero,
        .s3 = zero,
    };
    const __tca512 wideZero = zero2.lo;

    // Preparation of the weight kernels.
    __tca256 kernel_AH0, kernel_AH1, kernel_BH0, kernel_BH1, kernel_CH0, kernel_CH1;
    conv_u8_3x3_tca_prepare_kernel(&kernel_AH0, &kernel_AH1, kernel[0]);
    conv_u8_3x3_tca_prepare_kernel(&kernel_BH0, &kernel_BH1, kernel[1]);
    conv_u8_3x3_tca_prepare_kernel(&kernel_CH0, &kernel_CH1, kernel[2]);

    // Backup the scaling factor and set it to 8 (division by 256)
    unsigned long cs_orig = 0;
    unsigned long cs_value = 0;
    cs_orig = __builtin_kvx_get(COS_SFR_CS);
    cs_value = COS_SFR_SET_FIELD(cs_orig, CS_XDROP, 8);
    __builtin_kvx_set(COS_SFR_CS, cs_value);

    for (int col=0; col < width; col+=32) {
        __tca256 rowA0, rowB0, rowC0;
        __tca256 rowA1, rowB1, rowC1;
        __tca256 row_n_A, row_n_B, row_n_C;
        int y = 0;

        // Load the first three rows.
        rowA0 = *(__tca256*)&(src[SRC_IDX(y-1, col, width)]);
        rowB0 = *(__tca256*)&(src[SRC_IDX(y, col, width)]);
        rowC0 = *(__tca256*)&(src[SRC_IDX(y+1, col, width)]);
        // Note: We actually only need the first two elements. May be modified
        // to simple loads + move instructions to TCA if faster, or could load
        // 32 bytes more to work on 64 output pixels at once.
        row_n_A = *(__tca256*)&(src[SRC_IDX(y-1, col+32, width)]);
        row_n_B = *(__tca256*)&(src[SRC_IDX(y, col+32, width)]);
        row_n_C = *(__tca256*)&(src[SRC_IDX(y+1, col+32, width)]);

        // inserting one element from row*0 to the most significant place in row*1
        rowA1 = __builtin_kvx_alignv(rowA0, row_n_A, 1);
        rowB1 = __builtin_kvx_alignv(rowB0, row_n_B, 1);
        rowC1 = __builtin_kvx_alignv(rowC0, row_n_C, 1);

        // then shifting one elemnt out of row*0
        rowA0 = __builtin_kvx_alignv(*(__tca256*)&(src[SRC_IDX(y-1, col-32, width)]), rowA0, 31);
        rowB0 = __builtin_kvx_alignv(*(__tca256*)&(src[SRC_IDX(y, col-32, width)]),   rowB0, 31);
        rowC0 = __builtin_kvx_alignv(*(__tca256*)&(src[SRC_IDX(y+1, col-32, width)]), rowC0, 31);

        // Layout at this point:
        // rowX0: -1 0 [...] 30, rowX1: 1 2 [...] 32

        // acc_B_H0 is the 4x4 first/left half of 4x8 accumulator
        // acc_B_H1 is the 4x4 second/right half of 4x8 accumulator
        __tca512 acc_B_H0, acc_B_H1;

        // Initialize accumulators to 0
        acc_B_H0 = wideZero;
        acc_B_H1 = wideZero;

        //                           a  0  0  0
        //                           b  a  0  0
        //                           c  b  a  0
        //                           0  c  b  a
        //                           0  0  c  b
        //                           0  0  0  c
        //                           0  0  0  0
        //                           0  0  0  0
        //
        // -1  0  1  2  3  4  5  6   0  1  2  3
        //  7  8  9 10 11 12 13 14   8  9 10 11
        // 15 16 17 18 19 20 21 22  16 17 18 19
        // 23 24 25 26 27 28 29 30  24 25 26 27
        acc_B_H0 = __builtin_kvx_mma484ubw(acc_B_H0, rowA0, kernel_AH0);
        acc_B_H0 = __builtin_kvx_mma484ubw(acc_B_H0, rowB0, kernel_BH0);
        acc_B_H0 = __builtin_kvx_mma484ubw(acc_B_H0, rowC0, kernel_CH0);

        //                           0  0  0  0
        //                           0  0  0  0
        //                           a  0  0  0
        //                           b  a  0  0
        //                           c  b  a  0
        //                           0  c  b  a
        //                           0  0  c  b
        //                           0  0  0  c
        //
        //  1  2  3  4  5  6  7  8   4  5  6  7
        //  9 10 11 12 13 14 15 16  12 13 14 15
        // 17 18 19 20 21 22 23 24  20 21 22 23
        // 25 26 27 28 29 30 31 32  28 29 30 31
        acc_B_H1 =__builtin_kvx_mma484ubw(acc_B_H1, rowA1, kernel_AH1);
        acc_B_H1 =__builtin_kvx_mma484ubw(acc_B_H1, rowB1, kernel_BH1);
        acc_B_H1 = __builtin_kvx_mma484ubw(acc_B_H1, rowC1, kernel_CH1);

        // 32u -> 8u conversion, with division by 256.
        __tca1024 acc_B_32b;
        __tca256 acc_B_8b;
        __tca128B tmp = {
            .lo = acc_B_H0,
            .hi = acc_B_H1,
        };
        acc_B_8b = __builtin_kvx_convwbv(tmp.full, ".rz.satu");

        // Store the final result.
        *(__tca256*)&(dst[DST_IDX(y, col, width)]) = acc_B_8b;

        __tca256 row_c_C, row_p_C;
        row_p_C = *(__tca256*)&(src[SRC_IDX(y+2, col-32, width)]);
        row_c_C = *(__tca256*)&(src[SRC_IDX(y+2, col, width)]);
        row_n_C = *(__tca256*)&(src[SRC_IDX(y+2, col+32, width)]);

        for (y = 1; y < h; ++y) {
            // copying next row to current
            rowA0 = rowB0;
            rowA1 = rowB1;
            rowB0 = rowC0;
            rowB1 = rowC1;
            // TODO/FIXME manage last row
            rowC1 = __builtin_kvx_alignv(row_c_C, row_n_C, 1);
            rowC0 = __builtin_kvx_alignv(row_p_C, row_c_C, 31);

            // pre-loading next iteration next row
            row_p_C = *(__tca256*)&(src[SRC_IDX(y+2, col-32, width)]);
            row_c_C = *(__tca256*)&(src[SRC_IDX(y+2, col, width)]);
            row_n_C = *(__tca256*)&(src[SRC_IDX(y+2, col+32, width)]);

            acc_B_H0 = wideZero;
            acc_B_H1 = wideZero;

            acc_B_H0 = __builtin_kvx_mma484ubw(acc_B_H0, rowA0, kernel_AH0);
            acc_B_H1 =__builtin_kvx_mma484ubw(acc_B_H1, rowA1, kernel_AH1);
            acc_B_H0 = __builtin_kvx_mma484ubw(acc_B_H0, rowB0, kernel_BH0);
            acc_B_H1 =__builtin_kvx_mma484ubw(acc_B_H1, rowB1, kernel_BH1);
            acc_B_H0 = __builtin_kvx_mma484ubw(acc_B_H0, rowC0, kernel_CH0);
            acc_B_H1 = __builtin_kvx_mma484ubw(acc_B_H1, rowC1, kernel_CH1);

            tmp.lo = acc_B_H0;
            tmp.hi = acc_B_H1;
            acc_B_32b = tmp.full;
            acc_B_8b = __builtin_kvx_convwbv(acc_B_32b, ".rz.satu");

            *(__tca256*)&(dst[DST_IDX(y, col, width)]) = acc_B_8b;
        }
    }

    // Restore the previous XDROP factor.
    __builtin_kvx_set(COS_SFR_CS, cs_orig);
}

// Compute the result of a 3x3 u8 * u8 -> u8 convolution using a TCA
// implementation. Weights come from kx[x] * ky[y]. The sum of the
// weights should be 1.
void conv_u8_factor_u8_3x3(const unsigned char* src, unsigned char* dst,
          float *kx, float *ky, int start_line, int end_line,
          int width)
{
    const int nb_lines = end_line - start_line;

    // Compute the actual kernel from kx and ky.
    unsigned char kernel[3][3];
    int total = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            kernel[y][x] = 256*kx[x] * ky[y];
            total += kernel[y][x];
        }
    }

    // We want to have 256 as the sum of the weight.
    // The way we ensure it is by increasing the value of the weights
    // that have the biggest rounding error.
    while (total != 256) {
        int max_y = 0;
        int max_x = 0;
        float max_dist = 0;
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                float dist = 256 * kx[x] * ky[y] - kernel[y][x];
                if (dist > max_dist) {
                    max_x = x;
                    max_y = y;
                    max_dist = dist;
                }
            }
        }
        kernel[max_y][max_x]++;
        total++;
    }

    // Move our pointers so that they both point to the beginning of the
    // part where computation should occur.
    src += (width + 2) * start_line;
    dst += DST_IDX(start_line, 0, width);

    __cos_copro_enable();
    conv_u8_3x3_tca(src, dst, nb_lines, kernel, width);
    __cos_copro_disable();
}

// Compute the result of a 3x3 u8 * u8 -> u8 convolution using a "naive"
// implementation. Weights come from kx[x] * ky[y]. The sum of the weights
// should be 1.
void conv_u8_factor_u8_3x3_naive(const unsigned char* src, unsigned char* dst,
          float *kx, float *ky, int start_line, int end_line,
          int width)
{
    const int nb_lines = end_line - start_line;

    // Compute the actual kernel from kx and ky.
    unsigned char kernel[3][3];
    int total = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            kernel[y][x] = 256*kx[x] * ky[y];
            total += kernel[y][x];
        }
    }

    // We want to have 256 as the sum of the weight.
    // The way we ensure it is by increasing the value of the weights
    // that have the biggest rounding error.
    while (total != 256) {
        int max_y = 0;
        int max_x = 0;
        float max_dist = 0;
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                float dist = 256 * kx[x] * ky[y] - kernel[y][x];
                if (dist > max_dist) {
                    max_x = x;
                    max_y = y;
                    max_dist = dist;
                }
            }
        }
        kernel[max_y][max_x]++;
        total++;
    }

    // Move our pointers so that they both point to the beginning of the
    // part where computation should occur.
    src += (width + 2) * start_line;
    dst += DST_IDX(start_line, 0, width);

    for (int y = 0; y < nb_lines; y++) {
        #pragma clang loop unroll(enable)
        for (int x = 0; x < width; x++) {
            conv_u8_3x3_classic(src, dst, kernel, x, y, width);
        }
    }
}
