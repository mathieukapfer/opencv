#include <stdio.h>

#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b
#define clamp(val, min, max) MAX(MIN(val,max),min)

#define GRP_SIZEY 4
#define GRP_SIZEX 4

#define cols 10
#define rows 10

int src[GRP_SIZEX*cols + GRP_SIZEY*rows];
int smem[(GRP_SIZEX + 4) * (GRP_SIZEY + 4)];

#define src_offset 0
#define src_step GRP_SIZEX *cols
#define TYPE char

const int cn = 1;

#define mad24(a, b, c) (a * b + c)
#define loadpix(n) src[n]; printf("src[%d]",n);

void loadpixloop(int lidx, int lidy, int start_x, int start_y) {

//  int lidx = get_local_id(0);
//  int lidy = get_local_id(1);

//  int start_x = GRP_SIZEX * get_group_id(0);
//  int start_y = GRP_SIZEY * get_group_id(1);

  int i = lidx + lidy * GRP_SIZEX;
  for (int j = i; j < (GRP_SIZEX + 4) * (GRP_SIZEY + 4);
       j += GRP_SIZEX * GRP_SIZEY) {
    int x = clamp(start_x - 2 + (j % (GRP_SIZEX + 4)), 0, cols - 1);
    int y = clamp(start_y - 2 + (j / (GRP_SIZEX + 4)), 0, rows - 1);

    printf("smem[%d] = src[%d]  ",j, mad24(y, src_step, mad24(x, cn * (int)sizeof(TYPE), src_offset)));
    //smem[j] = loadpix(
    //    src + mad24(y, src_step, mad24(x, cn * (int)sizeof(TYPE), src_offset)));
  }
}

int main(int argc, char *argv[]) {

  for (int y=0; y<GRP_SIZEY;y++)
    for (int x=0; x<GRP_SIZEX; x++)
      {
        printf("\nlidx:%d, lidy:%d =>", x,y);
        loadpixloop(x,y,2,2);
      }

  return 0;

}
