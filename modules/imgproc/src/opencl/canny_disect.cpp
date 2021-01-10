#include <stdio.h>

#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b
#define min(a, b) a < b ? a : b
#define clamp(val, min, max) MAX(MIN(val,max),min)

#define GRP_SIZEY 4
#define GRP_SIZEX 4

#define cols 10
#define rows 10

int src[cols * rows];
int smem[(GRP_SIZEX + 4) * (GRP_SIZEY + 4)];

#define src_offset 0
#define src_step cols
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

    printf("smem[%d] = loadpix(%d)  ",j, mad24(y, src_step, mad24(x, cn * (int)sizeof(TYPE), src_offset)));
    //smem[j] = loadpix(
    //    src + mad24(y, src_step, mad24(x, cn * (int)sizeof(TYPE), src_offset)));
  }
}

void compute_sobel(int lidx, int lidy, int start_x, int start_y)
{
    int i = lidx + lidy * GRP_SIZEX;

    lidx++;
    lidy++;

    if (i < GRP_SIZEX + 2)
    {
        int grp_sizey = min(GRP_SIZEY + 1, rows - start_y);
        printf("mag[%2d] = sobel(%2d)  ", i, i);
        printf("mag[%2d] = sobel(%2d)  ",
               i + grp_sizey * (GRP_SIZEX + 2),
               i + grp_sizey * (GRP_SIZEX + 4));
    }
    if (i < GRP_SIZEY + 2)
    {
        int grp_sizex = min(GRP_SIZEX + 1, cols - start_x);
        printf("mag[%2d] = sobel(%2d)  ", i * (GRP_SIZEX + 2), i * (GRP_SIZEX + 4));
        printf("mag[%2d] = sobel(%2d)  ", i * (GRP_SIZEX + 2) + grp_sizex, i * (GRP_SIZEX + 4) + grp_sizex);
    }

    int idx = lidx + lidy * (GRP_SIZEX + 4);
    i = lidx + lidy * (GRP_SIZEX + 2);

    printf("mag[%2d] = sobel(%2d) ", i, idx);
}

 int prev[4][2] = {
    { 0, -1 },
    { -1, 1 },
    { -1, 0 },
    { -1, -1 }
};

 int next[4][2] = {
    { 0, 1 },
    { 1, -1 },
    { 1, 0 },
    { 1, 1 }
};


void compute_mag(int lidx, int lidy) {
  //float prev_mag = mag[(lidy + prev[dir][0]) * (GRP_SIZEX + 2) + lidx + prev[dir][1]];
  //float next_mag = mag[(lidy + next[dir][0]) * (GRP_SIZEX + 2) + lidx + next[dir][1]] + (dir & 1);

  for (int dir=0; dir < 4; dir++) {
    printf("\ndir=%d ",dir);
    printf("prev_mag = mag[%d], ", (lidy + prev[dir][0]) * (GRP_SIZEX + 2) + lidx + prev[dir][1] );
    printf("next_mag = mag[%d]", (lidy + next[dir][0]) * (GRP_SIZEX + 2) + lidx + next[dir][1] );
  }

}

int main(int argc, char *argv[]) {

  for (int y=0; y<GRP_SIZEY;y++)
    for (int x=0; x<GRP_SIZEX; x++)
      {
        printf("\nlidx:%2d, lidy:%2d =>", x,y);
        //loadpixloop(x,y,2,2);
        //compute_sobel(x,y,2,2);
        compute_mag(x,y);
      }

  return 0;

}
