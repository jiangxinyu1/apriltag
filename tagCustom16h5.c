#include <stdlib.h>
#include "tagCustom16h5.h"

static uint64_t codedata[27] = {
   0x00000000000091a0UL,
   0x0000000000009765UL,
   0x000000000000b43eUL,
   0x000000000000dca1UL,
   0x00000000000027a2UL,
   0x0000000000002d67UL,
   0x00000000000038f1UL,
   0x000000000000447bUL,
   0x00000000000089b7UL,
   0x0000000000009b06UL,
   0x000000000000c70aUL,
   0x0000000000005d0cUL,
   0x0000000000002136UL,
   0x0000000000005523UL,
   0x000000000000983bUL,
   0x0000000000000f40UL,
   0x00000000000018a6UL,
   0x000000000000d516UL,
   0x000000000000a4caUL,
   0x00000000000094f8UL,
   0x000000000000d24bUL,
   0x000000000000ba76UL,
   0x000000000000cf66UL,
   0x000000000000164eUL,
   0x000000000000c929UL,
   0x0000000000008052UL,
   0x00000000000091dbUL,
};
apriltag_family_t *tagCustom16h5_create()
{
   apriltag_family_t *tf = calloc(1, sizeof(apriltag_family_t));
   tf->name = strdup("tagCustom16h5");
   tf->h = 5;
   tf->ncodes = 27;
   tf->codes = codedata;
   tf->nbits = 16;
   tf->bit_x = calloc(16, sizeof(uint32_t));
   tf->bit_y = calloc(16, sizeof(uint32_t));
   tf->bit_x[0] = 1;
   tf->bit_y[0] = 1;
   tf->bit_x[1] = 2;
   tf->bit_y[1] = 1;
   tf->bit_x[2] = 3;
   tf->bit_y[2] = 1;
   tf->bit_x[3] = 2;
   tf->bit_y[3] = 2;
   tf->bit_x[4] = 4;
   tf->bit_y[4] = 1;
   tf->bit_x[5] = 4;
   tf->bit_y[5] = 2;
   tf->bit_x[6] = 4;
   tf->bit_y[6] = 3;
   tf->bit_x[7] = 3;
   tf->bit_y[7] = 2;
   tf->bit_x[8] = 4;
   tf->bit_y[8] = 4;
   tf->bit_x[9] = 3;
   tf->bit_y[9] = 4;
   tf->bit_x[10] = 2;
   tf->bit_y[10] = 4;
   tf->bit_x[11] = 3;
   tf->bit_y[11] = 3;
   tf->bit_x[12] = 1;
   tf->bit_y[12] = 4;
   tf->bit_x[13] = 1;
   tf->bit_y[13] = 3;
   tf->bit_x[14] = 1;
   tf->bit_y[14] = 2;
   tf->bit_x[15] = 2;
   tf->bit_y[15] = 3;
   tf->width_at_border = 6;
   tf->total_width = 8;
   tf->reversed_border = true;
   return tf;
}

void tagCustom16h5_destroy(apriltag_family_t *tf)
{
   free(tf->bit_x);
   free(tf->bit_y);
   free(tf->name);
   free(tf);
}
