#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../model_params.h"

static int emit_layer(FILE *wf, FILE *pf, const int8_t *weights,
                      const int32_t *bias, const int16_t *mult,
                      const int32_t *shift, int zx, int zout,
                      int out_channels, int k_size, int relu)
{
    int words = 0;
    int oc;
    int k;

    for (oc = 0; oc < out_channels; oc += 4) {
        for (k = 0; k < k_size; ++k) {
            uint32_t word = 0;
            int lane;
            for (lane = 0; lane < 4; ++lane) {
                if (oc + lane < out_channels)
                    word |= (uint32_t)(uint8_t)weights[(oc + lane) * k_size + k]
                            << (lane * 8);
            }
            fprintf(wf, "%08" PRIx32 "\n", word);
            ++words;
        }
    }

    for (oc = 0; oc < out_channels; ++oc) {
        int64_t sum = 0;
        int64_t corrected;
        uint64_t low;
        for (k = 0; k < k_size; ++k)
            sum += weights[oc * k_size + k];
        corrected = (int64_t)bias[oc] + (128 - zx) * sum;
        if (corrected < INT32_MIN || corrected > INT32_MAX) {
            fprintf(stderr, "corrected bias overflow at channel %d\n", oc);
            exit(1);
        }
        low = (uint32_t)(int32_t)corrected;
        low |= (uint64_t)(uint16_t)mult[oc] << 32;
        low |= (uint64_t)(uint8_t)shift[oc] << 48;
        low |= (uint64_t)(uint8_t)zout << 56;
        fprintf(pf, "%x%016" PRIx64 "\n", relu ? 1 : 0, low);
    }
    return words;
}

int main(void)
{
    FILE *wf = fopen("model_weights.hex", "w");
    FILE *pf = fopen("model_params_hw.hex", "w");
    int words = 0;

    if (!wf || !pf) {
        perror("model output");
        return 1;
    }

    words += emit_layer(wf, pf, c0_w, c0_b, c0_m, c0_s,
                        c0_Zx, c0_Zo, 4, 9, 1);
    words += emit_layer(wf, pf, c1_w, c1_b, c1_m, c1_s,
                        c1_Zx, c1_Zo, 8, 36, 1);
    words += emit_layer(wf, pf, fc_w, fc_b, fc_m, fc_s,
                        fc_Zx, fc_Zo, 10, 200, 0);
    fclose(wf);
    fclose(pf);

    if (words != 681) {
        fprintf(stderr, "weight word count %d, expected 681\n", words);
        return 1;
    }
    printf("MODEL PACK PASS weights=%d params=22\n", words);
    return 0;
}
