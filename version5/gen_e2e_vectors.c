#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../model_params.h"

#define MNIST_IMAGES "../data/MNIST/raw/t10k-images-idx3-ubyte"
#define MNIST_LABELS "../data/MNIST/raw/t10k-labels-idx1-ubyte"
#define INPUT_MEAN 0.1307f
#define INPUT_STD 0.3081f

static uint8_t clamp_u8(int64_t value)
{
    if (value < 0)
        return 0;
    if (value > 255)
        return 255;
    return (uint8_t)value;
}

static void conv(const uint8_t *input, uint8_t *output,
                 const int8_t *weights, const int32_t *bias,
                 const int16_t *mult, const int32_t *shift,
                 int zx, int zout, int in_c, int in_h, int in_w, int out_c)
{
    const int out_h = in_h - 2;
    const int out_w = in_w - 2;
    int oc, oh, ow, ic, kh, kw;

    for (oc = 0; oc < out_c; ++oc) {
        for (oh = 0; oh < out_h; ++oh) {
            for (ow = 0; ow < out_w; ++ow) {
                int32_t acc = bias[oc];
                int64_t q;
                for (ic = 0; ic < in_c; ++ic)
                    for (kh = 0; kh < 3; ++kh)
                        for (kw = 0; kw < 3; ++kw) {
                            int ii = ic * in_h * in_w +
                                     (oh + kh) * in_w + ow + kw;
                            int wi = oc * in_c * 9 + ic * 9 + kh * 3 + kw;
                            acc += ((int32_t)input[ii] - zx) * weights[wi];
                        }
                q = ((int64_t)acc * mult[oc] >> (15 + shift[oc])) + zout;
                if (q < zout)
                    q = zout;
                output[oc * out_h * out_w + oh * out_w + ow] = clamp_u8(q);
            }
        }
    }
}

static void pool(const uint8_t *input, uint8_t *output, int channels, int h, int w)
{
    const int out_h = h / 2;
    const int out_w = w / 2;
    int c, oh, ow, kh, kw;

    for (c = 0; c < channels; ++c)
        for (oh = 0; oh < out_h; ++oh)
            for (ow = 0; ow < out_w; ++ow) {
                uint8_t max_value = 0;
                for (kh = 0; kh < 2; ++kh)
                    for (kw = 0; kw < 2; ++kw) {
                        uint8_t value = input[c * h * w +
                                              (oh * 2 + kh) * w + ow * 2 + kw];
                        if ((kh == 0 && kw == 0) || value > max_value)
                            max_value = value;
                    }
                output[c * out_h * out_w + oh * out_w + ow] = max_value;
            }
}

static void fc(const uint8_t *input, uint8_t *output)
{
    int o, i;
    for (o = 0; o < 10; ++o) {
        int32_t acc = fc_b[o];
        for (i = 0; i < 200; ++i)
            acc += ((int32_t)input[i] - fc_Zx) * fc_w[o * 200 + i];
        output[o] = clamp_u8(((int64_t)acc * fc_m[o] >>
                              (15 + fc_s[o])) + fc_Zo);
    }
}

static void write_bytes(const char *path, const uint8_t *data, int count, int append)
{
    FILE *file = fopen(path, append ? "a" : "w");
    int i;
    if (!file) {
        perror(path);
        exit(1);
    }
    for (i = 0; i < count; ++i)
        fprintf(file, "%02x\n", data[i]);
    fclose(file);
}

int main(int argc, char **argv)
{
    uint8_t input[28 * 28];
    uint8_t conv1[4 * 26 * 26];
    uint8_t pool1[4 * 13 * 13];
    uint8_t conv2[8 * 11 * 11];
    uint8_t pool2[8 * 5 * 5];
    uint8_t fc_out[10];
    FILE *input_file;
    FILE *dataset;
    int first_index = argc > 1 ? atoi(argv[1]) : 0;
    int case_count = argc > 2 ? atoi(argv[2]) : 1;
    int case_id, index, i, best;
    uint8_t label;

    if (first_index < 0 || case_count <= 0 || first_index + case_count > 10000) {
        fprintf(stderr, "MNIST case range must be inside [0, 9999]\n");
        return 1;
    }

    for (case_id = 0; case_id < case_count; ++case_id) {
        index = first_index + case_id;
        best = 0;

        dataset = fopen(MNIST_IMAGES, "rb");
        if (!dataset || fseek(dataset, 16L + index * 28L * 28L, SEEK_SET)) {
            perror(MNIST_IMAGES);
            return 1;
        }
        for (i = 0; i < 28 * 28; ++i) {
            uint8_t pixel;
            float normalized;
            int32_t quantized;
            if (fread(&pixel, 1, 1, dataset) != 1) {
                fprintf(stderr, "short MNIST image read\n");
                return 1;
            }
            normalized = (pixel / 255.0f - INPUT_MEAN) / INPUT_STD;
            quantized = (int32_t)lrintf(normalized / in_scale) + in_zp;
            input[i] = clamp_u8(quantized);
        }
        fclose(dataset);

        dataset = fopen(MNIST_LABELS, "rb");
        if (!dataset || fseek(dataset, 8L + index, SEEK_SET) ||
            fread(&label, 1, 1, dataset) != 1) {
            perror(MNIST_LABELS);
            return 1;
        }
        fclose(dataset);

        conv(input, conv1, c0_w, c0_b, c0_m, c0_s,
             c0_Zx, c0_Zo, 1, 28, 28, 4);
        pool(conv1, pool1, 4, 26, 26);
        conv(pool1, conv2, c1_w, c1_b, c1_m, c1_s,
             c1_Zx, c1_Zo, 4, 13, 13, 8);
        pool(conv2, pool2, 8, 11, 11);
        fc(pool2, fc_out);

        input_file = fopen("e2e_input.hex", case_id ? "a" : "w");
        if (!input_file) {
            perror("e2e_input.hex");
            return 1;
        }
        for (i = 0; i < 28 * 28; i += 4) {
            uint32_t word = input[i] | ((uint32_t)input[i + 1] << 8) |
                            ((uint32_t)input[i + 2] << 16) |
                            ((uint32_t)input[i + 3] << 24);
            fprintf(input_file, "%08x\n", word);
        }
        fclose(input_file);

        write_bytes("e2e_conv1.hex", conv1, 4 * 26 * 26, case_id);
        write_bytes("e2e_pool1.hex", pool1, 4 * 13 * 13, case_id);
        write_bytes("e2e_conv2.hex", conv2, 8 * 11 * 11, case_id);
        write_bytes("e2e_pool2.hex", pool2, 8 * 5 * 5, case_id);
        write_bytes("e2e_fc.hex", fc_out, 10, case_id);
        write_bytes("e2e_label.hex", &label, 1, case_id);

        for (i = 1; i < 10; ++i)
            if (fc_out[i] > fc_out[best])
                best = i;
        if (best != label) {
            fprintf(stderr, "E2E VECTOR FAIL index=%d class=%d label=%u score=%u\n",
                    index, best, label, fc_out[best]);
            return 1;
        }
        printf("E2E VECTOR PASS index=%d class=%d label=%u score=%u\n",
               index, best, label, fc_out[best]);
    }

    printf("E2E VECTOR SUMMARY first=%d cases=%d\n", first_index, case_count);
    return 0;
}
