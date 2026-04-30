#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "model_params.h"   // ? 你生成的头文件

#define IN_H 28
#define IN_W 28

#define CONV1_OUT_C 16
#define CONV1_K 3
#define CONV1_OUT_H 26
#define CONV1_OUT_W 26

#define POOL1_H 13
#define POOL1_W 13

#define CONV2_OUT_C 32
#define CONV2_K 3
#define CONV2_OUT_H 11
#define CONV2_OUT_W 11

#define POOL2_H 5
#define POOL2_W 5

#define FC_IN (32 * 5 * 5)
#define FC_OUT 10

#define INPUT_MEAN 0.1307f
#define INPUT_STD 0.3081f

// ============================
// 结构体定义
// ============================
typedef struct {
    const int8_t  *W;
    const int32_t *b;
    const int32_t *mult;
    const int32_t *shift;
    int32_t Zx;
    int32_t Zout;
} LayerInt8;

typedef struct {
    LayerInt8 conv1;
    LayerInt8 conv2;
    LayerInt8 fc;
} Model;



	float S0 = in_scale;
	int32_t Z0 = in_zp;
// ============================
// 模型加载（从 .h 绑定）
// ============================
static Model load_model(void)
{
    Model m;

	
    m.conv1.W = c0_w;
    m.conv1.b = c0_b;
    m.conv1.mult = c0_m;
    m.conv1.shift = c0_s;
    m.conv1.Zx = c0_Zx;
    m.conv1.Zout = c0_Zo;

    m.conv2.W = c1_w;
    m.conv2.b = c1_b;
    m.conv2.mult = c1_m;
    m.conv2.shift = c1_s;
    m.conv2.Zx = c1_Zx;
    m.conv2.Zout = c1_Zo;

    m.fc.W = fc_w;
    m.fc.b = fc_b;
    m.fc.mult = fc_m;
    m.fc.shift = fc_s;
    m.fc.Zx = fc_Zx;
    m.fc.Zout = fc_Zo;

    return m;
}

// ============================
// 工具函数
// ============================
static uint8_t clamp_u8(int32_t x) {
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    return (uint8_t)x;
}

// ============================
// Conv（纯整数）
// ============================
static void conv2d_int8(
    const uint8_t *input,
    uint8_t *output,
    const LayerInt8 *layer,
    int in_c, int in_h, int in_w,
    int out_c, int K
) {
    int out_h = in_h - K + 1;
    int out_w = in_w - K + 1;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {

                int32_t acc = 0;

                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {

                            int in_idx =
                                ic*in_h*in_w +
                                (oh+kh)*in_w +
                                (ow+kw);

                            int w_idx =
                                oc*in_c*K*K +
                                ic*K*K +
                                kh*K +
                                kw;

                            int32_t x = (int32_t)input[in_idx] - layer->Zx;
                            int32_t w = (int32_t)layer->W[w_idx];

                            acc += x * w;
                        }
                    }
                }

                acc += layer->b[oc];

                int64_t tmp = (int64_t)acc * layer->mult[oc];
                int32_t q = (int32_t)(tmp >> (31 + layer->shift[oc]));

                q += layer->Zout;

                // ReLU
                if (q < layer->Zout) q = layer->Zout;

                output[oc*out_h*out_w + oh*out_w + ow] = clamp_u8(q);
            }
        }
    }
}

// ============================
// MaxPool
// ============================
static void maxpool2d_q8(
    const uint8_t *input,
    uint8_t *output,
    int c, int h, int w
) {
    int out_h = h / 2;
    int out_w = w / 2;

    for (int ic = 0; ic < c; ic++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {

                uint8_t max_v = 0;

                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int ih = oh*2 + kh;
                        int iw = ow*2 + kw;

                        uint8_t v = input[ic*h*w + ih*w + iw];

                        if (kh == 0 && kw == 0) max_v = v;
                        else if (v > max_v) max_v = v;
                    }
                }

                output[ic*out_h*out_w + oh*out_w + ow] = max_v;
            }
        }
    }
}

// ============================
// FC
// ============================
static void linear_int8(
    const uint8_t *input,
    uint8_t *output,
    const LayerInt8 *layer,
    int in_size,
    int out_size
) {
    for (int o = 0; o < out_size; o++) {

        int32_t acc = 0;

        for (int i = 0; i < in_size; i++) {
            int32_t x = (int32_t)input[i] - layer->Zx;
            int32_t w = (int32_t)layer->W[o*in_size + i];

            acc += x * w;
        }

        acc += layer->b[o];

        int64_t tmp = (int64_t)acc * layer->mult[o];
        int32_t q = (int32_t)(tmp >> (31 + layer->shift[o]));

        q += layer->Zout;

        output[o] = clamp_u8(q);
    }
}

// ============================
// MNIST 读取 + 量化
// ============================
static uint8_t *load_mnist_image_quantized(const char *filename, int index)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    fseek(fp, 16 + index * 28 * 28, SEEK_SET);

    uint8_t *img = malloc(28*28);

    for (int i = 0; i < 28*28; i++) {
        uint8_t p;
        fread(&p, 1, 1, fp);

        float x = (p / 255.0f - INPUT_MEAN) / INPUT_STD;
        int32_t q = (int32_t)lrintf(x / S0) + Z0;

        img[i] = clamp_u8(q);
    }

    fclose(fp);
    return img;
}

static int load_label(const char *filename, int index)
{
    FILE *fp = fopen(filename, "rb");
    fseek(fp, 8 + index, SEEK_SET);

    uint8_t l;
    fread(&l, 1, 1, fp);
    fclose(fp);

    return l;
}

// ============================
// 推理
// ============================
static int inference(uint8_t *input, Model *m)
{
    uint8_t conv1_out[16*26*26];
    uint8_t pool1_out[16*13*13];
    uint8_t conv2_out[32*11*11];
    uint8_t pool2_out[32*5*5];
    uint8_t fc_out[10];

    conv2d_int8(input, conv1_out, &m->conv1, 1, 28, 28, 16, 3);

    maxpool2d_q8(conv1_out, pool1_out, 16, 26, 26);

    conv2d_int8(pool1_out, conv2_out, &m->conv2, 16, 13, 13, 32, 3);
    	/* ===== DEBUG: Conv1 输出分布 ===== */
		int min = 255, max = 0;
		for (int i = 0; i < 16*26*26; i++) {
		    if (conv2_out[i] < min) min = conv2_out[i];
		    if (conv2_out[i] > max) max = conv2_out[i];
		}
		printf("conv2: min=%d max=%d\n", min, max);
		/* ================================= */
    
    maxpool2d_q8(conv2_out, pool2_out, 32, 11, 11);

    linear_int8(pool2_out, fc_out, &m->fc, 800, 10);

    int idx = 0;
    uint8_t maxv = fc_out[0];

    for (int i = 1; i < 10; i++) {
        if (fc_out[i] > maxv) {
            maxv = fc_out[i];
            idx = i;
        }
    }

    return idx;
}

// ============================
// main
// ============================
int main()
{
    Model model = load_model();

    int correct = 0;
    int total = 100;

    for (int i = 0; i < total; i++) {

        uint8_t *img = load_mnist_image_quantized(
            "data/MNIST/raw/t10k-images-idx3-ubyte", i);

        int label = load_label(
            "data/MNIST/raw/t10k-labels-idx1-ubyte", i);

        int pred = inference(img, &model);

        if (pred == label) correct++;

        printf("%d: pred=%d label=%d\n", i, pred, label);

        free(img);
    }

    printf("ACC = %.4f\n", (float)correct / total);

    return 0;
}
