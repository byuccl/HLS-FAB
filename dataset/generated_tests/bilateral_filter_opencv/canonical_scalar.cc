#include <cmath>
#include <cstdint>

#define HEIGHT 128
#define WIDTH 128
#define FILTER_SIZE 3

void bilateral_filter_opencv(uint8_t *in_img, double sigma_color, double sigma_space, uint8_t *out_img) {
    int half_filter = FILTER_SIZE / 2;
    double color_coeff = -0.5 / (sigma_color * sigma_color);
    double space_coeff = -0.5 / (sigma_space * sigma_space);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sum = 0.0;
            double norm_factor = 0.0;
            uint8_t center_pixel = in_img[y * WIDTH + x];

            for (int fy = -half_filter; fy <= half_filter; ++fy) {
                for (int fx = -half_filter; fx <= half_filter; ++fx) {
                    int ny = y + fy;
                    int nx = x + fx;

                    if (ny >= 0 && ny < HEIGHT && nx >= 0 && nx < WIDTH) {
                        uint8_t neighbor_pixel = in_img[ny * WIDTH + nx];
                        double color_diff = center_pixel - neighbor_pixel;
                        double space_diff = fy * fy + fx * fx;

                        double weight = exp(space_diff * space_coeff) * exp(color_diff * color_diff * color_coeff);
                        sum += weight * neighbor_pixel;
                        norm_factor += weight;
                    }
                }
            }

            out_img[y * WIDTH + x] = static_cast<uint8_t>(sum / norm_factor);
        }
    }
}