#include <cmath>
#include <cstdint>

#define HEIGHT 128
#define WIDTH 128
#define FILTER_SIZE 3

void bilateral_filter_opencv(uint8_t* in_img, double sigma_color, double sigma_space, uint8_t* out_img) {
    int half_size = FILTER_SIZE / 2;
    double sigma_color2 = 2 * sigma_color * sigma_color;
    double sigma_space2 = 2 * sigma_space * sigma_space;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sum = 0.0;
            double w_sum = 0.0;
            uint8_t center_pixel = in_img[y * WIDTH + x];

            for (int dy = -half_size; dy <= half_size; ++dy) {
                for (int dx = -half_size; dx <= half_size; ++dx) {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (ny >= 0 && ny < HEIGHT && nx >= 0 && nx < WIDTH) {
                        uint8_t neighbor_pixel = in_img[ny * WIDTH + nx];
                        double color_diff = center_pixel - neighbor_pixel;
                        double spatial_weight = exp(-(dx * dx + dy * dy) / sigma_space2);
                        double color_weight = exp(-(color_diff * color_diff) / sigma_color2);
                        double weight = spatial_weight * color_weight;

                        sum += neighbor_pixel * weight;
                        w_sum += weight;
                    }
                }
            }

            out_img[y * WIDTH + x] = static_cast<uint8_t>(sum / w_sum);
        }
    }
}