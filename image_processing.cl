#define PI (3.1415926535897)
__kernel void image_rotate_gpu(__global unsigned char *pixels,
                               __global unsigned char *pixels_result,
                               const unsigned int width,
                               const unsigned int height,
                               const float degree)
{
    int idx = get_global_id(0);
    float y_new = idx / width;
    float x_new = idx % width;
    int y_orig, x_orig;

    float rad = PI * degree / 180.f;
    rad *= -1.f;  // inverse transform
    
    float x_centered = x_new - width / 2.0;  
    float y_centered = y_new - height / 2.0; 

    x_orig = x_centered * cos(rad) - y_centered * sin(rad) + width / 2.0; 
    y_orig = x_centered * sin(rad) + y_centered * cos(rad) + height / 2.0; 

    if (x_orig < 0)         x_orig = -x_orig - 1;
    if (x_orig >= width)    x_orig = 2 * width - x_orig - 1;
    if (y_orig < 0)         y_orig = -y_orig - 1;
    if (y_orig >= height)   y_orig = 2 * height - y_orig - 1;

    int orig_idx = y_orig * width + x_orig;

    unsigned char r, g, b;
    b = pixels[orig_idx * 3 + 0];
    g = pixels[orig_idx * 3 + 1];
    r = pixels[orig_idx * 3 + 2];

    pixels_result[idx * 3 + 0] = b;
    pixels_result[idx * 3 + 1] = g;
    pixels_result[idx * 3 + 2] = r;
}
