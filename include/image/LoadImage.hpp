#ifndef IMAGELOADER_HPP
#define IMAGELOADER_HPP
#define STB_IMAGE_IMPLEMENTATION

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <algorithm>
#include <stb/stb_image.h>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::max;
using std::min;
using std::string;
using std::vector;

template <typename T>
class ImageLoader
{
public:
    static vector<Matrix<T, Dynamic, Dynamic>> LoadImageAsMatrix(const string &filePath)
    {
        int width, height, channels;
        unsigned char *data = stbi_load(filePath.c_str(), &width, &height, &channels, 0);

        if (data == nullptr)
        {
            throw std::runtime_error("Failed to load image: " + filePath);
        }

        if (channels < 3)
        {
            throw std::runtime_error("Image does not have enough color channels (RGB expected): " + filePath);
        }

        vector<Matrix<T, Dynamic, Dynamic>> matrices;
        for (int ch = 0; ch < 3; ++ch)
        {
            matrices.emplace_back(height, width);
        }

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                for (int ch = 0; ch < 3; ++ch)
                {
                    int index = (i * width + j) * channels + ch;
                    matrices[ch](i, j) = static_cast<T>(data[index]);
                }
            }
        }

        stbi_image_free(data);
        return matrices;
    }

    static vector<Matrix<T, Dynamic, Dynamic>> ResizeImage(
        const vector<Matrix<T, Dynamic, Dynamic>> &imageMatrices,
        int newWidth, int newHeight)
    {

        vector<Matrix<T, Dynamic, Dynamic>> resizedMatrices(3, Matrix<T, Dynamic, Dynamic>(newHeight, newWidth));

        double scaleX = static_cast<double>(imageMatrices[0].cols()) / newWidth;
        double scaleY = static_cast<double>(imageMatrices[0].rows()) / newHeight;

        for (int ch = 0; ch < 3; ++ch)
        {
            for (int y = 0; y < newHeight; ++y)
            {
                for (int x = 0; x < newWidth; ++x)
                {
                    double gx = ((x + 0.5) * scaleX) - 0.5;
                    double gy = ((y + 0.5) * scaleY) - 0.5;
                    int gxi = static_cast<int>(gx);
                    int gyi = static_cast<int>(gy);

                    T c00 = imageMatrices[ch](max(gyi, 0), max(gxi, 0));
                    T c10 = imageMatrices[ch](max(gyi, 0), min(gxi + 1, static_cast<int>(imageMatrices[ch].cols()) - 1));
                    T c01 = imageMatrices[ch](min(gyi + 1, static_cast<int>(imageMatrices[ch].rows()) - 1), max(gxi, 0));
                    T c11 = imageMatrices[ch](min(gyi + 1, static_cast<int>(imageMatrices[ch].rows()) - 1), min(gxi + 1, static_cast<int>(imageMatrices[ch].cols()) - 1));

                    T tx = gx - gxi;
                    T ty = gy - gyi;

                    T a = c00 * (1 - tx) + c10 * tx;
                    T b = c01 * (1 - tx) + c11 * tx;
                    T c = a * (1 - ty) + b * ty;

                    resizedMatrices[ch](y, x) = c;
                }
            }
        }

        return resizedMatrices;
    }
};

#endif // IMAGELOADER_HPP
