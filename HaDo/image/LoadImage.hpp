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

namespace hado {

/// @brief A template class for loading and resizing images into Eigen matrices.
/// @tparam T The data type of the matrix elements (e.g., float, double).
template <typename T>
class ImageLoader
{
public:
    /// @brief Loads an image from a file and converts it into a vector of matrices representing the RGB channels.
    /// @param filePath The path to the image file.
    /// @return A vector containing three matrices, each representing one of the RGB channels of the image.
    /// @throws std::runtime_error If the image fails to load or does not have at least 3 color channels (RGB).
    static vector<Matrix<T, Dynamic, Dynamic>> LoadImageAsMatrix(const string &filePath)
    {
        int width, height, channels;
        // Load image data from file
        unsigned char *data = stbi_load(filePath.c_str(), &width, &height, &channels, 0);

        // Check if image data was successfully loaded
        if (data == nullptr)
        {
            throw std::runtime_error("Failed to load image: " + filePath);
        }

        // Ensure the image has at least 3 color channels (RGB)
        if (channels < 3)
        {
            throw std::runtime_error("Image does not have enough color channels (RGB expected): " + filePath);
        }

        vector<Matrix<T, Dynamic, Dynamic>> matrices;
        // Prepare matrices for each color channel
        for (int ch = 0; ch < 3; ++ch)
        {
            matrices.emplace_back(height, width);
        }

        // Populate matrices with image data
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

        // Free the loaded image data
        stbi_image_free(data);
        return matrices;
    }

    /// @brief Resizes a vector of matrices representing an image's RGB channels to new dimensions using bilinear interpolation.
    /// @param imageMatrices The vector containing three matrices, each representing one of the RGB channels of the original image.
    /// @param newWidth The new width for the resized image.
    /// @param newHeight The new height for the resized image.
    /// @return A vector containing three resized matrices, each representing one of the RGB channels of the resized image.
    static vector<Matrix<T, Dynamic, Dynamic>> ResizeImage(
        const vector<Matrix<T, Dynamic, Dynamic>> &imageMatrices,
        int newWidth, int newHeight)
    {

        vector<Matrix<T, Dynamic, Dynamic>> resizedMatrices(3, Matrix<T, Dynamic, Dynamic>(newHeight, newWidth));

        double scaleX = static_cast<double>(imageMatrices[0].cols()) / newWidth;
        double scaleY = static_cast<double>(imageMatrices[0].rows()) / newHeight;

        // Resize each color channel using bilinear interpolation
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

                    // Get the values of the four surrounding pixels
                    T c00 = imageMatrices[ch](max(gyi, 0), max(gxi, 0));
                    T c10 = imageMatrices[ch](max(gyi, 0), min(gxi + 1, static_cast<int>(imageMatrices[ch].cols()) - 1));
                    T c01 = imageMatrices[ch](min(gyi + 1, static_cast<int>(imageMatrices[ch].rows()) - 1), max(gxi, 0));
                    T c11 = imageMatrices[ch](min(gyi + 1, static_cast<int>(imageMatrices[ch].rows()) - 1), min(gxi + 1, static_cast<int>(imageMatrices[ch].cols()) - 1));

                    // Perform bilinear interpolation
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

}

#endif // IMAGELOADER_HPP
