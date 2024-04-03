#ifndef MATRIX_SERIALIZER_HPP
#define MATRIX_SERIALIZER_HPP

#include "json/json.hpp"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
using json = nlohmann::json;

/**
 * Fully static class
*/
template<typename T=float>
class MatrixSerializer {

    // Very naughty of me but oh well
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;
public:

    static void to_json(json& j, const Matrix& matrix) {
        j = json::array();
        for (int i = 0; i < matrix.rows(); ++i) {
            j.push_back(json(matrix.row(i)));
        }
    }

    // Deserialize Eigen matrix from JSON
    static void from_json(const json& j, Matrix& matrix) {
        matrix.resize(j.size(), j[0].size());
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int k = 0; k < matrix.cols(); ++k) {
                matrix(i, k) = j[i][k];
            }
        }
    }

    static void write_to_file(const std::string& filename, const Matrix& matrix) {
        std::ofstream file(filename);
        json j;
        to_json(j, matrix);
        file << j.dump(4);
        file.close();
    }

    static void read_from_file(const std::string& filename, Matrix& matrix) {
        std::ifstream input(filename);
        json loaded_json;
        input >> loaded_json;
        input.close();
        from_json(loaded_json, matrix);
    }
};


#endif