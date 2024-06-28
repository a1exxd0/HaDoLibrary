#ifndef TENSOR_SERIALIZER_HPP
#define TENSOR_SERIALIZER_HPP

#include <json/json.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
using json = nlohmann::json;

namespace hado {

/**
 * Fully static class
*/
template<typename T=float>
class TensorSerializer {

    // Very naughty of me but oh well
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;
public:

    /**
     * @brief Serialize Eigen matrix to JSON
     * 
     * @param j JSON object to serialize into (by reference)
     * @param matrix Eigen matrix to serialize
    */
    static void to_json(json& j, const Matrix& matrix) {
        j = json::array();
        for (int i = 0; i < matrix.rows(); ++i) {
            j.push_back(json(matrix.row(i)));
        }
    }

    /**
     * @brief Deserialize JSON to Eigen matrix
     * 
     * @param j JSON object to deserialize from
     * @param matrix Eigen matrix to deserialize into (by reference)
    */
    static void from_json(const json& j, Matrix& matrix) {
        matrix.resize(j.size(), j[0].size());
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int k = 0; k < matrix.cols(); ++k) {
                matrix(i, k) = j[i][k];
            }
        }
    }

    /**
     * @brief Serialize vector of Eigen matrices to JSON
     * 
     * @param j JSON object to serialize into (by reference)
     * @param matrices vector of Eigen matrices to serialize
    */
    static void to_json(json& j, const std::vector<Matrix>& matrices) {
        j = json::array();
        for (const auto& matrix : matrices) {
            json matrix_json;
            to_json(matrix_json, matrix);
            j.push_back(matrix_json);
        }
    }

    /**
     * @brief Deserialize JSON to vector of Eigen matrices
     * 
     * @param j JSON object to deserialize from
     * @param matrices vector of Eigen matrices to deserialize into (by reference)
    */
    static void from_json(const json& j, std::vector<Matrix>& matrices) {
        matrices.clear(); // Clear the vector before populating it
        matrices.reserve(j.size()); // Reserve space for efficiency

        for (const auto& matrix_json : j) {
            Matrix matrix;
            from_json(matrix_json, matrix);
            matrices.push_back(matrix);
        }
    }


    static void write_to_file_test(const std::string& filename, const Matrix& matrix) {
        std::ofstream file(filename);
        json j;
        to_json(j, matrix);
        file << j.dump(4);
        file.close();
    }

    static void read_from_file_test(const std::string& filename, Matrix& matrix) {
        std::ifstream input(filename);
        json loaded_json;
        input >> loaded_json;
        input.close();
        from_json(loaded_json, matrix);
    }

    static void write_to_file_test(const std::string& filename, const std::vector<Matrix>& matrices) {
        std::ofstream file(filename);
        json j;
        to_json(j, matrices);
        file << j.dump(4);
        file.close();
    }

    static void read_from_file_test(const std::string& filename, std::vector<Matrix>& matrices) {
        std::ifstream input(filename);
        json loaded_json;
        input >> loaded_json;
        input.close();
        from_json(loaded_json, matrices);
    }
};


}


#endif // TENSOR_SERIALIZER_HPP
