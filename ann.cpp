#include<iostream>
#include<vector>
#include<cmath>
#include<random>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

using namespace std;

class Matrix {
public:
    vector<vector<double>> data;
    int rows, cols;

    Matrix(int rows, int cols);
    Matrix() : rows(0), cols(0) {};

    // Hàm khởi tạo ngẫu nhiên
    void randomize();
    void normalize();

    // Các phép toán ma trận
    Matrix transpose();
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(const Matrix& other) const;
    Matrix scalarMultiply(double scalar) const;
    int argmax(int row) const;

    // Hàm kích hoạt ReLU
    void applyReLU();

    // Hàm Softmax
    void applySoftmax();

    // In ma trận
    void print() const;
};

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    data.resize(rows, vector<double>(cols, 0));
}

void Matrix::normalize() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] /= 255.0;  // Chuyển đổi pixel sang phạm vi [0, 1]
        }
    }
}

// Hàm khởi tạo ngẫu nhiên
void Matrix::randomize() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dis(gen);
        }
    }
}
Matrix Matrix::transpose() {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

// Hàm cộng ma trận
Matrix Matrix::add(const Matrix& other) const {
    if (rows != other.rows && other.rows != 1) {
        throw invalid_argument("Matrix dimensions must match or the other matrix must have one row for broadcasting.");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] + other.data[i % other.rows][j];
        }
    }
    return result;
}

// Hàm trừ ma trận
Matrix Matrix::subtract(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw invalid_argument("Matrix dimensions must match for subtraction.");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

// Hàm nhân ma trận
Matrix Matrix::multiply(const Matrix& other) const {
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            for (int k = 0; k < cols; k++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

// Hàm nhân với số vô hướng
Matrix Matrix::scalarMultiply(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

// Hàm ReLU
void Matrix::applyReLU() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = max(0.0, data[i][j]);
        }
    }
}

// Hàm Softmax
void Matrix::applySoftmax() {
    for (int i = 0; i < rows; i++) {
        double max_val = data[i][0];
        for (int j = 1; j < cols; j++) {
            max_val = max(max_val, data[i][j]);
        }

        double sum_exp = 0.0;
        for (int j = 0; j < cols; j++) {
            sum_exp += exp(data[i][j] - max_val);
        }

        for (int j = 0; j < cols; j++) {
            data[i][j] = exp(data[i][j] - max_val) / sum_exp;
        }
    }
}

// In ma trận
void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << data[i][j] << " ";
        }
        cout << endl;
    }
}

int Matrix::argmax(int row) const {
    if (row < 0 || row >= rows) {
        throw std::out_of_range("Row index out of range");
    }

    int max_index = 0;
    double max_value = data[row][0];

    for (int col = 1; col < cols; col++) {
        if (data[row][col] > max_value) {
            max_value = data[row][col];
            max_index = col;
        }
    }
    return max_index;
}


// Tính Cross-Entropy Loss
double crossEntropyLoss(const Matrix& predictions, const Matrix& targets) {
    double epsilon = 1e-7; // Small value to prevent log(0)
    double loss = 0.0;
    for (int i = 0; i < predictions.rows; ++i) {
        for (int j = 0; j < predictions.cols; ++j) {
            double pred = predictions.data[i][j];
            double target = targets.data[i][j];
            loss -= target * log(pred + epsilon);
        }
    }
    return loss / predictions.rows;
}


class NeuralNetwork {
public:
    Matrix W1, b1, W_hidden2, b_hidden2, W2, b2;
    const int size = 50;
    // Khởi tạo net 1 input - 2 hidden - 1 output
    // b1, b2, b_hidden2 là bias và mặc định là 0
    NeuralNetwork(int input_size, int hidden_size1, int hidden_size2, int output_size)
        : W1(input_size, hidden_size1), b1(60000, hidden_size1),
        W_hidden2(hidden_size1, hidden_size2), b_hidden2(60000, hidden_size2),
        W2(hidden_size2, output_size), b2(60000, output_size) {
        
        // Giá trị bộ trọng số được khởi tạo ngẫu nhiên
        W1.randomize();
        

        W_hidden2.randomize();
    

        W2.randomize();
   
    }

    void train(const Matrix& input, const Matrix& target, double learning_rate, int epochs);
};

void NeuralNetwork::train(const Matrix& input, const Matrix& target, double learning_rate, int epochs) {
    Matrix X = input;  // Copy dữ liệu vào một biến mới
    X.normalize();      // Chuẩn hóa dữ liệu đầu vào

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        // Lớp ẩn đầu tiên
        Matrix Z1 = X.multiply(W1).add(b1);
        Z1.applyReLU();

        // Lớp ẩn thứ hai
        Matrix Z2_hidden = Z1.multiply(W_hidden2).add(b_hidden2);
        Z2_hidden.applyReLU();

        // Lớp đầu ra
        Matrix Z2 = Z2_hidden.multiply(W2).add(b2);
        Z2.applySoftmax();

        // Tính mất mát
        double loss = crossEntropyLoss(Z2, target);

        // Tính độ chính xác
        int correct_predictions = 0;
        for (int i = 0; i < Z2.rows; i++) {

            if (Z2.argmax(i) == target.argmax(i)) {
                correct_predictions++;
            }
        }
        double accuracy = static_cast<double>(correct_predictions) / Z2.rows;

        cout << "Epoch " << epoch << ", Loss: " << loss << ", Accuracy: " << accuracy * 100 << "%" << endl;
        cout << "Epoch " << epoch << ", Loss: " << loss << endl;

       // Backpropagation
        Matrix dZ_output = Z2.subtract(target); // Gradient tại output layer
        Matrix dW_output = Z2_hidden.transpose().multiply(dZ_output);
        Matrix db_output = dZ_output;

        // Gradient tại hidden layer 2
        Matrix dZ_hidden2 = dZ_output.multiply(W2.transpose());
        dZ_hidden2.applyReLU();

        Matrix dW_hidden2 = Z1.transpose().multiply(dZ_hidden2);
        Matrix db_hidden2 = dZ_hidden2;

        // Gradient tại hidden layer 1
        Matrix dZ_hidden1 = dZ_hidden2.multiply(W_hidden2.transpose());
        dZ_hidden1.applyReLU();

        Matrix dW_hidden1 = X.transpose().multiply(dZ_hidden1);
        Matrix db_hidden1 = dZ_hidden1;

        // Cập nhật trọng số và bias
        W1 = W1.subtract(dW_hidden1.scalarMultiply(learning_rate));
        b1 = b1.subtract(db_hidden1.scalarMultiply(learning_rate));
        W_hidden2 = W_hidden2.subtract(dW_hidden2.scalarMultiply(learning_rate));
        b_hidden2 = b_hidden2.subtract(db_hidden2.scalarMultiply(learning_rate));
        W2 = W2.subtract(dW_output.scalarMultiply(learning_rate));
        b2 = b2.subtract(db_output.scalarMultiply(learning_rate));

    }
}

void loadData(const string& filename, Matrix& X, Matrix& Y, int sample) {
    try {
        ifstream file(filename);
        if (!file.is_open()) {
            throw runtime_error("Error: Could not open file " + filename);
        }

        string line;

        // Skip the header line (first line)
        if (!getline(file, line)) {
            throw runtime_error("Error: The file is empty or cannot read the header.");
        }

        int row = 0;

        // Process the rest of the file
        while (getline(file, line) && (row < sample)) {

            stringstream ss(line);
            string value;
            vector<double> image_pixels;
            string label;

            // Đọc nhãn (label) của ảnh
            if (!getline(ss, label, ',')) {
                throw runtime_error("Error: Invalid label format in line " + to_string(row + 2)); // +2 to account for the header
            }

            // Đọc các giá trị pixel của ảnh
            while (getline(ss, value, ',')) {
                try {
                    image_pixels.push_back(stod(value));  // Chuyển string thành double
                }
                catch (const invalid_argument& e) {
                    throw runtime_error("Error: Invalid pixel value in line " + to_string(row + 2)); // +2 to account for the header
                }
            }

            // Lưu nhãn vào ma trận Y
            for (int i = 0; i < Y.cols; ++i) {
                Y.data[row][i] = (i == stoi(label)) ? 1.0 : 0.0;
            }

            // Lưu các pixel vào ma trận X
            if (image_pixels.size() != X.cols) {
                throw runtime_error("Error: Mismatched number of pixels in line " + to_string(row + 2)); // +2 to account for the header
            }

            for (int i = 0; i < X.cols; ++i) {
                X.data[row][i] = image_pixels[i];
            }

            ++row;
        }

        if (row == 0) {
            throw runtime_error("Error: No data found in file " + filename);
        }

        file.close();
    }
    catch (const exception& e) {
        cerr << e.what() << endl;
    }
}


int main() {
    int input_size = 784;  // 28x28 pixel cho mỗi ảnh
    int hidden_size1 = 128;
    int hidden_size2 = 128;
    int output_size = 10;  // 10 lớp cho 10 nhãn

    Matrix X_train(60000, input_size);  // Tập huấn luyện (60000 ảnh)
    Matrix Y_train(60000, output_size); // Nhãn tương ứng

    // Đọc dữ liệu từ file CSV
    loadData("fashion-mnist_train.csv", X_train, Y_train, 60000);

    cout << "X_train shape: " << X_train.rows << " " << X_train.cols << endl;
    cout << "Y_train shape: " << Y_train.rows << " " << Y_train.cols << endl;
    // Tạo và huấn luyện mạng
    NeuralNetwork nn(input_size, hidden_size1,hidden_size2, output_size);
    nn.train(X_train, Y_train, 0.0001, 10);

    return 0;
}