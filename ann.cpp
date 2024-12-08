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
    void initialBias();
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

    // Đạo hàm hàm kích hoạt ReLU
    void applyReLUDerivative();

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

    // Áp dụng He Initialization cho mỗi lớp
    double bound = sqrt(2.0 / rows);  // rows là số lượng neuron đầu vào của lớp

    uniform_real_distribution<> dis(-bound, bound);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dis(gen);
        }
    }
}

// Hàm khởi tạo ngẫu nhiên
void Matrix::initialBias() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = 0.01;
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

// Đạo hàm ReLU
void Matrix::applyReLUDerivative() {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Áp dụng đạo hàm ReLU: nếu giá trị > 0, đạo hàm = 1, nếu <= 0, đạo hàm = 0
            data[i][j] = data[i][j] > 0 ? 1.0 : 0.0;
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
Matrix crossEntropyLossMatrix(const Matrix& predictions, const Matrix& targets) {
    double epsilon = 1e-15; // Small value to prevent log(0)
    Matrix loss(predictions.rows, predictions.cols);
    for (int i = 0; i < predictions.rows; ++i) {
        for (int j = 0; j < predictions.cols; ++j) {
            double pred = predictions.data[i][j];
            double target = targets.data[i][j];
            loss.data[i][j] = target * log(pred + epsilon);
        }
    }
    return loss;
}


class NeuralNetwork {
public:
    Matrix W1, b1, W_hidden2, b_hidden2, W2, b2;
    int batch_size;

    NeuralNetwork(int input_size, int hidden_size1, int hidden_size2, int output_size, int batch_size)
        : W1(input_size, hidden_size1), b1(batch_size, hidden_size1),
        W_hidden2(hidden_size1, hidden_size2), b_hidden2(batch_size, hidden_size2),
        W2(hidden_size2, output_size), b2(batch_size, output_size) {

        W1.randomize();
        b1.initialBias();
        

        W_hidden2.randomize();
        b_hidden2.initialBias();

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
        Matrix lossMatrix = crossEntropyLossMatrix(Z2, target);
        // Calculate the average loss
        double averageLoss = 0.0;
        for (int i = 0; i < lossMatrix.rows; ++i) {
            for (int j = 0; j < lossMatrix.cols; ++j) {
                averageLoss -= lossMatrix.data[i][j];
            }
        }
        averageLoss /= lossMatrix.rows;

        // Tính độ chính xác
        int correct_predictions = 0;
        for (int i = 0; i < Z2.rows; i++) {

            if (Z2.argmax(i) == target.argmax(i)) {
                correct_predictions++;
            }
        }
        double accuracy = static_cast<double>(correct_predictions) / Z2.rows;

        cout << "Epoch " << epoch << ", Loss: " << averageLoss << ", Accuracy: " << accuracy * 100 << "%" << endl;
   

       // Backpropagation
        Matrix dZ_output = lossMatrix; // Gradient tại output layer
        Matrix dW_output = Z2_hidden.transpose().multiply(dZ_output);
        Matrix db_output = dZ_output;

        // Gradient tại hidden layer 2
        Matrix dZ_hidden2 = dZ_output.multiply(W2.transpose());
        dZ_hidden2.applyReLUDerivative();

        Matrix dW_hidden2 = Z1.transpose().multiply(dZ_hidden2);
        Matrix db_hidden2 = dZ_hidden2;

        // Gradient tại hidden layer 1
        Matrix dZ_hidden1 = dZ_hidden2.multiply(W_hidden2.transpose());
        dZ_hidden1.applyReLUDerivative();

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

void saveModel(const NeuralNetwork& nn, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file to save model." << endl;
        return;
    }

    auto saveMatrix = [&file](const Matrix& mat) {
        file << mat.rows << " " << mat.cols << endl;
        for (const auto& row : mat.data) {
            for (double val : row) {
                file << val << " ";
            }
            file << endl;
        }
        };

    saveMatrix(nn.W1);
    saveMatrix(nn.b1);
    saveMatrix(nn.W_hidden2);
    saveMatrix(nn.b_hidden2);
    saveMatrix(nn.W2);
    saveMatrix(nn.b2);

    file.close();
    cout << "Model saved to " << filename << endl;
}

void loadModel(NeuralNetwork& nn, const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file to load model." << endl;
        return;
    }

    auto loadMatrix = [&file](Matrix& mat) {
        file >> mat.rows >> mat.cols;
        mat.data.resize(mat.rows, vector<double>(mat.cols));
        for (auto& row : mat.data) {
            for (double& val : row) {
                file >> val;
            }
        }
        };

    loadMatrix(nn.W1);
    loadMatrix(nn.b1);
    loadMatrix(nn.W_hidden2);
    loadMatrix(nn.b_hidden2);
    loadMatrix(nn.W2);
    loadMatrix(nn.b2);

    file.close();
    cout << "Model loaded from " << filename << endl;
}

void predict(const NeuralNetwork& nn, const Matrix& input, Matrix& output) {
    Matrix X = input;
    X.normalize();

    // Forward pass
    Matrix Z1 = X.multiply(nn.W1).add(nn.b1);
    Z1.applyReLU();

    Matrix Z2_hidden = Z1.multiply(nn.W_hidden2).add(nn.b_hidden2);
    Z2_hidden.applyReLU();

    Matrix Z2 = Z2_hidden.multiply(nn.W2).add(nn.b2);
    Z2.applySoftmax();

    output = Z2;
}

void evaluate(const NeuralNetwork& nn, const Matrix& X_test, const Matrix& Y_test) {
    Matrix predictions;
    predict(nn, X_test, predictions);

    int correct_predictions = 0;
    for (int i = 0; i < predictions.rows; ++i) {
        if (predictions.argmax(i) == Y_test.argmax(i)) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / predictions.rows;
    cout << "Accuracy on test set: " << accuracy * 100 << "%" << endl;
}


int main() {
    int input_size = 784;  // 28x28 pixel cho mỗi ảnh
    int hidden_size1 = 128;
    int hidden_size2 = 128;
    int output_size = 10;  // 10 lớp cho 10 nhãn
    int batch_size = 1000;

    Matrix X_train(batch_size, input_size);  // Tập huấn luyện (60000 ảnh)
    Matrix Y_train(batch_size, output_size); // Nhãn tương ứng

    // Đọc dữ liệu từ file CSV
    loadData("fashion-mnist_train.csv", X_train, Y_train, batch_size);

    cout << "X_train shape: " << X_train.rows << " " << X_train.cols << endl;
    cout << "Y_train shape: " << Y_train.rows << " " << Y_train.cols << endl;
    // Tạo và huấn luyện mạng
    
    NeuralNetwork nn(input_size, hidden_size1,hidden_size2, output_size, batch_size);
    nn.train(X_train, Y_train, 0.001, 20);
    
    //saveModel(nn, "saved_model.txt");

    batch_size = 1000;
    // Testing model
    Matrix X_test(batch_size, input_size);  // 10000 ảnh kiểm tra
    Matrix Y_test(batch_size, output_size);

    loadData("fashion-mnist_test.csv", X_test, Y_test, batch_size);

    // Tải mô hình và đánh giá
    //NeuralNetwork nnTest(input_size, hidden_size1, hidden_size2, output_size, batch_size);
    //loadModel(nnTest, "saved_model.txt");

    evaluate(nn, X_test, Y_test);

    return 0;
}