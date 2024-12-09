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


    // Các phép toán ma trận
    Matrix transpose();
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(const Matrix& other) const;
    Matrix multiplyGrad(const Matrix& other,int m) const;
    Matrix scalarMultiply(double scalar) const;
    Matrix updateBias(int m) const;
    int argmax(int row) const;

    // Hàm kích hoạt ReLU
    Matrix applyReLU();

    // Đạo hàm hàm kích hoạt ReLU
    Matrix applyReLUDerivative();

    // Hàm Softmax
    Matrix applySoftmax();

    // In ma trận
    void print() const;
};

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    data.resize(rows, vector<double>(cols, 0));
}



// Hàm khởi tạo ngẫu nhiên
void Matrix::randomize() {
    random_device rd;
    mt19937 gen(rd());

    // He initialization
    double scale = sqrt(2.0 / rows);
    normal_distribution<double> d(0, scale);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = d(gen);
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
            result.data[i][j] = data[i][j] + other.data[0][j];
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
    


    if (cols == other.cols && rows == other.rows) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {              
                result.data[i][j] = data[i][j] * other.data[i][j];
              
            }
        }
        return result;

    }
    else {
        Matrix result(rows, other.cols);

        for (int i = 0; i < rows; i++) {

            for (int j = 0; j < other.cols; j++) {
                result.data[i][j] = 0; // Initialize the result cell
                for (int k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
   
}

// Hàm nhân ma trận để tính gradient
Matrix Matrix::multiplyGrad(const Matrix& other, int m ) const {
    if (cols != other.rows) {
        throw invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            result.data[i][j] = 0; // Initialize the result cell
            for (int k = 0; k < cols; k++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
            result.data[i][j] /= m;
        }
    }
    return result;
}

// Hàm Update bias
Matrix Matrix::updateBias(int m) const {
    Matrix result(1, cols);

    for (int j = 0; j < cols; j++) {
        double sumCols = 0.0;
        for (int i = 0; i < rows; i++) {
            sumCols += data[i][j]; // Cộng các phần tử trong cột j
        }
        // Gán giá trị tổng chia cho số hàng (rows) vào ma trận kết quả
        result.data[0][j] = sumCols / m;
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
 Matrix Matrix::applyReLU() {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = max(0.0, data[i][j]);
        }
    }
    return result;
}

// Đạo hàm ReLU
Matrix Matrix::applyReLUDerivative() {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Áp dụng đạo hàm ReLU: nếu giá trị > 0, đạo hàm = 1, nếu <= 0, đạo hàm = 0
            result.data[i][j] = data[i][j] > 0 ? 1.0 : 0.0;
        }
    }
    return result;
}

// Hàm Softmax
Matrix Matrix::applySoftmax() {
    Matrix result(rows, cols);
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
            result.data[i][j] = exp(data[i][j] - max_val) / sum_exp;
        }
    }
    return result;
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
    double loss = 0.0;
    for (int i = 0; i < targets.rows; i++) {
        // Find the index of the maximum value in the target (one-hot encoded)
        int targetIndex = 0;
        for (int j = 1; j < targets.cols; j++) {
            if (targets.data[i][j] > targets.data[i][targetIndex]) {
                targetIndex = j;
            }
        }

        // Compute the log likelihood
        double logLikelihood = -log(predictions.data[i][targetIndex]);
        loss += logLikelihood;
    }

    // Return the average loss over all samples
    return loss / targets.rows;
}

// Modified average loss calculation in train method
double calculateAverageLoss(const Matrix& lossMatrix) {
    double totalLoss = 0.0;
    int count = 0;

    for (int i = 0; i < lossMatrix.rows; ++i) {
        for (int j = 0; j < lossMatrix.cols; ++j) {
            if (lossMatrix.data[i][j] != 0) { // Only count non-zero losses
                totalLoss -= lossMatrix.data[i][j];
                count++;
            }
        }
    }

    return count > 0 ? totalLoss / count : 0.0;
}


class NeuralNetwork {
public:
    Matrix W1, b1, W2, b2, W3, b3;

    NeuralNetwork(int input_size, int hidden_size1, int hidden_size2, int output_size)
        : W1(input_size, hidden_size1), b1(1, hidden_size1),
        W2(hidden_size1, hidden_size2), b2(1, hidden_size2),
        W3(hidden_size2, output_size), b3(1, output_size) {

        W1.randomize();
        W2.randomize();
        W3.randomize();
    }

    void train(const Matrix& input, const Matrix& target, double learning_rate, int epochs);
};
void NeuralNetwork::train(const Matrix& input, const Matrix& target, double learning_rate, int epochs) {
    int m = input.rows;
    Matrix X = input;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward propagation like Python
        Matrix Z1 = input.multiply(W1).add(b1);
        Matrix A1 = Z1.applyReLU();

        Matrix Z2 = A1.multiply(W2).add(b2);
        Matrix A2 = Z2.applyReLU();

        Matrix Z3 = A2.multiply(W3).add(b3);
        Matrix A3 = Z3.applySoftmax();

        // Calculate loss
        double loss = crossEntropyLoss(A3, target);

        // Backpropagation matching Python implementation
        Matrix dZ3 = A3.subtract(target).scalarMultiply(loss); // Include loss scaling
        Matrix dW3 = A2.transpose().multiplyGrad(dZ3, m);
        Matrix db3 = dZ3.updateBias(m);

      
  
        Matrix dA2 = dZ3.multiply(W3.transpose());
        Matrix T = Z2.applyReLUDerivative();
        Matrix dZ2 = dA2.multiply(T);

        Matrix dW2 = A1.transpose().multiplyGrad(dZ2,m);
        Matrix db2 = dZ2.updateBias(m);

        Matrix dA1 = dZ2.multiply(W2.transpose());
        Matrix dZ1 = dA1.multiply(Z1.applyReLUDerivative());
        Matrix dW1 = X.transpose().multiplyGrad(dZ1,m);
        Matrix db1 = dZ1.updateBias(m);

        // Update weights with learning rate decay
        //double current_lr = learning_rate / (1 + 0.1 * epoch);
        W1 = W1.subtract(dW1.scalarMultiply(learning_rate));
        b1 = b1.subtract(db1.scalarMultiply(learning_rate));
        W2 = W2.subtract(dW2.scalarMultiply(learning_rate));
        b2 = b2.subtract(db2.scalarMultiply(learning_rate));
        W3 = W3.subtract(dW3.scalarMultiply(learning_rate));
        b3 = b3.subtract(db3.scalarMultiply(learning_rate));

        // Calculate and print accuracy
        int correct = 0;
        for (int i = 0; i < A3.rows; i++) {
            if (A3.argmax(i) == target.argmax(i)) correct++;
        }
        double accuracy = static_cast<double>(correct) / A3.rows * 100;
        cout << "Epoch " << epoch << ", Loss: " << loss << ", Accuracy: " << accuracy << "%" << endl;
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
                
                X.data[row][i] = image_pixels[i] / 255.0;
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
    saveMatrix(nn.W2);
    saveMatrix(nn.b2);
    saveMatrix(nn.W3);
    saveMatrix(nn.b3);

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
    loadMatrix(nn.W2);
    loadMatrix(nn.b2);
    loadMatrix(nn.W3);
    loadMatrix(nn.b3);

    file.close();
    cout << "Model loaded from " << filename << endl;
}

void predict(const NeuralNetwork& nn, const Matrix& input, Matrix& output) {

    Matrix Z1 = input.multiply(nn.W1).add(nn.b1);
    Matrix A1 = Z1.applyReLU();

    Matrix Z2 = A1.multiply(nn.W2).add(nn.b2);
    Matrix A2 = Z2.applyReLU();

    Matrix Z3 = A2.multiply(nn.W3).add(nn.b3);
    Matrix A3 = Z3.applySoftmax();

    output = A3;
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
    
    NeuralNetwork nn(input_size, hidden_size1,hidden_size2, output_size);
    nn.train(X_train, Y_train, 0.1, 30);
    
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