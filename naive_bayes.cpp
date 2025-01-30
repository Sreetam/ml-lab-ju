#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include <ctime>

// Function to read and preprocess data
void readData(const std::string &filePath, std::vector<std::string> &texts, std::vector<int> &labels) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return;
    }

    std::string line;
    bool isFirstLine = true;
    while (std::getline(file, line)) {
        if (isFirstLine) {
            isFirstLine = false;
            continue; // Skip header
        }

        std::stringstream ss(line);
        std::string negativeReview, positiveReview;
        std::getline(ss, negativeReview, ',');
        std::getline(ss, positiveReview, ',');

        if (negativeReview != "No Negative") {
            texts.push_back(negativeReview);
            labels.push_back(0);
        }

        if (positiveReview != "No Positive") {
            texts.push_back(positiveReview);
            labels.push_back(1);
        }
    }
    file.close();

    // Shuffle data
    std::srand(unsigned(std::time(nullptr)));
    std::vector<int> indices(texts.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 generator(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), generator);

    std::vector<std::string> shuffledTexts;
    std::vector<int> shuffledLabels;
    for (int idx : indices) {
        shuffledTexts.push_back(texts[idx]);
        shuffledLabels.push_back(labels[idx]);
    }
    texts = shuffledTexts;
    labels = shuffledLabels;
}

// Function to tokenize text
std::vector<std::string> tokenize(const std::string &text) {
    std::stringstream ss(text);
    std::string word;
    std::vector<std::string> tokens;
    while (ss >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        tokens.push_back(word);
    }
    return tokens;
}

// Function to train Naive Bayes classifier
void trainNaiveBayes(const std::vector<std::string> &X_train, const std::vector<int> &y_train,
                     std::map<int, std::map<std::string, int>> &wordCounts,
                     std::map<int, int> &classCounts,
                     std::unordered_set<std::string> &vocabulary) {
    for (size_t i = 0; i < X_train.size(); ++i) {
        auto tokens = tokenize(X_train[i]);
        classCounts[y_train[i]] += tokens.size();

        for (const auto &token : tokens) {
            vocabulary.insert(token);
            wordCounts[y_train[i]][token]++;
        }
    }
}

// Function to predict using Naive Bayes
int predictNaiveBayes(const std::string &text, const std::map<int, std::map<std::string, int>> &wordCounts,
                      const std::map<int, int> &classCounts, const std::unordered_set<std::string> &vocabulary) {
    auto tokens = tokenize(text);
    double totalWords = classCounts.at(0) + classCounts.at(1);
    std::map<int, double> logProbabilities;

    for (int label : {0, 1}) {
        logProbabilities[label] = std::log(classCounts.at(label) / totalWords);
        for (const auto &token : tokens) {
            double wordProbability = (wordCounts.at(label).count(token) ? wordCounts.at(label).at(token) + 1 : 1) /
                                     static_cast<double>(classCounts.at(label) + vocabulary.size());
            logProbabilities[label] += std::log(wordProbability);
        }
    }

    return logProbabilities[0] > logProbabilities[1] ? 0 : 1;
}

// Function to evaluate the model
double evaluateModel(const std::vector<std::string> &X_test, const std::vector<int> &y_test,
                     const std::map<int, std::map<std::string, int>> &wordCounts,
                     const std::map<int, int> &classCounts,
                     const std::unordered_set<std::string> &vocabulary) {
    int correct = 0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        if (predictNaiveBayes(X_test[i], wordCounts, classCounts, vocabulary) == y_test[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / X_test.size();
}

int main() {
    std::string filePath = "datasets/Hotel_Reviews.csv";
    std::vector<std::string> texts;
    std::vector<int> labels;

    // Read and preprocess data
    readData(filePath, texts, labels);

    // Split into train and test sets
    size_t trainSize = static_cast<size_t>(texts.size() * 0.8);
    std::vector<std::string> X_train(texts.begin(), texts.begin() + trainSize);
    std::vector<int> y_train(labels.begin(), labels.begin() + trainSize);
    std::vector<std::string> X_test(texts.begin() + trainSize, texts.end());
    std::vector<int> y_test(labels.begin() + trainSize, labels.end());

    // Train the model
    std::map<int, std::map<std::string, int>> wordCounts;
    std::map<int, int> classCounts;
    std::unordered_set<std::string> vocabulary;
    trainNaiveBayes(X_train, y_train, wordCounts, classCounts, vocabulary);

    // Evaluate on test set
    double accuracy = evaluateModel(X_test, y_test, wordCounts, classCounts, vocabulary);

    std::cout << "Accuracy: " << accuracy << std::endl;
    return 0;
}