/*
 * main.cpp
 *
 *  Created on: 12 mag 2017
 *      Author: angelo
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

void SVMevaluate(Mat &testResponse, float &count, float &accuracy,
		vector<int> &testLabels) {

	for (int i = 0; i < testResponse.rows; i++) {
		if (testResponse.at<float>(i, 0) == testLabels[i]) {
			count = count + 1;
		}
	}
	accuracy = (count / testResponse.rows) * 100;
}

void SVMtrain(CvSVM &svm, Mat &trainMat, vector<int> &trainLabels) {
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	CvMat trainingMat = trainMat;
	Mat trainLabelsMat(trainLabels.size(), 1, CV_32FC1);

	for (unsigned int i = 0; i < trainLabels.size(); i++) {
		trainLabelsMat.at<float>(i, 0) = trainLabels[i];
	}
	CvMat trainingLabelsMat = trainLabelsMat;
	svm.train(&trainingMat, &trainingLabelsMat, Mat(), Mat(), params);
	svm.save("svm_classifier.xml");
}

void convertVectorToMatrix(const vector<Mat> &resultLBP, Mat &mat) {
	for (unsigned int i = 0; i < resultLBP.size(); i++) {
		for (int j = 0; j < resultLBP[i].cols; j++) {
			mat.at<float>(i, j) = resultLBP[i].at<float>(0, j);
		}
	}
}

void histogram(const Mat &src, Mat &hist, int numPatterns) {
	hist = Mat::zeros(1, numPatterns, CV_32FC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int bin = src.at<unsigned char>(i, j);
			hist.at<float>(0, bin) += 1;
		}
	}
}

void OLBP(const Mat &src, Mat &dst) {
	dst = Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			float center = src.at<unsigned char>(i, j);
			unsigned char code = 0;
			code |= (src.at<unsigned char>(i - 1, j - 1) > center) << 7;
			code |= (src.at<unsigned char>(i - 1, j) > center) << 6;
			code |= (src.at<unsigned char>(i - 1, j + 1) > center) << 5;
			code |= (src.at<unsigned char>(i, j + 1) > center) << 4;
			code |= (src.at<unsigned char>(i + 1, j + 1) > center) << 3;
			code |= (src.at<unsigned char>(i + 1, j) > center) << 2;
			code |= (src.at<unsigned char>(i + 1, j - 1) > center) << 1;
			code |= (src.at<unsigned char>(i, j - 1) > center) << 0;
			dst.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
}

void ELBP(const Mat &src, Mat &dst, int radius, int neighbors) {
	neighbors = max(min(neighbors, 31), 1); // set bounds
	dst = Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
	for (int n = 0; n < neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius)
				* cos(2.0 * M_PI * n / static_cast<float>(neighbors));
		float y = static_cast<float>(radius)
				* -sin(2.0 * M_PI * n / static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;
		// iterate through your data
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				float t = w1 * src.at<unsigned char>(i + fy, j + fx)
						+ w2 * src.at<unsigned char>(i + fy, j + cx)
						+ w3 * src.at<unsigned char>(i + cy, j + fx)
						+ w4 * src.at<unsigned char>(i + cy, j + cx);
				// we are dealing with floating point precision, so add some little tolerance
				dst.at<unsigned int>(i - radius, j - radius) += ((t
						> src.at<unsigned char>(i, j))
						&& (abs(t - src.at<unsigned char>(i, j))
								> std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

void computeLBP(vector<Mat> &resultLBP, const vector<Mat> &img) {
	for (unsigned int i = 0; i < img.size(); i++) {
		Mat lbp;
		OLBP(img[i], lbp);
		normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);
		Mat hist;
		histogram(lbp, hist, 256); // 256 is the number of bins of the histogram. It changes with the neighbors
		resultLBP.push_back(hist);
	}
}

void loadLabels(vector<int> &labels, int pedNum, int vehiclesNum) {
	for (int i = 0; i < (pedNum + vehiclesNum); i++) {
		labels.push_back(i < pedNum ? 5 : 6); // Atm 5 means Pedestrian label and 6 means Vehicles
	}
}

void loadImages(vector<Mat> &images, int &pedNum, int &vehiclesNum,
		String pedPath, String vehPath) {
	vector<String> pedFilesNames;
	glob(pedPath, pedFilesNames, true);
	for (unsigned int i = 0; i < pedFilesNames.size(); i++) {
		Mat img = imread(pedFilesNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat resizedImg;
		resize(img, resizedImg, Size(100, 100));
		images.push_back(resizedImg);
	}
	pedNum = pedFilesNames.size();

	vector<String> vehFilesNames;
	glob(vehPath, vehFilesNames, true);
	for (unsigned int i = 0; i < vehFilesNames.size(); i++) {
		Mat img = imread(vehFilesNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat resizedImg;
		resize(img, resizedImg, Size(100, 100));
		images.push_back(resizedImg);
	}
	vehiclesNum = vehFilesNames.size();
}

int main(int argc, char** argv) {
	vector<Mat> trainImg;
	int trainPedNum, trainVehNum;
	loadImages(trainImg, trainPedNum, trainVehNum, "train_pedestrians/*.jpg",
			"train_vehicles/*.jpg");
	vector<int> trainLabels;
	loadLabels(trainLabels, trainPedNum, trainVehNum);

	vector<Mat> trainLBP;
	computeLBP(trainLBP, trainImg);

	Mat trainMat(trainLBP.size(), trainLBP[0].cols, CV_32FC1);
	convertVectorToMatrix(trainLBP, trainMat);
	CvSVM svm;
	SVMtrain(svm, trainMat, trainLabels);

	vector<Mat> testImg;
	int testPedNum, testVehNum;
	loadImages(testImg, testPedNum, testVehNum, "test_pedestrians/*.jpg",
			"test_vehicles/*.jpg");
	vector<int> testLabels;
	loadLabels(testLabels, testPedNum, testVehNum);

	vector<Mat> testLBP;
	computeLBP(testLBP, testImg);

	Mat testMat(testLBP.size(), testLBP[0].cols, CV_32FC1);
	convertVectorToMatrix(testLBP, testMat);
	Mat testResponse;
	svm.predict(testMat, testResponse);

	float count = 0;
	float accuracy = 0;
	SVMevaluate(testResponse, count, accuracy, testLabels);

	cout << "The accuracy is " << accuracy << "%" << endl;

	return (0);
}
