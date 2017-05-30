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

#include <iostream>

using namespace cv;
using namespace std;

#define LOAD_SVM 1

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

void convertVectorToMatrix(vector<vector<float> > &hogResult, Mat &mat) {
	int descriptor_size = hogResult[0].size();

	for (unsigned int i = 0; i < hogResult.size(); i++) {
		for (int j = 0; j < descriptor_size; j++) {
			mat.at<float>(i, j) = hogResult[i][j];
		}
	}
}

void computeHOG(vector<vector<float> > &hogResult, vector<Mat> &img,
		HOGDescriptor &hog) {
	for (unsigned int i = 0; i < img.size(); i++) {
		vector<float> descriptors;
		hog.compute(img[i], descriptors);
		hogResult.push_back(descriptors);
	}
}

void loadLabels(vector<int> &labels, int pedSize, int vehiclesSize) {
	for (int i = 0; i < (pedSize + vehiclesSize); i++) {
		labels.push_back(i < pedSize ? 5 : 6); // Atm 5 means Pedestrian label and 6 means Vehicles
	}
}

void loadImages(vector<Mat> &images, int &pedSize, int &vehiclesSize,
		String pedPath, String vehPath) {
	vector<String> pedFilesNames;
	glob(pedPath, pedFilesNames, true);
	for (unsigned int i = 0; i < pedFilesNames.size(); i++) {
		Mat img = imread(pedFilesNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat resizedImg;
		resize(img, resizedImg, Size(100, 100));
		images.push_back(resizedImg);
	}
	pedSize = pedFilesNames.size();

	vector<String> vehFilesNames;
	glob(vehPath, vehFilesNames, true);
	for (unsigned int i = 0; i < vehFilesNames.size(); i++) {
		Mat img = imread(vehFilesNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat resizedImg;
		resize(img, resizedImg, Size(100, 100));
		images.push_back(resizedImg);
	}
	vehiclesSize = vehFilesNames.size();
}

int main(int argc, char** argv) {
	// The 2nd and 4th params are fixed. Choose 1st and 3th such that (1st-2nd)/3th = 0
	HOGDescriptor hog(Size(100, 100), Size(16, 16), Size(4, 4), Size(8, 8), 9,
			-1, 0.2, true, 64);

	vector<Mat> testImg;
	int testPedSize, testVehSize;
	loadImages(testImg, testPedSize, testVehSize, "test_pedestrians/*.jpg",
			"test_vehicles/*.jpg");

	vector<int> testLabels;
	loadLabels(testLabels, testPedSize, testVehSize);

	vector<vector<float> > testHOG;
	computeHOG(testHOG, testImg, hog);

	int descriptorSize = testHOG[0].size();
	Mat testMat(testHOG.size(), descriptorSize, CV_32FC1);
	convertVectorToMatrix(testHOG, testMat);

	Mat testResponse;
	CvSVM svm;

	if (LOAD_SVM) {
		svm.load("svm_classifier.xml");
	} else {
		vector<Mat> trainImg;
		int trainPedSize, trainVehSize;
		loadImages(trainImg, trainPedSize, trainVehSize,
				"train_pedestrians/*.jpg", "train_vehicles/*.jpg");

		vector<int> trainLabels;
		loadLabels(trainLabels, trainPedSize, trainVehSize);

		vector<vector<float> > trainHOG;
		computeHOG(trainHOG, trainImg, hog);

		descriptorSize = trainHOG[0].size();
		Mat trainMat(trainHOG.size(), descriptorSize, CV_32FC1);
		convertVectorToMatrix(trainHOG, trainMat);
		SVMtrain(svm, trainMat, trainLabels);
	}

	svm.predict(testMat, testResponse);

	float count = 0;
	float accuracy = 0;
	SVMevaluate(testResponse, count, accuracy, testLabels);

	cout << "The accuracy is " << accuracy << "%" << endl;

	return (0);
}
