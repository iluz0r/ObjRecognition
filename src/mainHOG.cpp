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

void SVMevaluate1(Mat &testResponse, float &count, float &accuracy,
		vector<int> &testLabels) {

	for (int i = 0; i < testResponse.rows; i++) {
		if (testResponse.at<float>(i, 0) == testLabels[i]) {
			count = count + 1;
		}
	}
	accuracy = (count / testResponse.rows) * 100;
}

void SVMtrain1(CvSVM &svm, Mat &trainMat, vector<int> &trainLabels) {
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

void convertVectorToMatrix1(vector<vector<float> > &hogResult, Mat &mat) {
	int descriptor_size = hogResult[0].size();

	for (unsigned int i = 0; i < hogResult.size(); i++) {
		for (int j = 0; j < descriptor_size; j++) {
			mat.at<float>(i, j) = hogResult[i][j];
		}
	}
}

void computeHOG(vector<vector<float> > &hogResult, const vector<Mat> &img,
		HOGDescriptor &hog) {
	for (unsigned int i = 0; i < img.size(); i++) {
		vector<float> descriptors;
		hog.compute(img[i], descriptors);
		hogResult.push_back(descriptors);
	}
}

void loadLabels1(vector<int> &labels, int pedNum, int vehiclesNum) {
	for (int i = 0; i < (pedNum + vehiclesNum); i++) {
		labels.push_back(i < pedNum ? 5 : 6); // Atm 5 means Pedestrian label and 6 means Vehicles
	}
}

void loadImages1(vector<Mat> &images, int &pedNum, int &vehiclesNum,
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

int main1(int argc, char** argv) {
	// The 2nd and 4th params are fixed. Choose 1st and 3th such that (1st-2nd)/3th = 0
	HOGDescriptor hog(Size(100, 100), Size(16, 16), Size(4, 4), Size(8, 8), 9,
			-1, 0.2, true, 64);
	CvSVM svm;

	// If xml classifier exists it will be loaded, otherwise a new classifier will be trained
	if (LOAD_SVM) {
		svm.load("svm_classifier.xml");
	} else {
		vector<Mat> trainImg;
		int trainPedNum, trainVehSize;
		loadImages1(trainImg, trainPedNum, trainVehSize,
				"train_pedestrians/*.jpg", "train_vehicles/*.jpg");
		vector<int> trainLabels;
		loadLabels1(trainLabels, trainPedNum, trainVehSize);

		vector<vector<float> > trainHOG;
		computeHOG(trainHOG, trainImg, hog);

		int descriptorSize = trainHOG[0].size();
		Mat trainMat(trainHOG.size(), descriptorSize, CV_32FC1);
		convertVectorToMatrix1(trainHOG, trainMat);
		SVMtrain1(svm, trainMat, trainLabels);
	}

	vector<Mat> testImg;
	int testPedNum, testVehNum;
	loadImages1(testImg, testPedNum, testVehNum, "test_pedestrians/*.jpg",
			"test_vehicles/*.jpg");
	vector<int> testLabels;
	loadLabels1(testLabels, testPedNum, testVehNum);

	vector<vector<float> > testHOG;
	computeHOG(testHOG, testImg, hog);

	int descriptorSize = testHOG[0].size();
	Mat testMat(testHOG.size(), descriptorSize, CV_32FC1);
	convertVectorToMatrix1(testHOG, testMat);
	Mat testResponse;
	svm.predict(testMat, testResponse);

	float count = 0;
	float accuracy = 0;
	SVMevaluate1(testResponse, count, accuracy, testLabels);

	cout << "The accuracy is " << accuracy << "%" << endl;

	return (0);
}
