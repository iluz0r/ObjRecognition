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
#include <iomanip>
#include <string>

#include <rapidxml.hpp>
#include <rapidxml_utils.hpp>
#include <rapidxml_print.hpp>

using namespace cv;
using namespace std;
using namespace rapidxml;

int DESCRIPTOR_TYPE; // {0 = hog, 1 = lbp, 2 = bb, 3 = conc}
int LOAD_CLASSIFIER;
int USE_MES; // If MES is used, DESCRIPTOR_TYPE and LOAD_CLASSIFIER are not considered
int NUM_CLASS = 3; // Number of classes
int ACC_EVALUATION; // When this param is 1, the system loads the samples from test_pedestrians,
// test_vehicles and test_unknown folders and give as output the classification accuracy. If this
// param is 1, VIDEO_CLASS_OUTPUT is not considered.
String VIDEO_NAME;
int VIDEO_CLASS_OUTPUT; // {1 = generate xml, 2 = show video with classification}

void convertVectorToMatrix(const vector<vector<float> > &hogResult, Mat &mat);
void convertVectorToMatrix(const vector<Mat> &lbpResult, Mat &mat);
void convertVectorToMatrix(const vector<int> &labels, Mat &labelsMat);
void SVMevaluate(Mat &testResponse, float &count, float &accuracy,
		Mat &testLabelsMat);
void SVMtrain(CvSVM &svm, Mat &featureVecMat, Mat &labelsMat);
void histogram(const Mat &src, Mat &hist, int numPatterns);
void OLBP(const Mat &src, Mat &dst);
void ELBP(const Mat &src, Mat &dst, int radius, int neighbors);
void computeHOG(Mat &featureVecMat, const vector<Mat> &img);
void computeLBP(Mat &featureVecMat, const vector<Mat> &img);
void countWhitePixels(const Mat matrix, int &whitePixels);
void convertColorFromBGR2HSV(const Mat3b &bgr, Mat3b &hsv);
void computeBB(Mat &featureVecMat, const vector<Mat> &img);
void computeSingleHOG(vector<float> &featureVec, const Mat &img);
void computeSingleLBP(Mat &featureVec, const Mat &img);
void computeSingleBB(vector<float> &featureVec, const Mat &img);
void concatFeatureVectors(Mat &concatResult, const vector<Mat> &images);
void singleConcatFeatureVec(Mat &concatResult, const Mat &img);
void loadLabels(vector<int> &labels, int pedNum, int vehNum, int unkNum);
void loadImages(vector<Mat> &images, const vector<String> paths);
void createLabelsMat(Mat &labelsMat, vector<String> paths);
void createFeatureVectorsMat(Mat &featureVecMat, vector<String> paths);
void calculateFinalResponse(Mat &finalResponse,
		const vector<Mat> &votesMatrices,
		const vector<vector<float> > &accuracies);
void calculateSingleFinalResponse(float &finalResponse, const Mat &votesMatrix,
		const vector<vector<float> > &accuracies);
void createVotesMatrices(vector<Mat> &votesMatrices,
		const vector<Mat> &testResponses);
void createVotesMatrix(Mat &votesMatrix, const vector<float> &testResponse);
void calculateTestResponses(vector<Mat> &testResponses, const CvSVM &svm_hog,
		const CvSVM &svm_lbp, const CvSVM &svm_bb);
void calculateSingleTestResponse(vector<float> &testResponse, const Mat &img,
		const CvSVM &svm_hog, const CvSVM &svm_lbp, const CvSVM &svm_bb);
void calculateAccuracies(vector<vector<float> > &accuracies,
		const vector<Mat> confusionMatrices);
void createConfusionMatrices(vector<Mat> &confusionMatrices,
		const CvSVM &svm_hog, const CvSVM &svm_lbp, const CvSVM &svm_bb,
		const Mat &validLabelsMat);
void initMES(vector<vector<float> > &accuracies);
void computeMES(Mat &finalResponse, const vector<vector<float> > &accuracies,
		const CvSVM &svm_hog, const CvSVM &svm_lbp, const CvSVM &svm_bb);
void computeSingleMES(float &response, const Mat &img,
		const vector<vector<float> > &accuracies, const CvSVM &svm_hog,
		const CvSVM &svm_lbp, const CvSVM &svm_bb);
void classifySample(const Mat &img, const vector<vector<float> > &accuracies,
		const CvSVM &svm_hog, const CvSVM &svm_lbp, const CvSVM &svm_bb,
		const CvSVM &svm_concat, float &label);
void showVideoWithClassification(const String inputPath);
void saveClassificationAsXml(const String inputPath);
void classifySamplesFromVideo(const String path);
void classify();
void extractSamplesFromVideo(const String pathVideo, const String pathXml,
		const String pathSave);
void clearScreen();
void displayClassVideoMenu();
void displayAccEvaluationMenu();
void displayMainMenu();
void askBackToMainMenu();

int main(int argc, char** argv) {
	//extractSamplesFromVideo("prova.mp4", "prova.xgtf", "prova/");
	// Open the main menu
	displayMainMenu();

	return (0);
}

void convertVectorToMatrix(const vector<vector<float> > &hogResult, Mat &mat) {
	int featureVecSize = hogResult[0].size();

	for (unsigned int i = 0; i < hogResult.size(); i++) {
		for (int j = 0; j < featureVecSize; j++) {
			mat.at<float>(i, j) = hogResult[i][j];
		}
	}
}

void convertVectorToMatrix(const vector<Mat> &lbpResult, Mat &mat) {
	for (unsigned int i = 0; i < lbpResult.size(); i++) {
		for (int j = 0; j < lbpResult[i].cols; j++) {
			mat.at<float>(i, j) = lbpResult[i].at<float>(0, j);
		}
	}
}

void convertVectorToMatrix(const vector<int> &labels, Mat &labelsMat) {
	for (unsigned int i = 0; i < labels.size(); i++) {
		labelsMat.at<float>(i, 0) = labels[i];
	}
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy,
		Mat &testLabelsMat) {

	for (int i = 0; i < testResponse.rows; i++) {
		if (testResponse.at<float>(i, 0) == testLabelsMat.at<float>(i, 0)) {
			count = count + 1;
		}
	}
	accuracy = (count / testResponse.rows) * 100;
}

void SVMtrain(CvSVM &svm, Mat &featureVecMat, Mat &labelsMat) {
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	CvMat trainingDesc = featureVecMat;
	CvMat trainingLabels = labelsMat;
	svm.train(&trainingDesc, &trainingLabels, Mat(), Mat(), params);
	stringstream ss;
	ss << "svm_" << DESCRIPTOR_TYPE << "_classifier.xml";
	svm.save(ss.str().c_str());
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

void computeHOG(Mat &featureVecMat, const vector<Mat> &img) {
	// The 2nd and 4th params are fixed. Choose 1st and 3th such that (1st-2nd)%3th = 0
	HOGDescriptor hog(Size(100, 100), Size(16, 16), Size(4, 4), Size(8, 8), 9,
			-1, 0.2, true, 64);
	vector<vector<float> > hogResult;

	for (unsigned int i = 0; i < img.size(); i++) {
		// Convert image from BGR to Gray
		Mat grayImg;
		cvtColor(img[i], grayImg, COLOR_BGR2GRAY);
		// Calculate the feature vector of the gray image and push back it into the hogResult
		vector<float> featureVector;
		hog.compute(grayImg, featureVector);
		hogResult.push_back(featureVector);
	}

	// Converts the vector<vector<float>> into a Mat of float
	featureVecMat = Mat(hogResult.size(), hogResult[0].size(), CV_32FC1);
	convertVectorToMatrix(hogResult, featureVecMat);
}

void computeSingleHOG(vector<float> &featureVec, const Mat &img) {
	// The 2nd and 4th params are fixed. Choose 1st and 3th such that (1st-2nd)%3th = 0
	HOGDescriptor hog(Size(100, 100), Size(16, 16), Size(4, 4), Size(8, 8), 9,
			-1, 0.2, true, 64);
	// Convert image from BGR to Gray
	Mat grayImg;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	hog.compute(grayImg, featureVec);
}

void computeLBP(Mat &featureVecMat, const vector<Mat> &img) {
	vector<Mat> lbpResult;

	for (unsigned int i = 0; i < img.size(); i++) {
		// Convert image from BGR to Gray
		Mat grayImg;
		cvtColor(img[i], grayImg, COLOR_BGR2GRAY);
		// Tiny bit of smoothing is always a good idea
		GaussianBlur(grayImg, grayImg, Size(7, 7), 5, 3, BORDER_CONSTANT);
		Mat lbp;
		ELBP(grayImg, lbp, 4, 4);
		//OLBP(img[i], lbp);
		normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);
		Mat hist;
		histogram(lbp, hist, 256); // 256 is the number of bins of the histogram
		lbpResult.push_back(hist);
	}

	// Converts the vector<Mat> into a Mat of float
	featureVecMat = Mat(lbpResult.size(), lbpResult[0].cols, CV_32FC1);
	convertVectorToMatrix(lbpResult, featureVecMat);
}

void computeSingleLBP(Mat &featureVec, const Mat &img) {
	// Convert image from BGR to Gray
	Mat grayImg;
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	// Tiny bit of smoothing is always a good idea
	GaussianBlur(grayImg, grayImg, Size(7, 7), 5, 3, BORDER_CONSTANT);
	Mat lbp;
	ELBP(grayImg, lbp, 4, 4);
	//OLBP(img[i], lbp);
	normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);
	histogram(lbp, featureVec, 256); // 256 is the number of bins of the histogram
}

void countWhitePixels(const Mat matrix, int &whitePixels) {
	whitePixels = 0;
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			if (matrix.at<uchar>(i, j) == 255)
				whitePixels++;
		}
	}
}

// Mat3b bgr(Vec3b(100,24,90)) for example
void convertColorFromBGR2HSV(const Mat3b &bgr, Mat3b &hsv) {
	// Convert a BGR color to HSV
	cvtColor(bgr, hsv, COLOR_BGR2HSV);
}

void computeBB(Mat &featureVecMat, const vector<Mat> &img) {
	vector<Mat> hsvImg;
	vector<vector<float> > dimensions;

	for (unsigned int i = 0; i < img.size(); i++) {
		// Apply Gaussian blur to remove some noise
		Mat im;
		GaussianBlur(img[i], im, Size(7, 7), 5, 3);

		// Convert the image from BGR to HSV
		Mat hsvImg;
		cvtColor(im, hsvImg, COLOR_BGR2HSV);

		// Define range of colors in HSV
		vector<vector<Scalar> > colors;

		// The colors are choosen by H-+10, S from minS to 255, V from minV to 255
		vector<Scalar> brown;
		Scalar lower_brown(3, 60, 95);
		Scalar upper_brown(23, 255, 255);
		brown.push_back(lower_brown);
		brown.push_back(upper_brown);

		vector<Scalar> green;
		Scalar lower_green(30, 20, 40);
		Scalar upper_green(80, 255, 255);
		green.push_back(lower_green);
		green.push_back(upper_green);

		vector<Scalar> gray;
		Scalar lower_gray(0, 0, 80);
		Scalar upper_gray(255, 50, 220);
		gray.push_back(lower_gray);
		gray.push_back(upper_gray);

		vector<Scalar> white;
		Scalar lower_white(0, 0, 235);
		Scalar upper_white(255, 5, 255);
		white.push_back(lower_white);
		white.push_back(upper_white);

		colors.push_back(brown);
		colors.push_back(green);
		colors.push_back(gray);
		colors.push_back(white);

		int maxWhitePixels = 0;
		Mat bestThresh;
		for (unsigned int j = 0; j < colors.size(); j++) {
			// Threshold the HSV image to get only background of this color
			Mat colThresh;
			inRange(hsvImg, colors[j].at(0), colors[j].at(1), colThresh);

			int whitePixels;
			countWhitePixels(colThresh, whitePixels);
			if (whitePixels > maxWhitePixels) {
				maxWhitePixels = whitePixels;
				bestThresh = colThresh;
			}
		}

		// Invert the mask to get the object of interest
		bitwise_not(bestThresh, bestThresh);

		// Bitwise-AND mask and original image
		Mat res;
		bitwise_and(im, im, res, bestThresh);

		// Detect edges using Canny
		Mat canny_mat;
		Canny(res, canny_mat, 20, 70, 3, false);

		// Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(canny_mat, contours, hierarchy, CV_RETR_EXTERNAL,
				CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		// Merge all contours into one vector
		vector<Point> merged_contours_points;
		for (unsigned int j = 0; j < contours.size(); j++) {
			for (unsigned int k = 0; k < contours[j].size(); k++) {
				merged_contours_points.push_back(contours[j][k]);
			}
		}

		if (merged_contours_points.size() > 0) {
			// Merge all lines (contours) in one line through convex hull
			vector<Point> hull;
			convexHull(merged_contours_points, hull);

			// Get the rotated bounding box
			RotatedRect rotated_bounding = minAreaRect(hull);

			vector<float> dim;
			dim.push_back(rotated_bounding.size.width);
			dim.push_back(rotated_bounding.size.height);
			dimensions.push_back(dim);

			/*
			 // Draw the rotated bouding box
			 Mat drawing = Mat::zeros(canny_mat.size(), CV_8UC3);
			 Point2f rect_vertices[4];
			 rotated_bounding.points(rect_vertices);
			 for (int j = 0; j < 4; j++)
			 line(drawing, rect_vertices[j], rect_vertices[(j + 1) % 4],
			 Scalar(120, 200, 200), 1, 8);
			 imshow("Rotated bounding box", drawing);
			 waitKey(0);
			 */
		} else {
			vector<float> dim;
			dim.push_back(0);
			dim.push_back(0);
			dimensions.push_back(dim);
		}
	}
	// Converts the vector<vector<float>> into a Mat of float
	featureVecMat = Mat(dimensions.size(), dimensions[0].size(), CV_32FC1);
	convertVectorToMatrix(dimensions, featureVecMat);
}

void computeSingleBB(vector<float> &featureVec, const Mat &img) {
	// Apply Gaussian blur to remove some noise
	Mat im;
	GaussianBlur(img, im, Size(7, 7), 5, 3);

	// Convert the image from BGR to HSV
	Mat hsvImg;
	cvtColor(im, hsvImg, COLOR_BGR2HSV);

	// Define range of colors in HSV
	vector<vector<Scalar> > colors;

	// The colors are choosen by H-+10, S from minS to 255, V from minV to 255
	vector<Scalar> brown;
	Scalar lower_brown(3, 60, 95);
	Scalar upper_brown(23, 255, 255);
	brown.push_back(lower_brown);
	brown.push_back(upper_brown);

	vector<Scalar> green;
	Scalar lower_green(30, 20, 40);
	Scalar upper_green(80, 255, 255);
	green.push_back(lower_green);
	green.push_back(upper_green);

	vector<Scalar> gray;
	Scalar lower_gray(0, 0, 80);
	Scalar upper_gray(255, 50, 220);
	gray.push_back(lower_gray);
	gray.push_back(upper_gray);

	vector<Scalar> white;
	Scalar lower_white(0, 0, 235);
	Scalar upper_white(255, 5, 255);
	white.push_back(lower_white);
	white.push_back(upper_white);

	colors.push_back(brown);
	colors.push_back(green);
	colors.push_back(gray);
	colors.push_back(white);

	int maxWhitePixels = 0;
	Mat bestThresh;
	for (unsigned int j = 0; j < colors.size(); j++) {
		// Threshold the HSV image to get only background of this color
		Mat colThresh;
		inRange(hsvImg, colors[j].at(0), colors[j].at(1), colThresh);

		int whitePixels;
		countWhitePixels(colThresh, whitePixels);
		if (whitePixels > maxWhitePixels) {
			maxWhitePixels = whitePixels;
			bestThresh = colThresh;
		}
	}

	// Invert the mask to get the object of interest
	bitwise_not(bestThresh, bestThresh);

	// Bitwise-AND mask and original image
	Mat res;
	bitwise_and(im, im, res, bestThresh);

	// Detect edges using Canny
	Mat canny_mat;
	Canny(res, canny_mat, 20, 70, 3, false);

	// Find contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(canny_mat, contours, hierarchy, CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Merge all contours into one vector
	vector<Point> merged_contours_points;
	for (unsigned int j = 0; j < contours.size(); j++) {
		for (unsigned int k = 0; k < contours[j].size(); k++) {
			merged_contours_points.push_back(contours[j][k]);
		}
	}

	if (merged_contours_points.size() > 0) {
		// Merge all lines (contours) in one line through convex hull
		vector<Point> hull;
		convexHull(merged_contours_points, hull);

		// Get the rotated bounding box
		RotatedRect rotated_bounding = minAreaRect(hull);

		featureVec.push_back(rotated_bounding.size.width);
		featureVec.push_back(rotated_bounding.size.height);
	} else {
		featureVec.push_back(0);
		featureVec.push_back(0);
	}
}

void concatFeatureVectors(Mat &concatResult, const vector<Mat> &images) {
	Mat hogResult;
	computeHOG(hogResult, images);
	Mat lbpResult;
	computeLBP(lbpResult, images);
	Mat bbResult;
	computeBB(bbResult, images);

	Mat matArray[] = { hogResult, lbpResult, bbResult };
	hconcat(matArray, sizeof(matArray) / sizeof(*matArray), concatResult);
}

void singleConcatFeatureVec(Mat &concatResult, const Mat &img) {
	vector<float> hogFeatureVec;
	computeSingleHOG(hogFeatureVec, img);
	Mat hogResult(1, hogFeatureVec.size(), CV_32FC1, hogFeatureVec.data());
	Mat lbpResult;
	computeSingleLBP(lbpResult, img);
	vector<float> bbFeatureVec;
	computeSingleBB(bbFeatureVec, img);
	Mat bbResult(1, bbFeatureVec.size(), CV_32FC1, bbFeatureVec.data());

	Mat matArray[] = { hogResult, lbpResult, bbResult };
	hconcat(matArray, sizeof(matArray) / sizeof(*matArray), concatResult);
}

void loadLabels(vector<int> &labels, int pedNum, int vehNum, int unkNum) {
	// Atm 0 means Pedestrian label, 1 means Vehicles and 2 means Unknown
	for (int i = 0; i < (pedNum + vehNum + unkNum); i++) {
		if (i < pedNum)
			labels.push_back(0);
		else if (i >= pedNum && i < (pedNum + vehNum))
			labels.push_back(1);
		else if (i >= (pedNum + vehNum))
			labels.push_back(2);
	}
}

void loadImages(vector<Mat> &images, const vector<String> paths) {
	for (unsigned int i = 0; i < paths.size(); i++) {
		vector<String> fileNames;
		glob(paths[i], fileNames, true);
		for (unsigned int j = 0; j < fileNames.size(); j++) {
			Mat img = imread(fileNames[j], CV_LOAD_IMAGE_COLOR);
			// Don't resize in the bounding box case
			if (DESCRIPTOR_TYPE != 2)
				resize(img, img, Size(100, 100));
			images.push_back(img);
		}
	}
}

void createLabelsMat(Mat &labelsMat, vector<String> paths) {
	// Loads all the samples names to calculate the number of samples
	vector<String> pedFilesNames, vehFilesNames, unkFilesNames;
	glob(paths[0], pedFilesNames, true);
	glob(paths[1], vehFilesNames, true);
	glob(paths[2], unkFilesNames, true);

	vector<int> labels;
	loadLabels(labels, pedFilesNames.size(), vehFilesNames.size(),
			unkFilesNames.size());
	// Converts the vector<int> of labels into a Mat (a column vector) of float
	labelsMat = Mat(labels.size(), 1, CV_32FC1);
	convertVectorToMatrix(labels, labelsMat);
}

void createFeatureVectorsMat(Mat &featureVecMat, vector<String> paths) {
	// Loads samples and corresponding labels
	vector<Mat> images;
	loadImages(images, paths);

	switch (DESCRIPTOR_TYPE) {
	case 0: {
		// Computes HOG, calculating a matrix in which each row is a feature vector.
		computeHOG(featureVecMat, images);
	}
		break;
	case 1: {
		// Computes LBP, calculating a matrix in which each row is a feature vector.
		computeLBP(featureVecMat, images);
	}
		break;
	case 2: {
		// Computes the bounding boxes, calculating a matrix in which each row is a feature vector (width, height).
		computeBB(featureVecMat, images);
	}
		break;
	case 3: {
		// Computes the concatenation of different feature vectors (hog+lbp+bb).
		concatFeatureVectors(featureVecMat, images);
	}
		break;
	default:
		break;
	}
}

void calculateFinalResponse(Mat &finalResponse,
		const vector<Mat> &votesMatrices,
		const vector<vector<float> > &accuracies) {
	for (unsigned int i = 0; i < votesMatrices.size(); i++) {
		float maxReliability = 0;
		int bestClass;
		for (int j = 0; j < NUM_CLASS; j++) {
			float num = 0;
			float den = 0;
			for (int k = 0; k < 3; k++) {
				num += votesMatrices[i].at<int>(j, k) * accuracies[k][j];
				den += accuracies[k][j];
			}
			float reliability = num / den;
			if (reliability >= maxReliability) {
				maxReliability = reliability;
				bestClass = j;
			}
		}
		finalResponse.at<float>(i, 0) = bestClass;
	}
}

void calculateSingleFinalResponse(float &finalResponse, const Mat &votesMatrix,
		const vector<vector<float> > &accuracies) {
	float maxReliability = 0;
	int bestClass;
	for (int j = 0; j < NUM_CLASS; j++) {
		float num = 0;
		float den = 0;
		for (int k = 0; k < 3; k++) {
			num += votesMatrix.at<int>(j, k) * accuracies[k][j];
			den += accuracies[k][j];
		}
		float reliability = num / den;
		if (reliability >= maxReliability) {
			maxReliability = reliability;
			bestClass = j;
		}
	}
	finalResponse = bestClass;
}

void createVotesMatrices(vector<Mat> &votesMatrices,
		const vector<Mat> &testResponses) {
	int numSamples = testResponses[0].rows;
	for (int i = 0; i < numSamples; i++) {
		// votes has 3 rows as many classes and 3 columns as many classifiers (so in each column there will be only one 1)
		Mat votes = Mat::zeros(NUM_CLASS, 3, CV_32SC1);
		for (int j = 0; j < 3; j++) {
			votes.at<int>(testResponses[j].at<float>(i, 0), j)++;}
		votesMatrices.push_back(votes);
	}
}

void createVotesMatrix(Mat &votesMatrix, const vector<float> &testResponse) {
	// votes has 3 rows as many classes and 3 columns as many classifiers (so in each column there will be only one 1)
	Mat votes = Mat::zeros(NUM_CLASS, 3, CV_32SC1);
	for (int j = 0; j < 3; j++) {
		votes.at<int>(testResponse[j], j)++;}
	votesMatrix.push_back(votes);
}

void calculateTestResponses(vector<Mat> &testResponses, const CvSVM &svm_hog,
		const CvSVM &svm_lbp, const CvSVM &svm_bb) {
	for (int i = 0; i < 3; i++) {
		DESCRIPTOR_TYPE = i;

		// Create the matrix of all the feature vectors of testing samples
		Mat testFeatureVecMat;
		if (ACC_EVALUATION) {
			vector<String> paths;
			paths.push_back("test_pedestrians/*.jpg");
			paths.push_back("test_vehicles/*.jpg");
			paths.push_back("test_unknown/*.jpg");
			createFeatureVectorsMat(testFeatureVecMat, paths);
		}

		// Predict the testing samples class with the classifiers
		Mat testResponse;
		switch (DESCRIPTOR_TYPE) {
		case 0: {
			svm_hog.predict(testFeatureVecMat, testResponse);
		}
			break;
		case 1: {
			svm_lbp.predict(testFeatureVecMat, testResponse);
		}
			break;
		case 2: {
			svm_bb.predict(testFeatureVecMat, testResponse);
		}
			break;
		default:
			break;
		}
		testResponses.push_back(testResponse);
	}
}

void calculateSingleTestResponse(vector<float> &testResponse, const Mat &img,
		const CvSVM &svm_hog, const CvSVM &svm_lbp, const CvSVM &svm_bb) {
	for (int i = 0; i < 3; i++) {
		DESCRIPTOR_TYPE = i;

		// Predict the testing sample class with the classifiers
		float response;
		switch (DESCRIPTOR_TYPE) {
		case 0: {
			vector<float> featureVec;
			computeSingleHOG(featureVec, img);
			Mat featureVectorMat(1, featureVec.size(), CV_32FC1,
					featureVec.data());
			response = svm_hog.predict(featureVectorMat);
		}
			break;
		case 1: {
			Mat featureVec;
			computeSingleLBP(featureVec, img);
			response = svm_lbp.predict(featureVec);
		}
			break;
		case 2: {
			vector<float> featureVec;
			computeSingleBB(featureVec, img);
			Mat featureVectorMat(1, featureVec.size(), CV_32FC1,
					featureVec.data());
			response = svm_bb.predict(featureVectorMat);
		}
			break;
		default:
			break;
		}
		testResponse.push_back(response);
	}
}

// Calculate the weights w_k(i) as described into the Fire-Detection paper
void calculateAccuracies(vector<vector<float> > &accuracies,
		const vector<Mat> confusionMatrices) {
	for (unsigned int i = 0; i < confusionMatrices.size(); i++) {
		vector<float> accClassifier;
		for (int j = 0; j < NUM_CLASS; j++) {
			int rowSum = 0;
			for (int k = 0; k < NUM_CLASS; k++)
				rowSum += confusionMatrices[i].at<int>(j, k);
			float acc = (float) confusionMatrices[i].at<int>(j, j)
					/ (float) rowSum;
			accClassifier.push_back(acc);
		}
		accuracies.push_back(accClassifier);
	}
}

void createConfusionMatrices(vector<Mat> &confusionMatrices,
		const CvSVM &svm_hog, const CvSVM &svm_lbp, const CvSVM &svm_bb,
		const Mat &validLabelsMat) {
	for (int i = 0; i < 3; i++) {
		DESCRIPTOR_TYPE = i;

		// Create the matrix of all the feature vectors of validation samples and the matrix of all the labels of validation samples
		Mat validFeatureVecMat;
		vector<String> paths;
		paths.push_back("valid_pedestrians/*.jpg");
		paths.push_back("valid_vehicles/*.jpg");
		paths.push_back("valid_unknown/*.jpg");
		createFeatureVectorsMat(validFeatureVecMat, paths);

		// Predict the classes of validation samples with classifier
		Mat testResponse;
		switch (DESCRIPTOR_TYPE) {
		case 0: {
			svm_hog.predict(validFeatureVecMat, testResponse);
		}
			break;
		case 1: {
			svm_lbp.predict(validFeatureVecMat, testResponse);
		}
			break;
		case 2: {
			svm_bb.predict(validFeatureVecMat, testResponse);
		}
			break;
		default:
			break;
		}

		// Confusion matrix (aka classification matrix) has n x n dimension, where n is the number of classes
		// The confusion matrix has the actual classes on rows and the predicted classes on columns
		Mat confusionMat = Mat::zeros(NUM_CLASS, NUM_CLASS, CV_32SC1);
		for (int i = 0; i < validLabelsMat.rows; i++) {
			confusionMat.at<int>(validLabelsMat.at<float>(i, 0),
					testResponse.at<float>(i, 0))++;}
		confusionMatrices.push_back(confusionMat);
	}
}

void initMES(vector<vector<float> > &accuracies) {
	// Load the 4 trained classifiers
	CvSVM svm_hog, svm_lbp, svm_bb, svm_concat;
	svm_hog.load("svm_0_classifier.xml");
	svm_lbp.load("svm_1_classifier.xml");
	svm_bb.load("svm_2_classifier.xml");

	// Create the matrix containing all the labels for the validation samples
	Mat validLabelsMat;
	vector<String> paths;
	paths.push_back("valid_pedestrians/*.jpg");
	paths.push_back("valid_vehicles/*.jpg");
	paths.push_back("valid_unknown/*.jpg");
	createLabelsMat(validLabelsMat, paths);

	// Create the confusion matrices for the 3 classifiers (svm_hog, svm_lbp, svm_bb)
	vector<Mat> confusionMatrices;
	createConfusionMatrices(confusionMatrices, svm_hog, svm_lbp, svm_bb,
			validLabelsMat);

	// Calculate the accuracies for each class and each classifier by the confusion matrices.
	// accuracies contains 3 vector<float> (as many as classifiers), and each vector<float> contains NUM_CLASS float.
	calculateAccuracies(accuracies, confusionMatrices);
}

void computeMES(Mat &finalResponse, const vector<vector<float> > &accuracies,
		const CvSVM &svm_hog, const CvSVM &svm_lbp, const CvSVM &svm_bb) {
	// Predict the testing samples labels with the 3 classifiers
	vector<Mat> testResponses;
	calculateTestResponses(testResponses, svm_hog, svm_lbp, svm_bb);

	// Calculate the vector of matrices of votes (1 matrix of votes for each sample; the matrix size is 3x3, 3 classes and 3 classifiers)
	// delta_ik(b) is the vote of k-th classifier in relation to the i-th class for the sample b (so the vote can be 0 or 1)
	vector<Mat> votesMatrices;
	createVotesMatrices(votesMatrices, testResponses);

	// Calculate the best classes for each sample based on reliability of MES for each class (psi(0), psi(1) and psi(2); I choose 0,1 or 2 by argmax(psi(i)) on i)
	finalResponse = Mat(votesMatrices.size(), 1, CV_32FC1);
	calculateFinalResponse(finalResponse, votesMatrices, accuracies);
}

void computeSingleMES(float &response, const Mat &img,
		const vector<vector<float> > &accuracies, const CvSVM &svm_hog,
		const CvSVM &svm_lbp, const CvSVM &svm_bb) {
	// Predict the testing samples labels with the 3 classifiers
	vector<float> testResponse;
	calculateSingleTestResponse(testResponse, img, svm_hog, svm_lbp, svm_bb);

	// Calculate the vector of matrix of votes(the matrix size is 3x3, 3 classes and 3 classifiers)
	// delta_ik(b) is the vote of k-th classifier in relation to the i-th class for the sample b (so the vote can be 0 or 1)
	Mat votesMatrix;
	createVotesMatrix(votesMatrix, testResponse);

	// Calculate the best class based on reliability of MES for each class (psi(0), psi(1) and psi(2); I choose 0,1 or 2 by argmax(psi(i)) on i)
	calculateSingleFinalResponse(response, votesMatrix, accuracies);
}

void classifySample(const Mat &img, const vector<vector<float> > &accuracies,
		const CvSVM &svm_hog, const CvSVM &svm_lbp, const CvSVM &svm_bb,
		const CvSVM &svm_concat, float &label) {
	if (!USE_MES) {
		switch (DESCRIPTOR_TYPE) {
		case 0: {
			vector<float> featureVec;
			computeSingleHOG(featureVec, img);
			Mat featureVectorMat(1, featureVec.size(), CV_32FC1,
					featureVec.data());
			label = svm_hog.predict(featureVectorMat);
		}
			break;
		case 1: {
			Mat featureVec;
			computeSingleLBP(featureVec, img);
			label = svm_lbp.predict(featureVec);
		}
			break;
		case 2: {
			vector<float> featureVec;
			computeSingleBB(featureVec, img);
			Mat featureVectorMat(1, featureVec.size(), CV_32FC1,
					featureVec.data());
			label = svm_bb.predict(featureVectorMat);
		}
			break;
		case 3: {
			Mat featureVecMat;
			singleConcatFeatureVec(featureVecMat, img);
			label = svm_concat.predict(featureVecMat);
		}
			break;
		}
	} else {
		computeSingleMES(label, img, accuracies, svm_hog, svm_lbp, svm_bb);
	}
}

void showVideoWithClassification(const String inputPath) {
	// Load all the images
	vector<Mat> images;
	vector<String> path;
	path.push_back(inputPath);
	loadImages(images, path);

	// Load samples from videox_bboxes/ folder
	vector<String> fileNames;
	glob(inputPath, fileNames, true);

	// Load the 4 trained classifiers
	CvSVM svm_hog, svm_lbp, svm_bb, svm_concat;
	svm_hog.load("svm_0_classifier.xml");
	svm_lbp.load("svm_1_classifier.xml");
	svm_bb.load("svm_2_classifier.xml");
	svm_concat.load("svm_3_classifier.xml");

	// Init MES params if MES has been choosed
	vector<vector<float> > accuracies;
	if (USE_MES) {
		initMES(accuracies);
	}

	// Open videox
	stringstream sstr;
	sstr << VIDEO_NAME << "/" << "video.mp4";
	VideoCapture cap(sstr.str());
	if (!cap.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return;
	}

	// Classify on video
	namedWindow("Video con classificazione");
	startWindowThread();
	Mat frameImg, resized;
	int numFrame = 0;
	while (cap.isOpened()) {
		if (cap.grab()) {
			cap.retrieve(frameImg);
			resize(frameImg, resized, Size(960, 540));
			// Search for the samples at this frame
			for (unsigned int i = 0; i < fileNames.size(); i++) {
				// Get sample file name
				stringstream ss(fileNames[i]);
				string name;
				getline(ss, name, '/');
				getline(ss, name, '/');
				// ss1 contains the sample name
				stringstream ss1(name);
				// Declare frame, x, y, width and height
				string frame, x, y, width, height;
				getline(ss1, frame, '_');
				frame = frame.erase(0, frame.find_first_not_of('0'));
				int sampleFrame = atoi(frame.c_str());
				if (sampleFrame == numFrame) {
					getline(ss1, x, '_');
					int xInt = atoi(x.c_str());
					getline(ss1, y, '_');
					int yInt = atoi(y.c_str());
					getline(ss1, width, '_');
					int widthInt = atoi(width.c_str());
					getline(ss1, height, '_');
					int heightInt = atoi(height.c_str());
					stringstream ss2;
					ss2 << height;
					getline(ss2, height, '.');
					float label;
					classifySample(images[i], accuracies, svm_hog, svm_lbp,
							svm_bb, svm_concat, label);
					Scalar color;
					String result;
					if (label == 0){
						color = Scalar(255, 0, 0);
						result = "Pedestrian";
					}
					else if (label == 1){
						color = Scalar(0, 255, 0);
						result = "Car";
					}
					else if (label == 2){
						color = Scalar(0, 0, 255);
						result = "Unknown";
					}
					putText(resized, result, Point(xInt / 2, yInt / 2), FONT_HERSHEY_PLAIN, 2.0, color, 2.0, 8);
					rectangle(resized, Point(xInt / 2, yInt / 2),
							Point(xInt / 2 + widthInt / 2,
									yInt / 2 + heightInt / 2), color, 3);
				} else if (sampleFrame > numFrame) {
					break;
				}
			}
			imshow("Video con classificazione", resized);
			waitKey(1);
			numFrame++;
		}
	}
	destroyWindow("Video con classificazione");
}

void saveClassificationAsXml(const String inputPath) {
	vector<String> fileNames;
	glob(inputPath, fileNames, true);

	// Load all the images
	vector<Mat> images;
	vector<String> path;
	path.push_back(inputPath);
	loadImages(images, path);

	// Load the 4 trained classifiers
	CvSVM svm_hog, svm_lbp, svm_bb, svm_concat;
	svm_hog.load("svm_0_classifier.xml");
	svm_lbp.load("svm_1_classifier.xml");
	svm_bb.load("svm_2_classifier.xml");
	svm_concat.load("svm_3_classifier.xml");

	// Init MES params if MES has been choosed
	vector<vector<float> > accuracies;
	if (USE_MES) {
		initMES(accuracies);
	}

	// Crea the xml document
	xml_document<> doc;
	xml_node<>* declNode = doc.allocate_node(node_declaration);
	declNode->append_attribute(doc.allocate_attribute("version", "1.0"));
	declNode->append_attribute(doc.allocate_attribute("encoding", "utf-8"));
	doc.append_node(declNode);

	for (unsigned int i = 0; i < images.size(); i++) {
		float label;
		classifySample(images[i], accuracies, svm_hog, svm_lbp, svm_bb,
				svm_concat, label);
		string sampleClass;
		if (label == 0)
			sampleClass = "Pedestrian";
		else if (label == 1)
			sampleClass = "Vehicle";
		else if (label == 2)
			sampleClass = "Unknown";

		xml_node<>* dataBBNode = doc.allocate_node(node_element, "data:bbox");

		// Creates a string for each <data:bbox>..</> attribute value
		stringstream ss(fileNames[i]);
		string name;
		getline(ss, name, '/');
		getline(ss, name, '/');
		stringstream ss1(name);
		string frame, x, y, width, height;
		getline(ss1, frame, '_');
		getline(ss1, x, '_');
		getline(ss1, y, '_');
		getline(ss1, width, '_');
		getline(ss1, height, '_');
		stringstream ss2;
		ss2 << height;
		getline(ss2, height, '.');
		stringstream ss3;
		frame = frame.erase(0, frame.find_first_not_of('0'));
		ss3 << frame << ":" << frame;

		// Appends the attributes to the <data:bbox>..</> node
		dataBBNode->append_attribute(
				doc.allocate_attribute("framespan",
						doc.allocate_string(ss3.str().c_str())));
		ss3.str("");
		ss3 << height;
		dataBBNode->append_attribute(
				doc.allocate_attribute("height",
						doc.allocate_string(ss3.str().c_str())));
		ss3.str("");
		ss3 << width;
		dataBBNode->append_attribute(
				doc.allocate_attribute("width",
						doc.allocate_string(ss3.str().c_str())));
		ss3.str("");
		ss3 << x;
		dataBBNode->append_attribute(
				doc.allocate_attribute("x",
						doc.allocate_string(ss3.str().c_str())));
		ss3.str("");
		ss3 << y;
		dataBBNode->append_attribute(
				doc.allocate_attribute("y",
						doc.allocate_string(ss3.str().c_str())));

		dataBBNode->append_attribute(
				doc.allocate_attribute("class",
						doc.allocate_string(sampleClass.c_str())));
		doc.append_node(dataBBNode);
	}

	// Save to file
	String classifier;
	if (!USE_MES) {
		if (DESCRIPTOR_TYPE == 0)
			classifier = "hog";
		else if (DESCRIPTOR_TYPE == 1)
			classifier = "lbp";
		else if (DESCRIPTOR_TYPE == 2)
			classifier = "bb";
		else if (DESCRIPTOR_TYPE == 3)
			classifier = "concat";
	} else {
		classifier = "mes";
	}
	stringstream sst;
	sst << VIDEO_NAME << "_classification/svm_" << classifier << ".xml";
	ofstream file_stored(sst.str().c_str());
	file_stored << doc;
	file_stored.close();
	doc.clear();
}

void classifySamplesFromVideo(const String path) {
	if (VIDEO_CLASS_OUTPUT == 1) {
		saveClassificationAsXml(path);
		cout << "File xml generato con successo!" << endl;
	} else {
		showVideoWithClassification(path);
	}
}

void classify() {
	// If MES is not used
	if (!USE_MES) {
		CvSVM svm;

		if (!LOAD_CLASSIFIER) {
			// Create the matrix of all the feature vectors of training samples and the matrix of all the labels of training samples
			Mat featureVecMat, labelsMat;
			vector<String> paths;
			paths.push_back("train_pedestrians/*.jpg");
			paths.push_back("train_vehicles/*.jpg");
			paths.push_back("train_unknown/*.jpg");
			createFeatureVectorsMat(featureVecMat, paths);
			createLabelsMat(labelsMat, paths);
			// Train the SVM classifier
			SVMtrain(svm, featureVecMat, labelsMat);
		} else {
			stringstream ss;
			ss << "svm_" << DESCRIPTOR_TYPE << "_classifier.xml";
			svm.load(ss.str().c_str());
		}

		// Create the matrix of all the feature vectors of testing samples and the matrix of all the labels of testing samples
		Mat testFeatureVecMat, testLabelsMat;
		if (ACC_EVALUATION) {
			vector<String> paths;
			paths.push_back("test_pedestrians/*.jpg");
			paths.push_back("test_vehicles/*.jpg");
			paths.push_back("test_unknown/*.jpg");
			createFeatureVectorsMat(testFeatureVecMat, paths);
			createLabelsMat(testLabelsMat, paths);
		}

		// Predict the testing samples class with the classifier
		Mat testResponse;
		svm.predict(testFeatureVecMat, testResponse);

		if (ACC_EVALUATION) {
			// Evaluate the classifier accuracy
			float count = 0;
			float accuracy = 0;
			SVMevaluate(testResponse, count, accuracy, testLabelsMat);

			// Print the result
			cout << "The accuracy is " << accuracy << "%" << endl;
		}
	} else {
		// Load the 3 trained classifiers
		CvSVM svm_hog, svm_lbp, svm_bb;
		svm_hog.load("svm_0_classifier.xml");
		svm_lbp.load("svm_1_classifier.xml");
		svm_bb.load("svm_2_classifier.xml");

		// Init MES params
		vector<vector<float> > accuracies;
		initMES(accuracies);

		// Use MES
		Mat finalResponse;
		computeMES(finalResponse, accuracies, svm_hog, svm_lbp, svm_bb);

		if (ACC_EVALUATION) {
			// Create the matrix containing all the labels for the test samples
			Mat testLabelsMat;
			vector<String> paths;
			paths.push_back("test_pedestrians/*.jpg");
			paths.push_back("test_vehicles/*.jpg");
			paths.push_back("test_unknown/*.jpg");
			createLabelsMat(testLabelsMat, paths);

			// Evaluate the classifier accuracy
			float count = 0;
			float accuracy = 0;
			SVMevaluate(finalResponse, count, accuracy, testLabelsMat);

			// Print the result
			cout << "The accuracy is " << accuracy << "%" << endl;
		}
	}
}

void extractSamplesFromVideo(const String pathVideo, const String pathXml,
		const String pathSave) {
	// Open the video file
	VideoCapture cap(pathVideo);
	if (!cap.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return;
	}

	// Parse the xml file into doc
	file<> xmlFile(pathXml.c_str());
	xml_document<> doc;
	doc.parse<0>(xmlFile.data());

	// Read the <sourcefile>...</sourcefile> node
	xml_node<> *sourcefileNode =
			doc.first_node()->first_node("data")->first_node();

	// Iterate through all <object>...</object> nodes
	for (xml_node<> *objNode = sourcefileNode->first_node("object"); objNode;
			objNode = objNode->next_sibling()) {
		// Read the 2nd <attribute>...</attribute> node inside <object>...</object>
		xml_node<> *attributeNode = objNode->first_node()->next_sibling();

		// Iterate through all <data:bbox>...</data:bbox> nodes
		for (xml_node<> *bboxNode = attributeNode->first_node("data:bbox");
				bboxNode; bboxNode = bboxNode->next_sibling()) {
			// Read the frame number of the bbox
			stringstream ss(bboxNode->first_attribute("framespan")->value());
			string framespan;
			getline(ss, framespan, ':');
			double frame = atof(framespan.c_str());

			// Read the width, height, x and y of the bbox
			int height = atoi(bboxNode->first_attribute("height")->value());
			int width = atoi(bboxNode->first_attribute("width")->value());
			int x = atoi(bboxNode->first_attribute("x")->value());
			int y = atoi(bboxNode->first_attribute("y")->value());

			// Set the next frame
			cap.set(CV_CAP_PROP_POS_FRAMES, frame);

			// Save the frame into a Mat (image)
			Mat frameImg;
			bool success = cap.read(frameImg);
			if (!success) {
				cout << "Cannot read  frame " << endl;
				continue;
			}

			// Check if the bbox goes out of the img
			if (x >= 0 && y >= 0 && (x + width) <= frameImg.cols
					&& (y + height) <= frameImg.rows) {
				// Create the Rect to crop the bbox from the original frame
				Rect roi(Point(x, y), Point(x + width, y + height));

				// Crop the bbox from the original frame
				Mat cropImage = frameImg(roi);

				// Save the bbox crop as jpeg file
				stringstream s;
				s << setw(4) << setfill('0') << frame;// 0000, 0001, 0002, etc...
				string numFrame = s.str();
				stringstream st;
				st << pathSave << numFrame << "_" << x << "_" << y << "_"
						<< width << "_" << height << ".jpg";
				imwrite(st.str(), cropImage);
			}
		}
	}
}

void clearScreen() {
	system("clear");
}

void displayClassVideoMenu() {
	cout
			<< "Classificazione di Pedoni, Veicoli e Oggetti Sconosciuti presenti in video"
			<< endl << endl;
	cout << "Scegli il video su cui effettuare la classificazione:" << endl;
	cout << "1. Video 1;" << endl;
	cout << "2. Video 2;" << endl;
	cout << "3. Video 3." << endl;
	int videoNum;
	cin >> videoNum;

	while (videoNum < 1 || videoNum > 3 || cin.fail()) {
		cin.clear();
		cin.ignore(10000, '\n');
		cout << "Scelta errata! Scegliere un valore compreso tra 1 e 3!"
				<< endl;
		cin >> videoNum;
	}

	stringstream ss;
	ss << "video" << videoNum;
	VIDEO_NAME = ss.str();

	cout << endl << "Scegli la tipologia di output:" << endl;
	cout << "1. Genera il file xml contenente la classificazione;" << endl;
	cout << "2. Mostra il video con la classificazione." << endl;
	cin >> VIDEO_CLASS_OUTPUT;

	while (VIDEO_CLASS_OUTPUT < 1 || VIDEO_CLASS_OUTPUT > 2 || cin.fail()) {
		cin.clear();
		cin.ignore(10000, '\n');
		cout << "Scelta errata! Scegliere un valore compreso tra 1 e 2!"
				<< endl;
		cin >> VIDEO_CLASS_OUTPUT;
	}

	clearScreen();
	cout
			<< "Classificazione di Pedoni, Veicoli e Oggetti Sconosciuti presenti in video"
			<< endl << endl;
	cout << "Scegli la modalità desiderata:" << endl;
	cout << "0. Singolo classificatore;" << endl;
	cout << "1. MES (Sistema Multi Esperto)." << endl;
	cin >> USE_MES;

	while (USE_MES < 0 || USE_MES > 1 || cin.fail()) {
		cin.clear();
		cin.ignore(10000, '\n');
		cout << "Scelta errata! Scegliere un valore compreso tra 0 ed 1!"
				<< endl;
		cin >> USE_MES;
	}

	clearScreen();
	if (USE_MES) {
		cout
				<< "Classificazione di Pedoni, Veicoli e Oggetti Sconosciuti presenti in video utilizzando MES"
				<< endl << endl;
	} else {
		cout
				<< "Classificazione di Pedoni, Veicoli e Oggetti Sconosciuti presenti in video utilizzando un singolo classificatore"
				<< endl << endl;
		cout << "Scegli il descrittore da utilizzare:" << endl;
		cout << "0. HOG (forma);" << endl;
		cout << "1. LBP (texture);" << endl;
		cout << "2. Dimensione Bounding Box (dimensione);" << endl;
		cout << "3. Concatenazione HOG+LBP+DIM_BB." << endl;
		cin >> DESCRIPTOR_TYPE;

		while (DESCRIPTOR_TYPE < 0 || DESCRIPTOR_TYPE > 3 || cin.fail()) {
			cin.clear();
			cin.ignore(10000, '\n');
			cout << "Scelta errata! Scegliere un valore compreso tra 0 ed 3!"
					<< endl;
			cin >> DESCRIPTOR_TYPE;
		}
	}
	ss.str("");
	ss << VIDEO_NAME << "_bboxes/*.jpg";
	cout << "Sto generando l'output. Attendi qualche secondo." << endl;
	classifySamplesFromVideo(ss.str());
	askBackToMainMenu();
}

void displayAccEvaluationMenu() {
	cout << "Valutazione dell'accuratezza sul Test Set" << endl << endl;
	cout << "Scegli la modalità desiderata:" << endl;
	cout << "0. Singolo classificatore;" << endl;
	cout << "1. MES (Sistema Multi Esperto)." << endl;
	cin >> USE_MES;

	while (USE_MES < 0 || USE_MES > 1 || cin.fail()) {
		cin.clear();
		cin.ignore(10000, '\n');
		cout << "Scelta errata! Scegliere un valore compreso tra 0 ed 1!"
				<< endl;
		cin >> USE_MES;
	}

	clearScreen();
	if (USE_MES) {
		cout << "Valutazione dell'accuratezza sul Test Set utilizzando MES"
				<< endl << endl;
		cout
				<< "Sto calcolando l'accuratezza richiesta. Attendi qualche secondo."
				<< endl;
	} else {
		cout
				<< "Valutazione dell'accuratezza sul Test Set utilizzando un singolo classificatore"
				<< endl << endl;
		cout << "Scegli il descrittore da utilizzare:" << endl;
		cout << "0. HOG (forma);" << endl;
		cout << "1. LBP (texture);" << endl;
		cout << "2. Dimensione Bounding Box (dimensione);" << endl;
		cout << "3. Concatenazione HOG+LBP+DIM_BB." << endl;
		cin >> DESCRIPTOR_TYPE;

		while (DESCRIPTOR_TYPE < 0 || DESCRIPTOR_TYPE > 3 || cin.fail()) {
			cin.clear();
			cin.ignore(10000, '\n');
			cout << "Scelta errata! Scegliere un valore compreso tra 0 ed 3!"
					<< endl;
			cin >> DESCRIPTOR_TYPE;
		}

		cout << endl
				<< "Scegli se addestrare il classificatore o se caricarlo da file:"
				<< endl;
		cout
				<< "0. Addestra il classificatore e salvalo su file (WARN: il training richiede da 1 a 10 minuti a seconda del descrittore utilizzato);"
				<< endl;
		cout << "1. Carica il classificatore da file." << endl;
		cin >> LOAD_CLASSIFIER;

		while (LOAD_CLASSIFIER < 0 || LOAD_CLASSIFIER > 1 || cin.fail()) {
			cin.clear();
			cin.ignore(10000, '\n');
			cout << "Scelta errata! Scegliere un valore compreso tra 0 ed 1!"
					<< endl;
			cin >> LOAD_CLASSIFIER;
		}

		clearScreen();
		cout
				<< "Valutazione dell'accuratezza sul Test Set utilizzando un singolo classificatore"
				<< endl << endl;
		if (LOAD_CLASSIFIER) {
			cout
					<< "Sto calcolando l'accuratezza richiesta. Attendi qualche secondo."
					<< endl;
		} else {
			cout
					<< "Sto addestrando il classificatore e calcolando l'accuratezza richiesta. Attendi qualche secondo."
					<< endl;
		}
	}
	classify();
	askBackToMainMenu();
}

void displayMainMenu() {
	cout << "Menù principale" << endl << endl;
	cout << "Utilizza l'applicazione in modalità:" << endl;
	cout
			<< "0. Classificazione di Pedoni, Veicoli e Oggetti Sconosciuti presenti in video;"
			<< endl;
	cout << "1. Valutazione dell'accuratezza sul Test Set." << endl;
	cin >> ACC_EVALUATION;

	while (ACC_EVALUATION < 0 || ACC_EVALUATION > 1 || cin.fail()) {
		cin.clear();
		cin.ignore(10000, '\n');
		cout << "Scelta errata! Scegliere un valore compreso tra 0 ed 1!"
				<< endl;
		cin >> ACC_EVALUATION;
	}

	clearScreen();
	if (ACC_EVALUATION) {
		displayAccEvaluationMenu();
	} else {
		LOAD_CLASSIFIER = 1;
		displayClassVideoMenu();
	}
}

void askBackToMainMenu() {
	int ans;
	cout << endl << "Vuoi ritornare al menù principale?" << endl;
	cout << "0. No, esci dal programma" << endl;
	cout << "1. Sì." << endl;

	cin >> ans;

	while (ans < 0 || ans > 1 || cin.fail()) {
		cin.clear();
		cin.ignore(10000, '\n');
		cout << "Scelta errata! Scegliere un valore compreso tra 0 ed 1!"
				<< endl;
		cin >> ans;
	}

	if (ans) {
		clearScreen();
		displayMainMenu();
	}
}
