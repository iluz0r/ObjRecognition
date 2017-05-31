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
	neighbors = max(min(neighbors, 31), 1); // set bounds...
	// Note: alternatively you can switch to the new OpenCV Mat_
	// type system to define an unsigned int matrix... I am probably
	// mistaken here, but I didn't see an unsigned int representation
	// in OpenCV's classic typesystem...
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
				float t = w1 * src.at<int>(i + fy, j + fx)
						+ w2 * src.at<int>(i + fy, j + cx)
						+ w3 * src.at<int>(i + cy, j + fx)
						+ w4 * src.at<int>(i + cy, j + cx);
				// we are dealing with floating point precision, so add some little tolerance
				dst.at<unsigned int>(i - radius, j - radius) += ((t
						> src.at<float>(i, j))
						&& (abs(t - src.at<int>(i, j))
								> std::numeric_limits<float>::epsilon())) << n;
			}
		}
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
	vector<Mat> trainImg;
	int trainPedSize, trainVehSize;
	loadImages(trainImg, trainPedSize, trainVehSize, "train_pedestrians/*.jpg",
			"train_vehicles/*.jpg");

	for(unsigned int i = 0; i < trainImg.size(); i++) {
		Mat trainImgInt, resultLbp;
		//resize(trainImg[i], trainImg[i], Size(), 0.5, 0.5);
		trainImg[i].convertTo(trainImgInt, CV_32SC1);

		Mat lbp;
		ELBP(trainImgInt, lbp, 1, 16);
		lbp.convertTo(resultLbp, CV_8UC1);

		stringstream ss;
		ss << "LBP " << i;
		namedWindow(ss.str(),CV_WINDOW_AUTOSIZE);
		imshow(ss.str(), lbp);

		stringstream ss2;
		ss2 << "Original " << i;
		namedWindow(ss2.str(), CV_WINDOW_AUTOSIZE);
		imshow(ss2.str(), trainImg[i]);
		waitKey(0);
	}
	return (0);
}
