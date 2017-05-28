/*
 * main2.cpp
 *
 *  Created on: 19 mag 2017
 *      Author: angelo
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace cv;
using namespace std;

static string ann = "annotations/nexus11.txt";
static string video = "video/nexus11.mov";
static string ooi = "Car";
static string outputDir = "test_vehicles/";
//static int sampleWidth = 40;
//static int sampleHeight = 40;

typedef pair<Mat, String> Params;

void rclick_callback(int, int, int, int, void*);

int main2(int argc, char** argv) {
	ifstream infile(ann.c_str());
	VideoCapture cap(video);

	if (!cap.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	int id, xMin, yMin, xMax, yMax, numFrame, lost,
			occluded, generated;
	string label, attr0, attr1, attr2, attr3;
	int lastId = 999, countPerObj = 0;

	while (infile >> id >> xMin >> yMin >> xMax >> yMax
			>> numFrame >> lost >> occluded >> generated >> label /*>> attr0 >> attr1 >> attr2 >> attr3*/) {
		if (lost != 1 && occluded != 1
				&& label.compare(1, label.length() - 2, ooi) == 0) {
			Rect roi(Point(xMin, yMin),
					Point(xMax, yMax));
			//if (roi.width == roi.height) {
				if (lastId != id) {
					countPerObj = 0;
					lastId = id;
				}
				if (lastId == id && countPerObj < 20) {
					double frameRate = 29.97;
					double frameTime = 1000.0 * numFrame / frameRate;
					cap.set(CV_CAP_PROP_POS_MSEC, frameTime);
					Mat frame;
					bool success = cap.read(frame);
					if (!success) {
						cout << "Cannot read  frame " << endl;
						continue;
					}
					Mat cropImage = frame(roi);
					/*Mat resizedCropImage;
					resize(cropImage, resizedCropImage,
							Size(sampleWidth, sampleHeight));*/
					String video_name = video.substr(video.find("/") + 1, video.find(".") - video.find("/") - 1);
					stringstream ss;
					ss << video_name << "_" << id << "_" << numFrame;
					String fileName = outputDir + ss.str() + ".jpg";

					vector<int> quality_param;
					quality_param.push_back(CV_IMWRITE_JPEG_QUALITY);
					quality_param.push_back(100);
					imwrite(fileName, cropImage, quality_param);

					/*
					Params params(cropImage, fileName);
					imshow(fileName, cropImage);
					setMouseCallback(fileName, rclick_callback, &params);
					waitKey();
					*/

					countPerObj++;
				}
			//}
		}
	}

	return (0);
}

void rclick_callback(int event, int x, int y, int flags, void* ptr) {
	if (event == EVENT_RBUTTONDOWN) {
		Params *params = (Params*) (ptr);
		String img_name = params->second;
		vector<int> quality_param;
		quality_param.push_back(CV_IMWRITE_JPEG_QUALITY);
		quality_param.push_back(100);
		imwrite(img_name, params->first, quality_param);
		cout << "Image saved!" << endl;
	}
}
