#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<math.h>

using namespace std;
using namespace cv;

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define clip(_x, _min, _max) min(max(_x, _min), _max)

struct Tuple3D
{
	float x;
	float y;
	float z;
};

void dense(Mat &frame, vector< vector<Point3f>>landmarks, vector<Mat> Rs, Scalar color);

void sparse(Mat &frame, vector< vector<Point2f>>landmarks, vector<Mat> Rs, Scalar color);

void pose(Mat &frame, vector< vector<Point2f>>landmarks, vector<Mat> Rs, Scalar color);

void mesh(Mat &frame, float* landmarks, const int num_faces, const int num_pts, int* triangles, const int ntri);
