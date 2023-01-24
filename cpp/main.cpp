#define _CRT_SECURE_NO_WARNINGS
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include"utils.h"

using namespace std;
using namespace cv;
using namespace Ort;

typedef struct BoxInfo
{
	int x1;
	int y1;
	int x2;
	int y2;
	float score;
	int label;
} BoxInfo;

class Detect_Face
{
public:
	Detect_Face(float confThreshold);
	vector<BoxInfo> detect(Mat frame);
	void drawPred(Mat &frame, vector<BoxInfo> bboxes);
private:
	int inpWidth;
	int inpHeight;
	int num_proposal;
	int nout;
	
	float confThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Detect Face");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

Detect_Face::Detect_Face(float confThreshold)
{
	this->confThreshold = confThreshold;

	string model_path = "weights/RFB-320_240x320_post.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
}

void Detect_Face::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());

	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = (pix - 127.5) / 127.5;
			}
		}
	}
}

void Detect_Face::drawPred(Mat &frame, vector<BoxInfo> bboxes)
{
	for (size_t i = 0; i < bboxes.size(); ++i)
	{
		int xmin = bboxes[i].x1;
		int ymin = bboxes[i].y1;
		rectangle(frame, Point(xmin, ymin), Point(bboxes[i].x2, bboxes[i].y2), Scalar(0, 0, 255), 2);
		string label = format("Face:%.2f", bboxes[i].score);
		//label = "Face:" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

vector<BoxInfo> Detect_Face::detect(Mat frame)
{
	Mat dstimg;
	cvtColor(frame, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(this->inpWidth, this->inpHeight));
	
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(0);
	nout = pred_dims.at(1);

	vector<BoxInfo> bboxes;
	int n = 0; ///batchno , classid , score , x1y1x2y2
	const float* pdata = predictions.GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)   ///特征图尺度
	{
		const float class_socre = pdata[2];
		if (class_socre >= this->confThreshold)
		{
			int xmin = int(pdata[3] * (float)frame.cols);
			int ymin = int(pdata[4] * (float)frame.rows);
			int xmax = int(pdata[5] * (float)frame.cols);
			int ymax = int(pdata[6] * (float)frame.rows);

			bboxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, class_socre, int(pdata[1]) });
		}
		pdata += nout;
	}
	return bboxes;
}

class Face_Mesh
{
public:
	Face_Mesh(string mode);
	void detect(Mat &frame, vector<BoxInfo> faces);
	~Face_Mesh();  // 这是析构函数, 释放内存
private:
	int inpWidth;
	int inpHeight;
	string mode;
	int edge_size;
	const int ntri = 76073;
	int* triangles;

	vector<float> input_image_;
	void normalize_(vector<Mat> imgs);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Face Mesh");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

Face_Mesh::Face_Mesh(string mode)
{
	this->mode = mode;
	string model_path;
	if (mode == "pose" || mode == "sparse")
	{
		model_path = "weights/sparse_face_Nx3x120x120.onnx";
	}
	else if (mode == "dense" || mode == "mesh")
	{
		model_path = "weights/dense_face_Nx3x120x120.onnx";
	}
	else
	{
		cout << "input mode is error" << endl;
		exit(100);
	}

	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->edge_size = input_node_dims[0][3];

	const int len = this->ntri * 3;
	this->triangles = new int[len];
	FILE* fp = fopen("triangles.bin", "rb");
	fread(triangles, sizeof(int), len, fp);//导入数据
	fclose(fp);//关闭文件。
}

Face_Mesh::~Face_Mesh()
{
	delete[] triangles;
	triangles = NULL;
}

void Face_Mesh::normalize_(vector<Mat> imgs)
{
	const int imgnum = imgs.size();
	const int img_area = this->inpHeight * this->inpWidth;
	this->input_image_.resize(imgnum * img_area * 3);   ////也可以用opencv里的merge函数
	for (int n = 0; n < imgnum; n++)
	{
		for (int c = 0; c < 3; c++)
		{
			for (int i = 0; i < this->inpHeight; i++)
			{
				for (int j = 0; j < this->inpWidth; j++)
				{
					float pix = imgs[n].ptr<float>(i)[j * 3 + c];
					this->input_image_[n * img_area * 3 + c * img_area + i * this->inpWidth + j] = pix;   /// n, h, w, c
				}
			}
		}
	}
}

void Face_Mesh::detect(Mat &frame, vector<BoxInfo> faces)
{
	vector<Mat> inputs;
	vector<Mat> iMs;
	for (int i = 0; i < faces.size(); i++)
	{
		const float trans_distance = float(this->edge_size) * 0.5;
		const float maximum_edge = float(max(faces[i].x2 - faces[i].x1, faces[i].y2 - faces[i].y1)) * 2.7;
		const float scale = this->edge_size * 2.0 / maximum_edge;

		const float cx = trans_distance - scale * float(faces[i].x2 + faces[i].x1) * 0.5;
		const float cy = trans_distance - scale * float(faces[i].y2 + faces[i].y1) * 0.5;
		
		vector<float> M = { scale, 0, cx, 0, scale, cy };
		Mat warp_mat(2, 3, CV_32FC1, M.data());
		Mat cropped;
		Size outSize(this->inpWidth, this->inpHeight);
		warpAffine(frame, cropped, warp_mat, outSize);

		Mat rgb;
		cvtColor(cropped, rgb, COLOR_BGR2RGB);
		rgb.convertTo(rgb, CV_32FC3);
		normalize(rgb, rgb, -1, 1, NORM_MINMAX);
		inputs.push_back(rgb);

		Mat iM;
		invertAffineTransform(warp_mat, iM);
		iMs.push_back(iM);
	}

	if (inputs.size() > 0)
	{
		this->normalize_(inputs);
		array<int64_t, 4> input_shape_{ inputs.size(), 3, this->inpHeight, this->inpWidth };

		auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

		// 开始推理
		vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理

		auto camera_matrixes_dims = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape();
		const int camera_matrixes_h = camera_matrixes_dims.at(1);
		const int camera_matrixes_w = camera_matrixes_dims.at(2);
		auto landmarks_dims = ort_outputs.at(1).GetTensorTypeAndShapeInfo().GetShape();
		const int landmarks_h = landmarks_dims.at(1);
		const int landmarks_w = landmarks_dims.at(2);
		const int num_faces = iMs.size();

		float *camera_matrixes = ort_outputs[0].GetTensorMutableData<float>();
		float *landmarks = ort_outputs[1].GetTensorMutableData<float>();
		vector<Mat> Rs(num_faces);
		
		if (mode == "pose" || mode == "sparse")
		{
			vector< vector<Point2f>> faces_points(num_faces);
			for (int i = 0; i < num_faces; i++)
			{
				Mat R(camera_matrixes_h, camera_matrixes_w, CV_32FC1, camera_matrixes + i * camera_matrixes_h * camera_matrixes_w);
				vector<Point2f> points(landmarks_h);
				for (int j = 0; j < landmarks_h; j++)
				{
					float* plandmark = landmarks + i * landmarks_h*landmarks_w + j * landmarks_w;
					const float x = plandmark[0] * iMs[i].ptr<float>(0)[0] + iMs[i].ptr<float>(0)[2];
					const float y = plandmark[1] * iMs[i].ptr<float>(0)[0] + iMs[i].ptr<float>(1)[2];
					points[j] = Point2f(x, y);
				}
				faces_points[i] = points;
				Rs[i] = R;
			}
			if (mode == "sparse")
			{
				sparse(frame, faces_points, Rs, Scalar(224, 255, 255));
			}
			else
			{
				pose(frame, faces_points, Rs, Scalar(224, 255, 255));
			}
		}
		else 
		{
			if (mode == "dense")
			{
				vector< vector<Point3f>> faces_points(num_faces);
				for (int i = 0; i < num_faces; i++)
				{
					Mat R(camera_matrixes_h, camera_matrixes_w, CV_32FC1, camera_matrixes + i * camera_matrixes_h * camera_matrixes_w);
					vector<Point3f> points(landmarks_h);
					for (int j = 0; j < landmarks_h; j++)
					{
						float* plandmark = landmarks + i * landmarks_h*landmarks_w + j * landmarks_w;
						const float x = plandmark[0] * iMs[i].ptr<float>(0)[0] + iMs[i].ptr<float>(0)[2];
						const float y = plandmark[1] * iMs[i].ptr<float>(0)[0] + iMs[i].ptr<float>(1)[2];
						const float z = plandmark[2] * iMs[i].ptr<float>(0)[0];
						points[j] = Point3f(x, y, z);
					}
					faces_points[i] = points;
					Rs[i] = R;
				}
				dense(frame, faces_points, Rs, Scalar(224, 255, 255));
			}
			else
			{
				float* faces_points = new float[num_faces * landmarks_h * 3];
				for (int i = 0; i < num_faces; i++)
				{
					for (int j = 0; j < landmarks_h; j++)
					{
						float* plandmark = landmarks + i * landmarks_h*landmarks_w + j * landmarks_w;
						faces_points[i*landmarks_h * 3 + j * 3] = plandmark[0] * iMs[i].ptr<float>(0)[0] + iMs[i].ptr<float>(0)[2];
						faces_points[i*landmarks_h * 3 + j * 3 + 1] = plandmark[1] * iMs[i].ptr<float>(0)[0] + iMs[i].ptr<float>(1)[2];
						faces_points[i*landmarks_h * 3 + j * 3 + 2] = plandmark[2] * iMs[i].ptr<float>(0)[0];
					}
				}
				mesh(frame, faces_points, num_faces, landmarks_h, this->triangles, this->ntri);
				delete [] faces_points;
				faces_points = NULL;
			}
		}
	}
	
	
}
int main()
{
	Detect_Face detect_net(0.7);
	Face_Mesh mesh_net("mesh");  ///choices=["pose", "sparse", "dense", "mesh"]

	string imgpath = "images/4.jpg";
	Mat srcimg = imread(imgpath);
	vector<BoxInfo> bboxes = detect_net.detect(srcimg);
	//detect_net.drawPred(srcimg, bboxes);
	mesh_net.detect(srcimg, bboxes);

	static const string kWinName = "Deep learning Face Mesh in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}
