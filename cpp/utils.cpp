#include"utils.h"

void dense(Mat &frame, vector< vector<Point3f>>landmarks, vector<Mat> Rs, Scalar color)
{
	for (int i = 0; i < landmarks.size(); i++)
	{
		for (int j = 0; j < landmarks[i].size(); j++)
		{
			if (j % 6 != 0) continue;
			circle(frame, Point(int(landmarks[i][j].x), int(landmarks[i][j].y)), 1, color, -1);
		}
	}
}

void sparse(Mat &frame, vector< vector<Point2f>>landmarks, vector<Mat> Rs, Scalar color)
{
	for (int i = 0; i < landmarks.size(); i++)
	{
		const int num_pts = landmarks[i].size();
		for (int j = 0; j < num_pts; j++)
		{
			circle(frame, Point(int(landmarks[i][j].x), int(landmarks[i][j].y)), 2, color, 0);
		}

		vector<Point> pts(17);
		for (int j = 0; j < 17; j++)
		{
			pts[j] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, false, color, 1);

		pts.resize(5);
		for (int j = 17; j < 22; j++)
		{
			pts[j - 17] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, false, color, 1);

		pts.resize(5);
		for (int j = 22; j < 27; j++)
		{
			pts[j - 22] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, false, color, 1);

		pts.resize(4);
		for (int j = 27; j < 31; j++)
		{
			pts[j - 27] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, false, color, 1);

		pts.resize(5);
		for (int j = 31; j < 36; j++)
		{
			pts[j - 31] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, false, color, 1);

		pts.resize(6);
		for (int j = 36; j < 42; j++)
		{
			pts[j - 36] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, true, color, 1);

		pts.resize(6);
		for (int j = 42; j < 48; j++)
		{
			pts[j - 42] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, true, color, 1);

		pts.resize(12);
		for (int j = 48; j < 60; j++)
		{
			pts[j - 48] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, true, color, 1);

		pts.resize(num_pts - 60);
		for (int j = 60; j < num_pts; j++)
		{
			pts[j - 60] = Point(int(landmarks[i][j].x), int(landmarks[i][j].y));
		}
		polylines(frame, pts, true, color, 1);
	}
}

void pose(Mat &frame, vector< vector<Point2f>>landmarks, vector<Mat> Rs, Scalar color)
{
	const float factor = sqrt(2);
	for (int i = 0; i < landmarks.size(); i++)
	{
		float max_x = -10000;
		float max_y = -10000;
		float min_x = 10000;
		float min_y = 10000;
		float center_x = 0;
		float center_y = 0;
		const int num_pts = landmarks[i].size();
		for (int j = 0; j < num_pts; j++)
		{
			if (landmarks[i][j].x > max_x)
			{
				max_x = landmarks[i][j].x;
			}
			if (landmarks[i][j].y > max_y)
			{
				max_y = landmarks[i][j].y;
			}
			if (landmarks[i][j].x < min_x)
			{
				min_x = landmarks[i][j].x;
			}
			if (landmarks[i][j].y < min_y)
			{
				min_y = landmarks[i][j].y;
			}
			if (j < 27)
			{
				center_x += landmarks[i][j].x;
				center_y += landmarks[i][j].y;
			}
		}
		center_x /= 27.0;
		center_y /= 27.0;

		const float radius = max(max_x - min_x, max_y - min_y) / 2;
		const float front_size = factor * radius;
		const float front_depth = factor * radius;
		float a[8 * 3] = { -radius, -radius, 0,-radius, radius, 0,radius, radius, 0,radius, -radius, 0,-front_size, -front_size, front_depth,-front_size, front_size, front_depth,front_size, front_size, front_depth,front_size, -front_size, front_depth };
		Mat projections(8, 3, CV_32FC1, a);

		Rect rect(0, 0, 2, 3);
		Mat rotate_matrix = Rs[i](rect);
		rotate_matrix.ptr<float>(0)[1] *= -1;
		rotate_matrix.ptr<float>(1)[1] *= -1;
		rotate_matrix.ptr<float>(2)[1] *= -1;
		Mat M_points = projections * rotate_matrix;  ///8ÐÐ2ÁÐ

		const float inds[12][2] = { {0, 1},{1, 2},{2, 3},{3, 0},
									{0, 4},{1, 5},{2, 6},{3, 7},
									{4, 5},{5, 6},{6, 7},{7, 4} };
		vector<Point> pts(2);
		for (int j = 0; j < 12; j++)
		{
			float x = M_points.ptr<float>(inds[j][0])[0] + center_x;
			float y = M_points.ptr<float>(inds[j][0])[1] + center_y;
			pts[0] = Point(int(x), int(y));
			x = M_points.ptr<float>(inds[j][1])[0] + center_x;
			y = M_points.ptr<float>(inds[j][1])[1] + center_y;
			pts[1] = Point(int(x), int(y));
			polylines(frame, pts, false, color, 2);
		}
	}
}

void _render(const int *triangles,
	const int ntri,
	const float *light,
	const float *directional,
	const float *ambient,
	const float *vertices,
	const int nver,
	unsigned char *image,
	const int h, const int w)
{
	int tri_p0_ind, tri_p1_ind, tri_p2_ind;
	int color_index;
	float dot00, dot01, dot11, dot02, dot12;
	float cos_sum, det;

	struct Tuple3D p0, p1, p2;
	struct Tuple3D v0, v1, v2;
	struct Tuple3D p, start, end;

	struct Tuple3D ver_max = { -1.0e8, -1.0e8, -1.0e8 };
	struct Tuple3D ver_min = { 1.0e8, 1.0e8, 1.0e8 };
	struct Tuple3D ver_mean = { 0.0, 0.0, 0.0 };

	float *ver_normal = (float *)calloc(3 * nver, sizeof(float));
	float *colors = (float *)malloc(3 * nver * sizeof(float));
	float *depth_buffer = (float *)calloc(h * w, sizeof(float));

	for (int i = 0; i < ntri; i++)
	{
		tri_p0_ind = triangles[3 * i];
		tri_p1_ind = triangles[3 * i + 1];
		tri_p2_ind = triangles[3 * i + 2];

		// counter clockwise order
		start.x = vertices[tri_p1_ind] - vertices[tri_p0_ind];
		start.y = vertices[tri_p1_ind + 1] - vertices[tri_p0_ind + 1];
		start.z = vertices[tri_p1_ind + 2] - vertices[tri_p0_ind + 2];

		end.x = vertices[tri_p2_ind] - vertices[tri_p0_ind];
		end.y = vertices[tri_p2_ind + 1] - vertices[tri_p0_ind + 1];
		end.z = vertices[tri_p2_ind + 2] - vertices[tri_p0_ind + 2];

		p.x = start.y * end.z - start.z * end.y;
		p.y = start.z * end.x - start.x * end.z;
		p.z = start.x * end.y - start.y * end.x;

		ver_normal[tri_p0_ind] += p.x;
		ver_normal[tri_p1_ind] += p.x;
		ver_normal[tri_p2_ind] += p.x;

		ver_normal[tri_p0_ind + 1] += p.y;
		ver_normal[tri_p1_ind + 1] += p.y;
		ver_normal[tri_p2_ind + 1] += p.y;

		ver_normal[tri_p0_ind + 2] += p.z;
		ver_normal[tri_p1_ind + 2] += p.z;
		ver_normal[tri_p2_ind + 2] += p.z;
	}

	for (int i = 0; i < nver; ++i)
	{
		p.x = ver_normal[3 * i];
		p.y = ver_normal[3 * i + 1];
		p.z = ver_normal[3 * i + 2];

		det = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
		if (det <= 0)
			det = 1e-6;

		ver_normal[3 * i] /= det;
		ver_normal[3 * i + 1] /= det;
		ver_normal[3 * i + 2] /= det;

		ver_mean.x += p.x;
		ver_mean.y += p.y;
		ver_mean.z += p.z;

		ver_max.x = max(ver_max.x, p.x);
		ver_max.y = max(ver_max.y, p.y);
		ver_max.z = max(ver_max.z, p.z);

		ver_min.x = min(ver_min.x, p.x);
		ver_min.y = min(ver_min.y, p.y);
		ver_min.z = min(ver_min.z, p.z);
	}

	ver_mean.x /= nver;
	ver_mean.y /= nver;
	ver_mean.z /= nver;

	for (int i = 0; i < nver; ++i)
	{
		colors[3 * i] = vertices[3 * i];
		colors[3 * i + 1] = vertices[3 * i + 1];
		colors[3 * i + 2] = vertices[3 * i + 2];

		colors[3 * i] -= ver_mean.x;
		colors[3 * i] /= ver_max.x - ver_min.x;

		colors[3 * i + 1] -= ver_mean.y;
		colors[3 * i + 1] /= ver_max.y - ver_min.y;

		colors[3 * i + 2] -= ver_mean.z;
		colors[3 * i + 2] /= ver_max.z - ver_min.z;

		p.x = light[0] - colors[3 * i];
		p.y = light[1] - colors[3 * i + 1];
		p.z = light[2] - colors[3 * i + 2];

		det = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
		if (det <= 0)
			det = 1e-6;

		colors[3 * i] = p.x / det;
		colors[3 * i + 1] = p.y / det;
		colors[3 * i + 2] = p.z / det;

		colors[3 * i] *= ver_normal[3 * i];
		colors[3 * i + 1] *= ver_normal[3 * i + 1];
		colors[3 * i + 2] *= ver_normal[3 * i + 2];

		cos_sum = colors[3 * i] + colors[3 * i + 1] + colors[3 * i + 2];

		colors[3 * i] = clip(cos_sum * directional[0] + ambient[0], 0, 1);
		colors[3 * i + 1] = clip(cos_sum * directional[1] + ambient[1], 0, 1);
		colors[3 * i + 2] = clip(cos_sum * directional[2] + ambient[2], 0, 1);
	}

	for (int i = 0; i < ntri; ++i)
	{
		tri_p0_ind = triangles[3 * i];
		tri_p1_ind = triangles[3 * i + 1];
		tri_p2_ind = triangles[3 * i + 2];

		p0.x = vertices[tri_p0_ind];
		p0.y = vertices[tri_p0_ind + 1];
		p0.z = vertices[tri_p0_ind + 2];

		p1.x = vertices[tri_p1_ind];
		p1.y = vertices[tri_p1_ind + 1];
		p1.z = vertices[tri_p1_ind + 2];

		p2.x = vertices[tri_p2_ind];
		p2.y = vertices[tri_p2_ind + 1];
		p2.z = vertices[tri_p2_ind + 2];

		start.x = max(ceil(min(p0.x, min(p1.x, p2.x))), 0);
		end.x = min(floor(max(p0.x, max(p1.x, p2.x))), w - 1);

		start.y = max(ceil(min(p0.y, min(p1.y, p2.y))), 0);
		end.y = min(floor(max(p0.y, max(p1.y, p2.y))), h - 1);

		if (end.x < start.x || end.y < start.y)
			continue;

		v0.x = p2.x - p0.x;
		v0.y = p2.y - p0.y;
		v1.x = p1.x - p0.x;
		v1.y = p1.y - p0.y;

		// dot products np.dot(v0.T, v0)
		dot00 = v0.x * v0.x + v0.y * v0.y;
		dot01 = v0.x * v1.x + v0.y * v1.y;
		dot11 = v1.x * v1.x + v1.y * v1.y;

		// barycentric coordinates
		start.z = dot00 * dot11 - dot01 * dot01;
		if (start.z != 0)
			start.z = 1 / start.z;

		for (p.y = start.y; p.y <= end.y; p.y += 1.0)
		{
			for (p.x = start.x; p.x <= end.x; p.x += 1.0)
			{
				v2.x = p.x - p0.x;
				v2.y = p.y - p0.y;

				dot02 = v0.x * v2.x + v0.y * v2.y;
				dot12 = v1.x * v2.x + v1.y * v2.y;

				v2.z = (dot11 * dot02 - dot01 * dot12) * start.z;
				v1.z = (dot00 * dot12 - dot01 * dot02) * start.z;
				v0.z = 1 - v2.z - v1.z;

				// judge is_point_in_tri by below line of code
				if (v2.z >= 0 && v1.z >= 0 && v0.z > 0)
				{
					p.z = v0.z * p0.z + v1.z * p1.z + v2.z * p2.z;
					color_index = p.y * w + p.x;

					if (p.z > depth_buffer[color_index])
					{
						end.z = v0.z * colors[tri_p0_ind];
						end.z += v1.z * colors[tri_p1_ind];
						end.z += v2.z * colors[tri_p2_ind];
						image[3 * color_index] = end.z * 255;

						end.z = v0.z * colors[tri_p0_ind + 1];
						end.z += v1.z * colors[tri_p1_ind + 1];
						end.z += v2.z * colors[tri_p2_ind + 1];
						image[3 * color_index + 1] = end.z * 255;

						end.z = v0.z * colors[tri_p0_ind + 2];
						end.z += v1.z * colors[tri_p1_ind + 2];
						end.z += v2.z * colors[tri_p2_ind + 2];
						image[3 * color_index + 2] = end.z * 255;

						depth_buffer[color_index] = p.z;
					}
				}
			}
		}
	}

	free(depth_buffer);
	free(colors);
	free(ver_normal);
}

void mesh(Mat &frame, float* landmarks, const int num_faces, const int num_pts, int* triangles, const int ntri)
{
	const float light[3] = { 1, 1, 5 };
	const float directional[3] = { 0.6, 0.6, 0.6 };
	const float ambient[3] = { 0.6, 0.5, 0.4 };

	for (int i = 0; i < num_faces; i++)
	{
		_render(triangles, ntri, light, directional, ambient, landmarks + i * num_pts * 3, num_pts, (uchar*)frame.data, frame.rows, frame.cols);
	}
}