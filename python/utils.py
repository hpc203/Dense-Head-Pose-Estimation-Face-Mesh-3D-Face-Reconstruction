import cv2
import numpy as np


def sparse(frame, landmarks, params, color):
    landmarks = np.round(landmarks).astype(np.int32)
    _ = [
        cv2.circle(frame, tuple(p), 2, color, 0, cv2.LINE_AA) for p in landmarks
    ]
    draw_poly(frame, landmarks, color=color)


def dense(frame, landmarks, params, color):
    landmarks = np.round(landmarks).astype(np.int32)
    _ = [cv2.circle(frame, tuple(p), 1, color, 0, cv2.LINE_AA) for p in landmarks[::6, :2]]
    # _ = [cv2.circle(frame, tuple(p), 1, color, -1, cv2.LINE_AA) for p in landmarks[:, :2]]


def pose(frame, landmarks, params, color):
    # rotate matrix
    R = params[:3, :3].copy()
    # decompose matrix to ruler angle
    # euler = rotationMatrixToEulerAngles(R)
    # print(f"Pitch: {euler[0]} Yaw: {euler[1]} Roll: {euler[2]}")
    draw_projection(frame, R, landmarks, color)


def draw_poly(frame, landmarks, color=(128, 255, 255), thickness=1):
    cv2.polylines(frame, [
        landmarks[:17],
        landmarks[17:22],
        landmarks[22:27],
        landmarks[27:31],
        landmarks[31:36]], False, color, thickness=thickness)

    cv2.polylines(frame, [
        landmarks[36:42],
        landmarks[42:48],
        landmarks[48:60],
        landmarks[60:]
    ], True, color, thickness=thickness)


def draw_projection(frame, R, landmarks, color, thickness=2):
    # build projection matrix
    radius = np.max(np.max(landmarks, 0) - np.min(landmarks, 0)) // 2
    projections = build_projection_matrix(radius)
    # refine rotate matrix
    rotate_matrix = R[:, :2]
    rotate_matrix[:, 1] *= -1
    # 3D -> 2D
    center = np.mean(landmarks[:27], axis=0)
    points = projections @ rotate_matrix + center
    points = points.astype(np.int32)
    # draw poly
    cv2.polylines(frame, np.take(points, [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [4, 5], [5, 6], [6, 7], [7, 4]
    ], axis=0), False, color, thickness, cv2.LINE_AA)


def build_projection_matrix(rear_size, factor=np.sqrt(2)):
    rear_depth = 0
    front_size = front_depth = factor * rear_size
    projections = np.array([
        [-rear_size, -rear_size, rear_depth],
        [-rear_size, rear_size, rear_depth],
        [rear_size, rear_size, rear_depth],
        [rear_size, -rear_size, rear_depth],
        [-front_size, -front_size, front_depth],
        [-front_size, front_size, front_depth],
        [front_size, front_size, front_depth],
        [front_size, -front_size, front_depth],
    ], dtype=np.float32)
    return projections


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    else:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([x, y, z])

def clip(_x, _min, _max):
    return min(max(_x, _min), _max)

class Tuple3D:
    def __init__(self, data):
        self.x = float(data[0])
        self.y = float(data[1])
        self.z = float(data[2])

def mesh(frame, landmarks, triangles):
    light = [1, 1, 5]
    directional = [0.6, 0.6, 0.6]
    ambient = [0.6, 0.5, 0.4]
    nver = landmarks.shape[0]
    vertices = landmarks.flatten()

    p0, p1, p2 = Tuple3D([0, 0, 0]), Tuple3D([0, 0, 0]), Tuple3D([0, 0, 0])
    v0, v1, v2 = Tuple3D([0, 0, 0]), Tuple3D([0, 0, 0]), Tuple3D([0, 0, 0])
    p, start, end = Tuple3D([0, 0, 0]), Tuple3D([0, 0, 0]), Tuple3D([0, 0, 0])

    ver_max = Tuple3D([-1.0e8, -1.0e8, -1.0e8])
    ver_min = Tuple3D([1.0e8, 1.0e8, 1.0e8])
    ver_mean = Tuple3D([0.0, 0.0, 0.0])

    ver_normal = np.zeros(nver * 3, dtype=np.float32)
    colors = np.zeros(nver * 3, dtype=np.float32)
    h, w = frame.shape[:2]
    depth_buffer = np.zeros(h * w, dtype=np.float32)
    image = frame.flatten()

    ntri = triangles.shape[0]
    for i in range(ntri):
        tri_p0_ind, tri_p1_ind, tri_p2_ind = triangles[i, :]

        start.x = vertices[tri_p1_ind] - vertices[tri_p0_ind]
        start.y = vertices[tri_p1_ind + 1] - vertices[tri_p0_ind + 1]
        start.z = vertices[tri_p1_ind + 2] - vertices[tri_p0_ind + 2]

        end.x = vertices[tri_p2_ind] - vertices[tri_p0_ind]
        end.y = vertices[tri_p2_ind + 1] - vertices[tri_p0_ind + 1]
        end.z = vertices[tri_p2_ind + 2] - vertices[tri_p0_ind + 2]

        p.x = start.y * end.z - start.z * end.y
        p.y = start.z * end.x - start.x * end.z
        p.z = start.x * end.y - start.y * end.x

        ver_normal[tri_p0_ind] += p.x
        ver_normal[tri_p1_ind] += p.x
        ver_normal[tri_p2_ind] += p.x

        ver_normal[tri_p0_ind + 1] += p.y
        ver_normal[tri_p1_ind + 1] += p.y
        ver_normal[tri_p2_ind + 1] += p.y

        ver_normal[tri_p0_ind + 2] += p.z
        ver_normal[tri_p1_ind + 2] += p.z
        ver_normal[tri_p2_ind + 2] += p.z

    for i in range(nver):
        p.x = ver_normal[3 * i]
        p.y = ver_normal[3 * i + 1]
        p.z = ver_normal[3 * i + 2]

        det = np.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        if det <= 0:
            det = 1e-6

        ver_normal[3 * i] /= det
        ver_normal[3 * i + 1] /= det
        ver_normal[3 * i + 2] /= det

        ver_mean.x += p.x
        ver_mean.y += p.y
        ver_mean.z += p.z

        ver_max.x = max(ver_max.x, p.x)
        ver_max.y = max(ver_max.y, p.y)
        ver_max.z = max(ver_max.z, p.z)

        ver_min.x = min(ver_min.x, p.x)
        ver_min.y = min(ver_min.y, p.y)
        ver_min.z = min(ver_min.z, p.z)
    ver_mean.x /= nver
    ver_mean.y /= nver
    ver_mean.z /= nver

    for i in range(nver):
        colors[3 * i] = vertices[3 * i]
        colors[3 * i + 1] = vertices[3 * i + 1]
        colors[3 * i + 2] = vertices[3 * i + 2]

        colors[3 * i] -= ver_mean.x
        colors[3 * i] /= ver_max.x - ver_min.x

        colors[3 * i + 1] -= ver_mean.y
        colors[3 * i + 1] /= ver_max.y - ver_min.y

        colors[3 * i + 2] -= ver_mean.z
        colors[3 * i + 2] /= ver_max.z - ver_min.z

        p.x = light[0] - colors[3 * i]
        p.y = light[1] - colors[3 * i + 1]
        p.z = light[2] - colors[3 * i + 2]

        det = np.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        if det <= 0:
            det = 1e-6

        colors[3 * i] = p.x / det
        colors[3 * i + 1] = p.y / det
        colors[3 * i + 2] = p.z / det

        colors[3 * i] *= ver_normal[3 * i]
        colors[3 * i + 1] *= ver_normal[3 * i + 1]
        colors[3 * i + 2] *= ver_normal[3 * i + 2]

        cos_sum = colors[3 * i] + colors[3 * i + 1] + colors[3 * i + 2]

        colors[3 * i] = clip(cos_sum * directional[0] + ambient[0], 0, 1)
        colors[3 * i + 1] = clip(cos_sum * directional[1] + ambient[1], 0, 1)
        colors[3 * i + 2] = clip(cos_sum * directional[2] + ambient[2], 0, 1)

    for i in range(ntri):
        tri_p0_ind, tri_p1_ind, tri_p2_ind = triangles[i, :]

        p0.x = vertices[tri_p0_ind]
        p0.y = vertices[tri_p0_ind + 1]
        p0.z = vertices[tri_p0_ind + 2]

        p1.x = vertices[tri_p1_ind]
        p1.y = vertices[tri_p1_ind + 1]
        p1.z = vertices[tri_p1_ind + 2]

        p2.x = vertices[tri_p2_ind]
        p2.y = vertices[tri_p2_ind + 1]
        p2.z = vertices[tri_p2_ind + 2]

        start.x = max(np.ceil(min(p0.x, min(p1.x, p2.x))), 0)
        end.x = min(np.floor(max(p0.x, max(p1.x, p2.x))), w - 1)

        start.y = max(np.ceil(min(p0.y, min(p1.y, p2.y))), 0)
        end.y = min(np.floor(max(p0.y, max(p1.y, p2.y))), h - 1)

        if end.x < start.x or end.y < start.y:
            continue

        v0.x = p2.x - p0.x
        v0.y = p2.y - p0.y
        v1.x = p1.x - p0.x
        v1.y = p1.y - p0.y

        ## dot products
        dot00 = v0.x * v0.x + v0.y * v0.y
        dot01 = v0.x * v1.x + v0.y * v1.y
        dot11 = v1.x * v1.x + v1.y * v1.y

        ## barycentric coordinates
        start.z = dot00 * dot11 - dot01 * dot01
        if start.z != 0:
            start.z = 1 / start.z

        for j in range(0, int(end.y+1 - start.y)):
            p.y = start.y + j
            for k in range(0, int(end.x+1 - start.x)):
                p.x = start.x + k

                v2.x = p.x - p0.x
                v2.y = p.y - p0.y

                dot02 = v0.x * v2.x + v0.y * v2.y
                dot12 = v1.x * v2.x + v1.y * v2.y

                v2.z = (dot11 * dot02 - dot01 * dot12) * start.z
                v1.z = (dot00 * dot12 - dot01 * dot02) * start.z
                v0.z = 1 - v2.z - v1.z

                ### judge is_point_in_tri by below line of code
                if v2.z >= 0 and v1.z >= 0 and v0.z > 0:
                    p.z = v0.z * p0.z + v1.z * p1.z + v2.z * p2.z
                    color_index = int(p.y * w + p.x)

                    if p.z > depth_buffer[color_index]:
                        end.z = v0.z * colors[tri_p0_ind]
                        end.z += v1.z * colors[tri_p1_ind]
                        end.z += v2.z * colors[tri_p2_ind]
                        image[3 * color_index] = np.uint8(end.z * 255)

                        end.z = v0.z * colors[tri_p0_ind + 1]
                        end.z += v1.z * colors[tri_p1_ind + 1]
                        end.z += v2.z * colors[tri_p2_ind + 1]
                        image[3 * color_index + 1] = np.uint8(end.z * 255)

                        end.z = v0.z * colors[tri_p0_ind + 2]
                        end.z += v1.z * colors[tri_p1_ind + 2]
                        end.z += v2.z * colors[tri_p2_ind + 2]
                        image[3 * color_index + 2] = np.uint8(end.z * 255)

                        depth_buffer[color_index] = p.z
    return image.reshape(frame.shape)