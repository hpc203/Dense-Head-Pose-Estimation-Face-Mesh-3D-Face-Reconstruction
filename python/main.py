import argparse
import cv2
import numpy as np
import onnxruntime as ort
import copy
from utils import pose, sparse, dense, mesh

class Detect_Face():
    def __init__(self, confThreshold=0.8):
        # cv_net = cv2.dnn.readNet('weights/RFB-320_240x320_post.onnx')
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession('weights/RFB-320_240x320_post.onnx', so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        self.confThreshold = confThreshold

    def detect(self, frame):
        image_height, image_width = frame.shape[0], frame.shape[1]
        dstimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dstimg = cv2.resize(dstimg, (self.input_width, self.input_height))
        dstimg = (dstimg.astype(np.float32) - 127.5) / 127.5
        input_image = np.expand_dims(dstimg.transpose(2, 0, 1), axis=0)

        # Inference
        results = self.session.run(None, {self.input_name: input_image})

        # Post process
        bboxes, scores, class_ids = [], [], []
        for batchno_classid_score_x1y1x2y2 in results[0]:
            bbox = batchno_classid_score_x1y1x2y2[-4:].tolist()
            class_id = int(batchno_classid_score_x1y1x2y2[1])
            score = batchno_classid_score_x1y1x2y2[2]
            if score < self.confThreshold:
                continue

            bbox[0] = int(bbox[0] * image_width)
            bbox[1] = int(bbox[1] * image_height)
            bbox[2] = int(bbox[2] * image_width)
            bbox[3] = int(bbox[3] * image_height)
            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)
        return np.asarray(bboxes), np.asarray(scores), np.asarray(class_ids)

    def drawPred(self, frame, bboxes, scores, class_ids):
        for i in range(bboxes.shape[0]):
            left, top, right, bottom = bboxes[i, :]
            # Draw a bounding box.
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
            label = 'Face:'+str(round(scores[i], 2))

            # Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(frame, label, (left, top - 10), 0, 0.7, (0, 255, 0), thickness=2)
        return frame

class Face_Mesh():
    def __init__(self, mode):
        if mode in ['pose', 'sparse']:
            model_path = 'weights/sparse_face_Nx3x120x120.onnx'
        else:
            model_path = 'weights/dense_face_Nx3x120x120.onnx'
        # cv_net = cv2.dnn.readNet(model_path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        self.edge_size = int(self.input_shape[3])
        if mode == 'pose':
            self.alignment_draw_func = pose
        elif mode == 'sparse':
            self.alignment_draw_func = sparse
        elif mode == 'dense':
            self.alignment_draw_func = dense
        self.mode = mode
        self.triangles = np.load('triangles.npy')

    def detect(self, frame, faces):
        inputs = []
        Ms = []
        for face in faces:
            trans_distance = self.edge_size / 2.0
            maximum_edge = max(face[2:4] - face[:2]) * 2.7
            scale = self.edge_size * 2.0 / maximum_edge
            center = (face[2:4] + face[:2]) / 2.0
            cx, cy = trans_distance - scale * center
            M = np.array([[scale, 0, cx], [0, scale, cy]])
            cropped = cv2.warpAffine(frame, M, (self.input_height, self.input_width), borderValue=0.0)
            rgb = cropped[:, :, ::-1].astype(np.float32)
            cv2.normalize(rgb, rgb, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)
            inp = rgb.transpose(2, 0, 1)
            inputs.append(inp)
            Ms.append(M)
        # Inference
        camera_matrixes = []
        landmarks = []
        if len(inputs) > 0:
            camera_matrixes, landmarks = self.session.run(None, {self.input_name: np.asarray(inputs, dtype=np.float32)})
        faces_points = []
        Rs = []
        for camera_matrix, landmark, M_ in zip(camera_matrixes, landmarks, Ms):
            iM = cv2.invertAffineTransform(M_)
            R = copy.deepcopy(camera_matrix)
            points = copy.deepcopy(landmark)
            if self.mode in ['sparse', 'pose']:
                points *= iM[0, 0]
                points += iM[:, -1]
            else:
                points *= iM[0][0]
                points[:, :2] += iM[:, -1]
            faces_points.append(points)
            Rs.append(R)
        return faces_points, Rs

    def drawPred(self, frame, landmarks, Rs):
        if self.mode in ('pose', 'sparse', 'dense'):
            for landmark, R in zip(landmarks, Rs):
                self.alignment_draw_func(frame=frame, landmarks=landmark, params=R, color=(224, 255, 255))
        else:
            for landmark in landmarks:
                frame = mesh(frame, landmark, self.triangles)
        return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='images/4.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.7, type=float, help='class confidence')
    parser.add_argument("--mode", type=str, default='mesh', choices=['pose', 'sparse', 'dense', 'mesh'])
    args = parser.parse_args()

    detect_net = Detect_Face(confThreshold=args.confThreshold)
    mesh_net = Face_Mesh(args.mode)
    srcimg = cv2.imread(args.imgpath)
    result = detect_net.detect(srcimg)
    # srcimg = detect_net.drawPred(srcimg, *result)
    landmarks, Rs = mesh_net.detect(srcimg, result[0])
    dstimg = srcimg.copy()
    dstimg = mesh_net.drawPred(dstimg, landmarks, Rs)

    winName = 'Deep learning Face Mesh in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()