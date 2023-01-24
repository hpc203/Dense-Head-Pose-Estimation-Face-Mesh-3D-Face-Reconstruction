# Dense-Head-Pose-Estimation-Face-Mesh-3D-Face-Reconstruction
使用ONNXRuntime部署3D人脸重建，人脸Mesh，人头姿势估计，包含C++和Python两个版本的程序。

在本套程序里，包括人脸检测，人脸关键点检测，人头姿势估计，人脸网格Mesh生成，
3D人脸重建。其中3D人脸重建是本套程序里的重中之重。
本套程序对应的paper是ECCV2020里的一篇文章《Towards Fast, Accurate and Stable 3D Dense Face Alignment》

本套程序有3个onnx模型文件，链接：https://pan.baidu.com/s/1I2VzpDfrTuSfa9jJnMT7zA 
提取码：56ur


opencv的dnn模块读取RFB-320_240x320_post.onnx文件失败了，读取sparse_face_Nx3x120x120.onnx
和dense_face_Nx3x120x120.onnx是正常的。RFB-320_240x320_post.onnx文件是人脸检测模型，
有兴趣使用opencv部署的开发者，可以换用一个opencv部署的人脸检测模型到本套程序里，这样在本套
程序里，全流程都是使用opencv部署。
