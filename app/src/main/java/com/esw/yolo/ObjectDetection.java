// ---------------------------------------------------------------------
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.esw.yolo;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.osgi.OpenCVNativeLoader;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.stream.Collectors;

public class ObjectDetection  implements AutoCloseable {
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final List<String> labelList;
    private final int[] inputShape;
    private long preprocessingTime;
    private long inferenceTime;
    private long postprocessingTime;

    // --- MODIFIED: START ---
    // Removed old output shape variables
    // private final int[] outputBoxesShape;
    // private final int[] outputScoresShape;
    // private final int[] outputClassIdxShape;
    // private final boolean outputClassIs32bit;
    // private final int numBoxes;

    // Added new variables for single tensor output
    private final int numClasses;
    private final int outputShapeFeatures;
    private final int outputShapeProposals;
    private final float[][] postProcessOutput; // Re-usable buffer for raw output
    private final float[][] outputBoxes;       // Re-usable buffer for [x0,y0,x1,y1] boxes
    private final float[] outputScores;      // Re-usable buffer for scores
    private final int[] outputClasses;     // Re-usable buffer for class indices
    // --- MODIFIED: END ---


    // Re-usable memory
    private final ByteBuffer inputByteBuffer;
    private final float[] inputFloatArray;
    private final Mat inputMatAbgr;
    private final Mat inputMatRgb;

    private final static float INVALID_ANCHOR = -10000.0f;


    /**
     * Create an Object Detector from the given model.
     * Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context App context.
     * @param modelPath Model path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public ObjectDetection(Context context,
                           String modelPath,
                           String labelsPath,
                           TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Initialize OpenCV
        new OpenCVNativeLoader().init();

        // Load labels
        try (BufferedReader labelsFile = new BufferedReader(new InputStreamReader(context.getAssets().open(labelsPath)))) {
            labelList = labelsFile.lines().collect(Collectors.toCollection(ArrayList::new));
        }
        // --- MODIFIED: START ---
        numClasses = labelList.size();
        // --- MODIFIED: END ---

        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;

        // Validate TF Lite model fits requirements for this app
        assert tfLiteInterpreter.getInputTensorCount() == 1;

        int batchSize = 1;

        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        inputShape = inputTensor.shape();
        DataType inputType = inputTensor.dataType();
        assert inputShape.length == 4; // 4D Input Tensor: [Batch, Input Height, Input Width, Color Channels]
        assert inputShape[0] == batchSize;
        assert inputShape[3] == 3; // Input tensor should have 3 channels
        assert inputType == DataType.FLOAT32; // Requires an unquantized YOLO variant

        // --- MODIFIED: START ---
        // Validate the single output tensor, assuming [1, 4 + numClasses, numProposals]
        assert tfLiteInterpreter.getOutputTensorCount() == 1;

        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        DataType outputType = outputTensor.dataType();
        assert outputShape.length == 3; // 3D Output Tensor: [Batch, Features, Proposals]
        assert outputShape[0] == batchSize;
        // 4 box coordinates + N classes
        assert outputShape[1] == numClasses + 4;
        assert outputType == DataType.FLOAT32;

        outputShapeFeatures = outputShape[1];
        outputShapeProposals = outputShape[2];

        // Allocate re-usable memory for post-processing
        postProcessOutput = new float[outputShapeFeatures][outputShapeProposals];
        outputBoxes = new float[outputShapeProposals][4];
        outputScores = new float[outputShapeProposals];
        outputClasses = new int[outputShapeProposals];
        // --- MODIFIED: END ---

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        // Allocate re-usable memory
        inputByteBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * 3 * 4);
        inputByteBuffer.order(ByteOrder.nativeOrder());

        inputFloatArray = new float[inputHeight * inputWidth * 3];

        inputMatAbgr = new Mat(inputWidth, inputHeight, CvType.CV_8UC4);
        inputMatRgb = new Mat(inputWidth, inputHeight, CvType.CV_8UC3);
    }

    /**
     * @return neural network model input width
     */
    public int getInputWidth() {
        return inputShape[2];
    }

    /**
     * @return neural network model input height
     */
    public int getInputHeight() {
        return inputShape[1];
    }

    /**
     * Free resources used by the detector.
     */
    @Override
    public void close() {
        tfLiteInterpreter.close();
        for (Delegate delegate: tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    /**
     * @return last preprocessing time in microseconds.
     */
    public long getLastPreprocessingTime() {
        return preprocessingTime;
    }

    /**
     * @return last inference time in microseconds.
     */
    public long getLastInferenceTime() {
        return inferenceTime;
    }

    /**
     * @return last postprocessing time in microseconds.
     */
    public long getLastPostprocessingTime() {
        return postprocessingTime;
    }

    /**
     * Predicts and overlays
     * @param image Input image
     * @param sensorOrientation Sensor orientation in degrees. If input image is rotated this
     * number of degrees clockwise, the image should be upright.
     *
     * @return RGB bitmap of same size and orientation as input image, but with predictions overlay.
     */
    public void predict(Bitmap image, int sensorOrientation, ArrayList<RectangleBox> BBlist) {
        // The most common sensor orientation is 90, so we will use it for shape examples.
        // Image comes in requiring 90 degrees cw rotation to be correct;
        // its size is then 1024 x 2048 (width x height)

        long preStartTime = System.nanoTime();

        //
        // Preprocessing
        //

        // Copy input image into OpenCV Mat
        Utils.bitmapToMat(image, inputMatAbgr);

        // OpenCV loads the image as BGR, but the network expects RGB
        Imgproc.cvtColor(inputMatAbgr, inputMatRgb, Imgproc.COLOR_BGRA2RGB);

        // Rotate if necessary
        Mat correctRotInputImageRgb = new Mat();
        switch (sensorOrientation) {
            case 0:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_90_COUNTERCLOCKWISE);
                break;
            case 90:
                correctRotInputImageRgb = inputMatRgb;
                break;
            case 180:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_90_CLOCKWISE);
                break;
            case 270:
                Core.rotate(inputMatRgb, correctRotInputImageRgb, Core.ROTATE_180);
                break;
            default:
                break;
        }

        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        // Scale image to the network
        Mat scaledImage = new Mat(inputHeight, inputWidth, CvType.CV_8UC3);
        Imgproc.resize(correctRotInputImageRgb, scaledImage, scaledImage.size(), 0, 0, Imgproc.INTER_LINEAR);
        scaledImage.convertTo(scaledImage, CvType.CV_32FC3, 1 / 255f);

        //
        // TFLite inference
        //

        // Convert from OpenCV to TFLite expected ByteBuffer
        // To minimize IO overhead, we create a direct-allocated buffer in native order.

        scaledImage.get(0, 0, inputFloatArray);
        FloatBuffer inputFloatBuffer = inputByteBuffer.asFloatBuffer();
        inputFloatBuffer.put(inputFloatArray);

        long inferenceStartTime = System.nanoTime();
        preprocessingTime = inferenceStartTime - preStartTime;

        // Run inference
        ByteBuffer[] inputs = new ByteBuffer[]{inputByteBuffer};
        tfLiteInterpreter.runForMultipleInputsOutputs(inputs, new HashMap<>());

        //
        // Postprocessing
        //

        long postStartTime = System.nanoTime();
        inferenceTime = postStartTime - inferenceStartTime;


        // --- MODIFIED: START ---
        // Extract the single output tensor
        ByteBuffer outputBuffer = tfLiteInterpreter.getOutputTensor(0).asReadOnlyBuffer();
        outputBuffer.order(ByteOrder.nativeOrder());
        outputBuffer.rewind();

        // Copy buffer to our 2D float array
        for (int i = 0; i < outputShapeFeatures; i++) {
            for (int j = 0; j < outputShapeProposals; j++) {
                postProcessOutput[i][j] = outputBuffer.getFloat();
            }
        }

        // De-multiplex the single output tensor into boxes, scores, and classes
        // This logic is from PostProcessUtils.java
        for (int i = 0; i < outputShapeProposals; i++) {
            float maxScore = -Float.MAX_VALUE;
            int classIndex = -1;

            // Find the class with the highest score
            for (int c = 0; c < numClasses; c++) {
                float score = postProcessOutput[c + 4][i];
                if (score > maxScore) {
                    maxScore = score;
                    classIndex = c;
                }
            }

            outputScores[i] = maxScore;
            outputClasses[i] = classIndex;

            // Get box coordinates (cx, cy, w, h)
            float dx = postProcessOutput[0][i];
            float dy = postProcessOutput[1][i];
            float dw = postProcessOutput[2][i];
            float dh = postProcessOutput[3][i];

            // Convert [cx, cy, w, h] to [x0, y0, x1, y1]
            // This is what the original NMS and rotation logic expects
            outputBoxes[i][0] = dx - dw / 2; // x0
            outputBoxes[i][1] = dy - dh / 2; // y0
            outputBoxes[i][2] = dx + dw / 2; // x1
            outputBoxes[i][3] = dy + dh / 2; // y1
        }

        // Use the new `outputShapeProposals` instead of `numBoxes`
        float[][] updatedBoxes = new float[outputShapeProposals][4];
        for (int i = 0; i < outputShapeProposals; i++) {
            // Use the new `outputScores` array and original 0.2f threshold
            if (outputScores[i] >= 0.2) {
                // Use the new `outputBoxes` array
                float x0 = outputBoxes[i][0];
                float y0 = outputBoxes[i][1];
                float x1 = outputBoxes[i][2];
                float y1 = outputBoxes[i][3];

                // This sensor orientation logic now works as-is
                switch (sensorOrientation) {
                    case 0:
                        updatedBoxes[i][0] = inputHeight - y1;
                        updatedBoxes[i][1] = x0;
                        updatedBoxes[i][2] = inputHeight - y0;
                        updatedBoxes[i][3] = x1;
                        break;
                    case 90:
                        updatedBoxes[i][0] = x0;
                        updatedBoxes[i][1] = y0;
                        updatedBoxes[i][2] = x1;
                        updatedBoxes[i][3] = y1;
                        break;
                    case 180:
                        updatedBoxes[i][0] = y0;
                        updatedBoxes[i][1] = inputWidth - x1;
                        updatedBoxes[i][2] = y1;
                        updatedBoxes[i][3] = inputWidth - x0;
                        break;
                    case 270:
                        updatedBoxes[i][0] = inputWidth - x1;
                        updatedBoxes[i][1] = inputHeight - y1;
                        updatedBoxes[i][2] = inputWidth - x0;
                        updatedBoxes[i][3] = inputHeight - y0;
                        break;
                    default:
                        break;
                }
            } else {
                // Use the new `outputScores` array
                outputScores[i] = INVALID_ANCHOR;
            }
        }

        NMS nms = new NMS();
        // Pass the new `outputScores` array to NMS
        int[] result_indices = nms.nmsScoreFilter(updatedBoxes, outputScores, 20, 0.2f);
        // --- MODIFIED: END ---


        float scaleHeight = (float) image.getHeight() / getInputHeight();
        float scaleWidth = (float) image.getWidth() / getInputWidth();

        for (int index : result_indices) {
            if (index == 0) {
                continue;
            }

            float[] temp_boxes = updatedBoxes[index];
            RectangleBox tempbox = new RectangleBox();
            tempbox.left = temp_boxes[0] * scaleWidth;
            tempbox.bottom = temp_boxes[1] * scaleHeight;
            tempbox.right = temp_boxes[2] * scaleWidth;
            tempbox.top = temp_boxes[3] * scaleHeight;
            // --- MODIFIED: START ---
            // Use new scores and class arrays
            tempbox.confidence = outputScores[index];
            tempbox.classIdx = outputClasses[index];
            tempbox.label = labelList.get(outputClasses[index] % labelList.size());
            // --- MODIFIED: END ---
            BBlist.add(tempbox);
        }
        long endTime = System.nanoTime();
        postprocessingTime = endTime - postStartTime;
    }

    public class NMS {
        private float computeOverlapAreaRate(float[] anchor1, float[] anchor2){

            float xx1 = Math.max(anchor1[0], anchor2[0]);
            float yy1 = Math.max(anchor1[1], anchor2[1]);
            float xx2 = Math.min(anchor1[2], anchor2[2]);
            float yy2 = Math.min(anchor1[3], anchor2[3]);

            float w = xx2 - xx1 + 1;
            float h = yy2 - yy1 + 1;
            if(w<0||h<0){
                return 0;
            }

            float inter = w * h;

            float anchor1_area1 = (anchor1[2] - anchor1[0] + 1)*(anchor1[3] - anchor1[1] + 1);
            float anchor2_area1 = (anchor2[2] - anchor2[0] + 1)*(anchor2[3] - anchor1[1] + 1);

            return inter / (anchor1_area1 + anchor2_area1 - inter);
        }

        public  int[] nmsScoreFilter(float[][] anchors, float[] score, int topN, float thresh){
            int length = anchors.length;
            int count = 0;

            for(int i=0;i<length;i++){
                if(score[i]==INVALID_ANCHOR){
                    continue;
                }
                if (++count >= topN) {
                    break;
                }
                for(int j=i+1;j<length;j++){
                    if(score[j]!=INVALID_ANCHOR) {
                        if (computeOverlapAreaRate(anchors[i], anchors[j]) > thresh) {
                            score[j] = INVALID_ANCHOR;
                        }
                    }
                }
            }
            int outputIndex[] = new int[count];
            int j = 0;
            for(int i=0;i<length && count>0;i++){
                if(score[i]!=INVALID_ANCHOR){
                    outputIndex[j++] = i;
                    count--;
                }
            }
            return outputIndex;
        }
    }
}