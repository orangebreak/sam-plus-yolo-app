package com.ultralytics.yolo.predict.detect;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;



import com.ultralytics.yolo.ImageProcessing;
import com.ultralytics.yolo.models.LocalYoloModel;
import com.ultralytics.yolo.models.YoloModel;
import com.ultralytics.yolo.predict.PredictorException;


import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;
import org.tensorflow.lite.Delegate;
import android.util.Pair;
import java.security.NoSuchAlgorithmException;

public class TfliteDetector extends Detector {

    private Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;

    private static final int NUM_BYTES_PER_CHANNEL = 4;

    private final Bitmap pendingBitmapFrame;
    private int numClasses;
    private double confidenceThreshold = 0.25f;
    private double iouThreshold = 0.45f;
    private int numItemsThreshold = 30;
    private Interpreter interpreter;
    private Object[] inputArray;
    private int outputShape2;
    private int outputShape3;
    private float[][] output;

    private Map<Integer, Object> outputMap;
    private ObjectDetectionResultCallback objectDetectionResultCallback;
    private FloatResultCallback inferenceTimeCallback;
    private FloatResultCallback fpsRateCallback;

    private static final float Nanos2Millis = 1 / 1e6f;
    public class Stats {
        private float imageSetupTime;
        private float inferenceTime;
        private float postProcessTime;

    }
    public Stats stats;

    private ByteBuffer imgData;
    private int[] intValues;
    private byte[] bytes;

    private ByteBuffer outData;

    private ByteBuffer pixelBuffer;

    private ImageProcessing ip;


    public TfliteDetector(Context context) {
        super(context);

        pendingBitmapFrame = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);


        stats = new Stats();

        imgData = null;
        intValues = null;
        outData = null;
        pixelBuffer = null;

        ip = new ImageProcessing();

    }

    @Override
    public void loadModel(YoloModel yoloModel, boolean useGpu) throws Exception {
        if (!(yoloModel instanceof LocalYoloModel)) {
            throw new PredictorException("Only LocalYoloModel is supported for this detector.");
        }

        final LocalYoloModel localYoloModel = (LocalYoloModel) yoloModel;

        if (localYoloModel.modelPath == null || localYoloModel.modelPath.isEmpty() ||
                localYoloModel.metadataPath == null || localYoloModel.metadataPath.isEmpty()) {
            throw new PredictorException("Model or metadata path is empty.");
        }

        final AssetManager assetManager = context.getAssets();
        loadLabels(assetManager, localYoloModel.metadataPath);
        numClasses = labels.size();

        try {
            // 1. Load model file AND hash using the TFLiteHelpers class
            Pair<MappedByteBuffer, String> modelData = TFLiteHelpers.loadModelFile(assetManager, localYoloModel.modelPath);
            MappedByteBuffer modelFile = modelData.first;
            String modelHash = modelData.second;

            // 2. Get the default NPU-first delegate priority order
            // We ignore the 'useGpu' parameter and use the AI Hub default, which tries NPU first.
            TFLiteHelpers.DelegateType[][] delegatePriorityOrder = AIHubDefaults.delegatePriorityOrder;

            // 3. Create the Interpreter and Delegates
            Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                    modelFile,
                    delegatePriorityOrder,
                    AIHubDefaults.numCPUThreads, // Use default CPU threads from AIHubDefaults
                    context.getApplicationInfo().nativeLibraryDir,
                    context.getCacheDir().getAbsolutePath(),
                    modelHash
            );

            // 4. Store the new interpreter and delegates
            this.interpreter = iResult.first;
            this.tfLiteDelegateStore = iResult.second;

            // 5. This logic is from your original initDelegate and is still needed
            int[] outputShape = interpreter.getOutputTensor(0).shape();
            outputShape2 = outputShape[1];
            outputShape3 = outputShape[2];
            output = new float[outputShape2][outputShape3];

        } catch (IOException | NoSuchAlgorithmException e) {
            throw new PredictorException("Error loading model or creating interpreter: " + e.getMessage());
        } catch (Exception e) {
            throw new PredictorException("Error initializing model: " + e.getMessage());
        }
    }

    // Add this method to TfliteDetector.java

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
        if (tfLiteDelegateStore != null) {
            for (Delegate delegate : tfLiteDelegateStore.values()) {
                delegate.close();
            }
            tfLiteDelegateStore.clear();
        }
    }

    public Bitmap preprocess(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        return resizedBitmap;
    }



    @Override
    public ArrayList<DetectedObject> predict(Bitmap bitmap) {
        try {

            long startTime = System.nanoTime();
            // setInput(bitmap);
            setInputOptim(bitmap);
            stats.imageSetupTime = (System.nanoTime() - startTime) * Nanos2Millis;

            return runInference();
        } catch (Exception e) {
            return new ArrayList<>(); //float[0][];
        }
    }

    @Override
    public void setConfidenceThreshold(float confidence) {
        this.confidenceThreshold = confidence;
    }

    @Override
    public void setIouThreshold(float iou) {
        this.iouThreshold = iou;
    }

    @Override
    public void setNumItemsThreshold(int numItems) {
        this.numItemsThreshold = numItems;
    }

    @Override
    public void setObjectDetectionResultCallback(ObjectDetectionResultCallback callback) {
        objectDetectionResultCallback = callback;
    }

    @Override
    public void setInferenceTimeCallback(FloatResultCallback callback) {
        inferenceTimeCallback = callback;
    }

    @Override
    public void setFpsRateCallback(FloatResultCallback callback) {
        fpsRateCallback = callback;
    }





    private void setInput(Bitmap resizedbitmap) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * NUM_BYTES_PER_CHANNEL);
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

        resizedbitmap.getPixels(intValues, 0, resizedbitmap.getWidth(), 0, 0, resizedbitmap.getWidth(), resizedbitmap.getHeight());

        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                float r = (((pixelValue >> 16) & 0xFF)) / 255.0f;
                float g = (((pixelValue >> 8) & 0xFF)) / 255.0f;
                float b = ((pixelValue & 0xFF)) / 255.0f;
                imgData.putFloat(r);
                imgData.putFloat(g);
                imgData.putFloat(b);
            }
        }
        this.inputArray = new Object[]{imgData};
        this.outputMap = new HashMap<>();
        ByteBuffer outData = ByteBuffer.allocateDirect(outputShape2 * outputShape3 * NUM_BYTES_PER_CHANNEL);
        outData.order(ByteOrder.nativeOrder());
        outData.rewind();
        outputMap.put(0, outData);
    }


    private void setInputOptim(Bitmap bitmap) {

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        if (intValues == null) {
            intValues = new int[INPUT_SIZE * INPUT_SIZE];
            bytes = new byte[width * height * 3];

            int batchSize = 1;
            int RGB = 3;
            int numPixels = INPUT_SIZE * INPUT_SIZE;
            int bufferSize = batchSize * RGB * numPixels * NUM_BYTES_PER_CHANNEL;
            imgData = ByteBuffer.allocateDirect(bufferSize);
            imgData.order(ByteOrder.nativeOrder());

            outData = ByteBuffer.allocateDirect(outputShape2 * outputShape3 * NUM_BYTES_PER_CHANNEL);
            outData.order(ByteOrder.nativeOrder());

        }
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);

        ip.argb2yolo(
                intValues,
                imgData,
                width,
                height
        );

        imgData.rewind();

        this.inputArray = new Object[]{imgData};
        this.outputMap = new HashMap<>();
        outData.rewind();
        outputMap.put(0, outData);

    }



    private ArrayList<DetectedObject> runInference() {
        if (interpreter != null) {

            long startTime = System.nanoTime();

            interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

            stats.inferenceTime = (System.nanoTime() - startTime) * Nanos2Millis;

            ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
            if (byteBuffer != null) {
                byteBuffer.rewind();

                for (int j = 0; j < outputShape2; ++j) {
                    for (int k = 0; k < outputShape3; ++k) {
                        output[j][k] = byteBuffer.getFloat();
                    }
                }


                startTime = System.nanoTime();

                ArrayList<DetectedObject> ret = PostProcessUtils.postprocess(
                        output,
                        outputShape3,
                        outputShape2,
                        (float) confidenceThreshold,
                        (float) iouThreshold,
                        numItemsThreshold,
                        numClasses,
                        labels
                );


                stats.postProcessTime = (System.nanoTime() - startTime) * Nanos2Millis;

                return ret;

            }
        }
        //return new float[0][];
        return new ArrayList<>();
    }




}
