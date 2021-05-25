package com.example.alexnet_ipodorkeysdetection.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import static java.lang.Math.min;

public class Classifier {
    public static final String TAG = "ClassifierWithSupport";

    /** The loaded TensorFlow Lite model. */

    /** Image size along the x axis. */
    private final int imageSizeX;

    /** Image size along the y axis. */
    private final int imageSizeY;

    /** Optional GPU delegate for accleration. */
    private GpuDelegate gpuDelegate;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /** Input image TensorBuffer. */
    private TensorImage inputImageBuffer;

    /** Output probability TensorBuffer. */
    private final TensorBuffer outputProbabilityBuffer;

    /** Processer to apply post processing of the output probability. */
    private final TensorProcessor probabilityProcessor;

    /** Initializes a {@code Classifier}. */
    public Classifier(Activity activity) throws IOException {
        MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
        gpuDelegate = new GpuDelegate();
        /** Options for configuring the Interpreter. */
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.addDelegate(gpuDelegate);
        tfliteOptions.setNumThreads(-1);
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeX = imageShape[1];
        imageSizeY = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Creates the post processor for the output probability.
        probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
    }

    /** Gets the name of the model file stored in Assets. */
    protected String getModelPath() {
        return "model.tflite";
    }

    /** Gets the TensorOperator to nomalize the input image in preprocessing. */
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(0.0f, 255.0f);
    }

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(0.0f, 1.0f);
    }

    /** Runs inference and returns the classification results. */
    public float[] recognizeImage(final Bitmap bitmap, int sensorOrientation) {
        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("loadImage");
        long startTimeForLoadImage = SystemClock.uptimeMillis();
        inputImageBuffer = loadImage(bitmap, sensorOrientation);
        long endTimeForLoadImage = SystemClock.uptimeMillis();
        Trace.endSection();
        Log.v(TAG, "Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

        // Runs the inference call.
        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        Log.v(TAG, "Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

        Trace.endSection();

        TensorBuffer result = probabilityProcessor.process((outputProbabilityBuffer));
        return new float[] {
                result.getFloatValue(0),
                result.getFloatValue(1),
                result.getFloatValue(2)};
    }

    /** Loads input image, and applies preprocessing. */
    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new Rot90Op(-numRotation))
                        .add(new ResizeOp(imageSizeY, imageSizeX, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();

        return imageProcessor.process(inputImageBuffer);
    }

    /** Closes the interpreter and model to release resources. */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
    }
}
