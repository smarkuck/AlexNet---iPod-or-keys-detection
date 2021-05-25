/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.alexnet_ipodorkeysdetection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.media.ImageReader.OnImageAvailableListener;
import android.util.Size;
import android.widget.Toast;
import com.example.alexnet_ipodorkeysdetection.env.Logger;
import com.example.alexnet_ipodorkeysdetection.tflite.Classifier;
import java.io.IOException;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private Bitmap rgbFrameBitmap = null;
    private Integer sensorOrientation;
    private Classifier classifier;

    @Override
    protected int getLayoutId() {
        return R.layout.tfe_ic_camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        recreateClassifier();
        if (classifier == null) {
            LOGGER.e("No classifier on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    }

    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        runInBackground(() -> {
            if (classifier != null) {
                final float[] result =
                        classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
                runOnUiThread(() -> showLabel(result));
            }
            readyForNextImage();
        });
    }

    private void recreateClassifier() {
        if (classifier != null) {
            LOGGER.d("Closing classifier.");
            classifier.close();
            classifier = null;
        }
        try {
            LOGGER.d("Creating classifier");
            classifier = new Classifier(this);
        } catch (IOException | IllegalArgumentException e) {
            LOGGER.e(e, "Failed to create classifier.");
            runOnUiThread(
                    () -> Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show());
        }
    }
}
