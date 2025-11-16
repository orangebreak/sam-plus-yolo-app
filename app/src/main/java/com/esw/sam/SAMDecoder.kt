/*
 * Copyright (C) 2025 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.esw.sam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.util.EnumSet

class SAMDecoder {
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession

    // input and output node names for the decoder
    // ONNX model
    private lateinit var maskOutputName: String
    private lateinit var scoresOutputName: String

    private lateinit var imageEmbeddingInputName: String
    private lateinit var highResFeature0InputName: String
    private lateinit var highResFeature1InputName: String
    private lateinit var pointCoordinatesInputName: String
    private lateinit var pointLabelsInputName: String
    private lateinit var maskInputName: String
    private lateinit var hasMaskInputName: String

    suspend fun init(
        modelPath: String,
        useFP16: Boolean = false,
        useXNNPack: Boolean = false,
    ) = withContext(Dispatchers.IO) {
        ortEnvironment = OrtEnvironment.getEnvironment()
        val options =
            OrtSession.SessionOptions().apply {
                if (useFP16) {
                    addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                }
                if (useXNNPack) {
                    addXnnpack(
                        mapOf(
                            "intra_op_num_threads" to "2",
                        ),
                    )
                }
            }
        ortSession = ortEnvironment.createSession(modelPath, options)
        val decoderInputNames = ortSession.inputNames.toList()
        val decoderOutputNames = ortSession.outputNames.toList()
        Log.i(SAMDecoder::class.simpleName, "Decoder input names: $decoderInputNames")
        Log.i(SAMDecoder::class.simpleName, "Decoder output names: $decoderOutputNames")
        imageEmbeddingInputName = decoderInputNames[0]
        highResFeature0InputName = decoderInputNames[1]
        highResFeature1InputName = decoderInputNames[2]
        pointCoordinatesInputName = decoderInputNames[3]
        pointLabelsInputName = decoderInputNames[4]
        maskInputName = decoderInputNames[5]
        hasMaskInputName = decoderInputNames[6]

        maskOutputName = decoderOutputNames[0]
        scoresOutputName = decoderOutputNames[1]
    }

    suspend fun execute(
        encoderResults: SAMEncoder.SAMEncoderResults,
        pointCoordinates: FloatBuffer,
        pointLabels: FloatBuffer,
        numLabels: Long,
        numPoints: Long,
        inputImage: Bitmap,
    ): Pair<Bitmap, Bitmap> =
        withContext(Dispatchers.Default) {
            val imgHeight = inputImage.height
            val imgWidth = inputImage.width

            val imageEmbeddingTensor =
                OnnxTensor.createTensor(
                    ortEnvironment,
                    encoderResults.imageEmbedding,
                    longArrayOf(1, 256, 64, 64),
                )
            val highResFeature0Tensor =
                OnnxTensor.createTensor(
                    ortEnvironment,
                    encoderResults.highResFeature0,
                    longArrayOf(1, 32, 256, 256),
                )
            val highResFeature1Tensor =
                OnnxTensor.createTensor(
                    ortEnvironment,
                    encoderResults.highResFeature1,
                    longArrayOf(1, 64, 128, 128),
                )
            val hasMaskTensor =
                OnnxTensor.createTensor(
                    ortEnvironment,
                    FloatBuffer.wrap(floatArrayOf(0.0f)),
                    longArrayOf(1),
                )
            val origImageSizeTensor =
                OnnxTensor.createTensor(
                    ortEnvironment,
                    IntBuffer.wrap(intArrayOf(imgHeight, imgWidth)),
                    longArrayOf(2),
                )

            // Create a single mutable bitmap from the input image. This will be our canvas.
            val viewBitmap = inputImage.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(viewBitmap)
            val paint = Paint()

            // this will be the bitmap to export as a file
            val maskBitmap = Bitmap.createBitmap(imgWidth, imgHeight, Bitmap.Config.ARGB_8888)
            val maskBitmapPixels = IntArray(imgWidth * imgHeight, { _ -> Color.BLACK })


            // 1. Define a list of colors for the masks. You can add more colors here.
            val colors = listOf(
                Color.YELLOW, Color.CYAN, Color.GREEN, Color.MAGENTA, Color.RED,
                Color.WHITE, Color.BLUE, Color.BLACK
            )

            val batchSize = 10
            for (labelIndex in 0 until numLabels.toInt() step batchSize) {
                val batchEndIndex = (labelIndex + batchSize).coerceAtMost(numLabels.toInt())
                val currentBatchSize = batchEndIndex - labelIndex

                val pointCoordinatesSlice = pointCoordinates.slice()
                pointCoordinatesSlice.position(labelIndex * numPoints.toInt() * 2)
                pointCoordinatesSlice.limit((labelIndex + currentBatchSize) * numPoints.toInt() * 2)

                val pointLabelsSlice = pointLabels.slice()
                pointLabelsSlice.position(labelIndex * numPoints.toInt())
                pointLabelsSlice.limit((labelIndex + currentBatchSize) * numPoints.toInt())

                OnnxTensor.createTensor(
                    ortEnvironment,
                    pointCoordinatesSlice,
                    longArrayOf(currentBatchSize.toLong(), numPoints, 2),
                ).use { pointCoordinatesTensor ->
                    OnnxTensor.createTensor(
                        ortEnvironment,
                        pointLabelsSlice,
                        longArrayOf(currentBatchSize.toLong(), numPoints),
                    ).use { pointLabelsTensor ->
                        OnnxTensor.createTensor(
                            ortEnvironment,
                            FloatBuffer.wrap(FloatArray(currentBatchSize * 1 * 256 * 256) { 0f }),
                            longArrayOf(currentBatchSize.toLong(), 1, 256, 256),
                        ).use { maskTensor ->

                            val outputs =
                                ortSession.run(
                                    mapOf(
                                        imageEmbeddingInputName to imageEmbeddingTensor,
                                        highResFeature0InputName to highResFeature0Tensor,
                                        highResFeature1InputName to highResFeature1Tensor,
                                        pointCoordinatesInputName to pointCoordinatesTensor,
                                        pointLabelsInputName to pointLabelsTensor,
                                        maskInputName to maskTensor,
                                        hasMaskInputName to hasMaskTensor,
                                        "orig_im_size" to origImageSizeTensor,
                                    ),
                                )
                            outputs.use {
                                val mask = (outputs[maskOutputName].get() as OnnxTensor).floatBuffer
                                val scores =
                                    (outputs[scoresOutputName].get() as OnnxTensor).floatBuffer
                                val numPredictedMasks = scores.capacity() / currentBatchSize

                                for (batchItemIndex in 0 until currentBatchSize) {
                                    val currentLabel = labelIndex + batchItemIndex
                                    val colorForLabel = colors[currentLabel % colors.size]

                                    val semiTransparentColor = Color.argb(
                                        128,
                                        Color.red(colorForLabel),
                                        Color.green(colorForLabel),
                                        Color.blue(colorForLabel)
                                    )
                                    paint.color = semiTransparentColor

                                    val maskStartIndex =
                                        batchItemIndex * numPredictedMasks * imgHeight * imgWidth

                                    // Instead of slow setPixel, collect all mask points and draw them at once.
                                    val pointCloud = mutableListOf<Float>()
                                    for (i in 0..<imgHeight) {
                                        for (j in 0..<imgWidth) {
                                            // If the mask value for this pixel is > 0, it's part of an object.
                                            if (mask[maskStartIndex + j + i * imgWidth] > 0) {
                                                pointCloud.add(j.toFloat())
                                                pointCloud.add(i.toFloat())
                                                maskBitmapPixels[j + i * imgWidth] = Color.WHITE
                                            }
                                        }
                                    }
                                    canvas.drawPoints(pointCloud.toFloatArray(), paint)
                                }
                            }
                        }
                    }
                }
            }
            maskBitmap.setPixels(maskBitmapPixels, 0, imgWidth, 0, 0, imgWidth, imgHeight)

            imageEmbeddingTensor.close()
            highResFeature0Tensor.close()
            highResFeature1Tensor.close()
            hasMaskTensor.close()
            origImageSizeTensor.close()
            return@withContext Pair(viewBitmap, maskBitmap)
        }

    private fun saveBitmap(
        context: Context,
        image: Bitmap,
        name: String,
    ) {
        val fileOutputStream = FileOutputStream(File(context.filesDir.absolutePath + "/$name.png"))
        image.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream)
    }
}
