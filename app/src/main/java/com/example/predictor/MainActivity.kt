package com.example.predictor

import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.res.ResourcesCompat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.Tensor.allocateByteBuffer
import org.pytorch.Tensor.fromBlob
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() {

    lateinit var module: Module

    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val predButton:Button = findViewById<Button>(R.id.predict_button)
        predButton.setOnClickListener{
            predict()
        }

        module = Module.load(assetFilePath(this, "mymodel.pt"))
    }
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }

    private fun predict() {

        val resizedBitmap:Bitmap = Bitmap.createScaledBitmap(
            MyCanvasView.extraBitmap,
            28,
            28,
            true
        )

        val byteBuffer:ByteBuffer = allocateByteBuffer(784*4)
        val pixels = IntArray(28 * 28)
        resizedBitmap.getPixels(pixels, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        var i = 0
        val array = FloatArray(28*28)

        for (pixelValue in pixels) {

            val r = (pixelValue shr 16 and 0xFF)
            val g = (pixelValue shr 8 and 0xFF)
            val b = (pixelValue and 0xFF)

            // Convert RGB to grayscale and normalize pixel value to [0..1]
            val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
            byteBuffer.putFloat(normalizedPixelValue)
            array[i] = normalizedPixelValue
            i = i + 1

        }

        val inputTensor = Tensor.fromBlob(array, longArrayOf(1, 1, 28, 28))

        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray

        var maxScore: Float = -9999F
        var maxScoreIdx = -1
        var maxSecondScore: Float = -999F
        var maxSecondScoreIdx = -1


        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxSecondScore = maxScore
                maxSecondScoreIdx = maxScoreIdx
                maxScore = scores[i]
                maxScoreIdx = i
            }
        }

        val predictedValue:TextView = findViewById(R.id.predictedValue_text)
        predictedValue.text = maxScoreIdx.toString()

        MyCanvasView.extraCanvas.drawColor(ResourcesCompat.getColor(resources, R.color.colorBackground, null))
//        Toast.makeText(applicationContext,maxScoreIdx.toString(),Toast.LENGTH_SHORT).show()
    }
}
