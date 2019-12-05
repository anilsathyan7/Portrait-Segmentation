/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.tflitecamerademo2;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC3;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2Lab;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_Lab2BGR;

/** This segmentor works with the float mobile-unet model. */
public class ImageSegmentorFloatMobileUnet extends ImageSegmentor {

  /** The mobile net requires additional normalization of the used input. */
  private static final float IMAGE_MEAN = 127.5f;

  private static final float IMAGE_STD =127.5f;

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][] segmap = null;

  int opsize=128;

  Bitmap resbmp = Bitmap.createBitmap(513,513, Bitmap.Config.ARGB_8888);
  Bitmap tmpbmp = Bitmap.createBitmap(448,448, Bitmap.Config.ARGB_8888);

  /**
   * Initializes an {@code ImageSegmentorFloatMobileUnet}.
   *
   * @param activity
   */
  ImageSegmentorFloatMobileUnet(Activity activity) throws IOException {
    super(activity);
      segmap = new float[1][opsize*opsize];
  }

  @Override
  protected String getModelPath() {

    return "deconv_fin_munet.tflite";
  }


  @Override
  protected int getImageSizeX() {
    return 128;
  }

  @Override
  protected int getImageSizeY() {
    return 128;
  }

  @Override
  protected int getNumBytesPerChannel() {
    return 4; // Float.SIZE / Byte.SIZE;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN)/ IMAGE_STD);
    imgData.putFloat((((pixelValue >> 8) & 0xFF)  - IMAGE_MEAN)/ IMAGE_STD);
    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN)/ IMAGE_STD);
  }

  @Override
  protected void runInference( ) {

    tflite.run(imgData, segmap);
  }

  @Override
  public Bitmap imageblend(Bitmap fg_bmp, Bitmap bg_bmp, Boolean harmonize){

    Mat bgcpy = new Mat(448,448,CV_8UC3);

    Mat fg = new Mat(448,448,CV_8UC3);
    Mat bg = new Mat(448,448,CV_8UC3);
    Mat mskmat = new Mat (opsize, opsize, CV_32F);
    Mat invmskmat = new Mat(448, 448,CV_32FC3, new Scalar(1.0,1.0,1.0));

    Mat resmat = new Mat(448,448,CV_32FC3);

    if (segmap!=null){

      mskmat.put(0,0,segmap[0]);

      Core.multiply(mskmat,new Scalar(2.0),mskmat);
      Imgproc.threshold(mskmat, mskmat,1.0,1.0,Imgproc.THRESH_TRUNC);
      Core.pow(mskmat,2.0,mskmat);

      Imgproc.resize(mskmat,mskmat, new Size(448,448));
      Imgproc.cvtColor(mskmat,mskmat,Imgproc.COLOR_GRAY2BGR);

      Utils.bitmapToMat(fg_bmp, fg);
      Utils.bitmapToMat(bg_bmp, bg);

      Imgproc.cvtColor(fg,fg, COLOR_BGRA2BGR);
      Imgproc.cvtColor(bg,bg, COLOR_BGRA2BGR);

      if(harmonize) {
        // Deep copy the background mat
        bg.copyTo(bgcpy);
        fg = color_transfer(bgcpy, fg);
      }

      Log.d("FG", String.valueOf(fg.type()));
      Log.d("BG", String.valueOf(bg.type()));

      fg.convertTo(fg,CV_32FC3,1.0/255.0);
      bg.convertTo(bg,CV_32FC3,1.0/255.0);

      Core.subtract(invmskmat,mskmat,invmskmat);
      Core.multiply(fg,mskmat,fg);
      Core.multiply(bg,invmskmat, bg);
      Core.add(fg,bg,resmat);

      resmat.convertTo(resmat,CV_8UC3,255);
      Imgproc.resize(resmat, resmat, new Size(513,513));

      Utils.matToBitmap(resmat,resbmp);

    }
    return resbmp;
  }


  @Override
  public Bitmap videobokeh(Bitmap fg_bmp){


    Mat fg = new Mat(448,448,CV_8UC3);
    Mat bg = new Mat(448,448,CV_8UC3);
    Mat mskmat = new Mat (opsize, opsize, CV_32F);
    Mat invmskmat = new Mat(448, 448,CV_32FC3, new Scalar(1.0,1.0,1.0));

    Mat resmat = new Mat(448,448,CV_32FC3);

    if (segmap!=null){

      mskmat.put(0,0,segmap[0]);

      // Prepare the mask
      Core.multiply(mskmat,new Scalar(2.0),mskmat);
      Imgproc.threshold(mskmat, mskmat,1.0,1.0,Imgproc.THRESH_TRUNC);
      Core.pow(mskmat,2.0,mskmat);

      Imgproc.resize(mskmat,mskmat, new Size(448,448));
      Imgproc.cvtColor(mskmat,mskmat,Imgproc.COLOR_GRAY2BGR);

      Utils.bitmapToMat(fg_bmp, fg);
      Utils.bitmapToMat(fg_bmp, bg);

      Imgproc.cvtColor(fg,fg, COLOR_BGRA2BGR);
      Imgproc.cvtColor(bg,bg, COLOR_BGRA2BGR);

      // Blur the mask
      Imgproc.resize(bg,bg, new Size(224,224));
      Imgproc.blur(bg,bg,new Size(17,17));
      Imgproc.resize(bg,bg, new Size(448,448));

      fg.convertTo(fg,CV_32FC3,1.0/255.0);
      bg.convertTo(bg,CV_32FC3,1.0/255.0);

      // Alpha blend fg with bg, using the mask
      Core.subtract(invmskmat,mskmat,invmskmat);
      Core.multiply(fg,mskmat,fg);
      Core.multiply(bg,invmskmat, bg);
      Core.add(fg,bg,resmat);

      resmat.convertTo(resmat,CV_8UC3,255);
      Imgproc.resize(resmat, resmat, new Size(513,513));

      Utils.matToBitmap(resmat,resbmp);

    }
    return resbmp;
  }

  public Mat color_transfer(Mat src, Mat tgt){


    long startTimeb = SystemClock.uptimeMillis();

    Imgproc.cvtColor(src,src,COLOR_BGR2Lab,CV_32FC3);
    Imgproc.cvtColor(tgt,tgt,COLOR_BGR2Lab,CV_32FC3);


    //compute color statistics for the source image
    List<Mat> labsrc = new ArrayList<Mat>(3);
    Core.split(src, labsrc);


    MatOfDouble meansrc= new MatOfDouble();
    MatOfDouble stdsrc= new MatOfDouble();

    Core.meanStdDev(src, meansrc, stdsrc);

    Log.d("meansrc",meansrc.dump() );

    Log.d("meanval1", String.valueOf(meansrc.get(0,0)[0]));
    Log.d("meanval2", String.valueOf(meansrc.get(1,0)[0]));
    Log.d("meanval3", String.valueOf(meansrc.get(2,0)[0]));

    double lMeanSrc = meansrc.get(0,0)[0];
    double aMeanSrc = meansrc.get(1,0)[0];
    double bMeanSrc = meansrc.get(2,0)[0];

    double lStdSrc = stdsrc.get(0,0)[0];
    double aStdSrc = stdsrc.get(1,0)[0];
    double bStdSrc = stdsrc.get(2,0)[0];


    //compute color statistics for the target image
    List<Mat> labtgt = new ArrayList<Mat>(3);
    Core.split(tgt, labtgt);

    MatOfDouble meantgt = new MatOfDouble();
    MatOfDouble stdtgt = new MatOfDouble();

    Core.meanStdDev(src, meantgt, stdtgt);

    double lMeanTgt = meantgt.get(0,0)[0];
    double aMeanTgt = meantgt.get(1,0)[0];
    double bMeanTgt = meantgt.get(2,0)[0];


    double lStdTgt = stdtgt.get(0,0)[0];
    double aStdTgt = stdtgt.get(1,0)[0];
    double bStdTgt = stdtgt.get(2,0)[0];


    // subtract the means from the target image
    Core.subtract(labtgt.get(0),new Scalar(lMeanTgt),labtgt.get(0));
    Core.subtract(labtgt.get(1),new Scalar(aMeanTgt),labtgt.get(1));
    Core.subtract(labtgt.get(2),new Scalar(bMeanTgt),labtgt.get(2));

    // scale by the standard deviations
    Core.multiply(labtgt.get(0),new Scalar((lStdTgt/lStdSrc)),labtgt.get(0));
    Core.multiply(labtgt.get(1),new Scalar((aStdTgt/aStdSrc)),labtgt.get(1));
    Core.multiply(labtgt.get(2),new Scalar((bStdTgt/bStdSrc)),labtgt.get(2));

    // add in the source mean
    Core.add(labtgt.get(0),new Scalar(lMeanSrc),labtgt.get(0));
    Core.add(labtgt.get(1),new Scalar(aMeanSrc),labtgt.get(1));
    Core.add(labtgt.get(2),new Scalar(bMeanSrc),labtgt.get(2));

    Core.merge(labtgt,tgt);
    Log.d("tgt_size", tgt.size().toString());
    Log.d("tgt_type", String.valueOf(tgt.type()));

    Imgproc.cvtColor(tgt,tgt,COLOR_Lab2BGR);
    tgt.convertTo(tgt,CV_8UC3);

    long endTimeb = SystemClock.uptimeMillis();

    Log.d("color transfer", Long.toString(endTimeb - startTimeb));

    return tgt;
  }


  public Bitmap smoothblend(Bitmap fg_bmp, Bitmap bg_bmp, Boolean harmozize){


    Mat mskmat = new Mat (opsize, opsize, CV_32F);
    Mat resmat = new Mat(448,448,CV_32FC3);

    if (segmap!=null){

      //Binarize the mask
      mskmat.put(0,0,segmap[0]);
      Imgproc.threshold(mskmat,mskmat,0.5,1.0,Imgproc.THRESH_BINARY);

      //Gaussian blur
      mskmat.convertTo(mskmat,CV_8UC3,255);
      Imgproc.GaussianBlur(mskmat,mskmat,new Size(7,7),0);

      // Resize the mask
      Imgproc.resize(mskmat,mskmat, new Size(448,448));

      //Apply smooth and alpha blend filters
      Utils.matToBitmap(mskmat,tmpbmp);
      tmpbmp=Camera2BasicFragment.renderSmooth(bg_bmp,fg_bmp,tmpbmp);
      Utils.bitmapToMat(tmpbmp, resmat);

      // Resize the output
      Imgproc.resize(resmat, resmat, new Size(513,513));

      Utils.matToBitmap(resmat,resbmp);

    }
    return resbmp;
  }

}
